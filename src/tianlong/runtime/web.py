"""
[INPUT]: 依赖标准库 http.server / json / threading / argparse / contextlib，runtime/session 的 GameSession / TurnReport，
         runtime/suggest 的 suggestions，
         runtime/cli 的 load_dotenv / interpreter_for，runtime/webpage 的 PAGE，language/llm 的 llm_from_env / fast_llm_from_env，
         cognition/navigation 的 believed_place，scenarios 的 SCENARIOS
[OUTPUT]: 对外提供 WebGame（一局游戏的线程安全外壳：开场、回合、状态）、make_server()（本地 HTTP 服务）、
          main()（python -m tianlong.runtime.web [--port 8000]）
[POS]: runtime 的网页前端：只和 GameSession 打交道，与 CLI 平级。回合经 SSE 流式推给浏览器——主持人之声每通过闸门一句，
       页面就多一句；推送的只有玩家该看的文字（叙述、场外问答、落幕后的真相揭晓）与只凭玩家认知给出的行动建议，
       真相与 NPC 理由从不出这个进程。
       零依赖（标准库 ThreadingHTTPServer），单机单局：同一时刻只结算一个回合（锁），开场在后台先写好，
       开场与终章各只写一次，刷新页面接着玩同一局；--world 默认无量山普通人版（标题“天龙八部 · 无量山·普通人”），wuliang-duanyu 是旧版
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tianlong.cognition.navigation import believed_place
from tianlong.language.llm import fast_llm_from_env, llm_from_env
from tianlong.runtime.cli import interpreter_for, load_dotenv
from tianlong.runtime.session import GameSession, TurnReport
from tianlong.runtime.suggest import suggestions
from tianlong.runtime.webpage import PAGE
from tianlong.scenarios import SCENARIOS

log = logging.getLogger(__name__)

MAX_INPUT = 300      # 一句输入的上限（字）：比这更长的多半是粘贴错了


# ============================================================
#  一局游戏的外壳：开场只讲一次，回合串行结算
# ============================================================


class WebGame:
    def __init__(self, factory: Callable[[], GameSession], title: str = "") -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self.title = title
        self.session = factory()
        self._opening: str | None = None
        self._epilogue: str | None = None     # 终章只写一次（有模型时是一次调用），刷新页面原样再给
        self._warm()

    def _warm(self) -> None:
        """开场在后台先写好：浏览器打开页面时（有模型时那是一次调用）多半已经就绪，不必干等。"""
        threading.Thread(target=self.state, daemon=True).start()

    def restart(self) -> None:
        with self._lock:
            self.session = self._factory()
            self._opening = self._epilogue = None
        self._warm()

    def _closing(self) -> str | None:
        if self.session.ending is None:
            return None
        if self._epilogue is None:
            self._epilogue = self.session.epilogue()
        return self._epilogue

    def _place(self) -> str:
        s = self.session
        me = s.beliefs(s.player)
        here = believed_place(me, s.player)
        sk = me.sketch(here) if here else None
        return sk.name if sk else "某处"

    def _suggest(self) -> list[str]:
        s = self.session
        return [] if s.ending is not None else list(suggestions(s.beliefs(s.player)))

    def state(self) -> dict[str, Any]:
        """开场（只讲一次，刷新页面时原样再给）、时辰、以为自己在哪、是否已落幕。"""
        with self._lock:
            s = self.session
            if self._opening is None:
                parts = [s.scenario.setting] if s.scenario.setting and not s.resumed else []
                parts.append(s.intro())
                self._opening = "\n\n".join(p for p in parts if p)
            return {"title": self.title, "clock": s.clock(), "place": self._place(), "opening": self._opening,
                    "suggest": self._suggest(),
                    "hints": s.scenario.hints, "voice": s.llm is not None,
                    "ended": s.ending is not None, "ending": s.ending.title if s.ending else None,
                    "epilogue": self._closing()}

    def turn(self, text: str, on_text: Callable[[str], None]) -> dict[str, Any]:
        """结算一回合：叙述经 on_text 逐句交付；返回回合收尾的元数据（不含真相）。"""
        with self._lock:
            s = self.session
            report: TurnReport = s.turn(text, on_text=on_text)
            ended = report.ending is not None
            return {"clock": s.clock(), "place": self._place(), "kind": report.kind.value, "advanced": report.advanced,
                    "narration": report.narration, "first_text_ms": report.first_text_ms,
                    "ended": ended, "ending": report.ending.title if ended else None, "epilogue": self._closing(),
                    "suggest": self._suggest()}


# ============================================================
#  HTTP：GET / 页面，GET /api/state 开场与状态，POST /api/turn 流式回合（SSE），POST /api/restart 重开
# ============================================================


def _handler(game: WebGame) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Tianlong/1"

        def log_message(self, fmt: str, *args: Any) -> None:     # 安静：玩家的输入不写进访问日志
            log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict[str, Any]) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(200, game.state())
            else:
                self._json(404, {"error": "not found"})

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(min(n, 16384)) if n else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/restart":
                game.restart()
                self._json(200, game.state())
                return
            if self.path != "/api/turn":
                self._json(404, {"error": "not found"})
                return
            text = str(self._body().get("text") or "").strip()[:MAX_INPUT]
            if not text:
                self._json(400, {"error": "empty"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def event(name: str, data: dict[str, Any]) -> None:
                payload = json.dumps(data, ensure_ascii=False)
                self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode())
                self.wfile.flush()

            try:
                done = game.turn(text, lambda piece: event("text", {"t": piece}))
                event("done", done)
            except (BrokenPipeError, ConnectionResetError):
                pass                                  # 玩家关了页面：回合照常结算落库，只是没人看
            except Exception as e:  # noqa: BLE001 —— 一回合出错不该拖垮整个服务
                log.exception("回合出错")
                with contextlib.suppress(OSError):
                    event("error", {"message": f"这一回合出了岔子：{type(e).__name__}"})

    return Handler


def make_server(game: WebGame, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _handler(game))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.runtime.web", description="图世界文字游戏（网页）")
    ap.add_argument("--llm", choices=["auto", "none"], default="auto", help="auto：有 GEMINI_API_KEY 则启用")
    ap.add_argument("--world", choices=sorted(SCENARIOS), default="wuliang")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    load_dotenv()
    voice = llm_from_env() if args.llm == "auto" else None
    fast = fast_llm_from_env() if args.llm == "auto" else None

    def factory() -> GameSession:
        scenario = SCENARIOS[args.world](args.seed)
        return GameSession(scenario, llm=voice, fast_llm=fast, interpreter=interpreter_for(fast or voice, scenario))

    title = {"wuliang": "天龙八部 · 无量山·普通人", "wuliang-duanyu": "天龙八部 · 无量山", "warehouse": "仓库钥匙"}.get(
        args.world, args.world)
    server = make_server(WebGame(factory, title), args.host, args.port)
    print(f"在浏览器打开 http://{args.host}:{args.port}/ 开始游戏（{'Gemini 叙述' if voice else '模板叙述'}；Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
