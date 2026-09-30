"""
[INPUT]: 依赖标准库 http.server / json / threading / argparse / contextlib，runtime/session 的 GameSession / TurnReport，
         runtime/choice_service 的冻结菜单与结构化按钮执行，
         runtime/cli 的 load_dotenv / interpreter_for，runtime/webpage 的 PAGE，language/llm 的 llm_from_env / fast_llm_from_env，
         cognition/navigation 的 believed_place，scenarios 的 SCENARIOS
[OUTPUT]: 对外提供 WebGame（一局游戏的线程安全外壳：开场、回合、状态）、make_server()（本地 HTTP 服务）、
          main()（python -m tianlong.runtime.web [--port 8000]）
[POS]: runtime 的网页前端：只和 GameSession 打交道，与 CLI 平级。回合经 SSE 流式推给浏览器——主持人之声每通过闸门一句，
       页面就多一句；推送的只有玩家该看的文字（叙述、场外问答、落幕后的真相揭晓）与只凭玩家认知给出的行动建议，
       真相与 NPC 理由从不出这个进程。
       HTTP 使用标准库；WebSessions 按随机 HttpOnly Cookie 隔离玩家，并用 SQLite 保存世界与网页记录。
       回合携带 request_id；按钮只上传 decision_id/choice_id，由服务器保存的 Parsed 执行；断线只停止传送、不打断结算。
       冻结菜单随存档恢复，旧请求先重放、旧菜单拒绝；SSE 与 JSON 两种传输供不同代理使用。
       --allow-migration 显式接续旧规则存档，不补写过去未观察的事实。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tianlong.cognition.navigation import believed_place
from tianlong.language.llm import fast_llm_from_env, llm_from_env
from tianlong.persistence.sqlite_store import SQLiteWorldStore
from tianlong.persistence.store import RequestConflict, WorldStore
from tianlong.runtime.cli import interpreter_for, load_dotenv
from tianlong.runtime.session import GameSession, TurnReport
from tianlong.runtime.webpage import PAGE
from tianlong.scenarios import SCENARIOS

log = logging.getLogger(__name__)

MAX_INPUT = 300      # 一句输入的上限（字）：比这更长的多半是粘贴错了
MAX_BODY = 16384
COOKIE = "tianlong_session"
HISTORY_KEEP = 200


# ============================================================
#  一局游戏的外壳：开场只讲一次，回合串行结算
# ============================================================


class WebGame:
    def __init__(self, factory: Callable[[], GameSession], title: str = "", *,
                 saved: dict[str, Any] | None = None,
                 checkpoint: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self.title = title
        self.session = factory()
        saved = saved or {}
        self._opening: str | None = saved.get("opening")
        self._epilogue: str | None = saved.get("epilogue")
        self._history: list[dict[str, Any]] = saved.get("history", [])
        self._pending: dict[str, str] | None = saved.get("pending")
        self._checkpoint = checkpoint
        # 最后一回合叙述晚于内核提交，网页检查点补回这份最近正文与提示进度。
        if saved.get("runtime") and not self._pending:
            self.session._restore(saved["runtime"])
        self._warm()

    def _save(self) -> None:
        if self._checkpoint:
            s = self.session
            self._checkpoint({"branch": s.ref.branch_id, "opening": self._opening, "epilogue": self._epilogue,
                              "history": self._history, "pending": self._pending,
                              "runtime": s.session_state()})

    def _warm(self) -> None:
        """开场在后台先写好：浏览器打开页面时（有模型时那是一次调用）多半已经就绪，不必干等。"""
        threading.Thread(target=self.state, daemon=True).start()

    def restart(self) -> None:
        with self._lock:
            previous = self.session
            self.session = self._factory()
            self._opening = self._epilogue = None
            self._history = []
            self._pending = None
            self._save()
            self._close(previous)
        self._warm()

    @staticmethod
    def _close(session: GameSession) -> None:
        if session._pool:
            session._pool.shutdown(wait=True)
        client = getattr(session.index, "client", None)
        if client is not None:
            client.close()

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
        return [c["label"] for c in self.session.choices.current()["choices"]]

    def state(self) -> dict[str, Any]:
        """开场（只讲一次，刷新页面时原样再给）、时辰、以为自己在哪、是否已落幕。"""
        with self._lock:
            s = self.session
            if self._opening is None:
                parts = [s.scenario.setting] if s.scenario.setting and not s.resumed else []
                parts.append(s.intro())
                self._opening = "\n\n".join(p for p in parts if p)
            epilogue = self._closing()
            self._save()
            decision = s.choices.current()
            return {"title": self.title, "clock": s.clock(), "place": self._place(), "opening": self._opening,
                    "game_id": s.ref.branch_id,
                    **decision, "suggest": [c["label"] for c in decision["choices"]],
                    "history": [{"request_id": h["request_id"], "text": h["text"],
                                 "narration": h["done"]["narration"], "kind": h["done"]["kind"]} for h in self._history],
                    "pending": self._pending,
                    "hints": s.scenario.hints, "voice": s.llm is not None,
                    "ended": s.ending is not None, "ending": s.ending.title if s.ending else None,
                    "epilogue": epilogue}

    def turn(self, text: str, on_text: Callable[[str], None], request_id: str | None = None,
             game_id: str | None = None) -> dict[str, Any]:
        return self._turn({"text": text}, on_text, request_id, game_id)

    def choose(self, decision_id: str, choice_id: str, on_text: Callable[[str], None], request_id: str,
               game_id: str) -> dict[str, Any]:
        return self._turn({"decision_id": decision_id, "choice_id": choice_id}, on_text, request_id, game_id)

    def _turn(self, action: dict[str, str], on_text: Callable[[str], None], request_id: str | None,
              game_id: str | None) -> dict[str, Any]:
        """结算一回合：叙述经 on_text 逐句交付；返回回合收尾的元数据（不含真相）。"""
        with self._lock:
            s = self.session
            if game_id is not None and game_id != s.ref.branch_id:
                raise RequestConflict("这一局已重开，请刷新页面")
            request_id = request_id or secrets.token_urlsafe(18)
            for h in self._history:
                if h["request_id"] == request_id:
                    if h.get("action", {"text": h["text"]}) != action:
                        raise RequestConflict("同一请求编号不能绑定不同输入")
                    on_text(h["done"]["narration"])
                    return h["done"]
            if self._pending and (self._pending["request_id"] != request_id
                                  or {k: self._pending.get(k) for k in action} != action):
                raise RequestConflict("上一回合尚未完成，请先重试该回合")
            choice = "choice_id" in action
            if choice:
                prior = s.store.request(s.ref, request_id)
                text = prior.command if prior else s.choices.resolve(action["decision_id"], action["choice_id"]).label
            else:
                text = action["text"]
            self._pending = {"request_id": request_id, **action, **({"label": text} if choice else {})}
            self._save()
            try:
                report: TurnReport = (s.choose(action["decision_id"], action["choice_id"], request_id, on_text)
                                      if choice else s.turn(text, request_id=request_id, on_text=on_text))
            except RequestConflict:
                self._pending = None
                self._save()
                raise
            ended = report.ending is not None
            decision = s.choices.current()
            done = {"clock": s.clock(), "place": self._place(), "kind": report.kind.value, "advanced": report.advanced,
                    "narration": report.narration, "first_text_ms": report.first_text_ms,
                    "ended": ended, "ending": report.ending.title if ended else None, "epilogue": self._closing(),
                    **decision, "suggest": [c["label"] for c in decision["choices"]]}
            self._history.append({"request_id": request_id, "text": text, "action": action, "done": done})
            self._history = self._history[-HISTORY_KEEP:]
            self._pending = None
            self._save()
            return done


class WebSessions:
    """浏览器各有一局。缓存淘汰后与重启进程后都可从同一 SQLite 存档恢复。"""

    def __init__(self, factory: Callable[[WorldStore, str], GameSession], directory: str | Path,
                 title: str = "", capacity: int = 64) -> None:
        self.factory, self.title = factory, title
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._lock = threading.Lock()
        self._games: OrderedDict[str, WebGame] = OrderedDict()
        self._active: dict[str, int] = {}

    def release(self, session_id: str) -> None:
        with self._lock:
            self._active[session_id] = max(0, self._active.get(session_id, 0) - 1)

    def get(self, session_id: str | None) -> tuple[str, WebGame]:
        with self._lock:
            valid = bool(session_id and re.fullmatch(r"[A-Za-z0-9_-]{43}", session_id))
            if not valid or not (self.directory / f"{session_id}.sqlite3").is_file():
                session_id = secrets.token_urlsafe(32)
            if session_id in self._games:
                self._games.move_to_end(session_id)
                self._active[session_id] = self._active.get(session_id, 0) + 1
                return session_id, self._games[session_id]
            # 只淘汰空闲的会话；正在结算的对象不能同时被重建成第二个写入者。
            if len(self._games) >= self.capacity:
                for key, candidate in list(self._games.items()):
                    if not self._active.get(key) and candidate._lock.acquire(blocking=False):
                        try:
                            self._games.pop(key)
                            self._active.pop(key, None)
                            candidate._close(candidate.session)
                        finally:
                            candidate._lock.release()
                        break
                else:
                    raise RuntimeError("当前游玩人数较多，请稍后重试")
            store = SQLiteWorldStore(self.directory / f"{session_id}.sqlite3")
            saved = store.web_state()
            first = True

            def factory() -> GameSession:
                nonlocal first
                branch = saved.get("branch") if first else None
                first = False
                return self.factory(store, branch or secrets.token_urlsafe(18))

            game = WebGame(factory, self.title, saved=saved, checkpoint=store.save_web_state)
            self._games[session_id] = game
            self._active[session_id] = 1
            return session_id, game


class BadRequest(ValueError):
    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.code = code


# ============================================================
#  HTTP：GET / 页面，GET /api/state 开场与状态，POST /api/turn 流式回合（SSE），POST /api/restart 重开
# ============================================================


def _handler(games: WebGame | WebSessions, transport: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Tianlong/1"

        def _game(self) -> WebGame:
            if isinstance(games, WebGame):
                return games
            cookie = SimpleCookie()
            with contextlib.suppress(CookieError):
                cookie.load(self.headers.get("Cookie", ""))
            old = cookie.get(COOKIE)
            session_id, game = games.get(old.value if old else None)
            self._session_id = session_id
            self._cookie = f"{COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
            if self.headers.get("X-Forwarded-Proto") == "https":
                self._cookie += "; Secure"
            return game

        def finish(self) -> None:
            try:
                super().finish()
            finally:
                if isinstance(games, WebSessions) and getattr(self, "_session_id", None):
                    games.release(self._session_id)

        def _headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            if getattr(self, "_cookie", None):
                self.send_header("Set-Cookie", self._cookie)

        def log_message(self, fmt: str, *args: Any) -> None:     # 安静：玩家的输入不写进访问日志
            log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._headers()
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict[str, Any]) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/healthz":
                self._json(200, {"status": "ok"})
            elif path == "/api/state":
                try:
                    self._json(200, {**self._game().state(), "transport": transport})
                except RuntimeError as e:
                    self._json(503, {"error": str(e)})
            else:
                self._json(404, {"error": "not found"})

        def _body(self) -> dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n < 0 or n > MAX_BODY:
                    raise BadRequest("输入过长", 413)
                if self.headers.get_content_type() != "application/json":
                    raise BadRequest("请使用 JSON 请求")
                self.connection.settimeout(30)
                raw = self.rfile.read(n) if n else b"{}"
                data = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, ValueError) as e:
                if isinstance(e, BadRequest):
                    raise
                raise BadRequest("输入格式错误") from None
            if not isinstance(data, dict):
                raise BadRequest("输入格式错误")
            return data

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path not in ("/api/turn", "/api/restart"):
                self._json(404, {"error": "not found"})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self._json(403, {"error": "origin"})
                return
            try:
                body = self._body()
            except BadRequest as e:
                self._json(e.code, {"error": str(e)})
                return
            if path == "/api/restart":
                game = self._game()
                game.restart()
                self._json(200, {**game.state(), "transport": transport})
                return
            choice = "choice_id" in body or "decision_id" in body
            text = body.get("text")
            text = text.strip()[:MAX_INPUT] if isinstance(text, str) else ""
            if not choice and not text:
                self._json(400, {"error": "empty"})
                return
            request_id = body.get("request_id")
            if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)):
                self._json(400, {"error": "request_id"})
                return
            game = self._game()
            game_id = body.get("game_id")
            if game_id is not None and not isinstance(game_id, str):
                self._json(400, {"error": "game_id"})
                return
            if choice and (set(body) - {"decision_id", "choice_id", "request_id", "game_id"}
                        or not request_id or not game_id or any(
                            not isinstance(body.get(k), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", body[k])
                            for k in ("decision_id", "choice_id"))):
                self._json(400, {"error": "请选择当前页面提供的选项"})
                return

            def execute(on_text: Callable[[str], None]) -> dict[str, Any]:
                return (game.choose(body["decision_id"], body["choice_id"], on_text, request_id, game_id)
                        if choice else game.turn(text, on_text, request_id, game_id))

            if transport == "json" or self.headers.get("Accept") == "application/json":
                try:
                    done = execute(lambda _: None)
                except RequestConflict as e:
                    self._json(409, {"error": str(e)})
                    return
                except Exception:
                    log.exception("回合出错")
                    self._json(500, {"error": "这一回合出了岔子，请重试同一回合"})
                    return
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self._json(200, done)
                return
            disconnected = False
            started = False

            def event(name: str, data: dict[str, Any]) -> None:
                nonlocal disconnected, started
                if disconnected:
                    return
                payload = json.dumps(data, ensure_ascii=False)
                try:
                    if not started:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self._headers()
                        self.send_header("X-Accel-Buffering", "no")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                    self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    disconnected = True

            try:
                done = execute(lambda piece: event("text", {"t": piece}))
                event("done", done)
            except RequestConflict as e:
                if not started:
                    self._json(409, {"error": str(e)})
                else:
                    event("error", {"message": str(e), "conflict": True})
            except (BrokenPipeError, ConnectionResetError):
                pass                                  # 玩家关了页面：回合照常结算落库，只是没人看
            except Exception as e:  # noqa: BLE001 —— 一回合出错不该拖垮整个服务
                log.exception("回合出错")
                with contextlib.suppress(OSError):
                    event("error", {"message": f"这一回合出了岔子：{type(e).__name__}"})

    return Handler


def make_server(game: WebGame | WebSessions, host: str = "127.0.0.1", port: int = 8000,
                transport: str = "sse") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _handler(game, transport))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="tianlong.runtime.web", description="图世界文字游戏（网页）")
    ap.add_argument("--llm", choices=["auto", "none"], default="auto", help="auto：有 GEMINI_API_KEY 则启用")
    ap.add_argument("--world", choices=sorted(SCENARIOS), default="wuliang")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--data-dir", default=os.environ.get("TIANLONG_DATA_DIR", ".cache/web-saves"))
    ap.add_argument("--transport", choices=["sse", "json"], default="sse", help="不支持 SSE 的代理可用 json")
    ap.add_argument("--allow-migration", action="store_true", help="显式以当前规则接续旧存档；不补写过去未知的信息")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    voice = llm_from_env() if args.llm == "auto" else None
    fast = fast_llm_from_env() if args.llm == "auto" else None

    def factory(store: WorldStore, branch: str) -> GameSession:
        scenario = SCENARIOS[args.world](args.seed)
        return GameSession(scenario, store=store, branch_id=branch, llm=voice, fast_llm=fast,
                           interpreter=interpreter_for(fast or voice, scenario), allow_migration=args.allow_migration)

    title = {"wuliang": "天龙八部 · 无量山", "warehouse": "仓库钥匙"}.get(args.world, args.world)
    server = make_server(WebSessions(factory, args.data_dir, title), args.host, args.port, args.transport)
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
