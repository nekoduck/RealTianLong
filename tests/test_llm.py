"""
[INPUT]: 依赖 tianlong.language.llm 的 GeminiClient / CachedLLM / ScriptedLLM / LLMUnavailable / llm_from_env / fast_llm_from_env，
         tianlong.language.hedge 的 HedgedLLM，tianlong.language.render 的 RenderStatus，test_narrator_gm 的叫阵夹具与叙述辅助
[OUTPUT]: Gemini 接入验收（本地明文假端点代替 HTTPS，不联网、无密钥）：半截正文、半截错误正文、坏编码、怪形状的 JSON 一律是
          LLMUnavailable 且下一次调用照常；读超时与新连接上的挂断绝不重发，只有闲置长连接被对端关掉才重连一次；半读的响应不留在
          连接上（下一次请求只发一遍）；流内错误、提示词被拒、SAFETY/RECITATION/MAX_TOKENS/OTHER 与没有结束标记的流在交出已收到的
          文字之后抛出，叙述者据此保留已交付的整句、补上带句末标点的模板台词；思考档位被拒逐级降档（→ low → 预算 0 → 去掉）且看所发的请求体；
          generate 对来得快的 429/500/503 等 0.2 秒重试一次（至多一次，Retry-After 太久、失败太慢、stream 都不重试）；
          密钥只走请求头、不进路径与异常信息；坏缓存条目算没命中、写到一半被打断不留半截；
          llm_from_env 默认给叙述模型包一层首字对冲（备用与 fast_llm_from_env 同样装配、GEMINI_HEDGE_AFTER 定时限、0 不对冲、开缓存不对冲）
[POS]: tests 的模型接入层：证伪“传输异常越过 LLMUnavailable 让回合崩溃”“超时后盲目重发让等待与计费翻倍”“被截断的半句当成功交付”
       “思考档位被拒就退回最慢的默认深思”“一次 503 就丢掉一回合”“缓存坏一条就再也开不了局”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import contextlib
import http.client
import itertools
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from tianlong.language.hedge import HedgedLLM
from tianlong.language.llm import (
    CachedLLM,
    GeminiClient,
    LLMUnavailable,
    ScriptedLLM,
    fast_llm_from_env,
    llm_from_env,
)
from tianlong.language.render import RenderStatus

from .test_narrator_gm import (  # noqa: F401  （复用龚光杰叫阵的感知夹具）
    G1,
    G2,
    TAUNT,
    _prose,
    _run,
    _template,
    view,
)

KEY = "AIza-test-secret-key-0123456789"


# ============================================================
#  本地假端点：每个请求按顺序取一个剧本动作，动作直接往套接字里写原始字节，返回这条连接是否继续读下一个请求
# ============================================================


class FakeGemini:
    def __init__(self) -> None:
        self.log: list[tuple[int, str, dict, dict[str, str]]] = []      # (连接号, 路径, 请求体, 请求头)
        self.script: list = []
        self._ids = itertools.count(1)
        self._srv = socket.create_server(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                sock, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _serve(self, sock: socket.socket) -> None:
        cid = next(self._ids)
        f = sock.makefile("rb")
        try:
            while True:
                line = f.readline()
                if not line:
                    return
                headers: dict[str, str] = {}
                while (h := f.readline()) not in (b"\r\n", b"\n", b""):
                    k, v = h.decode().split(":", 1)
                    headers[k.strip().lower()] = v.strip()
                body = f.read(int(headers.get("content-length", 0)))
                self.log.append((cid, line.split()[1].decode(), json.loads(body), headers))
                act = self.script.pop(0) if self.script else ok("（剧本之外）")
                if not act(sock):
                    return
        except OSError:
            return
        finally:
            f.close()
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)             # 立即给客户端 EOF，不等引用计数
            sock.close()

    def sent_thinking(self) -> list:
        return [body["generationConfig"].get("thinkingConfig") for _, _, body, _ in self.log]

    def close(self) -> None:
        self._srv.close()


def _head(status: str, *headers: str) -> bytes:
    return ("\r\n".join([f"HTTP/1.1 {status}", *headers]) + "\r\n\r\n").encode()


def raw(code: int, body: bytes, *headers: str, keep: bool = True):
    def act(sock):
        sock.sendall(_head(f"{code} X", f"Content-Length: {len(body)}", "Content-Type: application/json", *headers) + body)
        return keep
    return act


def payload(text: str | None, finish: str | None = "STOP", thought: str | None = None) -> dict:
    cand: dict = {}
    if text is not None:
        parts = ([{"text": thought, "thought": True}] if thought else []) + [{"text": text}]
        cand["content"] = {"role": "model", "parts": parts}
    if finish:
        cand["finishReason"] = finish
    return {"candidates": [cand]}


def ok(text: str, *, finish: str = "STOP", thought: str | None = None, keep: bool = True):
    return raw(200, json.dumps(payload(text, finish, thought)).encode(), keep=keep)


def status(code: int, message: str, *headers: str):
    return raw(code, json.dumps({"error": {"code": code, "message": message}}).encode(), *headers)


def truncated(code: int):
    """说好 500 字节，只给了开头就关连接。"""
    def act(sock):
        sock.sendall(_head(f"{code} X", "Content-Length: 500") + b'{"candidates": [')
        return False
    return act


def hang_up(sock) -> bool:
    """读完请求，一个字节不回就挂断。"""
    return False


def stall(seconds: float):
    """一个字节不回，拖过客户端的超时。"""
    def act(sock):
        time.sleep(seconds)
        return False
    return act


def slow(seconds: float, then):
    def act(sock):
        time.sleep(seconds)
        return then(sock)
    return act


def body_stall(seconds: float):
    """头与半截正文先到，然后停住；醒来之后照旧在这条连接上读下一个请求。"""
    def act(sock):
        sock.sendall(_head("200 OK", "Content-Length: 500") + b'{"candidates": [')
        time.sleep(seconds)
        return True
    return act


def sse(*events: dict, end: bool = True):
    def act(sock):
        sock.sendall(_head("200 OK", "Content-Type: text/event-stream", "Transfer-Encoding: chunked"))
        for ev in events:
            data = b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\r\n\r\n"
            sock.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        if end:
            sock.sendall(b"0\r\n\r\n")
        return end
    return act


def piece(text: str | None, finish: str | None = None, thought: bool = False) -> dict:
    if thought:
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": text, "thought": True}]}}]}
    return payload(text, finish)


@pytest.fixture
def gemini(monkeypatch):
    """把 GeminiClient 的 HTTPS 连接改道到本地明文假端点（构造客户端本身不做任何网络 I/O）。"""
    fake = FakeGemini()

    class Plain(http.client.HTTPConnection):
        def __init__(self, host, port=None, timeout=None, context=None):
            super().__init__("127.0.0.1", fake.port, timeout=timeout)

    monkeypatch.setattr(http.client, "HTTPSConnection", Plain)
    yield fake
    fake.close()


def _client(timeout: float = 1.0, **kw) -> GeminiClient:
    c = GeminiClient(KEY, kw.pop("model", "gemini-3.8-flash"), timeout=timeout, **kw)
    c._proxy = None                        # 环境里的 HTTPS 代理不参与：直连假端点
    return c


def _drain(stream) -> tuple[list[str], BaseException | None]:
    got: list[str] = []
    try:
        for p in stream:
            got.append(p)
    except BaseException as e:  # noqa: BLE001 —— 由断言判断抛的是什么
        return got, e
    return got, None


# ============================================================
#  正常路径：长连接复用、思考片段不算正文、密钥只走请求头
# ============================================================


def test_generate_and_stream_reuse_one_connection_and_skip_thoughts(gemini):
    gemini.script[:] = [ok("第一句", thought="先想想"),
                        sse(piece("先想想", thought=True), piece("你走进大殿。"), piece("龚光杰冷笑。", "STOP"),
                            {"usageMetadata": {"totalTokenCount": 10}}),
                        ok("第三句")]
    c = _client()
    assert c.generate("一") == "第一句"
    assert list(c.stream("二")) == ["你走进大殿。", "龚光杰冷笑。"]
    assert c.generate("三") == "第三句"
    assert [cid for cid, *_ in gemini.log] == [1, 1, 1]
    assert [s.ok for s in c.stats.snapshot()] == [True, True, True]


def test_key_travels_only_in_the_header_and_never_in_errors(gemini):
    gemini.script[:] = [status(400, f"API key not valid: {KEY}")]
    c = _client()
    with pytest.raises(LLMUnavailable) as e:
        c.generate("一")
    assert KEY not in str(e.value) and "***" in str(e.value)
    assert all(KEY not in path for _, path, _, _ in gemini.log)
    assert gemini.log[0][3]["x-goog-api-key"] == KEY


# ============================================================
#  41：对外只抛 LLMUnavailable
# ============================================================


@pytest.mark.parametrize("act", [
    truncated(200),
    truncated(503),
    raw(200, b"\xff\xfe not utf-8"),
    raw(200, json.dumps({"candidates": [{"content": {"parts": ["x"]}, "finishReason": "STOP"}]}).encode()),
    raw(200, b"[1, 2]"),
], ids=["half-200", "half-503", "bad-utf8", "odd-parts", "not-object"])
def test_generate_raises_only_unavailable_and_recovers(gemini, act):
    gemini.script[:] = [act, ok("再来一次")]
    c = _client()
    with pytest.raises(LLMUnavailable):
        c.generate("一")
    assert c.generate("二") == "再来一次"
    assert [s.ok for s in c.stats.snapshot()] == [False, True]


@pytest.mark.parametrize("act", [truncated(200), truncated(503), sse(piece("你走进大殿。"), end=False)],
                         ids=["half-200", "half-503", "cut-mid-stream"])
def test_stream_raises_only_unavailable_and_recovers(gemini, act):
    gemini.script[:] = [act, ok("再来一次")]
    c = _client()
    _, err = _drain(c.stream("一"))
    assert isinstance(err, LLMUnavailable)
    assert c.generate("二") == "再来一次"


# ============================================================
#  42 / 44：超时与挂断不重发；只有闲置长连接被关才重连；半读的响应不留在连接上
# ============================================================


@pytest.mark.parametrize("call", ["generate", "stream"])
def test_read_timeout_is_never_resent(gemini, call):
    gemini.script[:] = [stall(1.5), ok("不该发第二遍")]
    c = _client(timeout=0.3)
    t0 = time.perf_counter()
    with pytest.raises(LLMUnavailable):
        c.generate("一") if call == "generate" else list(c.stream("一"))
    assert len(gemini.log) == 1, "请求已送达：超时后重发只会让等待与计费翻倍"
    assert time.perf_counter() - t0 < 1.0, "超时即放弃，不等对端"


def test_hang_up_on_a_fresh_connection_is_not_resent(gemini):
    gemini.script[:] = [hang_up, ok("不该发第二遍")]
    c = _client()
    with pytest.raises(LLMUnavailable):
        c.generate("一")
    assert len(gemini.log) == 1


def test_stale_keepalive_connection_reconnects_once(gemini):
    gemini.script[:] = [ok("第一句", keep=False), ok("第二句")]
    c = _client()
    assert c.generate("一") == "第一句"
    time.sleep(0.05)                                   # 对端已关掉这条闲置的长连接
    assert c.generate("二") == "第二句"
    assert [cid for cid, *_ in gemini.log] == [1, 2], "第二问只在新连接上发了一遍"


def test_half_read_response_does_not_make_the_next_request_run_twice(gemini):
    gemini.script[:] = [body_stall(0.6), ok("第二问的回答"), ok("剧本多出的一条")]
    c = _client(timeout=0.3)
    with pytest.raises(LLMUnavailable):
        c.generate("一")
    time.sleep(0.5)                                    # 对端醒来，还在旧连接上等下一个请求
    assert c.generate("二") == "第二问的回答"
    assert len(gemini.log) == 2 and gemini.log[1][0] != gemini.log[0][0]


# ============================================================
#  43：结束原因与流内错误
# ============================================================

CUT = [piece(None, "SAFETY"), piece("忽然", "RECITATION"), piece("", "MAX_TOKENS"), piece(None, "OTHER"),
       {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}}]


@pytest.mark.parametrize("tail", CUT, ids=["safety", "recitation", "max-tokens", "other", "error-event"])
def test_stream_cut_short_raises_after_the_delivered_text(gemini, tail):
    gemini.script[:] = [sse(piece("你环顾四周。"), piece("殿中众人"), tail)]
    c = _client()
    got, err = _drain(c.stream("一"))
    assert isinstance(err, LLMUnavailable)
    assert got[:2] == ["你环顾四周。", "殿中众人"]
    assert c.stats.snapshot()[-1].ok is False


def test_stream_without_a_finish_marker_is_truncated(gemini):
    gemini.script[:] = [sse(piece("你环顾四周。"), piece("殿中众人"))]
    got, err = _drain(_client().stream("一"))
    assert got == ["你环顾四周。", "殿中众人"] and isinstance(err, LLMUnavailable)


def test_blocked_prompt_is_unavailable_not_empty(gemini):
    gemini.script[:] = [sse({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})]
    got, err = _drain(_client().stream("一"))
    assert got == [] and isinstance(err, LLMUnavailable)


@pytest.mark.parametrize("finish", ["SAFETY", "MAX_TOKENS", "RECITATION"])
def test_generate_rejects_a_cut_short_answer(gemini, finish):
    gemini.script[:] = [ok("半句话", finish=finish)]
    with pytest.raises(LLMUnavailable):
        _client().generate("一")


@pytest.mark.parametrize("tail", [CUT[0], CUT[-1]], ids=["safety", "error-event"])
def test_narrator_keeps_whole_sentences_and_appends_the_template(gemini, view, tail):  # noqa: F811
    """被截断的叙述：已交付的整句留下，半句不交付，模板补上事实清单与龚光杰的叫阵。"""
    gemini.script[:] = [sse(piece(G1), piece(G2[:6]), tail)]
    r, got = _run(view, _client())
    assert r.status == RenderStatus.LLM_UNAVAILABLE
    assert got[0] == G1 and r.text == G1 + "\n" + _prose(_template(view))
    assert TAUNT in r.text


# ============================================================
#  45：思考档位被拒逐级降档，看所发的请求体
# ============================================================


def test_rejected_thinking_level_steps_down_to_low_and_is_remembered(gemini):
    gemini.script[:] = [status(400, "Thinking level MINIMAL is not supported for this model."), ok("好"), ok("还好")]
    c = _client(thinking="minimal")
    assert c.generate("一") == "好" and c.generate("二") == "还好"
    assert gemini.sent_thinking() == [{"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, {"thinkingLevel": "low"}]


def test_thinking_is_dropped_only_after_every_step_fails(gemini):
    no = status(400, "thinking config is not supported")
    gemini.script[:] = [no, no, no, ok("终于"), no]
    c = _client(thinking="medium")
    assert c.generate("一") == "终于"
    assert gemini.sent_thinking() == [{"thinkingLevel": "medium"}, {"thinkingLevel": "low"}, {"thinkingBudget": 0}, None]
    with pytest.raises(LLMUnavailable):
        c.generate("二")                                # 什么都没发还被拒：不再重试
    assert len(gemini.log) == 5


def test_thinking_retry_is_decided_by_the_body_that_was_sent(gemini):
    """并发的另一路已经改过共享的档位：这一路发出去的请求体仍带着思考字段，被拒就照样降档重试，而不是直接失败。"""
    c = _client(thinking="low")

    def rejected_after_a_peer_changed_it(sock):
        c.thinking = None
        return status(400, "Thinking level LOW is not supported")(sock)

    gemini.script[:] = [rejected_after_a_peer_changed_it, ok("照样成了")]
    assert c.generate("一") == "照样成了"


# ============================================================
#  46：来得快的忙时失败重试一次
# ============================================================


@pytest.mark.parametrize("code", [429, 500, 503])
def test_generate_retries_a_quick_busy_answer_once(gemini, code):
    gemini.script[:] = [status(code, "overloaded"), ok("第二次就好了")]
    c = _client()
    t0 = time.perf_counter()
    assert c.generate("一") == "第二次就好了"
    assert len(gemini.log) == 2 and time.perf_counter() - t0 >= 0.2


def test_busy_retry_happens_at_most_once(gemini):
    gemini.script[:] = [status(503, "overloaded"), status(503, "overloaded"), ok("不该到这")]
    with pytest.raises(LLMUnavailable):
        _client().generate("一")
    assert len(gemini.log) == 2


def test_long_retry_after_is_not_retried(gemini):
    gemini.script[:] = [status(429, "quota", "Retry-After: 30"), ok("不该到这")]
    with pytest.raises(LLMUnavailable):
        _client().generate("一")
    assert len(gemini.log) == 1


def test_slow_busy_answer_is_not_retried(gemini):
    gemini.script[:] = [slow(0.3, status(503, "overloaded")), ok("不该到这")]
    c = _client()
    c.quick_retry = 0.1
    with pytest.raises(LLMUnavailable):
        c.generate("一")
    assert len(gemini.log) == 1


def test_stream_is_not_retried_on_busy(gemini):
    """流不重试：对冲层看到主模型提前失败会立即换备用，比在这里等 0.2 秒再试更快。"""
    gemini.script[:] = [status(503, "overloaded"), ok("不该到这")]
    with pytest.raises(LLMUnavailable):
        list(_client().stream("一"))
    assert len(gemini.log) == 1


# ============================================================
#  48：坏缓存条目算没命中；写到一半被打断不留半截
# ============================================================


@pytest.mark.parametrize("junk", ['{"text": "半截', '{"model": "x"}', '{"text": 3}', "", "[]"])
def test_corrupt_cache_entry_is_a_miss_and_gets_rewritten(tmp_path, junk):
    def fresh() -> ScriptedLLM:
        return ScriptedLLM(lambda prompt, system, schema: "新鲜的文字")

    first = CachedLLM(fresh(), tmp_path)
    assert first.generate("问") == "新鲜的文字" and "".join(first.stream("讲")) == "新鲜的文字"
    for f in tmp_path.glob("*.json"):
        f.write_text(junk, "utf-8")
    again = CachedLLM(fresh(), tmp_path)
    assert again.generate("问") == "新鲜的文字" and "".join(again.stream("讲")) == "新鲜的文字"
    assert len(again.inner.prompts) == 2, "坏条目算没命中，重新问了模型"

    def boom(prompt, system, schema):
        raise AssertionError("修好的条目应当命中缓存")

    third = CachedLLM(ScriptedLLM(boom), tmp_path)
    assert third.generate("问") == "新鲜的文字" and "".join(third.stream("讲")) == "新鲜的文字"


def test_interrupted_cache_write_leaves_no_half_entry(tmp_path, monkeypatch):
    real = Path.write_text

    def half(self, data, *a, **kw):
        real(self, data[:7], *a, **kw)
        raise KeyboardInterrupt

    cached = CachedLLM(ScriptedLLM(lambda prompt, system, schema: "一段完整的文字"), tmp_path)
    monkeypatch.setattr(Path, "write_text", half)
    with pytest.raises(KeyboardInterrupt):
        cached.generate("问")
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []
    assert cached.generate("问") == "一段完整的文字"


# ============================================================
#  装配：叙述模型默认包一层首字对冲
# ============================================================

ENV = ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_THINKING", "GEMINI_FAST_MODEL", "GEMINI_FAST_THINKING",
       "GEMINI_HEDGE_AFTER", "TIANLONG_LLM_CACHE")


def _env(monkeypatch, **values: str) -> None:
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in values.items():
        monkeypatch.setenv(k, v)


def _shape(c: GeminiClient) -> tuple:
    return type(c), c.model, c.thinking, c.timeout, c._fallback


def test_narration_model_is_hedged_by_a_fast_backup_by_default(monkeypatch):
    _env(monkeypatch, GEMINI_API_KEY=KEY, GEMINI_MODEL="gemini-9-flash", GEMINI_FAST_MODEL="gemini-9-lite",
         GEMINI_FAST_THINKING="low")
    llm = llm_from_env(cache_dir=None)
    assert isinstance(llm, HedgedLLM) and llm.after == 2.5 and llm.model == "gemini-9-flash"
    assert _shape(llm.primary) == (GeminiClient, "gemini-9-flash", "low", 30.0, "gemini-flash-latest")
    fast = fast_llm_from_env(cache_dir=None)
    assert _shape(llm.backup) == _shape(fast) == (GeminiClient, "gemini-9-lite", "low", 15.0, "gemini-flash-lite-latest")
    assert llm.backup is not fast, "替补是独立的客户端，不与解释器共用"


@pytest.mark.parametrize("value, after", [("1.2", 1.2), ("", 2.5), ("abc", 2.5), ("inf", 2.5)])
def test_hedge_deadline_comes_from_env(monkeypatch, value, after):
    _env(monkeypatch, GEMINI_API_KEY=KEY, GEMINI_HEDGE_AFTER=value)
    llm = llm_from_env(cache_dir=None)
    assert isinstance(llm, HedgedLLM) and llm.after == after


@pytest.mark.parametrize("value", ["0", "0.0", "-1"])
def test_zero_hedge_deadline_returns_the_plain_client(monkeypatch, value):
    _env(monkeypatch, GEMINI_API_KEY=KEY, GEMINI_HEDGE_AFTER=value)
    assert type(llm_from_env(cache_dir=None)) is GeminiClient


def test_no_key_means_no_model(monkeypatch):
    _env(monkeypatch, GEMINI_HEDGE_AFTER="1")
    assert llm_from_env(cache_dir=None) is None and fast_llm_from_env(cache_dir=None) is None


def test_hedged_clients_over_the_wire_survive_a_stalled_primary(gemini):
    """主模型一个字节不回：替补准时上场讲完整段；主模型那一路只发过一次请求，超时后作废、不重发。"""
    gemini.script[:] = [stall(2.0), sse(piece("钟灵在梁上嗑着瓜子。", "STOP"))]
    primary, backup = _client(timeout=0.8), _client(timeout=0.8, model="gemini-9-lite", thinking="minimal")
    llm = HedgedLLM(primary, backup, after=0.1)
    t0 = time.perf_counter()
    assert "".join(llm.stream("写一段")) == "钟灵在梁上嗑着瓜子。"
    assert time.perf_counter() - t0 < 0.7 and llm.hedged == 1 and llm.switched == 1, "替补不等主模型超时"
    time.sleep(1.0)                                    # 主模型那一路超时收场
    assert [path.split("/")[-1] for _, path, _, _ in gemini.log] == [
        "gemini-3.8-flash:streamGenerateContent?alt=sse", "gemini-9-lite:streamGenerateContent?alt=sse"]


def test_cache_turns_hedging_off(monkeypatch, tmp_path):
    """缓存为的是逐字复现、重跑不花钱：谁先出字取决于网速，对冲会让重跑换一段文字，所以开缓存就只用主模型。"""
    _env(monkeypatch, GEMINI_API_KEY=KEY, TIANLONG_LLM_CACHE="1")
    cached = llm_from_env(cache_dir=tmp_path)
    assert isinstance(cached, CachedLLM) and type(cached.inner) is GeminiClient
    assert isinstance(llm_from_env(cache_dir=None), HedgedLLM), "cache_dir=None（评测）不开缓存，照样对冲"
