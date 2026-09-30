"""
[INPUT]: 依赖 tianlong.runtime.web 的 WebGame / make_server / MAX_INPUT，tianlong.runtime.session 的 GameSession，
         tianlong.scenarios 的 build_wuliang，标准库 http.client / threading
[OUTPUT]: 网页前端验收：本地起服务（临时端口），页面可取、开场只讲一次且刷新原样再给、开场与每回合都附行动建议；回合经 SSE 逐句推 text 事件、
          以 done 收尾，逐句拼起来就是整段叙述；元指令不推进时间；空输入 400；重开换一局；推送内容里没有 NPC 理由与真相
[POS]: tests 的网页前端：证伪“网页只能一次性拿到整段文字”“刷新页面开场重讲一遍、世界又从头来”“流里夹带了玩家不该看的东西”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import http.client
import json
import socket
import struct
import threading

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.web import MAX_INPUT, WebGame, WebSessions, make_server  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402


@pytest.fixture()
def served():
    game = WebGame(lambda: GameSession(build_wuliang(7)), "天龙八部 · 无量山")
    server = make_server(game, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield game, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _get(port: int, path: str) -> tuple[int, str, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
    finally:
        conn.close()


def _post(port: int, path: str, body: dict | None = None) -> tuple[int, str, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        raw = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
    finally:
        conn.close()


def _events(stream: str) -> list[tuple[str, dict]]:
    out = []
    for chunk in stream.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_page_and_opening_told_once(served):
    game, port = served
    code, ctype, page = _get(port, "/")
    assert code == 200 and ctype.startswith("text/html") and "/api/turn" in page
    code, _, raw = _get(port, "/api/state")
    first = json.loads(raw)
    assert code == 200 and first["opening"] and first["title"] == "天龙八部 · 无量山"
    assert first["clock"] and first["place"] and first["voice"] is False and first["ended"] is False
    assert 1 <= len(first["suggest"]) <= 3                  # 开场就有可点的行动建议
    again = json.loads(_get(port, "/api/state")[2])
    assert again["opening"] == first["opening"]            # 刷新页面：同一段开场，不是新的一局
    assert _get(port, "/nope")[0] == 404


def test_turn_streams_sentences_then_done(served):
    game, port = served
    _get(port, "/api/state")
    before = game.session.authority.head().clock
    code, ctype, stream = _post(port, "/api/turn", {"text": "环顾四周"})
    assert code == 200 and ctype.startswith("text/event-stream")
    events = _events(stream)
    texts = [d["t"] for e, d in events if e == "text"]
    done = [d for e, d in events if e == "done"]
    assert texts and len(done) == 1 and events[-1][0] == "done"
    assert "".join(texts).strip() == done[0]["narration"].strip()   # 逐句拼起来就是整段叙述
    assert done[0]["advanced"] is True and game.session.authority.head().clock > before
    # 推给浏览器的只有玩家该看的：没有 NPC 理由、真相，也没有叙述上下文
    assert set(done[0]) == {"clock", "place", "kind", "advanced", "narration", "first_text_ms", "ended", "ending",
                            "epilogue", "suggest", "decision_id", "choices"}
    assert done[0]["suggest"] and all(isinstance(t, str) and t for t in done[0]["suggest"])


def test_meta_does_not_advance_and_empty_is_rejected(served):
    game, port = served
    _get(port, "/api/state")
    before = game.session.authority.head().clock
    done = [d for e, d in _events(_post(port, "/api/turn", {"text": "/hint"})[2]) if e == "done"][0]
    assert done["kind"] == "meta" and done["advanced"] is False and done["narration"]
    assert game.session.authority.head().clock == before
    assert _post(port, "/api/turn", {"text": "   "})[0] == 400
    assert _post(port, "/api/turn", {"nope": 1})[0] == 400
    assert _post(port, "/api/other")[0] == 404


def test_overlong_input_is_cut(served):
    game, port = served
    seen: list[str] = []
    real = game.session.turn

    def spy(text, **kw):
        seen.append(text)
        return real(text, **kw)

    game.session.turn = spy
    _post(port, "/api/turn", {"text": "等" * (MAX_INPUT + 50)})
    assert len(seen) == 1 and len(seen[0]) == MAX_INPUT


def test_restart_starts_a_fresh_world(served):
    game, port = served
    _get(port, "/api/state")
    _post(port, "/api/turn", {"text": "等一会儿"})
    old = game.session
    code, _, raw = _post(port, "/api/restart")
    assert code == 200 and game.session is not old
    fresh = json.loads(raw)
    assert fresh["opening"] and fresh["ended"] is False
    start = build_wuliang(7).state.clock
    assert old.authority.head().clock > start and game.session.authority.head().clock == start


def _request(port, method, path, body=None, cookie=None, **headers):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    raw = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    if raw is not None:
        headers.setdefault("Content-Type", "application/json")
    if cookie:
        headers["Cookie"] = cookie
    try:
        conn.request(method, path, raw, headers)
        response = conn.getresponse()
        return response.status, response.getheader("Set-Cookie"), response.read().decode()
    finally:
        conn.close()


@pytest.fixture()
def multiple(tmp_path):
    factory = lambda store, branch: GameSession(build_wuliang(7), store=store, branch_id=branch)  # noqa: E731
    games = WebSessions(factory, tmp_path / "saves", "无量山", capacity=2)
    server = make_server(games, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield games, server.server_address[1], factory
    finally:
        server.shutdown()
        server.server_close()


def test_browser_sessions_isolated_and_refresh_restores_history(multiple):
    _, port, _ = multiple
    _, alice, raw = _request(port, "GET", "/api/state")
    start = json.loads(raw)
    _, bob, other = _request(port, "GET", "/api/state")
    assert alice != bob and json.loads(other)["game_id"] != start["game_id"]
    assert "HttpOnly" in alice and "SameSite=Lax" in alice
    _request(port, "POST", "/api/turn", {"text": "环顾四周", "request_id": "a1"}, alice, Accept="application/json")
    state = json.loads(_request(port, "GET", "/api/state", cookie=alice)[2])
    assert state["clock"] != start["clock"] and state["history"][0]["text"] == "环顾四周"
    assert state["history"][0]["narration"]
    b = json.loads(_request(port, "GET", "/api/state", cookie=bob)[2])
    assert b["clock"] == start["clock"] and b["history"] == []
    _request(port, "POST", "/api/restart", {}, alice)
    assert json.loads(_request(port, "GET", "/api/state", cookie=alice)[2])["history"] == []
    assert json.loads(_request(port, "GET", "/api/state", cookie=bob)[2])["game_id"] == b["game_id"]


def test_idempotent_json_turn_and_stale_tab_after_restart(multiple):
    games, port, _ = multiple
    _, cookie, raw = _request(port, "GET", "/api/state")
    game_id = json.loads(raw)["game_id"]
    body = {"text": "环顾四周", "request_id": "retry-1", "game_id": game_id}
    first = _request(port, "POST", "/api/turn", body, cookie, Accept="application/json")
    again = _request(port, "POST", "/api/turn", body, cookie, Accept="application/json")
    assert first[0] == again[0] == 200 and json.loads(first[2]) == json.loads(again[2])
    game = next(iter(games._games.values()))
    assert game.session.authority.head().version == 1
    assert len(game.state()["history"]) == 1
    assert _request(port, "POST", "/api/turn", {**body, "text": "去后院"}, cookie, Accept="application/json")[0] == 409
    _request(port, "POST", "/api/restart", {}, cookie)
    assert _request(port, "POST", "/api/turn", body, cookie, Accept="application/json")[0] == 409
    assert game.session.authority.head().version == 0


def test_sqlite_restores_world_transcript_and_latest_runtime(multiple):
    games, port, factory = multiple
    _, cookie, _ = _request(port, "GET", "/api/state")
    _request(port, "POST", "/api/turn", {"text": "去后院", "request_id": "persist-1"}, cookie, Accept="application/json")
    _request(port, "POST", "/api/turn", {"text": "/hint", "request_id": "hint-1"}, cookie, Accept="application/json")
    session_id = cookie.split(";", 1)[0].split("=", 1)[1]
    old = games._games[session_id]
    before = old.state()
    restored = WebSessions(factory, games.directory)
    _, new = restored.get(session_id)
    after = new.state()
    assert after["opening"] == before["opening"] and after["history"] == before["history"]
    assert new.session.authority.head().fingerprint() == old.session.authority.head().fingerprint()
    assert new.session._recent == old.session._recent and new.session._hint == old.session._hint
    assert new.turn("去后院", lambda _: None, "persist-1") == old._history[0]["done"]
    for agent in new.session.scenario.profiles:
        assert new.session.beliefs(agent) == old.session.beliefs(agent)


def test_cache_eviction_resumes_from_disk(multiple):
    games, port, _ = multiple
    _, cookie, _ = _request(port, "GET", "/api/state")
    _request(port, "POST", "/api/turn", {"text": "环顾四周", "request_id": "a"}, cookie, Accept="application/json")
    before = json.loads(_request(port, "GET", "/api/state", cookie=cookie)[2])
    for _ in range(3):
        assert _request(port, "GET", "/api/state")[0] == 200
    assert len(games._games) <= 2
    after = json.loads(_request(port, "GET", "/api/state", cookie=cookie)[2])
    assert after["history"] == before["history"] and after["clock"] == before["clock"]


def test_invalid_requests_rejected_without_advancing(served):
    game, port = served
    for value in [1, [], {}, None]:
        assert _post(port, "/api/turn", {"text": value})[0] == 400
    assert _request(port, "POST", "/api/turn", {"text": "等"}, **{"Content-Length": "bad"})[0] == 400
    assert _request(port, "POST", "/api/turn", {"text": "等"}, **{"Content-Length": "20000"})[0] == 413
    assert _request(port, "POST", "/api/turn", {"text": "等"}, Origin="https://another.example")[0] == 403
    assert game.session.authority.head().version == 0
    assert _get(port, "/healthz")[0] == 200


def test_disconnect_does_not_interrupt_recording_or_repeat_settlement(served):
    game, port = served
    released = threading.Event()
    completed = threading.Event()
    real = game.session.turn

    def slow(text, **kw):
        callback = kw["on_text"]

        def send(piece):
            callback(piece)
            assert released.wait(5)
            callback("续写" * 100000)
        result = real(text, **{**kw, "on_text": send})
        completed.set()
        return result

    game.session.turn = slow
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = {"text": "环顾四周", "request_id": "disconnect"}
    conn.request("POST", "/api/turn", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    assert response.status == 200 and response.read(1)
    response.fp.raw._sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    response.close()
    conn.close()
    released.set()
    assert completed.wait(5)
    # game.state 也等待完整网页检查点写完；关闭连接不能把叙述丢在半路。
    assert len(game.state()["history"]) == 1
    assert game.session.store.request(game.session.ref, "disconnect").narration
    done = json.loads(_request(port, "POST", "/api/turn", body, Accept="application/json")[2])
    assert done["narration"] and game.session.authority.head().version == 1


def test_http_choices_are_frozen_replayed_and_do_not_trust_client_actions(served):
    game, port = served
    state = json.loads(_get(port, "/api/state")[2])
    assert all(set(c) == {"id", "label"} for c in state["choices"])
    assert json.loads(_get(port, "/api/state")[2])["choices"] == state["choices"]

    class Bomb:
        def interpret(self, *_):
            raise AssertionError("按钮不能调用解释器")
    game.session.interpreter = Bomb()
    body = {"decision_id": state["decision_id"], "choice_id": state["choices"][0]["id"],
            "request_id": "http-choice", "game_id": state["game_id"]}
    assert _request(port, "POST", "/api/turn", {**body, "text": "攻击马五德"}, Accept="application/json")[0] == 400
    assert _request(port, "POST", "/api/turn", {**body, "op": "attack"}, Accept="application/json")[0] == 400
    assert game.session.authority.head().version == 0
    code, _, raw = _request(port, "POST", "/api/turn", body, Accept="application/json")
    assert code == 200
    done = json.loads(raw)
    version = game.session.authority.head().version
    again = _request(port, "POST", "/api/turn", body, Accept="application/json")
    assert again[0] == 200 and json.loads(again[2]) == done
    assert game.session.authority.head().version == version
    assert _request(port, "POST", "/api/turn", {**body, "request_id": "old-page"}, Accept="application/json")[0] == 409
    assert game.state()["pending"] is None
