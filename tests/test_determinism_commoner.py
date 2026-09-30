"""
[INPUT]: 依赖 runtime/session 的 GameSession（模板模式），persistence 的 InMemoryWorldStore，scenarios 的 build_wuliang_commoner，
         core 的 at，tests/test_resume 的 _fail_once（故障注入），tests/test_commoner 的 _session / stage
[OUTPUT]: 普通人版的确定性验收：同样的种子与输入，两次会话的世界指纹、事件日志、驱力标记与会话运行态逐项相同；
          后台预算（pipeline）开或关逐项相同；“等到天亮”这种长等待中途崩溃，新进程续跑与连续运行逐项相同（只走剩下的 tick）
[POS]: tests 的确定性层：驱力、看点账本（staged）都在 annotate 的副本上推进、与世界同一事务提交——相同存档 + 相同意图 → 相同指纹
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.core import at
from tianlong.persistence import InMemoryWorldStore
from tianlong.scenarios import build_wuliang_commoner

from .test_commoner import _session, stage
from .test_resume import _fail_once

INPUTS = ("扯了扯段公子的袖子", "等待", "问马五爷这位龚爷是什么来头", "等下去", "等下去", "去后院", "等一会儿", "跟上段公子",
          "等下去", "等下去", "等下去")


def _play(seed: int, pipeline: bool) -> tuple:
    from tianlong.runtime.session import GameSession
    s = GameSession(build_wuliang_commoner(seed), pipeline=pipeline)
    s.intro()
    reports = [s.turn(t) for t in INPUTS]
    events = tuple((e.id, e.op, e.outcome, e.changes, e.intent.utterance) for e in s.store.events(s.ref))
    return (s.authority.head().fingerprint(), events, s.session_state(), tuple(r.beats for r in reports),
            tuple(r.narration for r in reports))


def test_same_seed_same_inputs_same_world_and_marks():
    _session()                                         # 缺 LangGraph / Qdrant 即跳过
    first, second = _play(11, True), _play(11, True)
    assert first == second
    assert first[2]["drives"], "这几回合里确有驱力兑现：标记随运行态落库"


def test_pipeline_on_or_off_is_identical():
    _session()
    assert _play(7, True) == _play(7, False)


def test_waiting_for_dawn_resumes_after_a_crash_exactly_as_continuous_play(monkeypatch):
    from tianlong.runtime.session import GameSession

    _session()
    night = stage(build_wuliang_commoner(7), {"ashun": "jianhu"}, at(1, 23, 0))    # 深夜独自在崖底：一等就是四个时辰
    ref = GameSession(night, pipeline=False)
    ref.turn("等下去")                                         # 头一个 tick 就看见月下玉璧：等待先为它停一回
    reference = ref.turn("等到天亮", request_id="dawn")
    assert reference.advanced and len(ref.store.request(ref.ref, "dawn").versions) > 3

    store = InMemoryWorldStore()
    s = GameSession(night, store=store, pipeline=False)
    s.turn("等下去")
    _fail_once(monkeypatch, s.indexer, "drain", after=2)          # 第 3 个 tick 提交之后崩溃
    with pytest.raises(RuntimeError, match="注入的故障"):
        s.turn("等到天亮", request_id="dawn")
    assert not store.request(s.ref, "dawn").done
    monkeypatch.undo()
    retry = GameSession(night, store=store, pipeline=False).turn("等到天亮", request_id="dawn")
    assert not retry.replayed
    assert [e.id for e in store.events(s.ref)] == [e.id for e in ref.store.events(ref.ref)]
    assert store.head(s.ref).fingerprint() == ref.authority.head().fingerprint()
    assert store.session_state(s.ref) == ref.store.session_state(ref.ref), "驱力标记、看点账本与调度逐项相同"
    assert retry.narration == reference.narration and retry.beats == reference.beats
    assert ref.authority.head().clock <= at(2, 5, 0)
