"""
[INPUT]: GameSession、SQLiteWorldStore、冻结 ChoiceSpec 与原有世界内核
[OUTPUT]: 按钮不经解释器；路线/方式/话题/计划保真；过期、幂等、事务故障和跨进程菜单恢复验收
[POS]: tests 的玩家决策契约；检查实际意图、版本与认知，不能只比较选项文字。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""


import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.cognition.candidates import Candidate  # noqa: E402
from tianlong.core import Manner, Op, Rel, Social  # noqa: E402
from tianlong.language.parser import MoveKind, Parsed  # noqa: E402
from tianlong.persistence import RequestConflict, SQLiteWorldStore  # noqa: E402
from tianlong.persistence.store import ChoiceConflict  # noqa: E402
from tianlong.runtime.choice_model import ChoiceSpec, parsed_from, parsed_to  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402


@pytest.fixture
def session():
    s = GameSession(build_wuliang(7), pipeline=False)
    s.intro()
    yield s
    s.index.client.close()


def _freeze(s, spec):
    s.store.publish_decision(s.ref, {"schema": 1, "id": "test-decision", "version": s.authority.head().version,
                                   "choices": [spec.to_data()], "consumed_request": None}, s.authority.head().version)
    return s.choices.current()


def test_label_is_not_an_action_and_does_not_call_the_interpreter(session):
    class Bomb:
        def interpret(self, *_):
            raise AssertionError("点击按钮不应解析任何文字")
    session.interpreter = Bomb()
    parsed = Parsed(Candidate(Op.MOVE, "houyuan", "d_corridor", Manner.CAREFUL))
    spec = ChoiceSpec.of("这一显示文字即使换成研读易经，动作也不能变", parsed)
    decision = _freeze(session, spec)
    report = session.choose(decision["decision_id"], spec.id, "bound-action")
    it = next(e.intent for e in report.events if e.actor == session.player)
    assert (it.op, it.target, it.obj, it.manner) == (Op.MOVE, "houyuan", "d_corridor", Manner.CAREFUL)
    assert session.authority.head().target(session.player, Rel.AT) == "houyuan"


def test_parsed_round_trip_preserves_plan_and_wait_semantics():
    parsed = Parsed(Candidate(Op.TAKE, "yijing"), followups=(Candidate(Op.STUDY, "yijing"),),
                    repeat=30, until="night", utterance="先拿再读")
    assert parsed_to(parsed_from(parsed_to(parsed))) == parsed_to(parsed)


def test_replay_precedes_expiry_and_different_requests_cannot_consume_twice(session):
    d = session.choices.current()
    choice = d["choices"][0]
    first = session.choose(d["decision_id"], choice["id"], "one")
    version = session.authority.head().version
    session.choices.current()  # 现在菜单已是下一轮；旧请求仍可重放。
    again = session.choose(d["decision_id"], choice["id"], "one")
    assert again.replayed and again.narration == first.narration
    assert session.authority.head().version == version
    with pytest.raises(ChoiceConflict):
        session.choose(d["decision_id"], choice["id"], "two")
    with pytest.raises(RequestConflict):
        session.choose(d["decision_id"], d["choices"][1]["id"], "one")
    assert session.authority.head().version == version


def test_free_input_invalidates_old_menu(session):
    d = session.choices.current()
    session.turn("环顾四周")
    with pytest.raises(ChoiceConflict):
        session.choose(d["decision_id"], d["choices"][0]["id"], "stale")


@pytest.mark.parametrize("when", ["before", "after"])
def test_first_commit_and_choice_consumption_recover_together(tmp_path, when):
    path = tmp_path / "save.sqlite3"
    store = SQLiteWorldStore(path)
    s = GameSession(build_wuliang(7), store=store, pipeline=False)
    s.intro()
    d = _freeze(s, ChoiceSpec.of("向龚光杰解释", Parsed(Candidate(Op.TELL, "gongguangjie", social=Social.EXPLAIN),
                                                   "并无冒犯之意。", kind=MoveKind.SAY)))
    original = store.commit
    failed = False

    def fault(batch):
        nonlocal failed
        if failed:
            return original(batch)
        failed = True
        if when == "after":
            original(batch)
        raise RuntimeError("模拟进程在提交边界退出")

    store.commit = fault
    with pytest.raises(RuntimeError):
        s.choose(d["decision_id"], d["choices"][0]["id"], "resume")
    assert store.head(s.ref).version == (1 if when == "after" else 0)
    assert bool(store.decision(s.ref)["consumed_request"]) == (when == "after")
    assert (store.request(s.ref, "resume") is not None) == (when == "after")
    s.index.client.close()
    loaded = GameSession(build_wuliang(7), store=SQLiteWorldStore(path), pipeline=False)
    report = loaded.choose(d["decision_id"], d["choices"][0]["id"], "resume")
    assert report.narration and loaded.authority.head().version == 2
    assert loaded.store.decision(loaded.ref)["consumed_request"] == "resume"
    loaded.index.client.close()


def test_menu_is_frozen_and_restored_from_disk(tmp_path):
    path = tmp_path / "save.sqlite3"
    s = GameSession(build_wuliang(7), store=SQLiteWorldStore(path), pipeline=False)
    s.intro()
    first = s.choices.current()
    stored = s.store.decision(s.ref)
    changed = {**stored, "id": "untrusted-replacement", "choices": []}
    assert s.store.publish_decision(s.ref, changed, 0) == stored
    s.index.client.close()
    loaded = GameSession(build_wuliang(7), store=SQLiteWorldStore(path), pipeline=False)
    assert loaded.choices.current() == first
    loaded.index.client.close()
