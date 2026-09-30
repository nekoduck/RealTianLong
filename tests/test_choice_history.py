"""
[INPUT]: ChoiceHistory、GameSession、SQLite 与内核实际产生的玩家感知
[OUTPUT]: progress 续修、mastered 停止、失败跨 12 条经历/读档保持、相关变化解锁、所见放置恢复搜查、首 tick 原子性验收
[POS]: options 的长期有效性；用实际结果和玩家知识验证，不以整个世界版本重置失败。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.cognition import Candidate  # noqa: E402
from tianlong.core import Fact, Modality, Op, Outcome, PerceivedEvent, Percept, Proposition, Rel  # noqa: E402
from tianlong.persistence import SQLiteWorldStore  # noqa: E402
from tianlong.runtime.choice_builder import build_choices  # noqa: E402
from tianlong.runtime.choice_history import ChoiceHistory  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402

from .test_suggest import Idle, choice_scene, scene_beliefs  # noqa: E402


def _session(sc, store=None):
    return GameSession(sc, store=store, policies={a: Idle() for a in sc.npcs}, pipeline=False)


def _acts(s):
    d = s.choices.current()
    return [s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate for c in d["choices"]]


def test_real_progress_is_continuable_until_mastery_and_survives_reload(tmp_path):
    sc = choice_scene()
    path = tmp_path / "study.sqlite3"
    s = _session(sc, SQLiteWorldStore(path))
    s.turn("研读秘籍", "study-1")
    assert s.choice_history.latest(Candidate(Op.STUDY, "book")).reason == "progress"
    assert Candidate(Op.STUDY, "book") in _acts(s)
    s.index.client.close()
    s = _session(sc, SQLiteWorldStore(path))
    for i in range(2):
        d = s.choices.current()
        c = next(c for c in d["choices"] if s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate == Candidate(Op.STUDY, "book"))
        s.choose(d["decision_id"], c["id"], f"study-{i+2}")
    assert s.authority.head().attr("player", "evasion") is True
    assert s.choice_history.latest(Candidate(Op.STUDY, "book")).reason == "mastered"
    assert Candidate(Op.STUDY, "book") not in _acts(s)
    s.index.client.close()


def test_no_yield_failure_is_not_forgotten_after_buffer_rolls_or_process_restarts(tmp_path):
    sc = choice_scene(book_skill=None)
    path = tmp_path / "failure.sqlite3"
    s = _session(sc, SQLiteWorldStore(path))
    s.turn("研读秘籍", "failure")
    for _ in range(15):
        s.turn("环顾四周")
    me = s.beliefs("player")
    assert not any(ep.event.kind == Op.STUDY.value for ep in me.episodes)
    c = Candidate(Op.STUDY, "book")
    assert s.choice_history.blocked(me, c)
    assert c not in _acts(s)
    saved = s.choice_history.to_data()
    s.index.client.close()
    loaded = _session(sc, SQLiteWorldStore(path))
    assert loaded.choice_history.to_data() == saved
    assert loaded.choice_history.blocked(loaded.beliefs("player"), c)
    assert c not in _acts(loaded)
    loaded.index.client.close()


def test_unrelated_background_change_does_not_reset_failure_but_relevant_knowledge_does():
    me = scene_beliefs(choice_scene(book_skill=None))
    c = Candidate(Op.STUDY, "book")
    p = Percept(me.last_tick, Modality.SELF, PerceivedEvent("study", "hall", "player", "book",
                                                        outcome=Outcome.FAILURE, reason="nothing_to_learn"))
    history = ChoiceHistory().note(me, (p,), c)
    after = me.revise(p)[0]
    # 其他人在后台移动或时间经过，都不是这本书的可学习条件。
    unrelated = Percept(me.last_tick + 30, Modality.SIGHT,
                        facts=(Fact(Proposition.rel("npc", Rel.AT, "road")),))
    after = after.revise(unrelated)[0]
    assert history.blocked(after, c)
    changed = after.revise(Percept(after.last_tick, Modality.SELF,
                                 facts=(Fact(Proposition.attr("book", "teaches", "evasion")),)))[0]
    assert not history.blocked(changed, c)
    assert c in [spec.parsed.candidate for spec in build_choices(changed, history=history)]


def test_seen_new_placement_restores_search_and_clue_survives_buffer_rollover():
    me = scene_beliefs(choice_scene())
    c = Candidate(Op.INSPECT, "table")
    p = Percept(me.last_tick, Modality.SELF, PerceivedEvent("inspect", "hall", "player", "table", outcome=Outcome.SUCCESS),
                scopes=("table",), facts=(Fact(Proposition.rel("cup", Rel.AT, "table")),))
    history = ChoiceHistory().note(me, (p,), c)
    me = me.revise(p)[0]
    assert history.blocked(me, c)
    put = Percept(me.last_tick + 1, Modality.SIGHT,
                  PerceivedEvent("put", "hall", "npc", "table", "pill", outcome=Outcome.SUCCESS))
    # 看到了“往石桌放东西”本身已是线索；不能等后台全知位置来恢复。
    history = history.note(me, (put,), Candidate(Op.WAIT))
    me = me.revise(put)[0]
    for _ in range(15):
        p = Percept(me.last_tick + 1, Modality.SELF, PerceivedEvent("wait", "hall", "player", outcome=Outcome.SUCCESS))
        history = history.note(me, (p,), Candidate(Op.WAIT))
        me = me.revise(p)[0]
    assert not any(ep.event.kind == "put" for ep in me.episodes)
    assert not history.blocked(me, c)
    menu = build_choices(me, history=history)
    assert c in [spec.parsed.candidate for spec in menu]
    assert ChoiceHistory.from_data(history.to_data()) == history


def test_day_night_change_can_restore_inspection_without_resetting_study():
    me = scene_beliefs(choice_scene(book_skill=None))
    history = ChoiceHistory()
    for c, reason in ((Candidate(Op.INSPECT, "table"), None), (Candidate(Op.STUDY, "book"), "nothing_to_learn")):
        p = Percept(me.last_tick, Modality.SELF, PerceivedEvent(c.op.value, "hall", "player", c.target,
                    outcome=Outcome.FAILURE if reason else Outcome.SUCCESS, reason=reason))
        history = history.note(me, (p,), c)
    night = replace(me, last_tick=20*60)
    assert not history.blocked(night, Candidate(Op.INSPECT, "table"))
    assert history.blocked(night, Candidate(Op.STUDY, "book"))


@pytest.mark.parametrize("when", ["before", "after"])
def test_attempt_and_first_tick_are_atomic(tmp_path, when):
    sc = choice_scene()
    store = SQLiteWorldStore(tmp_path / "atomic.sqlite3")
    s = _session(sc, store)
    original = store.commit
    def fault(batch):
        if when == "after":
            original(batch)
        raise RuntimeError("crash")
    store.commit = fault
    with pytest.raises(RuntimeError):
        s.turn("研读秘籍", "atomic")
    persisted = ChoiceHistory.from_data((store.session_state(s.ref) or {}).get("choice_history"))
    assert bool(persisted.latest(Candidate(Op.STUDY, "book"))) == (when == "after")
    assert store.head(s.ref).version == (1 if when == "after" else 0)
    # 未确认提交，不在内存中提前记进度。
    assert s.choice_history.latest(Candidate(Op.STUDY, "book")) is None
    s.index.client.close()


def test_failed_not_holding_offers_real_take_then_restores_study():
    from tianlong.kernel.perception import make_percept
    sc = choice_scene(book_at="table")
    # 过时的认知以为秘籍仍在自己手里，实际已在石桌上。
    priors = dict(sc.priors)
    priors["player"] += (make_percept(sc.state, Modality.SELF,
                                     facts=(Fact(Proposition.rel("book", Rel.AT, "player")),)),)
    sc = replace(sc, priors=priors)
    s = _session(sc)
    def choose(op, request):
        d = s.choices.current()
        c = next(c for c in d["choices"] if s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate.op == op)
        return s.choose(d["decision_id"], c["id"], request)
    first = choose(Op.STUDY, "not-held")
    assert any(e.actor == "player" and e.reason == "not_holding" for e in first.events)
    assert Candidate(Op.STUDY, "book") not in _acts(s)
    assert Candidate(Op.TAKE, "book") in _acts(s)
    choose(Op.TAKE, "prepare")
    assert s.authority.head().target("book", Rel.AT) == "player"
    assert Candidate(Op.STUDY, "book") in _acts(s)
    assert any(e.actor == "player" and e.reason == "progress" for e in choose(Op.STUDY, "retry").events)
    s.index.client.close()


def test_failed_locked_route_offers_key_unlock_then_restores_exact_route():
    from tianlong.core import Entity, Kind, Relation, WorldState
    from tianlong.kernel.perception import make_percept, scene_percept
    sc = choice_scene()
    ents = [Entity.make(e.id, e.kind, e.name, **{**dict(e.attrs), **({"locked": True} if e.id == "d1" else {})})
            for e in sc.state.entities.values()]
    ents.append(Entity.make("key", Kind.ITEM, "钥匙", small=True))
    state = WorldState.build(sc.state.seed, sc.state.clock, ents,
                            (*sc.state.relations, Relation("key", Rel.AT, "table"), Relation("key", Rel.MATCHES, "d1")))
    priors = {a: (scene_percept(state, a),) for a in sc.profiles}
    priors["player"] += (make_percept(state, Modality.SELF,
                                      facts=(Fact(Proposition.rel("key", Rel.MATCHES, "d1")),)),)
    s = _session(replace(sc, state=state, priors=priors))
    def choose(op, target, request):
        d = s.choices.current()
        c = next(c for c in d["choices"] if
                 (s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate.op,
                  s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate.target) == (op, target))
        return s.choose(d["decision_id"], c["id"], request)
    first = choose(Op.MOVE, "yard", "locked")
    assert any(e.actor == "player" and e.reason == "door_locked" for e in first.events)
    assert Candidate(Op.MOVE, "yard", "d1") not in _acts(s)
    choose(Op.TAKE, "key", "key")
    choose(Op.UNLOCK, "d1", "unlock")
    assert not s.authority.head().attr("d1", "locked")
    assert Candidate(Op.MOVE, "yard", "d1") in _acts(s)
    choose(Op.MOVE, "yard", "leave")
    assert s.authority.head().target("player", Rel.AT) == "yard"
    s.index.client.close()
