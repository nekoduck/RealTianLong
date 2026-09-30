"""
[INPUT]: 玩家认知、结构化生成器、GameSession 与确定性小场景
[OUTPUT]: 当前问题、多种社交态度、不同路线、被制仍可交流、真正施治、回答话题、真相隔离与三分支验收
[POS]: options 的情境选择测试；断言冻结动作和落库状态，不再要求展示文字被二次解释。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.agents.policy_kit import Choice  # noqa: E402
from tianlong.cognition import BeliefStore  # noqa: E402
from tianlong.cognition.agenda import Obligation, SocialCue  # noqa: E402
from tianlong.core import (  # noqa: E402
    Entity,
    Fact,
    Kind,
    Modality,
    Op,
    Proposition,
    Rel,
    Relation,
    Social,
    WorldState,
    at,
)
from tianlong.core.profiles import Profile  # noqa: E402
from tianlong.kernel.perception import make_percept, scene_percept  # noqa: E402
from tianlong.runtime.choice_builder import build_choices  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.suggest import suggestions  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402
from tianlong.scenarios.base import Scenario  # noqa: E402


class Idle:
    def choose(self, sit):
        return Choice(next(i for i, c in enumerate(sit.candidates) if c.op == Op.WAIT), "test idle")


def choice_scene(*, poisoned=False, book_skill="evasion", book_at="player", secret_martial=0.1):
    entities = [Entity.make("hall", Kind.PLACE, "大厅"), Entity.make("yard", Kind.PLACE, "后院"),
                Entity.make("road", Kind.PLACE, "山道"), Entity.make("secret", Kind.PLACE, "幕后"),
                Entity.make("d1", Kind.DOOR, "侧门"), Entity.make("d2", Kind.DOOR, "正门"),
                Entity.make("player", Kind.PERSON, "段誉"), Entity.make("npc", Kind.PERSON, "钟灵", poisoned=poisoned),
                Entity.make("hidden", Kind.PERSON, "幕后人", martial=secret_martial),
                Entity.make("table", Kind.SURFACE, "石桌"), Entity.make("cup", Kind.ITEM, "茶碗"),
                Entity.make("pill", Kind.ITEM, "解药", cures="poisoned"),
                Entity.make("book", Kind.ITEM, "秘籍", teaches=book_skill, difficulty=3)]
    relations = [Relation("d1", Rel.CONNECTS, "hall"), Relation("d1", Rel.CONNECTS, "yard"),
                 Relation("d2", Rel.CONNECTS, "hall"), Relation("d2", Rel.CONNECTS, "road"),
                 Relation("player", Rel.AT, "hall"), Relation("npc", Rel.AT, "hall"),
                 Relation("hidden", Rel.AT, "secret"), Relation("table", Rel.AT, "hall"),
                 Relation("cup", Rel.AT, "table"), Relation("pill", Rel.AT, "player"),
                 Relation("book", Rel.AT, book_at)]
    state = WorldState.build(7, at(1, 12, 0), entities, relations)
    profiles = {a: Profile(a, "test", "test", is_player=a == "player", allies=("npc",) if a == "player" else ())
                for a in ("player", "npc", "hidden")}
    priors = {a: (replace(scene_percept(state, a), tick=state.clock - 1),) for a in profiles}
    tactile = make_percept(state, Modality.SELF, facts=(Fact(Proposition.attr("book", "teaches", book_skill)),))
    priors["player"] += (tactile,)
    return Scenario("choices", state, profiles, priors)


def scene_beliefs(sc):
    return BeliefStore("player", allies=("npc",)).revise_all(sc.priors["player"])[0]


def test_opening_has_three_distinct_deterministic_supported_actions():
    me = scene_beliefs(choice_scene())
    menu = build_choices(me)
    assert len(menu) == 3 and len({c.semantic_key for c in menu}) == 3
    assert build_choices(me) == menu
    assert suggestions(me) == tuple(c.label for c in menu)
    # 两条目的不同的路线同时保留，没有“每动作族只能一个”的限制。
    routes = {(c.parsed.candidate.target, c.parsed.candidate.obj) for c in menu if c.parsed.candidate.op == Op.MOVE}
    assert routes == {("yard", "d1"), ("road", "d2")}


def test_threat_changes_the_problem_and_does_not_force_apology_first():
    s = GameSession(build_wuliang(7), pipeline=False)
    s.intro()
    before = s.choices.current()
    s.turn("环顾四周")
    after = s.choices.current()
    specs = [s.choices.resolve(after["decision_id"], c["id"]) for c in after["choices"]]
    assert after != before
    acts = [c.parsed.candidate for c in specs]
    assert acts[0].social != Social.APOLOGIZE
    assert {Social.EXPLAIN, Social.REFUSE} <= {c.social for c in acts if c.op == Op.TELL}
    assert any(c.op == Op.MOVE for c in acts) and all(c.op != Op.STUDY for c in acts)
    s.index.client.close()


def test_subdued_still_has_relevant_speech():
    me = scene_beliefs(choice_scene())
    me = me.revise(make_percept(choice_scene().state, Modality.SELF,
                               facts=(Fact(Proposition.attr("player", "subdued", True)),)))[0]
    me = replace(me, cues=(SocialCue("npc", Op.TELL, Social.THREATEN, me.last_tick, to="player"),))
    acts = [c.parsed.candidate for c in build_choices(me)]
    assert len(acts) == 3
    assert sum(c.op == Op.TELL for c in acts) >= 2
    assert all(c.op in (Op.TELL, Op.ASK, Op.WAIT) for c in acts)


def test_known_cure_is_a_real_use_and_changes_health():
    sc = choice_scene(poisoned=True)
    s = GameSession(sc, policies={a: Idle() for a in sc.npcs}, pipeline=False)
    d = s.choices.current()
    chosen = next(c for c in d["choices"] if s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate.op == Op.USE)
    frozen = s.choices.resolve(d["decision_id"], chosen["id"]).parsed.candidate
    assert (frozen.target, frozen.obj) == ("npc", "pill")
    report = s.choose(d["decision_id"], chosen["id"], "cure")
    assert any(e.actor == "player" and e.op == Op.USE for e in report.events)
    assert not s.authority.head().attr("npc", "poisoned")
    assert not s.beliefs("player").holds(Proposition.attr("npc", "poisoned", True))
    s.index.client.close()


def test_answer_uses_the_actual_question_and_known_answer():
    me = scene_beliefs(choice_scene())
    asked = Fact(Proposition.rel("cup", Rel.AT, None))
    me = replace(me, obligations=(Obligation("answer", "npc", asked, me.last_tick),))
    answer = next(c.parsed.candidate for c in build_choices(me) if c.parsed.candidate.topic)
    assert answer.op == Op.TELL and answer.target == "npc"
    assert answer.topic == Fact(Proposition.rel("cup", Rel.AT, "table"))


def test_hidden_truth_does_not_change_menu_content_or_order():
    a, b = choice_scene(secret_martial=0.01), choice_scene(secret_martial=0.99)
    assert a.state.fingerprint() != b.state.fingerprint()
    assert build_choices(scene_beliefs(a)) == build_choices(scene_beliefs(b))


def test_three_independent_branches_have_meaningful_state_and_opportunity_differences():
    sc = choice_scene()
    outcomes = []
    for i in range(3):
        s = GameSession(sc, policies={a: Idle() for a in sc.npcs}, pipeline=False)
        d = s.choices.current()
        choice = d["choices"][i]
        spec = s.choices.resolve(d["decision_id"], choice["id"])
        s.choose(d["decision_id"], choice["id"], f"branch-{i}")
        state = s.authority.head()
        me = s.beliefs("player")
        outcomes.append((state.target("player", Rel.AT), state.target("cup", Rel.AT),
                         tuple((c.parsed.candidate.op, c.parsed.candidate.target) for c in build_choices(me))))
        assert state.clock == sc.state.clock + 1
        assert spec.parsed.candidate.op in (Op.MOVE, Op.STUDY, Op.TAKE)
        s.index.client.close()
    assert len(set(outcomes)) == 3
