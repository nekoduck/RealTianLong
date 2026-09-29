"""
[INPUT]: 依赖 tianlong.agents.predictors 的 HeuristicPredictor / BranchValuer / branch_value / direct_effects / imagine / PRED_FIELDS，
         tianlong.agents.policies 的 ScriptedPolicy / Situation，tianlong.cognition 的 BeliefStore / Belief / candidates，
         tianlong.core.goals 的 UnsupportedGoal，tianlong.kernel 的 Kernel，tianlong.learning.task 的 TaskConfig
         （GNN 用例另需 torch + torch_geometric）
[OUTPUT]: 验收 Prediction v2 的共享假想快路径：启发式与 GNN 预测在 TaskConfig().scenario(seed) 世界里与逐候选完整假想
          （branch_value 的定义，即改动前的路径）逐位相同——含“此刻”晚于认知最后一刻、并存说法、时间闸门未到的目标；
          没有已激活目标不做假想，碰不到任何目标读集的候选不做假想，相同的假想事实只做一次；
          反查（谁拥有）按整个谓词记入读集；不合法的目标与定义在同一时机报错（候选全无事实时两边都不报）；
          目标求值若读了读集追踪器不认识的查询，当场报错而不是漏记依赖
[POS]: tests 的预测器等价层：快路径只许省功，不许改值（比较 float.hex，连 -0.0 与 0.0 都分得开）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tianlong.agents import predictors
from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import (
    PRED_FIELDS,
    BranchValuer,
    HeuristicPredictor,
    branch_value,
    direct_effects,
)
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.cognition.beliefs import Belief
from tianlong.core import EntitySketch, Fact, Kind, Manner, Modality, Op, Percept, Proposition, Rel, make_id
from tianlong.core.goals import UnsupportedGoal
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel import Kernel
from tianlong.learning.task import TaskConfig

SEEDS = range(8)
LATER = 45          # 另在“此刻晚于认知最后一刻”时再问一遍：假想认知的此刻前移，时间闸门也可能打开


def definition(values: list[float]):
    """改动前的路径：每个候选各自完整假想一次（branch_value 的定义本身），顺手记下每个势能差。"""

    class Definition:
        def __init__(self, store: BeliefStore, now: int, profile: Profile | None) -> None:
            self.args = (store, now, profile)

        def __call__(self, facts) -> float:
            values.append(branch_value(*self.args, facts))
            return values[-1]

    return Definition


def bits(preds) -> list[tuple]:
    return [(p.note, *(float.hex(float(getattr(p, f))) for f in PRED_FIELDS)) for p in preds]


def decisions(seeds=SEEDS, steps: int = 10):
    """脚本策略在程序化世界里走几步：每一刻每个角色的（认知, 此刻, 设定, 候选）。世界真的在动——
    有人走动、拿取、动手、传话，认知里才会出现并存的说法与变了的下落。"""
    task, kernel, policy, heur = TaskConfig(), Kernel(), ScriptedPolicy(), HeuristicPredictor()
    out = []
    for seed in seeds:
        sc = task.scenario(seed)
        state = sc.state
        stores = {a: BeliefStore(a, trust=dict(p.trust)).revise_all(sc.priors.get(a, ()))[0]
                  for a, p in sc.profiles.items()}
        for _ in range(steps):
            intents = []
            for a, prof in sorted(sc.profiles.items()):
                cands = candidates(stores[a], prof.interests())
                out.append((stores[a], state.clock, prof, cands))
                preds = tuple(heur.predict(stores[a], state.clock, cands, prof.interests(), profile=prof))
                choice = policy.choose(Situation(a, prof, stores[a], state.clock, cands, preds))
                intents.append(cands[choice.index].to_intent(make_id("eq", seed, a, state.version), a, state.version))
            r = kernel.step(state, intents)
            for o in r.observations:
                stores[o.observer] = stores[o.observer].revise(o.percept)[0]
            state = r.state
    return out


def compare(predict, target, monkeypatch, cases) -> tuple[list, list, list[float]]:
    """同一批决策各跑一遍快路径与定义：两边的逐位表示，以及定义算出的全部势能差。"""
    fast = [bits(predict(s, now, c, p.interests(), profile=p)) for s, now, p, c in cases]
    values: list[float] = []
    with monkeypatch.context() as m:
        m.setattr(target, "BranchValuer", definition(values))
        slow = [bits(predict(s, now, c, p.interests(), profile=p)) for s, now, p, c in cases]
    return fast, slow, values


def test_heuristic_predictions_are_bitwise_identical_to_the_definition(monkeypatch):
    base = decisions()
    cases = base + [(s, now + LATER, p, c) for s, now, p, c in base]
    fast, slow, values = compare(HeuristicPredictor().predict, predictors, monkeypatch, cases)
    assert fast == slow
    # 不是空转：定义确实给出过进展与损失，也确实遇到过“此刻前移 + 读到并存说法”的目标
    assert any(v > 0 for v in values) and any(v < 0 for v in values)
    valuers = []
    for s, now, p, c in cases:
        v = BranchValuer(s, now, p)
        for x in c:
            v(direct_effects(s, x))
        valuers.append(v)
    assert any(v.shifted and any(t.ordered for t in v.traces) for v in valuers)


def test_gnn_predictions_are_bitwise_identical_to_the_definition(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from tianlong.learning import predictor as gnn_module
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.predictor import GNNPredictor

    class Scramble(DynamicsModel):
        """确定性地打乱位置指针头（只在合法容纳者与 UNKNOWN/GONE/NEW 之间挑）：每个候选都“预测”出一串
        换了容纳者（有的落到自己手上）或确知不在原处的事实。"""

        def forward(self, data, encoded=None):
            out = super().forward(data, encoded)
            pick = torch.sin(torch.arange(out.holder.numel(), dtype=torch.float32).reshape(out.holder.shape) * 1.7)
            out.holder = torch.where(torch.isfinite(out.holder), pick, float("-inf"))
            return out

    torch.manual_seed(0)
    gnn = GNNPredictor(Scramble(16))
    base = decisions(range(4), steps=5)
    cases = base + [(s, now + LATER, p, c) for s, now, p, c in base]
    fast, slow, values = compare(gnn.predict, gnn_module, monkeypatch, cases)
    assert fast == slow
    assert any(v > 0 for v in values) and any(v < 0 for v in values), "定义确实给出过进展与损失"


# ============================================================
#  跳过是真跳过：没有已激活目标、或碰不到目标读集，就不做假想
# ============================================================

NAMES = (EntitySketch("hall", Kind.PLACE, "大堂"), EntitySketch("yard", Kind.PLACE, "后院"),
         EntitySketch("d1", Kind.DOOR, "边门"), EntitySketch("me", Kind.PERSON, "我"),
         EntitySketch("cup", Kind.ITEM, "茶杯"), EntitySketch("coin", Kind.ITEM, "铜钱"))


def small_mind() -> BeliefStore:
    layout = Percept(0, Modality.SCENE, None, (
        Fact(Proposition.rel("d1", Rel.CONNECTS, "hall")), Fact(Proposition.rel("d1", Rel.CONNECTS, "yard")),
        Fact(Proposition.rel("me", Rel.AT, "hall")), Fact(Proposition.rel("cup", Rel.AT, "hall")),
        Fact(Proposition.rel("coin", Rel.AT, "hall"))), ("hall",), NAMES)
    return BeliefStore("me").revise(layout)[0]


def test_skipped_branches_are_never_imagined(monkeypatch):
    imagined: list[tuple] = []
    real = predictors.imagine
    monkeypatch.setattr(predictors, "imagine", lambda s, now, facts: imagined.append(tuple(facts)) or real(s, now, facts))
    mind = small_mind()
    cands = [Candidate(Op.TAKE, "coin"), Candidate(Op.TAKE, "cup"), Candidate(Op.MOVE, "yard", "d1"),
             Candidate(Op.MOVE, "yard", "d1", Manner.CAREFUL), Candidate(Op.WAIT)]
    later = Profile("me", "x", "x", (Goal(GoalKind.ACQUIRE, "cup", not_before=100),))

    HeuristicPredictor().predict(mind, 5, cands, profile=later)
    assert imagined == [], "目标还没到时辰：没有已激活的目标，一条分支也不假想"

    preds = HeuristicPredictor().predict(mind, 100, cands, profile=later)
    assert imagined == [(Fact(Proposition.rel("cup", Rel.AT, "me")),), (Fact(Proposition.rel("me", Rel.AT, "yard")),)], \
        "拿铜钱碰不到“茶杯在谁手里、我在哪”：不假想；同一去处的两种走法只假想一次"
    assert preds[0].progress == 0.0 and preds[1].progress > 0.0 and preds[2].progress == preds[3].progress


def test_a_later_now_that_reorders_coexisting_claims_is_recomputed():
    """并存说法的先后按“衰减后的可信度”排：经 revise() 形成的认知里它与此刻无关，但直接构造的认知可以有晚于 last_tick
    学到的信念——此刻一前移，先后就换了。定义照实反映这一点（拿铜钱也“有进展”），快路径必须同样重算而不是沿用基线。"""
    mind = small_mind()
    here, there = Proposition.rel("cup", Rel.AT, "hall"), Proposition.rel("cup", Rel.AT, "yard")
    beliefs = {p: b for p, b in mind.beliefs.items() if p != here}
    beliefs[here] = Belief(here, True, 0.8, Modality.SPEECH, 50, "x")      # 晚于 last_tick=0 学到
    beliefs[there] = Belief(there, True, 0.9, Modality.SPEECH, 0, "y")
    odd = replace(mind, beliefs=beliefs)
    assert odd.location_of("cup") == "yard" and odd.last_tick == 0
    prof = Profile("me", "x", "x", (Goal(GoalKind.ACQUIRE, "cup"),))
    coin = (Fact(Proposition.rel("coin", Rel.AT, "me")),)
    expected = branch_value(odd, 100, prof, coin)
    assert expected != 0.0, "此刻前移让“茶杯就在大堂”排到了前面"
    assert float.hex(BranchValuer(odd, 100, prof)(coin)) == float.hex(expected)


def test_reverse_lookups_depend_on_the_whole_predicate():
    """owners() 反查扫过 OWNS 谓词下的全部信念：假想“我成了失主”写在 (me, OWNS) 槽位，守护目标从没按这个槽位读过，
    却照样受影响——读集必须按谓词记下反查。"""
    mind = small_mind().revise(Percept(1, Modality.SELF, None, (Fact(Proposition.rel("cup", Rel.AT, "me")),)))[0]
    prof = Profile("me", "x", "x", (Goal(GoalKind.PROTECT, "cup", home="yard"),))
    owns = (Fact(Proposition.rel("me", Rel.OWNS, "cup")),)
    expected = branch_value(mind, 1, prof, owns)
    assert expected > 0.0, "成了失主，东西在自己手上就算守住了"
    assert float.hex(BranchValuer(mind, 1, prof)(owns)) == float.hex(expected)


def test_an_invalid_goal_fails_exactly_where_the_definition_does():
    broken = Profile("me", "x", "x", (Goal(GoalKind.PROTECT, "cup"),))       # 守护却没说守在哪（缺 home）
    wait, take = (Candidate(Op.WAIT),), (Candidate(Op.TAKE, "cup"),)
    for branches in (BranchValuer, lambda s, n, p: lambda f: branch_value(s, n, p, f)):
        assert branches(small_mind(), 1, broken)(direct_effects(small_mind(), wait[0])) == 0.0, "没有事实就不求目标"
        with pytest.raises(UnsupportedGoal):
            branches(small_mind(), 1, broken)(direct_effects(small_mind(), take[0]))


def test_goal_reads_outside_the_trace_fail_loudly():
    trace = predictors._ReadTrace(small_mind())
    assert trace.location_of("cup") == "hall" and ("cup", Rel.AT.value) in trace.slots
    with pytest.raises(AttributeError):
        trace.beliefs        # noqa: B018  没登记的读法不许绕过读集
