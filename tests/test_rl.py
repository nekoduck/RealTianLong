"""
[INPUT]: 依赖 tianlong.learning.rl 的 env / observation / rewards / module / imitation / evaluation / train / policy，
         tianlong.learning.task 的 TaskConfig，tianlong.core.goals 的 GoalRegistry / UnsupportedGoal，tianlong.scenarios
[OUTPUT]: 强化学习层测试：观测合乎空间、掩码只屏蔽空位；奖励分项语义（任务跃迁/塑形/成本、搜身落空）；
          验收 T01–T04（场景配置贯通到环境、寻仇按 until 给任务奖励、持续保护按窗口判定、未注册目标显式失败）
          与 B01–B04（等待权重边界无 NaN、批组成不改变贡献、专家在不知下落时去探索、留出世界上的分动作指标）；
          学得的策略接入决策图、选中等待时标 idle；训练期消融（预测/记忆列）在环境、tag、上线策略里是同一个定义；PPO 冒烟（slow）
[POS]: tests 的 RL 层；PPO 冒烟用例标记 slow（默认不跑，`pytest -m slow` 显式运行）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")   # 学习层 extras 未装时整模块跳过（核心零依赖）

torch = pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

from tianlong.agents.policies import ScriptedPolicy, Situation  # noqa: E402
from tianlong.agents.predictors import HeuristicPredictor  # noqa: E402
from tianlong.cognition import BeliefStore, candidates  # noqa: E402
from tianlong.core import (  # noqa: E402
    Entity,
    Fact,
    Intent,
    Kind,
    Modality,
    Op,
    Proposition,
    Rel,
    Relation,
    WorldState,
    at,
)
from tianlong.core.goals import GoalRegistry, UnsupportedGoal  # noqa: E402
from tianlong.core.profiles import Goal, GoalKind, Profile  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.kernel.perception import make_percept, scene_percept  # noqa: E402
from tianlong.learning.rl.env import TianlongEnv  # noqa: E402
from tianlong.learning.rl.evaluation import evaluate, net_policy, wait_policy  # noqa: E402
from tianlong.learning.rl.imitation import (  # noqa: E402
    Demo,
    bc_weights,
    behavior_clone,
    collect_demos,
    holdout_metrics,
    weighted_batch_loss,
)
from tianlong.learning.rl.module import GraphPolicyNet  # noqa: E402
from tianlong.learning.rl.rewards import GoalTracker, RewardWeights, step_reward  # noqa: E402
from tianlong.learning.rl.train import RLConfig  # noqa: E402
from tianlong.learning.task import TaskConfig  # noqa: E402
from tianlong.scenarios import build_warehouse  # noqa: E402

pytestmark = pytest.mark.rl


def test_observations_fit_space_and_mask_only_padding():
    env = TianlongEnv({"horizon": 5})
    obs, _ = env.reset(seed=11)
    for a in env.agents:
        assert env.observation_spaces[a].contains(obs[a])
        n_valid = int(obs[a]["action_mask"].sum())
        assert n_valid == len(env._cands[a]), "掩码 = 保留的候选集，不多不少"
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(env.expert_actions())
    assert trunc["__all__"] and set(rew) == set(env.agents)
    assert all(math.isclose(rew[a], info[a]["reward"]["total"]) for a in env.agents), "总奖励 = 分项之和"


# ============================================================
#  奖励分项
# ============================================================


def _tracker(state: WorldState, goals: dict[str, tuple[Goal, ...]], allies=None) -> GoalTracker:
    profiles = {a: Profile(a, "x", "x", g, allies=(allies or {}).get(a, ())) for a, g in goals.items()}
    t = GoalTracker(GoalRegistry(), profiles)
    t.start(state)
    return t


def test_reward_components_on_warehouse():
    sc = build_warehouse()
    s0, k = sc.state, Kernel()
    goals = {"player": (Goal(GoalKind.ACQUIRE, "key"),), "guard": (Goal(GoalKind.PROTECT, "key", home="table"),)}
    tr = _tracker(s0, goals)
    r = k.step(s0, [Intent("t", "player", Op.TAKE, "key", based_on=0)])
    parts = step_reward(tr, s0, r.state, r.events)
    assert parts["player"].task == 1.0, "拿到想要的东西：任务达成的跃迁"
    assert parts["guard"].task == -1.0, "守护的东西被拿走：持续目标被破坏"
    assert parts["player"].shaping > 0 > parts["guard"].shaping
    # 守卫搜一个身上没有钥匙的人 = 搜身落空（成本项，不混进任务奖励）
    tr2 = _tracker(s0, goals)
    s1 = k.step(s0, [Intent("m", "guard", Op.MOVE, "warehouse", "door_main", based_on=0)]).state
    tr2.update(s1)
    r2 = k.step(s1, [Intent("i", "guard", Op.INSPECT, "player", based_on=1)])
    p2 = step_reward(tr2, s1, r2.state, r2.events)
    assert p2["guard"].search == -RewardWeights().search_penalty and p2["guard"].task == 0.0


def duel(foe_martial: float = 1.0) -> WorldState:
    ents = [Entity.make("hall", Kind.PLACE, "厅"), Entity.make("a", Kind.PERSON, "甲", martial=foe_martial),
            Entity.make("b", Kind.PERSON, "乙", martial=0.0), Entity.make("c", Kind.PERSON, "丙", martial=0.0),
            Entity.make("salve", Kind.ITEM, "金创药", small=True, cures="wounded")]
    rels = [Relation("a", Rel.AT, "hall"), Relation("b", Rel.AT, "hall"), Relation("c", Rel.AT, "hall"),
            Relation("salve", Rel.AT, "c")]
    return WorldState.build(2, at(1, 9, 0), ents, rels)


def test_t02_hostile_until_wounded_gets_task_reward_not_item_fallback():
    s0, k = duel(), Kernel()
    tr = _tracker(s0, {"a": (Goal(GoalKind.HOSTILE, person="b", until="wounded"),), "b": (), "c": ()})
    r = k.step(s0, [Intent("x", "a", Op.ATTACK, "b", based_on=0)])
    assert r.state.attr("b", "wounded"), "前提：身手 1.0 对 0.0 必然得手"
    parts = step_reward(tr, s0, r.state, r.events)
    rec = tr.records["a"][0]
    assert parts["a"].task == 1.0 and rec.newly_achieved and rec.achieved_at == r.state.clock
    assert tr.achieved("a")


def test_t03_defend_is_judged_over_the_window_not_by_a_snapshot():
    s0, k = duel(), Kernel()
    tr = _tracker(s0, {"a": (), "b": (), "c": (Goal(GoalKind.DEFEND, person="b"),)})
    rec = tr.records["c"][0]
    assert rec.initial is True and tr.achieved("c"), "开局乙安然无恙：初态满足（不是学来的）"
    s1 = k.step(s0, [Intent("w", "c", Op.WAIT, based_on=0)]).state
    assert step_reward(tr, s0, s1, ())["c"].task == 0.0
    r = k.step(s1, [Intent("x", "a", Op.ATTACK, "b", based_on=1)])
    assert step_reward(tr, s1, r.state, r.events)["c"].task == -1.0 and rec.violated_at is not None
    r2 = k.step(r.state, [Intent("u", "c", Op.USE, "b", "salve", based_on=2)])
    assert not r2.state.attr("b", "wounded")
    assert step_reward(tr, r.state, r2.state, r2.events)["c"].task == 1.0, "恢复有奖励"
    assert rec.satisfied and rec.maintained is False and not tr.achieved("c"), "窗口里被破坏过：不算守住"


def test_t04_unregistered_goal_type_fails_loudly():
    s0 = duel()
    only_items = GoalRegistry(e for e in GoalRegistry()._by_kind.values()
                              if e.kind in (GoalKind.PROTECT, GoalKind.ACQUIRE, GoalKind.DELIVER))
    with pytest.raises(UnsupportedGoal):
        GoalTracker(only_items, {"a": Profile("a", "x", "x", (Goal(GoalKind.HOSTILE, person="b"),))})
    with pytest.raises(UnsupportedGoal):
        GoalTracker(GoalRegistry(), {"a": Profile("a", "x", "x", (Goal(GoalKind.HOSTILE),))})   # 缺字段
    # 取样服从启用的目标族：只启用物品目标时，江湖世界里也不会抽到寻仇/护人/守地/潜逃
    task = TaskConfig(jianghu=1.0, goals=("protect", "acquire", "deliver"))
    env = TianlongEnv({"task": task.to_dict()})
    for sd in range(20):
        env.reset(seed=sd)
        assert all(g.kind.value in task.goals for p in env.scenario.profiles.values() for g in p.goals)
    assert env.coverage["goal:protect"] > 0 and "goal:hostile" not in env.coverage
    # 取样与奖励注册表不一致（显式注入了没注册的目标族）→ reset 明确报错，而不是静默套用别的奖励
    full = TianlongEnv({"task": TaskConfig(jianghu=1.0).to_dict()})
    full.registry = only_items
    seed = next(sd for sd in range(200) if any(g.kind == GoalKind.HOSTILE for p in
                                               TaskConfig(jianghu=1.0).scenario(sd).profiles.values() for g in p.goals))
    with pytest.raises(UnsupportedGoal):
        full.reset(seed=seed)
    assert s0.version == 0
    with pytest.raises(ValueError, match="规模"):
        TaskConfig(max_persons=9)


def test_t01_task_config_reaches_the_ppo_environment():
    cfg = RLConfig(jianghu=1.0, horizon=4, scroll_held=1.0)
    env = TianlongEnv(cfg.env_config())          # PPO 的每个 env runner 用的就是这份 env_config
    assert env.task == cfg.task() and env.task.jianghu == 1.0 and env.horizon == 4
    for sd in range(5):
        env.reset(seed=sd)
        assert env.state.fingerprint() == TaskConfig(jianghu=1.0, horizon=4, scroll_held=1.0).scenario(sd).state.fingerprint()
    assert env.coverage["episodes"] == 5 and env.coverage["jianghu_worlds"] == 5, "报告的场景覆盖与配置一致"
    plain = TianlongEnv(RLConfig(jianghu=0.0).env_config())
    for sd in range(5):
        plain.reset(seed=sd)
    assert plain.coverage["jianghu_worlds"] == 0


# ============================================================
#  模仿学习
# ============================================================


def test_b01_wait_share_bounds_are_defined_and_never_nan():
    acts = [0] * 9 + [3]
    assert bc_weights(acts, 0.5)[2] == 0.5
    w_wait, w_act, _ = bc_weights(acts, 0.5)
    assert math.isclose(9 * w_wait, 5.0) and math.isclose(w_act, 5.0), "权重使等待总份额恰为 0.5、平均权重为 1"
    assert bc_weights(acts, 0.0)[:2] == (0.0, 10.0) and bc_weights(acts, 1.0)[:2] == (10 / 9, 0.0)
    assert bc_weights([0, 0, 0], 0.2) == (1.0, 1.0, 1.0), "只有等待：份额全归等待，如实报告"
    assert bc_weights([2, 5], 0.7) == (1.0, 1.0, 0.0)
    for bad in (-0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bc_weights(acts, bad)
    env = TianlongEnv({"horizon": 4})
    demos = collect_demos(env, 2, seed=1)
    all_wait = [Demo(d.obs, 0, "idle", d.world) for d in demos]
    for share in (0.0, 0.5, 1.0):
        rep = behavior_clone(GraphPolicyNet(16), all_wait, 1, wait_share=share, log=lambda *_: None)
        assert all(math.isfinite(r["loss"]) for r in rep)


def test_b02_epoch_contribution_does_not_depend_on_batch_composition():
    g = torch.Generator().manual_seed(0)
    per = torch.rand(40, generator=g)
    is_wait = torch.tensor([i % 5 != 0 for i in range(40)])
    w_wait, w_act, share = bc_weights([0 if w else 1 for w in is_wait.tolist()], 0.5)

    def epoch(order, size):
        tot = wait = 0.0
        for i in range(0, 40, size):
            idx = order[i:i + size]
            tot += float(weighted_batch_loss(per[idx], is_wait[idx], w_wait, w_act, 8))
            wait += float(weighted_batch_loss(per[idx] * is_wait[idx], is_wait[idx], w_wait, w_act, 8))
        return tot, wait

    a = epoch(torch.arange(40), 8)
    b = epoch(torch.argsort(is_wait.int(), stable=True), 5)       # 批按类别扎堆：有全等待的批
    assert a == pytest.approx(b), "一个 epoch 的总贡献与分批方式无关"
    weight_mass = float(torch.where(is_wait, w_wait, w_act).sum())
    assert float(torch.where(is_wait, w_wait, 0.0).sum()) / weight_mass == pytest.approx(share)


def explore_world() -> tuple[WorldState, BeliefStore]:
    """码头—货栈—账房一条线；玉佩藏在账房，阿福在码头，只知道地图与眼前。"""
    ents = [Entity.make("p0", Kind.PLACE, "码头"), Entity.make("p1", Kind.PLACE, "货栈"),
            Entity.make("p2", Kind.PLACE, "账房"), Entity.make("d0", Kind.DOOR, "门甲"), Entity.make("d1", Kind.DOOR, "门乙"),
            Entity.make("h0", Kind.PERSON, "阿福"), Entity.make("jade", Kind.ITEM, "玉佩", small=True)]
    rels = [Relation("d0", Rel.CONNECTS, "p0"), Relation("d0", Rel.CONNECTS, "p1"), Relation("d1", Rel.CONNECTS, "p1"),
            Relation("d1", Rel.CONNECTS, "p2"), Relation("h0", Rel.AT, "p0"), Relation("jade", Rel.AT, "p2")]
    s = WorldState.build(4, at(1, 9, 0), ents, rels)
    told = make_percept(s, Modality.SCENE, facts=tuple(Fact(Proposition.of(r)) for r in s.sorted_relations()
                                                       if r.type == Rel.CONNECTS))
    store = BeliefStore("h0").revise_all([told, scene_percept(s, "h0")])[0]
    heard = make_percept(s, Modality.SCENE, facts=())         # 只知道有块玉佩（名字），不知在哪
    from dataclasses import replace

    from tianlong.core import EntitySketch
    store = store.revise(replace(heard, sketches=(EntitySketch("jade", Kind.ITEM, "玉佩", (), seen=False),)))[0]
    return s, store


def test_b03_expert_explores_by_its_own_map_instead_of_waiting_forever():
    s, store = explore_world()
    prof = Profile("h0", "寻宝", "x", (Goal(GoalKind.ACQUIRE, "jade"),))
    policy, k, pred = ScriptedPolicy(), Kernel(), HeuristicPredictor()
    tags = []
    for _ in range(12):
        cands = candidates(store, prof.interests())
        choice = policy.choose(Situation("h0", prof, store, s.clock, cands, tuple(pred.predict(store, s.clock, cands))))
        tags.append(choice.tag)
        r = k.step(s, [cands[choice.index].to_intent(f"i{s.version}", "h0", s.version)])
        store = store.revise_all(o.percept for o in r.observations if o.observer == "h0")[0]
        s = r.state
        if s.target("jade", Rel.AT) == "h0":
            break
    assert s.target("jade", Rel.AT) == "h0", tags
    assert "explore" in tags and "stuck_unknown" not in tags, "不知下落时去找，而不是卡住等待"


def test_b04_holdout_metrics_on_other_worlds():
    env = TianlongEnv({"horizon": 12})
    demos = collect_demos(env, 30, seed=3)
    held = collect_demos(env, 8, seed=97)
    assert {d.world for d in demos} and not {id(d) for d in held} & {id(d) for d in demos}
    torch.manual_seed(0)                       # 先定种子再建网络：初始化不能取决于先跑了哪些测试
    net = GraphPolicyNet(32)
    reps = behavior_clone(net, demos, 6, seed=0, wait_share=0.5, log=lambda *_: None)
    assert reps[-1]["declared_wait_share"] == 0.5 and math.isfinite(reps[-1]["wait_loss_share"])
    m = holdout_metrics(net, held)
    assert set(m) >= {"wait_precision", "wait_recall", "act_precision", "act_recall", "act_exact_acc", "by_expert_tag"}
    assert m["act_recall"] > 0.0, "留出世界上确实学到了非等待动作（永远等待的基线召回为 0）"
    base = evaluate(env, wait_policy, 3)
    learned = evaluate(env, net_policy(net), 3)
    assert set(base) == set(learned) and "search_miss_rate" in learned


def test_learned_policy_plugs_into_npc_pipeline():
    pytest.importorskip("langgraph")
    from tianlong.agents.npc_graph import NpcContext
    from tianlong.agents.orchestrator import Orchestrator
    from tianlong.agents.port import AgentPort
    from tianlong.learning.rl.policy import LearnedPolicy
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority

    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    s = auth.head()
    policy = LearnedPolicy(GraphPolicyNet(32))
    ctx = {a: NpcContext(AgentPort(a, sc.profiles[a], auth.ref.world_id, auth.ref.branch_id, s.version, s.clock,
                                   beliefs=lambda a=a: auth.store.beliefs(auth.ref, a)),
                         policy=policy, max_candidates=policy.spec.max_cands)
           for a in sc.npcs}
    ds = Orchestrator().decide(ctx)
    assert all("策略网络" in d.rationale for d in ds)
    auth.settle([d.intent for d in ds])   # 学得的策略产出的意图照样只能经内核结算


def test_learned_policy_tags_its_waits_idle():
    from tianlong.learning.rl.observation import build_observation
    from tianlong.learning.rl.policy import LearnedPolicy

    env = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 4}})
    env.reset(seed=5)
    sit = env.situation(env.agents[0])
    policy = LearnedPolicy(GraphPolicyNet(32))
    shown = build_observation(sit.beliefs, sit.now, sit.profile, sit.candidates, sit.predictions, policy.spec,
                              sit.memory).candidates
    for k, cand in enumerate(shown):
        policy.net = lambda batch, k=k: (torch.nn.functional.one_hot(torch.tensor([k]), len(batch["action_mask"][0])
                                                                     ).float(), None)
        choice = policy.choose(sit)
        assert sit.candidates[choice.index] == cand
        assert choice.tag == ("idle" if cand.op == Op.WAIT else ""), "选中等待标 idle，外层驱力层据此试 IDLE"
    assert any(c.op == Op.WAIT for c in shown)


@pytest.mark.slow
def test_ppo_smoke():
    from tianlong.learning.rl.train import train_ppo
    cfg = RLConfig(ppo_iterations=1, train_batch=200, horizon=10, hidden=32, jianghu=1.0)
    net = train_ppo(cfg, GraphPolicyNet(32), log=lambda *_: None)
    env = TianlongEnv(cfg.env_config())
    obs, _ = env.reset(seed=1)
    act = net_policy(net)(env, obs)
    assert all(obs[a]["action_mask"][i] == 1 for a, i in act.items())
    assert np.isfinite(evaluate(env, net_policy(net), 2)["mean_return"])


def test_time_gated_goal_done_before_activation_is_credited_and_initial_is_not_farmable():
    """not_before 目标：激活前自己办成的，激活时照样记功；开局就满足的一次性目标，拆了再办也不给奖励。"""
    s0, k = duel(), Kernel()
    tr = _tracker(s0, {"a": (Goal(GoalKind.HOSTILE, person="b", until="wounded", not_before=s0.clock + 2),),
                       "b": (), "c": ()})
    r = k.step(s0, [Intent("x", "a", Op.ATTACK, "b", based_on=0)])          # 激活前就出了手
    assert step_reward(tr, s0, r.state, r.events)["a"].task == 0.0, "未激活：不求值"
    s2 = k.step(r.state, [Intent("w", "a", Op.WAIT, based_on=1)]).state
    assert step_reward(tr, r.state, s2, ())["a"].task == 1.0, "激活那一刻已办成：记功"
    rec = tr.records["a"][0]
    assert rec.initial is False and rec.newly_achieved
    # 开局即满足的一次性目标：achieved_at 记在开局，永不再给奖励
    sc = build_warehouse()
    tr2 = _tracker(sc.state, {"player": (Goal(GoalKind.ACQUIRE, "key"),)})
    assert tr2.records["player"][0].initial is False
    s_have = k.step(sc.state, [Intent("t", "player", Op.TAKE, "key", based_on=0)]).state
    tr3 = _tracker(s_have, {"player": (Goal(GoalKind.ACQUIRE, "key"),)})
    s_drop = k.step(s_have, [Intent("p", "player", Op.PUT, "harbor", "key", based_on=1)]).state
    s_back = k.step(s_drop, [Intent("t2", "player", Op.TAKE, "key", based_on=2)]).state
    total = step_reward(tr3, s_have, s_drop, ())["player"].task + step_reward(tr3, s_drop, s_back, ())["player"].task
    assert tr3.records["player"][0].initial is True and total == 0.0, "开局就有：放下再拿起不能刷奖励"


def test_demo_holdout_and_eval_seed_namespaces_are_disjoint():
    from tianlong.learning.rl.imitation import DEMO_SEED_FLOOR, demo_seed
    demo = {demo_seed("demo", s, ep) for s in range(12) for ep in range(50)}
    held = {demo_seed("bc_holdout", s, ep) for s in range(12) for ep in range(20)}
    assert not demo & held and min(demo | held) >= DEMO_SEED_FLOOR > RLConfig().eval_seed + 10_000


def test_expert_finds_a_hidden_goal_item_it_only_knows_by_name():
    """目标物品从没见过、还藏着：专家凭名字去找，每到一处先仔细翻查，最终拿到（评审回归：以前只会随意环顾）。"""
    s, store = explore_world()
    s = WorldState.build(s.seed, s.clock, [e.with_attr("hidden", True) if e.id == "jade" else e
                                           for e in s.entities.values()], s.relations)
    prof = Profile("h0", "寻宝", "x", (Goal(GoalKind.ACQUIRE, "jade"),))
    policy, k, pred = ScriptedPolicy(), Kernel(), HeuristicPredictor()
    for _ in range(16):
        cands = candidates(store, prof.interests())
        choice = policy.choose(Situation("h0", prof, store, s.clock, cands, tuple(pred.predict(store, s.clock, cands))))
        r = k.step(s, [cands[choice.index].to_intent(f"i{s.version}", "h0", s.version)])
        store = store.revise_all(o.percept for o in r.observations if o.observer == "h0")[0]
        s = r.state
        if s.target("jade", Rel.AT) == "h0":
            break
    assert s.target("jade", Rel.AT) == "h0"


def test_expert_detours_around_a_door_it_knows_is_locked():
    """自己的地图上有绕得开锁门的路，就绕；不再在锁门前放弃（评审回归）。"""
    ents = [Entity.make("a", Kind.PLACE, "甲地"), Entity.make("b", Kind.PLACE, "乙地"), Entity.make("c", Kind.PLACE, "丙地"),
            Entity.make("dab", Kind.DOOR, "甲乙门", locked=True), Entity.make("dac", Kind.DOOR, "甲丙门"),
            Entity.make("dcb", Kind.DOOR, "丙乙门"), Entity.make("h", Kind.PERSON, "行者")]
    rels = [Relation("dab", Rel.CONNECTS, "a"), Relation("dab", Rel.CONNECTS, "b"), Relation("dac", Rel.CONNECTS, "a"),
            Relation("dac", Rel.CONNECTS, "c"), Relation("dcb", Rel.CONNECTS, "c"), Relation("dcb", Rel.CONNECTS, "b"),
            Relation("h", Rel.AT, "a")]
    s = WorldState.build(5, at(1, 9, 0), ents, rels)
    layout = make_percept(s, Modality.SCENE, facts=tuple(Fact(Proposition.of(r)) for r in s.sorted_relations()
                                                         if r.type == Rel.CONNECTS)
                          + (Fact(Proposition.attr("dab", "locked", True)),))
    store = BeliefStore("h").revise_all([layout, scene_percept(s, "h")])[0]
    prof = Profile("h", "x", "x", (Goal(GoalKind.ESCAPE, home="b"),))
    cands = candidates(store, [])
    c = cands[ScriptedPolicy().choose(Situation("h", prof, store, s.clock, cands, ())).index]
    assert (c.op, c.target, c.obj) == (Op.MOVE, "c", "dac"), c


# ============================================================
#  评审回归：训练期消融是一个定义，训练、评测、上线同用
# ============================================================


def test_training_time_ablation_zeroes_whole_columns_in_env():
    env = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 6}, "ablate": ["memory", "predictions"]})
    obs, _ = env.reset(seed=3)
    for _ in range(4):
        for o in obs.values():
            assert not o["memory"].any() and not o["cand_pred"].any()
        obs, *_ = env.step(env.expert_actions())
    with pytest.raises(ValueError, match="未知的消融"):
        TianlongEnv({"task": {"horizon": 2}, "ablate": ["beliefs"]}).reset(seed=0)


def test_rl_config_names_ablations_in_tag_and_env_config():
    cfg = RLConfig(ablate_memory=True)
    assert cfg.tag() == "policy_ppo_noMem_s0" and cfg.env_config()["ablate"] == ["memory"]
    both = RLConfig(ablate_predictions=True, ablate_memory=True, seed=2)
    assert both.tag() == "policy_ppo_noPred_noMem_s2" and both.ablations() == ("predictions", "memory")


def test_deployed_policy_applies_the_ablation_it_was_trained_with(tmp_path):
    from dataclasses import asdict

    from tianlong.learning.rl.observation import ObsSpec
    from tianlong.learning.rl.policy import LearnedPolicy
    from tianlong.learning.schema import SCHEMA

    net = GraphPolicyNet(32)
    path = tmp_path / "p.pt"
    torch.save({"state_dict": net.state_dict(), "config": {"hidden": 32, "ablate_predictions": True},
                "schema": SCHEMA, "view": "policy", "obs_spec": asdict(ObsSpec()), "ablate": ["predictions"]}, path)
    pol = LearnedPolicy.load(path)
    assert pol.ablations == ("predictions",)
    env = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 4}})
    env.reset(seed=5)
    sit = env.situation(env.agents[0])
    seen = {}

    def spy(policy, name):
        inner = policy.net

        def forward(batch):
            seen[name] = batch["cand_pred"].clone()
            return inner(batch)
        policy.net = forward
        return policy

    spy(pol, "ablated").choose(sit)
    spy(LearnedPolicy(net, pol.spec), "plain").choose(sit)
    assert seen["plain"].any(), "对照：未消融的策略确实看得到预测"
    assert not seen["ablated"].any(), "训练时没见过预测：上线时预测列同样置零"
