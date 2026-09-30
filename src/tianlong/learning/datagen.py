"""
[INPUT]: 依赖 kernel 的 Kernel，cognition 的 BeliefStore / BeliefChange / candidates，agents 的 ScriptedPolicy / HeuristicPredictor / Situation，
         learning/task 的 TaskConfig，learning/samples 的 env_sample / agent_sample，learning/parallel 的 ordered_map，core 的变化类型
[OUTPUT]: 对外提供 RolloutConfig、Rollouts、collect()（workers > 1 时按世界分给子进程，与顺序收集逐项相同）、
          own_effect_slots()、observation_gain()
[POS]: learning 的数据工厂：在程序化小世界（一半江湖化）里用“脚本策略 + 分层随机探索”行动，由内核实际执行，
       同时记录环境样本与角色样本。每个 tick 只让一个角色行动，使环境标签只反映这一个行动的后果（孤立行动效果模型：
       其他角色同时行动时的主观预测不在此列）。“有效新观察数”= 行动者信念里发生变化的槽位，扣除本行动的直接效果——
       自己走到了哪、拿起了什么、学到了几成都不算“获知”，看见新房间里的东西、发现藏匿物、得知门锁着才算
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import HeuristicPredictor
from tianlong.cognition import BeliefChange, BeliefStore, Candidate, candidates
from tianlong.core import (
    ATTR_PREFIX,
    TACTILE_ATTRS,
    AddRelation,
    Change,
    Event,
    Op,
    Outcome,
    Rel,
    RemoveRelation,
    SetAttr,
    derive_seed,
    make_id,
)
from tianlong.kernel import Kernel
from tianlong.learning.parallel import ordered_map
from tianlong.learning.task import TaskConfig

if TYPE_CHECKING:
    from tianlong.learning.samples import Sample


@dataclass(frozen=True)
class RolloutConfig:
    worlds: int = 300
    steps: int = 24
    epsilon: float = 0.8      # 随机探索比例：覆盖失败与少见行动
    seed: int = 0
    task: TaskConfig = TaskConfig()   # 任务分布（江湖化比例、规模、修习覆盖旋钮）：与 RL 环境、评测同一份


@dataclass
class Rollouts:
    env: list[Sample] = field(default_factory=list)
    agent: list[Sample] = field(default_factory=list)
    world_of: list[int] = field(default_factory=list)   # 样本所属世界：按世界切分训练/验证，检验对新布局的泛化


def _stratified(rng: random.Random, cands: tuple[Candidate, ...]) -> Candidate:
    """先均匀选操作、再在该操作内选对象：言语候选数量庞大，不分层就会淹没其余行动；等待交给脚本策略去产生。"""
    by_op: dict = defaultdict(list)
    for c in cands:
        if c.op != Op.WAIT or len(cands) == 1:
            by_op[c.op].append(c)
    op = rng.choice(sorted(by_op, key=lambda o: o.value))
    return rng.choice(by_op[op])


def own_effect_slots(changes: tuple[Change, ...], event: Event | None = None) -> frozenset[tuple[str, str]]:
    """本行动直接造成的、以及随得手必然附带的信念槽位 (主语, 谓词)：
    世界变化本身；拿到手的东西随之而来的手感（锋利、淬毒、所载武功）；走过的那扇门“没锁”。"""
    out: set[tuple[str, str]] = set()
    for c in changes:
        if isinstance(c, AddRelation | RemoveRelation):
            out.add((c.rel.src, c.rel.type.value))
            if event is not None and isinstance(c, AddRelation) and c.rel.type == Rel.AT and c.rel.dst == event.actor:
                out.update((c.rel.src, ATTR_PREFIX + k) for k in TACTILE_ATTRS)
        elif isinstance(c, SetAttr):
            out.add((c.entity, ATTR_PREFIX + c.key))
            if c.key == "subdued_until":
                out.add((c.entity, ATTR_PREFIX + "subdued"))
    if event is not None and event.op == Op.MOVE and event.outcome == Outcome.SUCCESS and event.intent.obj:
        out.add((event.intent.obj, ATTR_PREFIX + "locked"))
    return frozenset(out)


def observation_gain(changes: list[BeliefChange], own: frozenset[tuple[str, str]]) -> int:
    """变化了的信念槽位数（同一槽位的“旧值作废 + 新值确立”只算一次），扣除本行动的直接效果。"""
    slots = {(b.prop.subject, b.prop.predicate) for c in changes for b in (c.before, c.after) if b is not None}
    return len(slots - own)


def _collect_world(job: tuple[RolloutConfig, int]) -> tuple[list[Sample], list[Sample]]:
    """一个世界的全部样本（环境视角，角色视角）：只依赖 (配置, 世界序号) 派生的种子。"""
    from tianlong.learning.samples import agent_sample, env_sample

    cfg, w = job
    kernel, policy, predictor = Kernel(), ScriptedPolicy(), HeuristicPredictor()
    sc = cfg.task.scenario(cfg.seed * 1_000_003 + w)
    rng = random.Random(derive_seed("rollout", cfg.seed, w))
    state = sc.state
    stores = {a: BeliefStore(a, trust=dict(p.trust)).revise_all(sc.priors.get(a, ()))[0]
              for a, p in sc.profiles.items()}
    agents = sorted(sc.profiles)
    env_out: list[Sample] = []
    agent_out: list[Sample] = []
    for _ in range(cfg.steps):
        actor = rng.choice(agents)
        profile = sc.profiles[actor]
        interests = list(profile.interests())
        cands = candidates(stores[actor], interests)
        if rng.random() < cfg.epsilon:
            cand = _stratified(rng, cands)
        else:
            preds = tuple(predictor.predict(stores[actor], state.clock, cands, interests, profile=profile))
            cand = cands[policy.choose(Situation(actor, profile, stores[actor], state.clock, cands, preds)).index]
        intent = cand.to_intent(make_id("int", sc.world_id, actor, state.version), actor, state.version)
        result = kernel.step(state, [intent])
        mine = next(e for e in result.events if e.intent.id == intent.id)
        success = mine.outcome == Outcome.SUCCESS
        new_stores = dict(stores)
        changed: list[BeliefChange] = []
        for o in result.observations:
            new_stores[o.observer], cs = new_stores[o.observer].revise(o.percept)
            if o.observer == actor:
                changed.extend(cs)
        gain = observation_gain(changed, own_effect_slots(mine.changes, mine))
        env_out.append(env_sample(state, result.state, actor, cand, success))
        agent_out.append(agent_sample(stores[actor], new_stores[actor], state.clock, actor, cand, success, gain))
        state, stores = result.state, new_stores
    return env_out, agent_out


def collect(cfg: RolloutConfig, workers: int = 1) -> Rollouts:
    """workers > 1 时各世界分给子进程，按世界序号拼回（与顺序收集逐项相同）；0 = 本机全部核。"""
    out = Rollouts()
    for w, (env_s, agent_s) in enumerate(ordered_map(_collect_world, [(cfg, w) for w in range(cfg.worlds)], workers)):
        out.env += env_s
        out.agent += agent_s
        out.world_of += [w] * len(env_s)
    return out
