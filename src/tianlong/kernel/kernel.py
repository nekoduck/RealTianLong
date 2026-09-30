"""
[INPUT]: 依赖 core 的 WorldState / Intent / Event / Observation / signature_error，kernel/rules 的规则注册表，
         kernel/perception 的 Witnessing / scene_percept，kernel/invariants 的 assert_invariants
[OUTPUT]: 对外提供 Kernel（纯函数式结算器）、StepResult、KERNEL_VERSION（规则语义版本，写进存档）
[POS]: kernel 的心脏：同一版本 → 并行意图 → 统一排序与裁定 → 新版本 + 事件 + 观察。无 IO、无全局随机，可独立回放
       kernel-v3 增加物品请求类型准入；请求、回应、递交与使用各自经同一内核结算。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tianlong.core import (
    TICK_MINUTES,
    Event,
    Intent,
    Kind,
    Modality,
    Observation,
    Op,
    Outcome,
    PerceivedEvent,
    WorldState,
    derive_seed,
    make_id,
)
from tianlong.core.grammar import signature_error
from tianlong.kernel import space
from tianlong.kernel.invariants import assert_invariants
from tianlong.kernel.perception import Witnessing, make_percept, scene_percept
from tianlong.kernel.resolution import fail
from tianlong.kernel.rules import ActionRule, default_rules
from tianlong.kernel.rules.base import MANNER_INITIATIVE

# 规则语义版本：准入、先手、结算、感知投影任何一处的语义改变都要手动递增。
# 它随存档写入；读档时不一致即拒绝（除非调用方显式迁移）——旧存档里的事件是按旧规则裁定的
KERNEL_VERSION = "kernel-v3"   # v3：物品请求有明确受益人；请求、同意、真实 GIVE、USE 各自结算


@dataclass(frozen=True, slots=True)
class StepResult:
    state: WorldState                      # 结算后的新版本（version + 1，clock + 1 tick）
    events: tuple[Event, ...]              # 含被拒绝的意图：幂等与调试都需要它们
    observations: tuple[Observation, ...]  # 每条都指明观察者；角色只会拿到其中的 percept


class Kernel:
    """世界规则内核。step() 是纯函数：相同 (state, intents) 永远得到相同结果。"""

    def __init__(self, rules: Mapping[Op, ActionRule] | None = None) -> None:
        self._rules: dict[Op, ActionRule] = dict(rules) if rules is not None else default_rules()

    def rule(self, op: Op) -> ActionRule:
        return self._rules[op]

    # ============================================================
    #  结算主流程
    # ============================================================

    def step(self, s: WorldState, intents: Sequence[Intent]) -> StepResult:
        events: list[Event] = []
        observations: list[Observation] = []
        admitted: list[Intent] = []

        # ---- 1. 准入：语法、版本、每人每 tick 至多一个意图 ----
        seen_actors: set[str] = set()
        for it in sorted(intents, key=lambda i: i.id):
            reason = self._admission_error(s, it, seen_actors)
            seen_actors.add(it.actor)
            if reason is None:
                admitted.append(it)
                continue
            evt = Event(make_id("evt", s.seed, s.version, it.id), s.clock, it,
                        self._place(s, it.actor), Outcome.REJECTED, reason)
            events.append(evt)
            if self._is_person(s, it.actor):
                view = PerceivedEvent(it.op.value, evt.place or "", it.actor, it.target, it.obj, Outcome.REJECTED,
                                      reason=reason)
                percept = make_percept(s, Modality.SELF, view, vantage=evt.place)
                observations.append(Observation(make_id("obs", evt.id, it.actor, 0), it.actor, evt.id, percept))

        # ---- 2. 按先手度排序后逐个生效：冲突即由顺序裁定 ----
        #  两人同抢一把钥匙：先手者成功，后手者面对的已是“钥匙在别人手里”的世界
        working = s
        for it in sorted(admitted, key=lambda i: (-self._initiative(s, i), i.id)):
            rule = self._rules[it.op]
            place = space.place_of(working, it.actor)
            if space.is_subdued(working, it.actor) and not rule.usable_when_subdued:
                res = fail("subdued")      # 穴道被制，动弹不得
            else:
                res = rule.resolve(working, it)
            after = working.apply(res.changes) if res.changes else working
            evt = Event(make_id("evt", s.seed, s.version, it.id), s.clock, it, place,
                        res.outcome, res.reason, res.changes)
            events.append(evt)
            witnessing = Witnessing(working, after, evt, res, rule.loudness(it))
            for idx, (observer, percept) in enumerate(rule.perceive(witnessing)):
                observations.append(Observation(make_id("obs", evt.id, observer, idx), observer, evt.id, percept))
            working = after

        # ---- 3. 不变量闸门 ----
        assert_invariants(working)

        # ---- 4. 推进时间，所有人环顾四周 ----
        new = working.stamp(s.version + 1, s.clock + TICK_MINUTES)
        for person in new.of_kind(Kind.PERSON):
            oid = make_id("obs", s.seed, new.version, "scene", person.id)
            observations.append(Observation(oid, person.id, None, scene_percept(new, person.id)))

        return StepResult(new, tuple(events), tuple(observations))

    # ============================================================
    #  准入与先手
    # ============================================================

    def _admission_error(self, s: WorldState, it: Intent, seen_actors: set[str]) -> str | None:
        if not self._is_person(s, it.actor):
            return "unknown_actor"
        if it.actor in seen_actors:
            return "duplicate_actor"
        if it.based_on != s.version:
            return "stale"
        if it.op not in self._rules:
            return "unknown_op"

        def kind_of(eid: str) -> Kind | None:
            return s.kind(eid) if s.has_entity(eid) else None

        err = signature_error(it.op, kind_of, it.target, it.obj, it.topic, it.beneficiary, it.request_ref)
        return f"syntax: {err}" if err else None

    def _initiative(self, s: WorldState, it: Intent) -> float:
        rule = self._rules[it.op]
        agility = float(s.attr(it.actor, "agility", 0.5))
        jitter = random.Random(derive_seed(s.seed, s.clock, "initiative", it.actor)).random() * 0.1
        return rule.initiative + MANNER_INITIATIVE[it.manner] + 0.5 * agility + jitter

    @staticmethod
    def _is_person(s: WorldState, eid: str) -> bool:
        return s.has_entity(eid) and s.kind(eid) == Kind.PERSON

    def _place(self, s: WorldState, eid: str) -> str | None:
        return space.place_of(s, eid) if self._is_person(s, eid) else None
