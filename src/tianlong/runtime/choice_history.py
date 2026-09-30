"""
[INPUT]: 仅玩家意图、玩家所得 Percept、BeliefStore 与冻结动作编解码
[OUTPUT]: ChoiceHistory：有界的动作结果/相关条件记录与亲眼所见放置线索；随世界同一事务落库
[POS]: 决策派生状态。失败不会随 12 条经历滚掉；条件摘要只含该动作相关的玩家知识，不含世界版本、NPC 私密目标或后台时间。
       progress 可继续；无收益/已学成/不可达只在相同条件下阻挡；探索因看见的新放置或昼夜变化恢复。
       NPC 明确拒绝或暂缓作为玩家听见的反馈保留，相关条件未变不重新推荐同一请求。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tianlong.cognition import BeliefStore, Candidate
from tianlong.core import FrozenMap, Modality, Op, Outcome, Percept, Rel, digest, is_night
from tianlong.runtime.choice_model import candidate_from, candidate_to

log = logging.getLogger(__name__)
CAPACITY = 256
_BLOCKING = frozenset({"nothing_to_learn", "already_learned", "mastered", "no_effect", "not_holding",
                       "out_of_reach", "not_found", "door_locked", "wrong_key", "held_by_other", "already_there",
                       "already_held", "route_not_here", "route_mismatch", "not_adjacent", "one_way", "subdued"})
_REQUEST_BLOCKING = frozenset({"request_refused", "request_deferred"})


def action_key(c: Candidate) -> str:
    return digest("choice-attempt-v1", candidate_to(c))


def conditions(me: BeliefStore, c: Candidate, clues: Mapping[str, int] = (), reason: str | None = None) -> str:
    """只取会影响这个动作的知识槽位；排除置信度、记录时间与不相干角色活动。"""
    slots: set[tuple[str, str]] = set()
    extra: list = []

    def slot(eid: str | None, key: str):
        if eid:
            slots.add((eid, key))

    def location(eid: str | None):
        seen = set()
        while eid and eid not in seen and me.knows(eid):
            seen.add(eid)
            slot(eid, Rel.AT.value)
            eid = me.location_of(eid)

    def attr(eid: str | None, key: str):
        slot(eid, "attr." + key)
        sk = me.sketch(eid) if eid else None
        extra.append((eid, key, dict(sk.attrs).get(key) if sk else None))

    target, obj = c.target, c.obj
    if reason == "subdued":
        attr(me.owner, "subdued")
    if c.op == Op.STUDY:
        slot(target, Rel.AT.value)
        attr(target, "teaches")
        sk = me.sketch(target) if target else None
        best = me.best(target, "attr.teaches") if target else None
        skill = best.prop.value if best else (dict(sk.attrs).get("teaches") if sk else None)
        if isinstance(skill, str):
            attr(me.owner, skill)
            # 进境本身不构成失败重试的借口，成功 progress 总会放行。
    else:
        location(me.owner)
        location(target)
        location(obj)
        if c.op in (Op.MOVE, Op.LOCK, Op.UNLOCK):
            door = obj if c.op == Op.MOVE else target
            for key in ("locked", "oneway"):
                attr(door, key)
            slot(door, Rel.CONNECTS.value)
            if c.op in (Op.LOCK, Op.UNLOCK):
                slot(obj, Rel.MATCHES.value)
        elif c.op == Op.INSPECT:
            extra.extend((b.prop.sort_key(), b.holds) for b in me.sorted_beliefs()
                         if b.prop.predicate == Rel.AT.value and b.prop.value == target)
            extra.append(("observed-placement", dict(clues).get(target)))
            extra.append(("daylight", is_night(me.last_tick)))
        elif c.op == Op.TAKE:
            attr(target, "hidden")
            holder = me.location_of(target) if target else None
            attr(holder, "subdued")
        elif c.op == Op.USE:
            attr(obj, "cures")
            attr(target, "poisoned")
            attr(target, "wounded")
        elif c.op == Op.REQUEST_ITEM:
            location(c.beneficiary)
            attr(c.beneficiary, "poisoned")
            attr(c.beneficiary, "wounded")
            extra.append(("attitude", target, me.attitude(target)))
        if c.topic:
            slot(c.topic.prop.subject, c.topic.prop.predicate)
    values = [(b.prop.sort_key(), b.holds) for b in me.sorted_beliefs() if b.prop.slot in slots]
    return digest("choice-conditions-v1", values, extra)


@dataclass(frozen=True, slots=True)
class Attempt:
    candidate: Candidate
    condition: str
    outcome: Outcome
    reason: str | None
    tick: int

    def to_data(self) -> dict:
        return {"candidate": candidate_to(self.candidate), "condition": self.condition,
                "outcome": self.outcome.value, "reason": self.reason, "tick": self.tick}

    @classmethod
    def from_data(cls, d: Mapping) -> Attempt:
        return cls(candidate_from(d["candidate"]), str(d["condition"]), Outcome(d["outcome"]), d.get("reason"), int(d["tick"]))


@dataclass(frozen=True, slots=True)
class ChoiceHistory:
    attempts: Mapping[str, Attempt] = field(default_factory=dict)
    surface_changes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "attempts", FrozenMap(self.attempts))
        object.__setattr__(self, "surface_changes", FrozenMap(self.surface_changes))

    def latest(self, c: Candidate) -> Attempt | None:
        return self.attempts.get(action_key(c))

    def unchanged(self, me: BeliefStore, c: Candidate) -> bool:
        prev = self.latest(c)
        return prev is not None and prev.condition == conditions(me, c, self.surface_changes, prev.reason)

    def blocked(self, me: BeliefStore, c: Candidate) -> bool:
        prev = self.latest(c)
        if prev is None or not self.unchanged(me, c):
            return False
        if c.op == Op.INSPECT and prev.outcome == Outcome.SUCCESS:
            return True
        return prev.reason in _BLOCKING or prev.reason in _REQUEST_BLOCKING

    def note(self, before: BeliefStore, percepts: Sequence[Percept], c: Candidate) -> ChoiceHistory:
        after = before.revise_all(percepts)[0]
        clues = dict(self.surface_changes)
        for p in percepts:
            ev = p.event
            if (p.modality == Modality.SIGHT and ev and ev.kind == Op.PUT.value and ev.target
                    and ev.outcome == Outcome.SUCCESS and after.knows(ev.target)):
                clues[ev.target] = p.tick
        attempts = dict(self.attempts)
        # NPC 的拒绝/暂缓是实际听见的反馈，“请求说出口了”不表示已获得物品。
        for o in after.obligations:
            if o.kind == "requested_item" and o.state in ("refused", "deferred"):
                for key, a in list(attempts.items()):
                    cand = a.candidate
                    if (cand.op == Op.REQUEST_ITEM and (cand.target, cand.obj, cand.beneficiary) ==
                            (o.counterpart, o.item, o.beneficiary)):
                        reason = "request_refused" if o.state == "refused" else "request_deferred"
                        if any(p.modality == Modality.SPEECH and p.event and p.event.request_ref == o.request_ref
                               for p in percepts):
                            attempts[key] = Attempt(cand, conditions(after, cand, clues, reason),
                                                    a.outcome, reason, after.last_tick)
        result = next((p for p in percepts if p.modality == Modality.SELF and p.event
                       and p.event.actor == before.owner and p.event.kind == c.op.value), None)
        if result is not None and c.op != Op.WAIT:
            ev = result.event
            if ev.outcome is not None:
                attempts[action_key(c)] = Attempt(c, conditions(after, c, clues, ev.reason), ev.outcome, ev.reason, result.tick)
        if len(attempts) > CAPACITY:
            log.warning("选项尝试记录超过 %s 项，淘汰最旧的一项", CAPACITY)
            attempts = dict(sorted(attempts.items(), key=lambda kv: (kv[1].tick, kv[0]))[-CAPACITY:])
        clues = dict(sorted(clues.items(), key=lambda kv: (kv[1], kv[0]))[-CAPACITY:])
        return ChoiceHistory(attempts, clues)

    def to_data(self) -> dict:
        return {"schema": 1, "attempts": [a.to_data() for _, a in sorted(self.attempts.items())],
                "surface_changes": dict(sorted(self.surface_changes.items()))}

    @classmethod
    def from_data(cls, d: Mapping | None) -> ChoiceHistory:
        if not d:
            return cls()
        if d.get("schema") != 1:
            raise ValueError("不兼容的选项尝试记录")
        attempts = [Attempt.from_data(a) for a in d.get("attempts", ())]
        return cls({action_key(a.candidate): a for a in attempts}, d.get("surface_changes", {}))
