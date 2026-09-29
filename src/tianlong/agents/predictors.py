"""
[INPUT]: 依赖 cognition 的 BeliefStore / Belief / Candidate / BeliefReader，core 的 Op / Rel / Proposition / Kind / Fact / Modality /
         Percept / EntitySketch，core/goals 的 GoalRegistry，core/profiles 的 Goal / Profile
[OUTPUT]: 对外提供 Prediction（v2：成功率、有效新观察、目标进展、风险、不确定性）、PRED_FIELDS、OutcomePredictor 协议、
          HeuristicPredictor（基于信念的先验预测器）、imagine()（在自己的认知上假想一条短分支）、goal_potential()、
          direct_effects()（候选得手时的直接效果）、branch_value()（假想分支势能差的定义）、
          BranchValuer（一次决策所有候选共用的 branch_value，逐位相同）、entropy()、STALE
[POS]: agents 的后果预测接口；回答“这个候选行动可能发生什么”，只看角色认知，保留不确定性。
       预期获知 ≠ 位移：确定地走到刚看过的地方几乎没有新观察，原地仔细翻查一处没翻过的地方却可能有；
       目标进展与风险来自“在自己的认知上假想行动得手后的样子”，再用同一套目标语义（core/goals + BeliefReader）求势能之差——
       假想分支只存在于这次预测里，从不写回认知，更不碰真相。预测不替代结算：它是孤立行动的主观估计（旁人同时行动不在其中）。
       branch_value() 是定义，两个预测器都走 BranchValuer：基线势能每次决策只求一次并由 _ReadTrace 记下每个目标读过的槽位，
       候选的假想事实碰不到的目标沿用基线势能、全都碰不到就不假想，相同的假想事实只做一次——只省功、不改值。
       依赖集取自目标求值实际读了什么，而不是手抄一份“哪些实体与目标有关”：认得的 AT 链、守地时在场的每个人、潜逃时的每扇门都自动在内。
       learning 训练出的 GNN 预测器实现同一协议后即可替换，策略代码不动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.beliefs import Belief
from tianlong.cognition.goals import BeliefReader
from tianlong.core import EntitySketch, Fact, Kind, Modality, Op, Percept, Proposition, Rel
from tianlong.core.goals import GoalRegistry
from tianlong.core.profiles import Goal, Profile

STALE = 20         # 多久没看过的地方又值得一看（分钟），与脚本策略的探索一致


@dataclass(frozen=True, slots=True)
class Prediction:
    success: float           # 行动成功的主观概率
    info_gain: float         # 预期有效新观察（0~1）：不含行动本身的直接效果（自己走到了哪、拿起了什么）
    note: str = ""
    progress: float = 0.0    # 预期的目标进展（主观势能的增加，0~1）
    risk: float = 0.0        # 预期的损失：目标势能下降或自己挨打受伤的可能（0~1）
    uncertainty: float = 0.0  # 成败的不确定性（二元熵，0~1）


# 进入策略观测的预测字段（0~1）：观测契约按它排列列，新增字段只需在这里登记
PRED_FIELDS = ("success", "info_gain", "progress", "risk", "uncertainty")


class OutcomePredictor(Protocol):
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = (),
        profile: Profile | None = None,
    ) -> list[Prediction]: ...


# ============================================================
#  假想分支：在自己的认知上折叠一条“如果得手”的自我感知，求目标势能
# ============================================================

_REGISTRY = GoalRegistry()


def imagine(store: BeliefStore, now: int, facts: Sequence[Fact]) -> BeliefStore:
    """只存在于这次预测里的假想认知：不写回、不留经历。"""
    if not facts:
        return store
    return store.revise(Percept(now, Modality.SELF, None, tuple(facts)))[0]


def _active(profile: Profile | None, now: int) -> tuple[list[Goal], float]:
    """已激活的目标与权重总量；总量为 0 就是没有可追求的目标。"""
    if profile is None:
        return [], 0.0
    active = [g for g in profile.goals if g.active(now)]
    return active, sum(abs(g.weight) for g in active)


def _weighted(active: Sequence[Goal], pots: Iterable[float], total: float) -> float:
    """按权重归一的势能。定义与快路径共用这一处算式：同样的逐目标势能 → 逐位相同的结果。"""
    return sum(g.weight * p for g, p in zip(active, pots, strict=True)) / total


def goal_potential(store: BeliefStore, now: int, profile: Profile | None) -> float:
    """已激活目标的主观势能（按权重归一到 0~1）；没有目标就是 0。"""
    active, total = _active(profile, now)
    if not total:
        return 0.0
    reader = BeliefReader(store)
    return _weighted(active, [_REGISTRY.potential(reader, store.owner, g, profile.allies) for g in active], total)


def direct_effects(store: BeliefStore, c: Candidate) -> list[Fact]:
    """候选得手时的直接效果（按自己的认知写成事实）。"""
    me = store.owner
    if c.op == Op.MOVE and c.target:
        return [Fact(Proposition.rel(me, Rel.AT, c.target))]
    if c.op == Op.TAKE and c.target:
        return [Fact(Proposition.rel(c.target, Rel.AT, me))]
    if c.op in (Op.PUT, Op.GIVE) and c.target and c.obj:
        return [Fact(Proposition.rel(c.obj, Rel.AT, c.target))]
    if c.op in (Op.UNLOCK, Op.LOCK) and c.target:
        return [Fact(Proposition.attr(c.target, "locked", True), c.op == Op.LOCK)]
    if c.op == Op.ATTACK and c.target:
        hurt = store.holds(Proposition.attr(c.target, "wounded", True))
        return [Fact(Proposition.attr(c.target, "subdued" if hurt else "wounded", True))]
    if c.op == Op.USE and c.target and c.obj:
        sk = store.sketch(c.obj)
        cure = dict(sk.attrs).get("cures") if sk else None
        return [Fact(Proposition.attr(c.target, str(cure), True), False)] if cure else []
    return []


def entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def branch_value(store: BeliefStore, now: int, profile: Profile | None, facts: Sequence[Fact]) -> float:
    """假想分支的势能差（得手时）：正为进展，负为损失。这是定义；预测器走 BranchValuer，与它逐位相同。"""
    if profile is None or not facts:
        return 0.0
    return goal_potential(imagine(store, now, facts), now, profile) - goal_potential(store, now, profile)


# ============================================================
#  一次决策共用的假想：与逐候选调用 branch_value() 逐位相同，只省重复功
#  - 基线势能只求一次，同时记下每个目标读过的槽位
#  - 假想只改动事实所在的槽位（自我感知不带场景、事件与草图），外加把“此刻”推到 max(last_tick, now)；
#    某目标读过的槽位一个也没碰到，它的求值就是同一串读、同一串结果——势能原样沿用。
#    “此刻”只影响同槽多条并存说法的先后，所以此刻前移且读到过并存说法的目标一律重算
#  - 所有目标都没碰到：假想与基线逐位相同，差恰为 0，连假想都不必做
#  - 相同的假想事实（同一去处的不同走法、不同方式）只求一次
# ============================================================


class _ReadTrace:
    """冒充 BeliefStore 的只读面，记下目标求值（BeliefReader + navigation）读过哪些槽位。

    只放行目标语义今天用到的查询：以后谁在目标求值里读了别的东西，这里当场 AttributeError，
    而不是漏记一个依赖、让快路径悄悄偏离定义。实体草图不记——假想只折叠事实，从不改动认得谁。"""

    __slots__ = ("_store", "owner", "entities", "slots", "predicates", "ordered")

    def __init__(self, store: BeliefStore) -> None:
        self._store, self.owner, self.entities = store, store.owner, store.entities
        self.slots: set[tuple[str, str]] = set()     # 读过的槽位 (主语, 谓词)
        self.predicates: set[str] = set()            # 反查过的谓词：subjects() 扫过该谓词下的全部信念
        self.ordered = False                         # 读到过同槽多条并存的说法：先后随“此刻”而变

    def sketch(self, eid: str) -> EntitySketch | None:
        return self._store.sketch(eid)

    def believed(self, prop: Proposition) -> Belief | None:
        self.slots.add(prop.slot)
        return self._store.believed(prop)

    def positives(self, subject: str, predicate: str) -> tuple[Belief, ...]:
        self.slots.add((subject, predicate))
        found = self._store.positives(subject, predicate)
        self.ordered = self.ordered or len(found) > 1
        return found

    def subjects(self, predicate: str, value: str) -> tuple[str, ...]:
        self.predicates.add(predicate)
        return self._store.subjects(predicate, value)

    # 派生查询原样借用 BeliefStore 的定义：它们只经上面的原语读信念，两边不会各写一份而走样
    best = BeliefStore.best
    location_of = BeliefStore.location_of

    def affected(self, slots: set[tuple[str, str]], predicates: set[str], shifted: bool) -> bool:
        """假想写到的槽位或谓词与读集相交，或此刻前移而读到过并存说法 → 这个目标的势能可能变。"""
        return (shifted and self.ordered) or not (self.slots.isdisjoint(slots) and self.predicates.isdisjoint(predicates))


class BranchValuer:
    """一次预测里所有候选共用：BranchValuer(store, now, profile)(facts) 与 branch_value(store, now, profile, facts)
    逐位相同。没有已激活的目标就什么也不求；基线等到头一个带事实的候选才求——与定义同一时机（目标不合法时同样在那里报错），
    候选全无事实的决策也不白算。"""

    def __init__(self, store: BeliefStore, now: int, profile: Profile | None) -> None:
        self.store, self.now = store, now
        self.allies = profile.allies if profile is not None else ()
        self.active, self.total = _active(profile, now)
        self.shifted = now > store.last_tick         # 假想认知的“此刻”是 max(last_tick, now)
        self.traces: list[_ReadTrace] = []
        self.pots: list[float] = []
        self.base = 0.0
        self._memo: dict[tuple[Fact, ...], float] = {}

    def __call__(self, facts: Sequence[Fact]) -> float:
        if not self.total or not facts:
            return 0.0
        if not self.traces:
            self._trace_base()
        key = tuple(facts)
        if key not in self._memo:
            self._memo[key] = self._delta(key)
        return self._memo[key]

    def _trace_base(self) -> None:
        for g in self.active:
            trace = _ReadTrace(self.store)
            self.pots.append(_REGISTRY.potential(BeliefReader(trace), self.store.owner, g, self.allies))
            self.traces.append(trace)
        self.base = _weighted(self.active, self.pots, self.total)

    def _delta(self, facts: tuple[Fact, ...]) -> float:
        slots = {f.prop.slot for f in facts}
        predicates = {f.prop.predicate for f in facts}
        dirty = [i for i, t in enumerate(self.traces) if t.affected(slots, predicates, self.shifted)]
        if not dirty:
            return 0.0
        mind = imagine(self.store, self.now, facts)
        reader = BeliefReader(mind)
        pots = list(self.pots)
        for i in dirty:
            pots[i] = _REGISTRY.potential(reader, mind.owner, self.active[i], self.allies)
        return _weighted(self.active, pots, self.total) - self.base


# ============================================================
#  启发式先验
# ============================================================


class HeuristicPredictor:
    """手写先验：门锁信念决定能否通过，位置可信度决定能否拿到；没看过/没翻过的地方才有新观察；
    进展与风险来自在自己认知上的假想分支。"""

    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = (),
        profile: Profile | None = None,
    ) -> list[Prediction]:
        missing = [i for i in interests if store.location_of(i) is None]
        branches = BranchValuer(store, now, profile)
        out = []
        for c in cands:
            p, info, note = self._one(store, now, c, missing)
            delta = branches(direct_effects(store, c))
            risk = max(0.0, -delta) * p
            if c.op == Op.ATTACK:
                risk = max(risk, 1.0 - p)        # 落空就可能招来还手
            out.append(Prediction(p, info, note, min(1.0, max(0.0, delta) * p), min(1.0, risk), entropy(p)))
        return out

    @staticmethod
    def _fresh(store: BeliefStore, now: int, place: str | None, table: str = "surveyed") -> bool:
        if place is None:
            return False
        last = getattr(store, table).get(place)
        return last is not None and now - last <= STALE

    def _one(self, store: BeliefStore, now: int, c: Candidate, missing: list[str]) -> tuple[float, float, str]:
        me = store.owner
        if c.op == Op.MOVE:
            door = c.obj
            locked = store.believed(Proposition.attr(door, "locked", True)) if door else None
            p = 0.8 if locked is None else (0.1 if locked.holds else 0.95)
            heard = any(ep.modality == Modality.SOUND and ep.event.place == c.target and now - ep.tick <= 5
                        for ep in store.episodes)         # 只算听到的响动：自己走过那里不是“那边有动静”
            if heard:
                return p, 0.7, "那边刚有动静"
            if self._fresh(store, now, c.target):
                return p, 0.05, "刚看过那边"
            return p, 0.6 if c.target not in store.surveyed else 0.4, ""
        if c.op == Op.TAKE:
            best = store.best(c.target, Rel.AT.value) if c.target else None
            return (0.9 * best.confidence if best else 0.2), 0.1, ""
        if c.op in (Op.PUT, Op.GIVE):
            return (0.95 if store.location_of(c.obj or "") == me else 0.1), 0.0, ""
        if c.op in (Op.UNLOCK, Op.LOCK):
            match = store.believed(Proposition.rel(c.obj or "", Rel.MATCHES, c.target or ""))
            return (0.9 if match and match.holds else 0.4), 0.3, ""
        if c.op == Op.INSPECT:
            sk = store.sketch(c.target or "")
            if sk is not None and sk.kind == Kind.PERSON:
                return 1.0, (0.6 if missing else 0.1), ("也许东西在他身上" if missing else "")
            if self._fresh(store, now, c.target, "searched"):
                return 1.0, 0.05, "刚翻过"
            return 1.0, (0.6 if missing else 0.2), ""
        if c.op == Op.ATTACK:
            helpless = store.holds(Proposition.attr(c.target or "", "subdued", True))
            return (1.0 if helpless else 0.5), 0.0, ("" if helpless else "胜负难料")
        if c.op == Op.ASK:
            return 0.95, 0.4, ""
        if c.op == Op.TELL:
            return 0.95, 0.0, ""
        return 1.0, 0.0, ""
