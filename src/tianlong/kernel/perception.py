"""
[INPUT]: 依赖 core 的 WorldState / Event / Percept / Fact 等，kernel/space 的空间查询，kernel/resolution 的 Resolution
[OUTPUT]: 对外提供 Fragment（行动留下的可感知片段）、Witnessing（一次事件的目击上下文）、scene_percept()、self_knowledge()、
          make_percept()、change_facts()、attr_fact()、sketches_for()、in_sight()、audibility()、SILENT_FAILURES
[POS]: kernel 的感知物理：决定“谁以何种方式、获得事件的哪一部分”。观察不按行动的“目标”投影，而按行动在物理世界里
       实际留下的片段投影——失败的移动没有抵达，目的地的人就什么也看不见；内部失败原因与未说出口的意图默认不进目击。
       外观只给亲眼所见：言语提到的实体、隔墙听见的地点、门那头的地点都只有名字（seen=False），不读取真实外观。
       自我感知（内力、所学、进度、手中之物的手感）只进本人的环顾，永不给旁人。
       规则通过覆写 fragments()/perceive() 组合这些积木，加新行动不改本模块
       请求的物品、受益人和语义编号只传给真正听见内容者，耳语旁观者看不到这些字段。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace

from tianlong.core import (
    OBSERVABLE_ATTRS,
    SKILLS,
    STATUS_ATTRS,
    TACTILE_ATTRS,
    AddRelation,
    Change,
    EntitySketch,
    Event,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    RemoveRelation,
    SetAttr,
    WorldState,
    derive_seed,
    is_private_attr,
    make_id,
)
from tianlong.core.attributes import true_value
from tianlong.kernel import space
from tianlong.kernel.resolution import Resolution

# ============================================================
#  声音传播参数
# ============================================================

HOP_ATTENUATION = 0.3   # 每隔一道门衰减
MAX_HEARING_HOPS = 1    # 声音只穿过一道门

# 尚未触及世界就失败的尝试：没有可见的动作，也没有响动（伸手去拿、东西却不在；想去的路不在这里）
SILENT_FAILURES = frozenset({
    "out_of_reach", "not_found", "not_holding", "already_there", "already_held", "self_target", "subdued",
    "nothing_to_learn", "already_learned", "route_not_here", "route_mismatch", "not_adjacent",
})


def audibility(loudness: float, hop_count: int, alertness: float) -> float:
    """听见概率：响度随门衰减，警觉度（0~1，0.5 为常人）加减。"""
    return max(0.0, min(1.0, loudness - HOP_ATTENUATION * hop_count + (alertness - 0.5)))


# ============================================================
#  事实与草图的通用构造
# ============================================================


def change_facts(changes: Iterable[Change]) -> tuple[Fact, ...]:
    """把世界变化翻译为目击者能获知的事实。顺序保留：先负后正，便于信念修正覆盖。"""
    out: list[Fact] = []
    for c in changes:
        if isinstance(c, AddRelation):
            out.append(Fact(Proposition.of(c.rel), True))
        elif isinstance(c, RemoveRelation):
            out.append(Fact(Proposition.of(c.rel), False))
        elif isinstance(c, SetAttr):
            if c.key == "subdued_until":
                # 点穴的时限是私密数值；旁人看到的只是“他被制住了”
                out.append(Fact(Proposition.attr(c.entity, "subdued", True), True))
            elif not is_private_attr(c.key):
                out.append(attr_fact(c.entity, c.key, c.old, c.new))
    return tuple(out)


def attr_fact(entity: str, key: str, old: object, new: object) -> Fact:
    """布尔属性统一规范为 (attr=True, 极性)：“开锁”与“推门发现没锁”必须落在同一个命题上。"""
    if isinstance(new, bool) or (new is None and isinstance(old, bool)):
        return Fact(Proposition.attr(entity, key, True), bool(new))
    if new is None:
        return Fact(Proposition.attr(entity, key, old), False)  # type: ignore[arg-type]
    return Fact(Proposition.attr(entity, key, new), True)  # type: ignore[arg-type]


Seen = Callable[[str], bool]


def sketches_for(s: WorldState, ids: Iterable[str | None], seen: Seen | None = None) -> tuple[EntitySketch, ...]:
    """为感知中出现的实体生成草图。seen(eid) 为假的实体只有名字与种类——外观未知，而不是“没有”。"""
    out: dict[str, EntitySketch] = {}
    for eid in ids:
        if eid is None or eid in out or not s.has_entity(eid):
            continue
        e = s.entity(eid)
        if seen is None or seen(eid):
            out[eid] = EntitySketch(e.id, e.kind, e.name, tuple((k, v) for k, v in e.attrs if k in OBSERVABLE_ATTRS))
        else:
            out[eid] = EntitySketch(e.id, e.kind, e.name, (), seen=False)
    return tuple(out[k] for k in sorted(out))


def in_sight(states: Iterable[WorldState], vantage: str, revealed: Iterable[str] = ()) -> Seen:
    """站在 vantage 能亲眼看见的：这个地点本身、此处（行动前或后）看得见的人与物、连着此处的明门，
    以及本次感知的事实亲眼揭示出来的（翻查找到的藏匿物、手里的东西）。
    藏匿的物件、揣在别人身上的小物件、暗门，只是“在这里”并不等于看得见——在话题里被提起也不行。"""
    states = tuple(states)
    shown = frozenset(revealed)

    def seen(eid: str) -> bool:
        if eid == vantage or eid in shown:
            return True
        for st in states:
            if not st.has_entity(eid):
                continue
            if st.kind(eid) == Kind.DOOR:
                if vantage in st.targets(eid, Rel.CONNECTS) and not st.attr(eid, "hidden", False):
                    return True
            elif space.place_of(st, eid) == vantage and _visible(st, eid):
                return True
        return False

    return seen


def _visible(st: WorldState, eid: str) -> bool:
    if st.kind(eid) != Kind.ITEM:
        return True
    if st.attr(eid, "hidden", False) or space.is_concealed(st, eid):
        return False
    holder = space.holder_of(st, eid)
    return holder is None or st.kind(holder) != Kind.ITEM


def _referenced(s: WorldState, facts: Iterable[Fact]) -> Iterator[str]:
    for f in facts:
        yield f.prop.subject
        if not f.prop.is_attr and isinstance(f.prop.value, str) and s.has_entity(f.prop.value):
            yield f.prop.value


def _view_ids(view: PerceivedEvent | None) -> tuple[str | None, ...]:
    if view is None:
        return ()
    topic_ids: tuple[str | None, ...] = ()
    if view.topic is not None:
        v = view.topic.prop.value
        topic_ids = (view.topic.prop.subject, v if isinstance(v, str) else None)
    return (view.place, view.actor, view.target, view.obj, view.beneficiary, *topic_ids)


def make_percept(
    s: WorldState,
    modality: Modality,
    view: PerceivedEvent | None = None,
    facts: tuple[Fact, ...] = (),
    scopes: tuple[str, ...] = (),
    informant: str | None = None,
    vantage: str | None = None,
    also: WorldState | None = None,
) -> Percept:
    """vantage：感知发生的地点，决定哪些实体是亲眼所见（None 只用于场景作者给出的“过去所见”）。
    言语只让说话双方被看见——话里提到的东西只闻其名；响动连地点也只是听说。"""
    ids = (*_view_ids(view), *_referenced(s, facts), *scopes, informant)
    seen: Seen | None
    if modality == Modality.SOUND:
        seen = lambda _: False  # noqa: E731
    elif modality == Modality.SPEECH:
        present = {informant, view.actor if view else None, view.target if view else None, vantage}
        seen = lambda eid: eid in present  # noqa: E731
    elif vantage is not None:
        # 本次感知亲眼确立的事实的主语（找到的藏匿物、手里的东西）才算揭示；
        # 事实的宾语（他走向的地点、门那头）与话题里提到的只是被提起
        revealed = [f.prop.subject for f in facts if f.holds]
        seen = in_sight((s, also) if also is not None else (s,), vantage, revealed)
    else:
        seen = None
    return Percept(s.clock, modality, view, facts, scopes, sketches_for(s, ids, seen), informant)


# ============================================================
#  Fragment：行动在物理世界里实际留下的、可被感知的片段
#  成功的移动在出发地留下“离开”、在目的地留下“抵达”；推不开的门只在门这一侧留下“推门未果”与响动；
#  压根没有路可走的尝试什么也不留下——观察由片段决定，而不是由行动的目标决定
# ============================================================


@dataclass(frozen=True, slots=True)
class Fragment:
    place: str
    view: PerceivedEvent               # 在场者看得见的部分：不含未说出口的意图与内部失败原因
    facts: tuple[Fact, ...] = ()       # 在场者因此获知的事实
    loudness: float = 0.0              # 从这里传出的响动（0 = 不出声）


# ============================================================
#  Witnessing：一次事件的目击上下文
# ============================================================


@dataclass(frozen=True, slots=True)
class Witnessing:
    before: WorldState
    after: WorldState
    event: Event
    resolution: Resolution
    loudness: float

    @property
    def actor(self) -> str:
        return self.event.actor

    def full_view(self, with_topic: bool = True) -> PerceivedEvent:
        """行动者自己眼中的完整事件：意图、目标、成败与原因。"""
        it = self.event.intent
        return PerceivedEvent(
            kind=it.op.value,
            place=self.event.place or "",
            actor=it.actor,
            target=it.target,
            obj=it.obj if with_topic or it.op != Op.REQUEST_ITEM else None,
            outcome=self.event.outcome,
            topic=it.topic if with_topic else None,
            reason=self.event.reason,
            utterance=it.utterance if with_topic else None,
            social=it.social if with_topic else None,
            beneficiary=it.beneficiary if with_topic else None,
            request_ref=(make_id("request", it.actor, it.target, it.obj, it.beneficiary, self.event.tick)
                         if it.op == Op.REQUEST_ITEM else it.request_ref) if with_topic else None,
        )

    def public_view(self, public_reasons: frozenset[str] = frozenset(), with_topic: bool = True) -> PerceivedEvent:
        """旁观者眼中的事件：失败原因只有看得出来的才给（被挡开、被闪开），暗处的东西不给 ID。"""
        v = self.full_view(with_topic)
        reason = v.reason if v.reason in public_reasons else None
        obj = None if v.obj and self.before.has_entity(v.obj) and self.before.attr(v.obj, "hidden", False) \
            and self.before.kind(v.obj) == Kind.DOOR else v.obj
        return replace(v, reason=reason, obj=obj)

    def actor_percept(self) -> tuple[str, Percept]:
        facts = (*self.resolution.learned, *change_facts(self.resolution.changes))
        here = space.place_of(self.after, self.actor) or self.event.place
        p = make_percept(self.after, Modality.SELF, self.full_view(), facts, self.resolution.scopes,
                         vantage=here, also=self.before)
        # 感知时刻是行动发生的 tick，而非结算后的新时钟
        return self.actor, _at_tick(p, self.event.tick)

    def witnesses(self, places: Iterable[str]) -> tuple[str, ...]:
        """这些地点里（行动前或行动后）在场、且不是行动者本人的人。"""
        found: set[str] = set()
        for place in places:
            found.update(space.persons_in(self.before, place))
            found.update(space.persons_in(self.after, place))
        found.discard(self.actor)
        return tuple(sorted(found))

    def sight(self, view: PerceivedEvent, facts: tuple[Fact, ...], place: str) -> Percept:
        p = make_percept(self.after, Modality.SIGHT, view, facts, vantage=place, also=self.before)
        return _at_tick(p, self.event.tick)

    def speech(self, facts: tuple[Fact, ...]) -> Percept:
        p = make_percept(self.after, Modality.SPEECH, self.full_view(), facts, informant=self.actor,
                         vantage=self.event.place)
        return _at_tick(p, self.event.tick)

    def sounds(self, place: str, loudness: float, exclude: Iterable[str], salt: int = 0) -> Iterator[tuple[str, Percept]]:
        """隔壁（经门）的人按概率听到“某处有响动”——不知道是谁、做了什么。"""
        if loudness <= 0 or not place:
            return
        skip = set(exclude) | {self.actor}
        dist = space.hops(self.before, place, MAX_HEARING_HOPS)
        for other in sorted(dist):
            if dist[other] == 0:
                continue
            for listener in space.persons_in(self.before, other):
                if listener in skip:
                    continue
                alertness = float(self.before.attr(listener, "alertness", 0.5))
                p = audibility(loudness, dist[other], alertness)
                parts = (self.before.seed, self.event.id, listener, "sound") + ((salt,) if salt else ())
                if random.Random(derive_seed(*parts)).random() < p:
                    view = PerceivedEvent(kind="noise", place=place)
                    yield listener, _at_tick(make_percept(self.before, Modality.SOUND, view), self.event.tick)

    def observe(self, fragments: Iterable[Fragment]) -> Iterator[tuple[str, Percept]]:
        """在场者看见各自所在处的片段；片段的响动再传到隔壁（看见了的人不必再听见）。"""
        fragments = tuple(fragments)
        seers: set[str] = set()
        for f in fragments:
            here = self.witnesses((f.place,))
            for w in here:
                yield w, self.sight(f.view, f.facts, f.place)
            seers.update(here)
        for i, f in enumerate(fragments):
            yield from self.sounds(f.place, f.loudness, seers, salt=i)

    def default_fragments(self, public_reasons: frozenset[str], facts: tuple[Fact, ...] | None = None
                          ) -> tuple[Fragment, ...]:
        """默认片段：行动者所在处一次可见的动作；尚未触及世界就失败的尝试什么也不留下。"""
        if not self.event.place:
            return ()
        if self.event.outcome == Outcome.FAILURE and self.event.reason in SILENT_FAILURES:
            return ()
        fs = change_facts(self.resolution.changes) if facts is None else facts
        return (Fragment(self.event.place, self.public_view(public_reasons), fs, self.loudness),)


def _at_tick(p: Percept, tick: int) -> Percept:
    if p.tick == tick:
        return p
    return Percept(tick, p.modality, p.event, p.facts, p.scopes, p.sketches, p.informant)


# ============================================================
#  环顾：每个 tick 结束时，每个人对所在地点的被动视觉
#  scopes = 地点 + 其中台面：这些容纳者“可见内容”被完整看清，
#  于是认知层可以据此推出“原本以为在这里的东西不见了”
# ============================================================


def scene_percept(s: WorldState, observer: str) -> Percept:
    place = space.place_of(s, observer)
    if place is None:
        return make_percept(s, Modality.SCENE)
    facts: list[Fact] = [Fact(Proposition.rel(observer, Rel.AT, place))]
    for e in space.visible_in(s, place, observer):
        holder = space.holder_of(s, e)
        if holder is not None:
            facts.append(Fact(Proposition.rel(e, Rel.AT, holder)))
    # 自己身上的东西自己清楚（包括藏在身上的小物件）
    facts.extend(Fact(Proposition.rel(i, Rel.AT, observer)) for i in space.contents(s, observer))
    # 在场者（含自己）的身体状态一目了然：受伤、中毒、被制住，带极性进入信念
    for p in dict.fromkeys((observer, *space.persons_in(s, place))):
        facts.extend(Fact(Proposition.attr(p, st, True), space.status_of(s, p, st)) for st in STATUS_ATTRS)
    # 暗门不在环顾之列：只能靠仔细查看发现
    for door, other in space.neighbors(s, place, visible_only=True):
        facts.append(Fact(Proposition.rel(door, Rel.CONNECTS, place)))
        facts.append(Fact(Proposition.rel(door, Rel.CONNECTS, other)))
    facts.extend(self_knowledge(s, observer))
    scopes = (place, *space.surfaces_in(s, place))
    return make_percept(s, Modality.SCENE, None, tuple(facts), scopes, vantage=place)


def self_knowledge(s: WorldState, observer: str) -> tuple[Fact, ...]:
    """自我感知：自己的内力、所学、修习进度，以及手里东西的手感（锋利、淬毒、所载武功）。
    这些事实只进本人的环顾——私密属性“不外泄”指的是不给旁人，而不是连自己都不知道自己。"""
    me = s.entity(observer)
    out = [Fact(Proposition.attr(observer, "martial", round(float(true_value(me, "martial", s.clock)), 3)))]
    for skill in SKILLS:
        out.append(Fact(Proposition.attr(observer, skill, True), bool(me.get(skill, False))))
        out.append(Fact(Proposition.attr(observer, f"progress_{skill}", int(me.get(f"progress_{skill}", 0) or 0))))
    for item in space.contents(s, observer):
        e = s.entity(item)
        for key in sorted(TACTILE_ATTRS):
            v = true_value(e, key, s.clock)
            out.append(Fact(Proposition.attr(item, key, True), v) if isinstance(v, bool)
                       else Fact(Proposition.attr(item, key, v)))
    return tuple(out)
