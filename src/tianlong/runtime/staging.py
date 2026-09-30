"""
[INPUT]: 依赖 core 的 Percept / Modality / Outcome / Rel，scenarios 的 Beat
[OUTPUT]: 对外提供 recognize()（玩家本回合感知到了哪些看点）、stops_wait()（等待该不该在这个 tick 停下）、witnessed()（B1：看点编号集合）
[POS]: runtime 的看点识别：真相只用于呈现——这里只读玩家自己的感知（已结算的事件、环顾所见），从不读世界真相，
       也从不进入任何 NPC 的 Situation（只由会话与评测脚本调用）。识别器由场景给出（Scenario.beats）：
       事件看点按行动/施动者/对象/物件/门/地点/言语行为/原因匹配（缺省只认成功的）；景观看点（lore）只看环顾：
       玩家在该处、时钟已到、看得见那件陈设。once 的看点一局只让等待停一次（已停过的记在会话运行态 staged 里）。
       天色 sky() 与写法卡、细节卡组留给 M3
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from tianlong.core import Modality, Outcome, Percept, Rel
from tianlong.scenarios import Beat


def _from(beat: Beat, moments: Mapping[str, int]) -> int | None:
    """看点的时钟下限：整数照用，字符串按场景的时刻换算（场景没有这个时刻就永不成立）。"""
    if beat.clock_from is None or isinstance(beat.clock_from, int):
        return beat.clock_from
    return moments.get(beat.clock_from, 1 << 60)


def _where(p: Percept, player: str) -> str | None:
    return next((str(f.prop.value) for f in p.facts if f.holds and f.prop.subject == player
                 and f.prop.predicate == Rel.AT.value), None)


def matches(beat: Beat, p: Percept, player: str, moments: Mapping[str, int] | None = None) -> bool:
    """这条感知是否就是这个看点：景观只看环顾，事件只看感知到的那一件（看不清是谁的响动不算）。"""
    floor = _from(beat, moments or {})
    if floor is not None and p.tick < floor:
        return False
    if beat.lore:
        eid = beat.lore.split("@", 1)[0]
        return (p.modality == Modality.SCENE and (not beat.place or _where(p, player) in beat.place)
                and any(f.holds and f.prop.subject == eid for f in p.facts))
    ev = p.event
    if ev is None or p.modality == Modality.SCENE or ev.actor is None:
        return False
    door_ok = beat.door is None or ev.obj == beat.door or any(f.prop.subject == beat.door for f in p.facts)
    return ((beat.op is None or ev.kind == beat.op.value) and (not beat.actors or ev.actor in beat.actors)
            and (not beat.target or ev.target in beat.target) and (beat.obj is None or ev.obj == beat.obj)
            and door_ok and (not beat.place or ev.place in beat.place)
            and (beat.social is None or ev.social == beat.social) and (beat.reason is None or ev.reason == beat.reason)
            and (not beat.success or ev.outcome == Outcome.SUCCESS))


def recognize(beats: Iterable[Beat], percepts: Iterable[Percept], player: str, clock: int | None = None,
              moments: Mapping[str, int] | None = None) -> tuple[Beat, ...]:
    """玩家这些感知里认得出的看点，每个编号只留第一条（按感知的先后）。clock 只作没有时刻的感知的兜底。"""
    beats = tuple(beats)
    found: dict[str, Beat] = {}
    for p in percepts:
        if clock is not None and not p.tick:
            p = Percept(clock, p.modality, p.event, p.facts, p.scopes, p.sketches, p.informant)
        for b in beats:
            if b.key not in found and matches(b, p, player, moments):
                found[b.key] = b
    return tuple(found.values())


def stops_wait(beats: Iterable[Beat], percepts: Iterable[Percept], player: str,
               moments: Mapping[str, int] | None = None, staged: Collection[str] = ()) -> bool:
    """等待在玩家感知到看点的那个 tick 停下；once 的看点停过一次（staged 里有它）就不再为它停。"""
    return any(not (b.once and b.key in staged) for b in recognize(beats, percepts, player, moments=moments))


def witnessed(beats: Iterable[Beat], percepts: Iterable[Percept], player: str,
              moments: Mapping[str, int] | None = None) -> frozenset[str]:
    """B1：这些感知里玩家目击了哪些看点（编号集合）。"""
    return frozenset(b.key for b in recognize(beats, percepts, player, moments=moments))
