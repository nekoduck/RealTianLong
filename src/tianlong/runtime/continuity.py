"""
[INPUT]: 依赖 cognition 的 BeliefStore / believed_place，core 的 Kind / Modality / Op / Outcome / Proposition / Rel，
         language/templates 的 ATTR_WORDS / render_experience，persistence 的 TurnEnvelope
[OUTPUT]: 对外提供 Continuity（condition / notes / present / statuses / afflicted / newcomers）、continuity()（本回合的前后照应）、
          lately()（某个 NPC 近来亲历的事，用他自己的话）
[POS]: runtime 的前后照应，gm.build_brief 的帮手。纯模型主持人最常被挑的错是“前后接不上”：人明明被点了穴，下一幕毫发无伤地走来；
       玩家挨了一掌，此后再没人提。这里只拿玩家自己的认知做比对——本回合之前以为的（before）vs 此刻看见的（me）——
       写成几句给叙述者的提醒：你自己的伤毒被制、刚好了的伤、你原以为某人被制在别处却在眼前见到了他、一路相随的同伴没跟来。
       意外出现、或刚走到眼前的人（玩家自己看见他进来不过半个时辰）开口时，另给他自己近来的经历（lately），
       好让他亲口说出怎么脱身、怎么找来的——意外往往在下一回合才被问起，所以不只看本回合。
       从不看世界真相：玩家不知道的变化不会出现在这里
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tianlong.cognition import BeliefStore
from tianlong.cognition.navigation import believed_place
from tianlong.core import Kind, Modality, Op, Outcome, Proposition, Rel
from tianlong.language.templates import ATTR_WORDS, render_experience
from tianlong.persistence import TurnEnvelope

AILMENTS = ("wounded", "poisoned", "subdued")
_HEALED = {"wounded": "先前的伤已无大碍", "poisoned": "身上的毒已经解了", "subdued": "被制住的穴道已经解开"}
_WAS = {"wounded": "受了伤", "poisoned": "中了毒", "subdued": "被点了穴道、动弹不得"}
LATELY_TICKS = 60        # 近来：一个时辰之内
LATELY_KEEP = 4          # 至多说这么多件
ARRIVED_TICKS = 30       # 刚来到眼前：玩家自己看见他走进此地不过这么多分钟


@dataclass(frozen=True, slots=True)
class Continuity:
    condition: str | None = None
    notes: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    statuses: frozenset[str] = frozenset()
    afflicted: tuple[tuple[str, str], ...] = ()
    newcomers: frozenset[str] = frozenset()       # 出人意料地出现、或刚来到眼前的人（ID）：他开口时带上 lately


def _ails(store: BeliefStore, eid: str) -> list[str]:
    return [a for a in AILMENTS if store.holds(Proposition.attr(eid, a, True))]


def _name(store: BeliefStore, eid: str | None) -> str | None:
    sk = store.sketch(eid) if eid else None
    return sk.name if sk else None


def _seen(env: TurnEnvelope, player: str, here: str | None) -> list[str]:
    """本回合亲眼在此地看见的人：环顾里的在场者、在此地动作的人（先后次序，去重）。"""
    out: dict[str, None] = {}
    for p in env.percepts:
        if p.modality == Modality.SCENE:
            for f in p.facts:
                if f.holds and f.prop.predicate == Rel.AT.value and f.prop.value == here and f.prop.subject != player:
                    out.setdefault(f.prop.subject)
        elif p.event is not None and p.modality in (Modality.SIGHT, Modality.SPEECH) and p.event.actor \
                and p.event.actor != player and p.event.place == here:
            out.setdefault(p.event.actor)
    return list(out)


def continuity(env: TurnEnvelope, me: BeliefStore, before: BeliefStore | None, friends: Iterable[str] = ()) -> Continuity:
    """me 是此刻的认知，before 是本回合开始前的认知（重试补写叙述时没有，就只给身体状况与眼前）；friends 是玩家的同伴。"""
    player = me.owner
    now = _ails(me, player)
    condition = ("你" + "，".join(ATTR_WORDS[a][0] for a in now)) if now else None
    notes: list[str] = []
    statuses = set(now)
    here = believed_place(me, player)
    surprising: set[str] = set()
    if before is not None:
        healed = [a for a in _ails(before, player) if a not in now]
        notes += [_HEALED[a] for a in healed]
        statuses |= set(healed)
        for p in _seen(env, player, here):
            name = _name(me, p)
            if name is None or here is None or not before.knows(p):
                continue
            parts = []
            was_held = before.holds(Proposition.attr(p, "subdued", True)) and not me.holds(Proposition.attr(p, "subdued", True))
            if was_held:
                parts.append(_WAS["subdued"])
                statuses.add("subdued")
            was_at = believed_place(before, p)
            if was_at is not None and was_at != here and _name(before, was_at):
                parts.append(f"还在{_name(before, was_at)}")
            if parts:
                notes.append(f"你原以为{name}" + "，".join(parts) + "——此刻却在眼前")
                surprising.add(p)
        moved = any(p.modality == Modality.SELF and p.event is not None and p.event.actor == player
                    and p.event.kind == Op.MOVE.value and p.event.outcome == Outcome.SUCCESS for p in env.percepts)
        if moved:
            for f in sorted(set(friends)):
                if f in before.company and believed_place(me, f) != here and _name(me, f):
                    notes.append(f"{_name(me, f)}没有跟来")
    people = sorted(_name(me, p) for p, sk in me.entities.items()
                    if sk.kind == Kind.PERSON and p != player and here and believed_place(me, p) == here)
    present = tuple(x for x in (_name(me, here), *people) if x)
    arrived = {ep.event.actor for ep in me.episodes
               if ep.event.kind == Op.MOVE.value and ep.event.outcome == Outcome.SUCCESS and ep.event.actor
               and ep.event.actor != player and here and ep.event.target == here and ep.tick >= me.last_tick - ARRIVED_TICKS}
    return Continuity(condition, tuple(notes), present, frozenset(statuses),
                      tuple((me.sketch(player).name, a) for a in now) if me.sketch(player) else (),
                      frozenset(surprising | arrived))


def lately(mind: BeliefStore, who: str, now: int) -> str:
    """此人近来亲历的事（他自己的经历，用“我”说）：走过的路、挨过的打、被制与脱身——只取他自己的行动与冲着他来的事。"""
    rows: list[str] = []
    for ep in mind.episodes:
        ev = ep.event
        if ep.tick < now - LATELY_TICKS or ev.kind == "noise":
            continue
        mine = ep.modality == Modality.SELF and ev.actor == who and ev.kind in (Op.MOVE.value, Op.ATTACK.value, Op.USE.value)
        at_me = ev.target == who and ev.kind in (Op.ATTACK.value, Op.USE.value)
        if mine or at_me:
            rows.append(render_experience(ep.modality, ev, mind.entities, who, me="我"))
    return "；".join(dict.fromkeys(rows[-LATELY_KEEP:]))
