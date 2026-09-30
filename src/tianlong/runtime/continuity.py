"""
[INPUT]: 依赖 cognition 的 BeliefStore / believed_place，core 的 Kind / Modality / Op / Outcome / Proposition / Rel，
         language/templates 的 ATTR_WORDS / render_experience，persistence 的 TurnEnvelope
[OUTPUT]: 对外提供 Continuity（condition / notes / present / statuses / afflicted / newcomers / nearby）、continuity()（本回合的前后照应）、
          lately()（某个 NPC 近来亲历的事，用他自己的话）
[POS]: runtime 的前后照应，gm.build_brief 的帮手。纯模型主持人最常被挑的错是“前后接不上”：人明明被点了穴，下一幕毫发无伤地走来；
       玩家挨了一掌，此后再没人提。这里只拿玩家自己的认知做比对——本回合之前以为的（before）vs 此刻看见的（me）——
       写成几句给叙述者的提醒：你自己的伤毒被制、刚好了的伤、你原以为某人被制在别处却在眼前见到了他、一路相随的同伴没跟来。
       意外出现、或刚走到眼前的人（玩家自己看见他进来不过半个时辰）开口时，另给他自己近来的经历（lately），
       好让他亲口说出怎么脱身、怎么找来的——意外往往在下一回合才被问起，所以不只看本回合。
       从不看世界真相：玩家不知道的变化不会出现在这里
       lately 只到他开口那一刻为止、只给一路走到此地的几步与冲着他来的动手（挨的打优先留着），只在他自己的引语里算出处。
       “原以为被制”的提醒写明是谁制住的（玩家亲眼看见的那一下），并把“被制”列为说到的状态：叙述者写出“她明明被左子穆点了穴道，
       怎地此刻就在眼前”正是我们要的觉察，不能被闸门当成凭空点名与状态升级丢掉；nearby 让“回头是剑湖宫”“蒲团上那卷帛书”可以点名
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
_HEALED = {"wounded": "你身上已不怎么疼了", "poisoned": "你体内那股麻痒已经退了", "subdued": "你手脚又能动了"}
LATELY_TICKS = 60        # 近来：一个时辰之内
LATELY_KEEP = 4          # 至多说这么多件
LATELY_HITS = 2          # 其中冲着他来的动手至多这么多件（一路走来的几步占其余）
ARRIVED_TICKS = 30       # 刚来到眼前：玩家自己看见他走进此地不过这么多分钟


@dataclass(frozen=True, slots=True)
class Continuity:
    condition: str | None = None
    notes: tuple[str, ...] = ()
    present: tuple[str, ...] = ()
    statuses: frozenset[str] = frozenset()
    afflicted: tuple[tuple[str, str], ...] = ()
    newcomers: frozenset[str] = frozenset()       # 出人意料地出现、或刚来到眼前的人（ID）：他开口时带上 lately
    nearby: tuple[str, ...] = ()                  # 此地叫得出名字的东西、门与门那头的地点：可以点名，不进提示词


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


def _subduer(store: BeliefStore, who: str) -> str | None:
    """玩家亲眼看见的、最近一次得手制住（或打伤）此人的是谁：“你原以为钟灵被左子穆制住了”——谁动的手也是玩家记得的事。"""
    for ep in reversed(store.episodes):
        ev = ep.event
        if ev.kind == Op.ATTACK.value and ev.target == who and ev.outcome == Outcome.SUCCESS and ev.actor:
            return "你" if ev.actor == store.owner else _name(store, ev.actor)
    return None


def _nearby(me: BeliefStore, here: str | None) -> tuple[str, ...]:
    """此地叫得出名字的东西（物件、陈设）、通向别处的门与门那头的地点：说“回头是剑湖宫”“蒲团上那卷帛书”不算凭空点名。"""
    if here is None:
        return ()
    out: dict[str, None] = {}
    for eid, sk in sorted(me.entities.items()):
        if sk.kind in (Kind.ITEM, Kind.SURFACE) and believed_place(me, eid) == here:
            out.setdefault(sk.name)
        elif sk.kind == Kind.DOOR:
            ends = {str(b.prop.value) for b in me.positives(eid, Rel.CONNECTS.value)}
            if here in ends:
                out.setdefault(sk.name)
                out.update(dict.fromkeys(n for n in (_name(me, e) for e in sorted(ends - {here})) if n))
    return tuple(out)


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
        notes += [_HEALED[a] for a in _ails(before, player) if a not in now]       # 措辞不带状态词：不必另开许可
        for p in _seen(env, player, here):
            name = _name(me, p)
            if name is None or here is None or not before.knows(p) or believed_place(me, p) != here:
                continue                         # 本回合又走了的人不算“此刻在眼前”
            was_held = before.holds(Proposition.attr(p, "subdued", True)) and not me.holds(Proposition.attr(p, "subdued", True))
            was_at = believed_place(before, p)
            elsewhere = was_at is not None and was_at != here and _name(before, was_at) is not None
            by = _subduer(before, p) if was_held else None
            if was_held:
                statuses.add("subdued")               # 提醒里说到“被制住”：叙述者据此写出觉察，不算状态升级
            if elsewhere:
                held = (f"被{by}制住了" if by else "被制住了") + "，还困在" if was_held else "还在"
                notes.append(f"你原以为{name}{held}{_name(before, was_at)}——此刻却在眼前")
                surprising.add(p)
            elif was_held:
                notes.append(f"你原以为{name}{'被' + by + '制住、' if by else ''}动弹不得——此刻却已能动了")
        moved = any(p.modality == Modality.SELF and p.event is not None and p.event.actor == player
                    and p.event.kind == Op.MOVE.value and p.event.outcome == Outcome.SUCCESS for p in env.percepts)
        if moved:
            for f in sorted(set(friends)):
                if f in before.company and believed_place(me, f) != here and _name(me, f) \
                        and not before.holds(Proposition.attr(f, "subdued", True)):     # 被制住的人本就跟不来
                    notes.append(f"{_name(me, f)}没有跟来")
    people = sorted(_name(me, p) for p, sk in me.entities.items()
                    if sk.kind == Kind.PERSON and p != player and here and believed_place(me, p) == here)
    present = tuple(x for x in (_name(me, here), *people) if x)
    arrived = {ep.event.actor for ep in me.episodes
               if ep.event.kind == Op.MOVE.value and ep.event.outcome == Outcome.SUCCESS and ep.event.actor
               and ep.event.actor != player and here and ep.event.target == here and ep.tick >= me.last_tick - ARRIVED_TICKS}
    return Continuity(condition, tuple(notes), present, frozenset(statuses),
                      tuple((me.sketch(player).name, a) for a in now) if me.sketch(player) else (),
                      frozenset(surprising | arrived), _nearby(me, here))


def lately(mind: BeliefStore, who: str, now: int) -> str:
    """此人怎么来到眼前（他自己的话，用“我”说）：只取他开口之前（不晚于 now）的经历里，一路走到此地的那几步，
    以及冲着他来的动手与施救。他自己对别人的动手、去过又离开的地方（私奔去营地之类的隐情）一概不给——
    这是给他解释“怎么找来的”，不是替他交代行踪。"""
    eps = [ep for ep in mind.episodes if now - LATELY_TICKS <= ep.tick <= now and ep.event.kind != "noise"]
    path: list = []
    for ep in reversed(eps):                    # 从开口那一刻往回找：最后一步走到的地方，和一路连着走过来的那几步
        ev = ep.event
        if ep.modality == Modality.SELF and ev.actor == who and ev.kind == Op.MOVE.value and ev.outcome == Outcome.SUCCESS:
            if path and ev.target != path[-1].event.place:
                break
            path.append(ep)
    hits = [ep for ep in eps if ep.event.target == who and ep.event.kind in (Op.ATTACK.value, Op.USE.value)]
    hits = hits[-LATELY_HITS:]                  # 挨的打最要紧（“我被点了穴”），一路走来的几步占其余，免得被路程挤掉
    steps = list(reversed(path))[-(LATELY_KEEP - len(hits)):]
    rows = [render_experience(ep.modality, ep.event, mind.entities, who, me="我")
            for ep in sorted([*steps, *hits], key=lambda e: e.tick)]
    return "；".join(dict.fromkeys(rows))
