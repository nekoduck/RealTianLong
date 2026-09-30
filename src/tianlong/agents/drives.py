"""
[INPUT]: 依赖 agents/policy_kit 的 Situation / Choice / Policy / WAIT_REASONS / SPEAK / CHATTER，agents/tactics 的 MartialTactics，
         cognition 的 BeliefStore / Candidate，cognition/navigation 的 believed_place / route_to，
         core 的 Kind / Manner / Op / Fact / Proposition / Rel / derive_seed，core/profiles 的 Goal / GoalKind，
         core/drives 的驱力词表（Level / 条件 / 行动 / Drive）
[OUTPUT]: 对外提供 Driven（Policy 装饰器：把角色的驱力套在任何策略外面）、holds()（条件求值）、realize()（行动落到候选集上）、
          oneway_doors()、Marks
[POS]: agents 的驱力层：性情不是剧本。条件只读该角色自己的认知（Situation）与自己的驱力标记，行动只落到 PolicyKit 已有的积木上——
       在候选集中挑、沿自己的地图走一步、对眼前的人开口、带字的姿态（等待 + 原话）、临时目标交给 MartialTactics 的寻仇/护人、
       以为锁着才开锁、以为没锁才上锁（钥匙照 PolicyKit 的取舍）。
       驱力的原话挂在 Choice.line 上（chosen() 之后），随意图落库；实现不了就返回 None，退回里面那个策略的选择。
       优先级：一次性提议 → URGENT（表序第一条：成立、不在冷却、实现得了）→ 里面的策略 →
       它闲着（等待原因或闲谈；学得的策略选中等待时也标 "idle"）时试 IDLE → 对最终结果套 VETO。
       没有驱力时 Driven 恒等于里面那个策略。
       平手与台词轮换一律由 derive_seed 派生；不 import kernel / persistence / runtime（测试钉死导入图）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import singledispatch

from tianlong.agents.policy_kit import CHATTER, SPEAK, WAIT_REASONS, Choice, Policy, Situation
from tianlong.agents.tactics import MartialTactics
from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import believed_place, route_to
from tianlong.core import Fact, Kind, Manner, Op, Proposition, Rel, derive_seed
from tianlong.core.drives import (
    PLAYER,
    SELF,
    TARGET,
    Alone,
    AnyOf,
    Arrived,
    Ask,
    At,
    Between,
    Cross,
    Drive,
    Fired,
    Flee,
    Follow,
    Fond,
    Give,
    Go,
    Heard,
    HeldBy,
    Here,
    Hold,
    Holds,
    Inspect,
    Knows,
    Level,
    Lock,
    Lost,
    Menaced,
    Near,
    Not,
    Pose,
    Pursue,
    Saw,
    Say,
    Searched,
    Status,
    Study,
    Take,
    Unlock,
    Use,
)
from tianlong.core.profiles import Goal, GoalKind

Marks = Mapping[str, Sequence[int]]     # 一个角色的驱力标记：驱力 → 兑现成功的时刻（旧的在前）


def oneway_doors(b: BeliefStore) -> frozenset[str]:
    """认为是单向的门（断崖、隧道）：走过去就回不来。"""
    return frozenset(d for d, sk in b.entities.items() if sk.kind == Kind.DOOR and b.positives(d, "attr.oneway"))


class _Wary(MartialTactics):
    """追人、赶路时不走认为单向的门：把它们当作过不去的门（探索与带路都绕开，连去面对锁门的退路也绕开），
    也不拿钥匙去试；以为非经单向门到不了的地方就不去。"""

    def _hard_avoid(self, sit: Situation) -> frozenset[str]:
        return oneway_doors(sit.beliefs)

    def _blocked_doors(self, sit: Situation) -> frozenset[str]:
        return super()._blocked_doors(sit) | oneway_doors(sit.beliefs)

    def _keys_for(self, b: BeliefStore, door: str) -> list[str]:
        return [] if door in oneway_doors(b) else super()._keys_for(b, door)

    def _go_towards(self, sit: Situation, place: str | None, why: str, manner: Manner | None = None) -> Choice | None:
        if place and route_to(sit.beliefs, place, oneway_doors(sit.beliefs)) is None:
            return None                  # 以为要经单向的门才到得了：不去（连门边也不去）
        return super()._go_towards(sit, place, why, manner)


_KIT, _WARY = MartialTactics(), _Wary()


def _who(who: str | None, sit: Situation, target: str | None = None) -> str | None:
    return {SELF: sit.agent, PLAYER: sit.player, TARGET: target}.get(who, who) if who is not None else None


# ============================================================
#  条件：只读 Situation 与本人的驱力标记
# ============================================================


@singledispatch
def holds(cond, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    """条件在该角色自己的认知上是否成立；target 是 VETO 被否决那一步的对象（who="target" 时用）。"""
    raise TypeError(f"未知的驱力条件：{cond!r}")


@holds.register(At)
def _at(c: At, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return _KIT._here(sit.beliefs) == c.place


@holds.register(Between)
def _between(c: Between, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return c.start <= sit.now and (c.end is None or sit.now < c.end)


@holds.register(Here)
def _present(c: Here, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return _who(c.who, sit, target) in _KIT._persons_here(sit.beliefs)


@holds.register(Near)
def _near(c: Near, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    """亲眼所见的下落（环顾、看见他的举动）不出 within 个 tick：他刚才还在身边，或刚从眼前走开；听人说的不算。"""
    who = _who(c.who, sit, target)
    seen = sit.beliefs.best(who, Rel.AT.value) if who is not None else None
    return seen is not None and not seen.hearsay and sit.now - seen.learned_at <= c.within


@holds.register(Status)
def _state(c: Status, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    who = _who(c.who, sit, target)
    return who is not None and _KIT._status(sit.beliefs, who, c.status)


def _witnessed(sit: Situation, within: int, target: str | None, *, op=None, actor=None, obj=None, socials=(),
               outcome=None, to=None, spoken: bool = False, here: bool = False) -> bool:
    b, me = sit.beliefs, sit.agent
    actor, to = _who(actor, sit, target), _who(to, sit, target)
    place = _KIT._here(b) if here else None
    for ep in b.episodes:
        ev = ep.event
        if ev.actor in (None, me) or sit.now - ep.tick > within or (here and ev.place != place):
            continue
        if spoken and ev.utterance is None and ev.social is None:
            continue
        if (op is None or ev.kind == op.value) and (actor is None or ev.actor == actor) \
                and (to is None or ev.target == to) and (obj is None or ev.obj == obj) \
                and (not socials or ev.social in socials) and (outcome is None or ev.outcome == outcome):
            return True
    return False


@holds.register(Saw)
def _saw(c: Saw, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return _witnessed(sit, c.within, target, op=c.op, actor=c.actor, to=c.target, obj=c.obj, socials=c.socials,
                      outcome=c.outcome, here=c.here)


@holds.register(Heard)
def _heard(c: Heard, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return _witnessed(sit, c.within, target, actor=c.speaker, to=c.to, socials=c.socials, spoken=True)


@holds.register(Holds)
def _holding(c: Holds, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return sit.beliefs.location_of(c.item) == sit.agent


@holds.register(HeldBy)
def _held_by(c: HeldBy, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    loc = sit.beliefs.location_of(c.item)
    if c.who is None:
        return loc in _KIT._persons_here(sit.beliefs)
    return loc is not None and loc == _who(c.who, sit, target)


@holds.register(Knows)
def _knows(c: Knows, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return sit.beliefs.knows(c.eid)


@holds.register(Fond)
def _fond(c: Fond, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    who = _who(c.who, sit, target)
    return who is not None and sit.beliefs.attitude(who) >= c.at_least


def _menacing(sit: Situation, at_most: int) -> list[str]:
    """眼前态度 ≤ at_most、以为既没中毒也没被制的人：真能再下手的仇人。"""
    b = sit.beliefs
    return [p for p in _KIT._persons_here(b) if b.attitude(p) <= at_most
            and not _KIT._status(b, p, "poisoned") and not _KIT._status(b, p, "subdued")]


@holds.register(Menaced)
def _menaced(c: Menaced, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return bool(_menacing(sit, c.at_most))


@holds.register(Alone)
def _alone(c: Alone, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    allowed = {_who(a, sit, target) for a in c.allowed}
    return all(p in allowed for p in _KIT._persons_here(sit.beliefs))


@holds.register(Arrived)
def _arrived(c: Arrived, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    b = sit.beliefs
    allowed = {_who(a, sit, target) for a in c.allowed}
    return any((since := b.company.get(p)) is not None and since < sit.now and sit.now - since <= c.within
               for p in _KIT._persons_here(b) if p not in allowed)


@holds.register(Searched)
def _searched(c: Searched, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return c.eid in sit.beliefs.searched


@holds.register(Fired)
def _fired(c: Fired, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    ticks = [t for t in marks.get(c.key, ()) if c.within is None or sit.now - t <= c.within]
    return len(ticks) >= c.times


@holds.register(Lost)
def _lost(c: Lost, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    who = _who(c.who, sit, target)
    if who is None or who in _KIT._persons_here(sit.beliefs):
        return False
    where = believed_place(sit.beliefs, who)
    if where is None:
        return _WARY._explore(sit, who) is None
    return _WARY._go_towards(sit, where, "") is None


@holds.register(Not)
def _not(c: Not, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return not holds(c.cond, sit, marks, target)


@holds.register(AnyOf)
def _any(c: AnyOf, sit: Situation, marks: Marks, target: str | None = None) -> bool:
    return any(holds(x, sit, marks, target) for x in c.conds)


# ============================================================
#  行动：只落到 PolicyKit 的积木上；实现不了返回 None
# ============================================================


@singledispatch
def realize(act, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    """把驱力的行动落到候选集上（或候选之外的一句话、一个带字的姿态）；key 用于派生平手的种子。"""
    raise TypeError(f"未知的驱力行动：{act!r}")


@realize.register(Go)
def _go(a: Go, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    kit = _WARY if a.avoid_oneway else _KIT
    return kit._go_towards(sit, a.place, f"往{kit._name(sit.beliefs, a.place)}去",
                           Manner.CAREFUL if a.careful else Manner.NORMAL)


@realize.register(Flee)
def _flee(a: Flee, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    """在认为走得通的门里逃：不走单向的门、不撞认为锁着的门，不往（以为）有仇人的地方去；
    先按 prefer，再按没去过的，平手由种子定；无路可逃返回 None。"""
    b = sit.beliefs
    feared = {believed_place(b, p) for p, sk in b.entities.items()
              if sk.kind == Kind.PERSON and p != sit.agent and b.attitude(p) <= -2}
    shut = oneway_doors(b) | _KIT._blocked_doors(sit)
    options = [(i, c) for i, c in enumerate(sit.candidates) if c.op == Op.MOVE and c.manner == Manner.NORMAL
               and c.obj not in shut and c.target not in feared]
    if not options:
        return None
    rng = random.Random(derive_seed("drive", sit.agent, key, sit.now))
    jitter = {t: rng.random() for t in sorted({c.target for _, c in options})}

    def rank(ic: tuple[int, Candidate]) -> tuple:
        c = ic[1]
        pref = a.prefer.index(c.target) if c.target in a.prefer else len(a.prefer)
        return pref, c.target in b.surveyed, jitter[c.target], c.obj or ""

    i, c = min(options, key=rank)
    return Choice(i, f"往{_KIT._name(b, c.target or '')}逃")


@realize.register(Follow)
def _follow(a: Follow, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    who = _who(a.who, sit, target)
    return _KIT._defend(sit, Goal(GoalKind.DEFEND, person=who)) if who else None


@realize.register(Pursue)
def _pursue(a: Pursue, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    who = _who(a.who, sit, target)
    kit = _WARY if a.avoid_oneway else _KIT
    return kit._hostile(sit, Goal(GoalKind.HOSTILE, person=who)) if who else None


@realize.register(Ask)
def _ask(a: Ask, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    b = sit.beliefs
    about = _who(a.about, sit, target)
    if about is None or believed_place(b, about) is not None:
        return None                                        # 以为知道下落就不问
    q = Fact(Proposition.rel(about, Rel.AT, None), True)
    asked = [_who(a.who, sit, target)] if a.who is not None else _KIT._persons_here(b)
    for p in asked:
        if p is not None and p != about and not _KIT._did_recently(sit, Op.ASK, p):
            choice = _KIT._pick(sit, f"向{_KIT._name(b, p)}打听{_KIT._name(b, about)}的下落", Op.ASK, p, topic=q)
            if choice is not None:
                return choice
    return None


@realize.register(Cross)
def _cross(a: Cross, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    for i, c in enumerate(sit.candidates):
        if c.op == Op.MOVE and c.obj == a.door and c.manner == Manner.NORMAL:
            return Choice(i, f"经{_KIT._name(sit.beliefs, a.door)}过去")
    return None


@realize.register(Inspect)
def _inspect(a: Inspect, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _KIT._pick(sit, f"细看{_KIT._name(sit.beliefs, a.eid)}", Op.INSPECT, a.eid)


@realize.register(Take)
def _take(a: Take, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _KIT._pick(sit, f"拿起{_KIT._name(sit.beliefs, a.item)}", Op.TAKE, a.item, manner=Manner.NORMAL)


@realize.register(Study)
def _study(a: Study, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _KIT._pick(sit, f"研读{_KIT._name(sit.beliefs, a.item)}", Op.STUDY, a.item)


@realize.register(Use)
def _use(a: Use, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _KIT._pick(sit, f"用{_KIT._name(sit.beliefs, a.item)}", Op.USE, _who(a.on, sit, target), a.item)


@realize.register(Give)
def _give(a: Give, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _KIT._pick(sit, f"把{_KIT._name(sit.beliefs, a.item)}递过去", Op.GIVE, _who(a.to, sit, target), a.item)


def _turn_key(sit: Situation, door: str, op: Op) -> Choice | None:
    """以为锁着才开、以为没锁才锁（免得白试一回）；钥匙照 PolicyKit 的取舍（认为配的在前）。"""
    b = sit.beliefs
    if _KIT._status(b, door, "locked") != (op == Op.UNLOCK):
        return None
    for key in _KIT._keys_for(b, door):
        choice = _KIT._pick(sit, f"用{_KIT._name(b, key)}{'开' if op == Op.UNLOCK else '锁'}{_KIT._name(b, door)}", op, door, key)
        if choice is not None:
            return choice
    return None


@realize.register(Unlock)
def _unlock(a: Unlock, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _turn_key(sit, a.door, Op.UNLOCK)


@realize.register(Lock)
def _lock(a: Lock, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return _turn_key(sit, a.door, Op.LOCK)


@realize.register(Pose)
def _pose(a: Pose, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return Choice(_KIT._wait_index(sit), a.text, SPEAK, social=a.social, line=a.text)


@realize.register(Say)
def _say(a: Say, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    who = _who(a.to, sit, target)
    if who is None or who not in _KIT._persons_here(sit.beliefs):
        return None
    manner = Manner.CAREFUL if a.careful else Manner.NORMAL
    return Choice(_KIT._wait_index(sit), f"对{_KIT._name(sit.beliefs, who)}开口", SPEAK,
                  free=Candidate(Op.TELL, who, manner=manner, social=a.social))


@realize.register(Hold)
def _hold(a: Hold, sit: Situation, key: str = "", target: str | None = None) -> Choice | None:
    return Choice(_KIT._wait_index(sit), "按兵不动")


# ============================================================
#  Driven：把驱力套在任何策略外面
# ============================================================


class Driven:
    def __init__(self, inner: Policy, drives: Sequence[Drive] = (), marks: Marks | None = None,
                 proposal: Drive | None = None) -> None:
        """inner：里面那个策略（脚本或学得的）；marks：本人的驱力标记；proposal：一次性提议（同一条实现路径与合法性检查）。"""
        self.inner = inner
        self.drives = tuple(drives)
        self.marks: Marks = marks or {}
        self.proposal = proposal

    @property
    def reads_memories(self) -> bool:
        return bool(getattr(self.inner, "reads_memories", False))

    def choose(self, sit: Situation) -> Choice:
        if not self.drives and self.proposal is None:
            return self.inner.choose(sit)
        choice = self._try(self.proposal, sit) if self.proposal is not None else None
        choice = choice or self._first(Level.URGENT, sit)
        if choice is None:
            choice = self.inner.choose(sit)
            if choice.tag in WAIT_REASONS or choice.tag == CHATTER:
                choice = self._first(Level.IDLE, sit) or choice
        return self._veto(sit, choice)

    # ------------------------------------------------------------

    def _ready(self, d: Drive, sit: Situation) -> bool:
        ticks = self.marks.get(d.key, ())
        if d.once and ticks:
            return False
        return not (d.cooldown and ticks and sit.now - ticks[-1] < d.cooldown)

    def _try(self, d: Drive, sit: Situation, target: str | None = None) -> Choice | None:
        """条件成立、不在冷却就依次试它的行动；VETO 的行动都实现不了（或没写）时原地不动。"""
        if not self._ready(d, sit) or not all(holds(c, sit, self.marks, target) for c in d.when):
            return None
        for act in (*d.do, Hold()) if d.level == Level.VETO else d.do:
            choice = realize(act, sit, d.key, target)
            if choice is not None:
                return self._stamp(choice, d, sit)
        return None

    def _first(self, level: Level, sit: Situation) -> Choice | None:
        return next((c for d in self.drives if d.level == level and (c := self._try(d, sit)) is not None), None)

    def _veto(self, sit: Situation, choice: Choice) -> Choice:
        """最终选择若是某条 VETO 否决的行动、且它的条件（对被否决那一步的对象）成立：改做它的行动，缺省原地不动。"""
        vetoes = [d for d in self.drives if d.level == Level.VETO]
        if not vetoes:
            return choice
        cand = choice.free or sit.candidates[choice.index]
        for d in vetoes:
            if cand.op == d.veto and (instead := self._try(d, sit, cand.target)) is not None:
                return instead
        return choice

    def _stamp(self, choice: Choice, d: Drive, sit: Situation) -> Choice:
        """挂上驱力的原话与出处：驱力的选择从不标 CHATTER，带原话或开口的标 SPEAK。"""
        line = self._line(d, sit) or choice.line
        tag = SPEAK if line or choice.free is not None or choice.tag in (SPEAK, CHATTER) else choice.tag
        return replace(choice, rationale=f"{d.gloss or d.key}：{choice.rationale}", tag=tag, line=line, drive=d.key)

    def _line(self, d: Drive, sit: Situation) -> str | None:
        """台词：lines 按兑现次数轮换，起点由 (角色, 驱力) 派生；否则是 line。"""
        if d.lines:
            n = len(self.marks.get(d.key, ()))
            return d.lines[(derive_seed("line", sit.agent, d.key) + n) % len(d.lines)]
        return d.line or None
