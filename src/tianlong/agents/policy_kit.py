"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / navigation，core 的 Fact / Kind / Manner / Modality / Op / Proposition / Rel，
         core/profiles 的 Profile，agents/predictors 的 Prediction，memory/view 的 MemoryView
[OUTPUT]: 对外提供 Situation（可带主角 player）/ Choice（含结构化标签 tag、言语行为 social、候选之外的言语 free、驱力的原话 line 与出处 drive、
          chosen()）/ Policy 协议、
          PolicyKit（规则策略共享的“在候选集中挑选”、沿自己的地图带路（认为锁着的门先试着开、打不开就不去撞；
          子类可经 _hard_avoid() 指定怎样都不走的门）、凭个人勘察记录探索、
          开口积木 _say()（候选之外的闲话）/ _tell()（候选之外、带自己相信的命题的答话），index 都指向 WAIT，
          与 _last_spoke()（最近一次对谁开口）、信念查询积木）、WAIT_REASONS、SPEAK、CHATTER、RECENT、STALE
[POS]: agents 的决策契约与策略工具箱：策略只能在候选集中选（Choice.index），唯一的例外是 Choice.free——对某人开口：
       不带命题的闲话，或一句 TELL 答话，其命题取自说话者自己认为为真的信念（chosen() 对照其认知核验，ASK 永不带命题）；
       一切判断来自信念与近期经历。
       探索只凭自己的地图与勘察记录（BeliefStore.surveyed/searched），从不读真相里的最短路或藏匿处。
       ScriptedPolicy 与 MartialTactics 都建立在这些积木之上，保证脚本行为与 RL 面对的是同一套候选与同一份认知
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from tianlong.agents.predictors import Prediction
from tianlong.cognition import BeliefStore, Candidate, effective_confidence
from tianlong.cognition.navigation import believed_distance, route_to
from tianlong.core import Fact, Kind, Manner, Modality, Op, Proposition, Rel, Social
from tianlong.core.profiles import Profile
from tianlong.memory.view import MemoryView

RECENT = 5
STALE = 20          # 多久没看过的地方值得再去看一眼（分钟）
LOCK_DOUBT = 0.3    # 对“门锁着”的把握低于此（记忆已旧）就再去推一推
# 示范者等待的结构化原因：模仿学习据此区分合理等待与卡住
WAIT_REASONS = ("idle", "goal_inactive", "goal_done", "stuck_unknown", "no_candidate", "expert_no_action")
SPEAK = "speak"     # 开口（回话、答话、叫阵、喝止）的标签：index 指向 WAIT，只认下标的学习层把它当作等待
CHATTER = "chatter" # 没人搭话、自己找话说（见礼、说笑）的标签：同样指向 WAIT；编排据此每处每 tick 只留一句


@dataclass(frozen=True)
class Situation:
    agent: str
    profile: Profile
    beliefs: BeliefStore
    now: int
    candidates: tuple[Candidate, ...]
    predictions: tuple[Prediction, ...]
    memories: tuple[str, ...] = ()
    memory: MemoryView | None = None     # 长期记忆的结构化摘要：谁的话被证实/证伪过、哪些实体常被提起
    player: str | None = None            # 主角是谁（在场的人都看得见的公开身份，不是秘密）：搭话与见义出声以他为准；
                                         # None 时按眼前的人取舍，且不给任何人“回话不设限”的例外


@dataclass(frozen=True, slots=True)
class Choice:
    index: int        # 候选集下标：策略永远只能在候选集中选
    rationale: str
    tag: str = ""     # 结构化标签：等待时为 WAIT_REASONS 之一，探索时为 "explore"，开口时为 SPEAK，自己找话说时为 CHATTER
    social: Social | None = None        # 给所选候选附上言语行为（“答话”“叫阵”）：修辞，不改变行动本身
    free: Candidate | None = None       # 候选集之外唯一允许的行动：对某人说的 TELL/ASK——
                                        # 不带命题的闲话（回话、叫阵、搭话：只有原话与言语行为）；或一句 TELL 答话，
                                        # 命题须是说话者自己认为为真的信念（被问到候选话题之外的事也照实答）；ASK 永不带命题。
                                        # 不占策略的动作编号（候选规则版本不变）：index 须指向 WAIT 候选，
                                        # 只认 index 的消费者（学习层的数据生成、示范）把它当作等待
    line: str | None = None             # 驱力给的原话：在 chosen() 之后挂上（不挂在候选上），成为任何行动的 utterance；
                                        # 只是修辞，只认 index 的学习层看不见
    drive: str | None = None            # 这一步出自哪条驱力（驱力标记据此只记兑现成功的）

    def chosen(self, cands: Sequence[Candidate], beliefs: BeliefStore | None = None) -> Candidate:
        """最终交给内核的行动：free 优先，否则取候选并附上言语行为。
        free 带命题时须给出说话者的认知 beliefs 以核验“说的是自己相信的事”；不合规的 free 一律抛 ValueError。"""
        if self.free is not None:
            f = self.free
            if f.op not in (Op.TELL, Op.ASK) or f.target is None:
                raise ValueError(f"free 只接受对某人说的 TELL/ASK：{f}")
            if f.topic is not None and not (f.op == Op.TELL and f.topic.holds and f.topic.prop.value is not None
                                            and beliefs is not None and beliefs.holds(f.topic.prop)):
                raise ValueError(f"free 只能带说话者自己认为为真的命题，且只用于 TELL：{f}")
            return f
        if not 0 <= self.index < len(cands):
            raise ValueError(f"策略越界选择了候选 {self.index}")
        cand = cands[self.index]
        return replace(cand, social=self.social) if self.social is not None else cand


class Policy(Protocol):
    def choose(self, situation: Situation) -> Choice: ...


class PolicyKit:
    # ------------------------------------------------------------
    #  在候选集中挑选
    # ------------------------------------------------------------

    @staticmethod
    def _pick(sit: Situation, why: str, op: Op, target: str | None = None, obj: str | None = None,
              manner: Manner | None = None, topic: Fact | None = None) -> Choice | None:
        for i, c in enumerate(sit.candidates):
            if c.op == op and c.target == target and (obj is None or c.obj == obj) \
                    and (manner is None or c.manner == manner) and (topic is None or c.topic == topic):
                return Choice(i, why)
        return None

    # ------------------------------------------------------------
    #  开口：候选之外的闲话（只有言语行为，不带命题）
    # ------------------------------------------------------------

    @staticmethod
    def _wait_index(sit: Situation) -> int:
        return next((i for i, c in enumerate(sit.candidates) if c.op == Op.WAIT), 0)

    def _say(self, sit: Situation, target: str, social: Social, why: str, tag: str = SPEAK) -> Choice:
        """对 target 说一句只有言语行为的话（回话、叫阵、搭话）：措辞留给主持人之声，事实一概不传。"""
        return Choice(self._wait_index(sit), why, tag, free=Candidate(Op.TELL, target, social=social))

    def _tell(self, sit: Situation, target: str, fact: Fact, why: str) -> Choice | None:
        """候选里没有这句话时，对 target 照实说出自己相信的 fact（被问到了才用）：自己不信的一概不说。"""
        if not (fact.holds and sit.beliefs.holds(fact.prop)):
            return None
        return Choice(self._wait_index(sit), why, SPEAK, free=Candidate(Op.TELL, target, topic=fact))

    @staticmethod
    def _last_spoke(sit: Situation, listener: str | None = None,
                    socials: Collection[Social] | None = None) -> int | None:
        """最近一次对 listener（None = 对任何人）开口的时刻，socials 限定言语行为：
        持久的 said 记录（说过的话与闲话）加近期经历里自己的问话（问话不进 said）。"""
        ticks = [s.tick for s in sit.beliefs.said
                 if (listener is None or s.listener == listener) and (socials is None or s.social in socials)]
        ticks += [ep.tick for ep in sit.beliefs.episodes
                  if ep.modality == Modality.SELF and ep.event.kind in (Op.TELL.value, Op.ASK.value)
                  and (listener is None or ep.event.target == listener)
                  and (socials is None or ep.event.social in socials)]
        return max(ticks) if ticks else None

    def _go_towards(self, sit: Situation, place: str | None, why: str, manner: Manner | None = None) -> Choice | None:
        """沿自己以为的地图走一步：目的地与路线（门）都来自认知。
        认为那扇门锁着：手里有（认为配或不知配不配的）钥匙就先开锁；没有就不去撞门——除非这份记忆已经很旧。"""
        if not place:
            return None
        b = sit.beliefs
        blocked = self._blocked_doors(sit)
        # 有绕得开锁门的路就绕，没有才去面对那扇门（怎样都不走的门始终绕开）
        hop = route_to(b, place, blocked) or route_to(b, place, self._hard_avoid(sit))
        if not hop:
            return None
        nxt, door = hop
        if door in blocked:
            for key in self._keys_for(b, door):
                choice = self._pick(sit, f"门锁着，用{self._name(b, key)}开锁", Op.UNLOCK, door, key)
                if choice is not None:
                    return choice
            return None
        return self._pick(sit, why, Op.MOVE, nxt, door, manner)

    def _hard_avoid(self, sit: Situation) -> frozenset[str]:
        """怎样都不走的门（连去面对锁门的退路也不走）：缺省没有；驱力层的 _Wary 用它绕开单向的门。"""
        return frozenset()

    @staticmethod
    def _blocked_doors(sit: Situation) -> frozenset[str]:
        """自知过不去的门：确信锁着（记忆还不旧）。"""
        b = sit.beliefs
        out = set()
        for d, sk in b.entities.items():
            if sk.kind != Kind.DOOR:
                continue
            lk = b.believed(Proposition.attr(d, "locked", True))
            if lk is not None and lk.holds and effective_confidence(lk, sit.now) >= LOCK_DOUBT:
                out.add(d)
        return frozenset(out)

    @staticmethod
    def _keys_for(b: BeliefStore, door: str) -> list[str]:
        """手里可能开这扇门的东西：认为配的在前，其次是不知配不配、看着也不像兵刃药物秘籍的；确知不配的不试。"""
        ranked = []
        for i, sk in sorted(b.entities.items()):
            if sk.kind != Kind.ITEM or b.location_of(i) != b.owner:
                continue
            match = b.believed(Proposition.rel(i, Rel.MATCHES, door))
            looks = dict(sk.attrs)
            if match is not None:
                if match.holds:
                    ranked.append((0, i))
            elif not (looks.get("weapon") or looks.get("cures") or looks.get("teaches")):
                ranked.append((1, i))
        return [i for _, i in sorted(ranked)]

    def _explore(self, sit: Situation, what: str) -> Choice | None:
        """不知道要找的东西/人在哪：先把此处仔细翻一遍（找物件时），再去最近的、没看过或很久没看过的地方。
        一切依据是自己的地图与勘察记录——不知道的地方不在候选里，全知的最短路与藏匿处从不参与。"""
        b = sit.beliefs
        here = self._here(b)
        name = self._name(b, what)
        sk = b.sketch(what)
        if here and sk is not None and sk.kind == Kind.ITEM and here not in b.searched \
                and not self._did_recently(sit, Op.INSPECT, here):
            choice = self._pick(sit, f"{name}下落不明，先把这里仔细翻一遍", Op.INSPECT, here)
            if choice is not None:
                return Choice(choice.index, choice.rationale, "explore")
        if here is None:
            return None
        options = []
        blocked = self._blocked_doors(sit)
        for p, psk in sorted(b.entities.items()):
            last = b.surveyed.get(p)
            if psk.kind != Kind.PLACE or p == here or (last is not None and sit.now - last <= STALE):
                continue
            d = believed_distance(b, here, p, blocked)
            d = d if d is not None else believed_distance(b, here, p)
            if d is not None:
                options.append((d, last if last is not None else -1, p))
        for _, _, p in sorted(options):
            choice = self._go_towards(sit, p, f"{name}下落不明，去{self._name(b, p)}找找")
            if choice is not None:
                return Choice(choice.index, choice.rationale, "explore")
        # 该看的地方都看过了（小物件可能揣在谁身上）：向眼前的人打听
        q = Fact(Proposition.rel(what, Rel.AT, None), True)
        for p in self._persons_here(b):
            if p != what and not self._did_recently(sit, Op.ASK, p):
                choice = self._pick(sit, f"打听{name}的下落", Op.ASK, p, topic=q)
                if choice is not None:
                    return Choice(choice.index, choice.rationale, "explore")
        return None

    # ------------------------------------------------------------
    #  近期经历
    # ------------------------------------------------------------

    @staticmethod
    def _did_recently(sit: Situation, op: Op, target: str) -> bool:
        return any(
            ep.modality == Modality.SELF and ep.event.kind == op.value and ep.event.target == target
            and sit.now - ep.tick <= RECENT
            for ep in sit.beliefs.episodes
        )

    @staticmethod
    def _said(sit: Situation, listener: str, fact: Fact) -> bool:
        """是否已经对此人说过这句话（持久记录，不随经历缓冲滚掉；说过就不必追着再说）。"""
        return any(s.listener == listener and s.fact == fact for s in sit.beliefs.said)

    @staticmethod
    def _attackers_of(sit: Situation, victim: str, window: int = 3) -> list[str]:
        """近期（亲眼所见或亲身所受）对 victim 动过手的人，新近者在前。"""
        seen: list[str] = []
        for ep in reversed(sit.beliefs.episodes):
            ev = ep.event
            if ev.kind == Op.ATTACK.value and ev.target == victim and ev.actor and sit.now - ep.tick <= window \
                    and ev.actor not in seen:
                seen.append(ev.actor)
        return seen

    # ------------------------------------------------------------
    #  信念查询
    # ------------------------------------------------------------

    @staticmethod
    def _status(b: BeliefStore, person: str, status: str) -> bool:
        return b.holds(Proposition.attr(person, status, True))

    @staticmethod
    def _here(b: BeliefStore) -> str | None:
        return b.location_of(b.owner)

    def _persons_here(self, b: BeliefStore) -> list[str]:
        here = self._here(b)
        return [p for p, sk in sorted(b.entities.items())
                if sk.kind == Kind.PERSON and p != b.owner and here is not None and b.location_of(p) == here]

    @staticmethod
    def _whereabouts(sit: Situation, eid: str) -> str | None:
        """矛盾的说法并存时按“可信度 × 说话者的可靠度（来自长期记忆）”取舍；没有记忆就是最可信的那一条。"""
        b = sit.beliefs
        ps = b.positives(eid, Rel.AT.value)
        if not ps:
            return None
        if sit.memory is None or len(ps) == 1:
            return ps[0].prop.value  # type: ignore[return-value]
        mem = sit.memory

        def score(bl) -> float:
            return effective_confidence(bl, sit.now) * (mem.reliability(bl.informant) if bl.hearsay else 1.0)

        best = sorted(ps, key=lambda bl: (-score(bl), -bl.learned_at, bl.prop.sort_key()))[0]
        return best.prop.value  # type: ignore[return-value]

    @staticmethod
    def _held_items(b: BeliefStore) -> list[str]:
        return [i for i, sk in sorted(b.entities.items()) if sk.kind == Kind.ITEM and b.location_of(i) == b.owner]

    @staticmethod
    def _cures(b: BeliefStore, item: str) -> str | None:
        sk = b.sketch(item)
        return dict(sk.attrs).get("cures") if sk else None  # type: ignore[return-value]

    @staticmethod
    def _name(b: BeliefStore, eid: str) -> str:
        sk = b.sketch(eid)
        return sk.name if sk else eid
