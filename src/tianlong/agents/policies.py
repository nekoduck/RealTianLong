"""
[INPUT]: 依赖 agents/policy_kit 的 Situation / Choice / Policy / PolicyKit / CHATTER，agents/tactics 的 MartialTactics / reply_act / HOT /
         DEFIANT_SOCIAL，cognition 的 Candidate，cognition/navigation 的 believed_place，cognition/agenda 的 REPLY_TTL，
         cognition/goals 的 BeliefReader，core/goals 的 GoalRegistry，
         core 的 Op / Manner / Rel / Kind / Modality / Fact / Proposition / Social / HOSTILE_SOCIAL / derive_seed，core/profiles 的 Goal / GoalKind
[OUTPUT]: 对外提供 ScriptedPolicy（角色条件化的规则策略）、hush_chatter()（每处每 tick 至多一句闲谈的确定性裁决）、CHAT_COOLDOWN、QUIET，
          并再导出 Situation / Choice / Policy
[POS]: agents 的决策作曲者：自救 → 还手 → 救治盟友 → 回话 → 回应提问 → 按目标（守护/获取/递送/守地/寻仇（先礼后兵）/灭口/护人，
       受时间闸门约束）→ 见义出声 → 闲谈 → 查探响动 → 等待。
       回话按性情 × 态度 × 对方的言语行为选言语行为（reply_act；火爆且积怨已深者被骂即动手；见过礼的再被招呼只寒暄），
       对方是在回我的话就到此为止（主角除外），寻仇对象的话留给先礼后兵（只在先礼后兵还要出手时；气已出了就自己回话）、
       闯入者的顶撞留给守卫；
       被问到的事知道就如实相告（候选话题里没有这句，就趁热以 free 照实说出自己相信的命题），不知道就说不知道
       （不带命题的 EXPLAIN）并勾销，一个问题答不上来也不耽误后面的问题；
       话多的角色在主角（或眼前的人）身边按 (角色, tick) 派生的确定性闸门搭话，头一回见礼、此后说笑，有冷却；
       别人刚开过口（QUIET 个 tick 内）不插嘴；几个人同时想闲谈、或同处已有人正经开口，由编排按 hush_chatter() 只留一句。
       开口只是修辞（Choice.free，index 指向 WAIT）：只认下标的学习层看到的是等待。
       目标所需的东西或人下落不明时，凭自己的地图与勘察记录去找（而不是原地干等）；
       等待带结构化原因（没事可做/目标未到时辰/自以为已达成/不知道而卡住/没有可行候选/找不到动作）。
       它是阶段 A 的初始策略，也是阶段 C 模仿学习的示范者；RL 策略实现同一协议即可替换
       请求物品按本人知识、能力和目标给予/拒绝/暂缓；先答应、下一 tick 才尝试 GIVE；请求也占用实际言语话头。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import replace

from tianlong.agents.policy_kit import CHATTER, Choice, Policy, Situation
from tianlong.agents.tactics import DEFIANT_SOCIAL, HOT, MartialTactics, reply_act
from tianlong.cognition import Candidate
from tianlong.cognition.agenda import REPLY_TTL
from tianlong.cognition.goals import BeliefReader
from tianlong.cognition.navigation import believed_place
from tianlong.core import (
    HOSTILE_SOCIAL,
    Fact,
    Kind,
    Manner,
    Modality,
    Op,
    Proposition,
    Rel,
    Social,
    derive_seed,
)
from tianlong.core.goals import GoalRegistry
from tianlong.core.profiles import Goal, GoalKind

_REGISTRY = GoalRegistry()
CHAT_COOLDOWN = 6        # 对同一个人（没有主角时：对任何人）搭过话之后，多少个 tick 内不再找话说
QUIET = 1                # 同处有别人开过口之后，多少个 tick 内不插嘴找话说：一句闲谈不会引出一串
_TALK = (Op.TELL, Op.ASK, Op.REQUEST_ITEM)

__all__ = ["CHAT_COOLDOWN", "QUIET", "Choice", "Policy", "ScriptedPolicy", "Situation", "hush_chatter"]


def hush_chatter(now: int, decided: Mapping[str, tuple[str | None, str, Candidate]]) -> frozenset[str]:
    """每处每 tick 至多一句闲谈：decided 是 角色 → (自己认为的所在, Choice.tag, 交给内核的行动)，返回该按捺住、改为等待的人。
    同处已有人正经开口（回话、答话、叫阵、喝止……）闲谈一概让出话头；几个人同时想闲谈，
    按 derive_seed("floor", 地点, tick, 角色) 取一人——确定性，与扇出顺序无关；话多的人想开口的回数多，轮到的也多。"""
    places: dict[str | None, list[tuple[str, str, Candidate]]] = {}
    for agent, (place, tag, cand) in sorted(decided.items()):
        places.setdefault(place, []).append((agent, tag, cand))
    hushed: set[str] = set()
    for place, here in places.items():
        chat = [a for a, tag, _ in here if tag == CHATTER]
        if not chat:
            continue
        busy = place is None or any(tag != CHATTER and c.op in _TALK for _, tag, c in here)
        keep = None if busy else min(chat, key=lambda a: derive_seed("floor", place, now, a))
        hushed |= {a for a in chat if a != keep}
    return frozenset(hushed)


# ============================================================
#  ScriptedPolicy
#  所有判断都来自信念与近期经历；“最近做过”由 SELF 经历判断，避免反复盘问同一个人
# ============================================================


class ScriptedPolicy(MartialTactics):
    def choose(self, sit: Situation) -> Choice:
        steps: list[Callable[[Situation], Choice | None]] = [
            self._cure_self, self._retaliate, self._heal_allies, self._request_item, self._reply, self._answer_questions,
        ]
        steps += [self._goal_step(g) for g in sit.profile.goals if g.active(sit.now)]
        steps += [self._witness, self._chatter, self._investigate_noise]
        for step in steps:
            choice = step(sit)
            if choice is not None:
                return choice
        return self._wait(sit)

    def _request_item(self, sit: Situation) -> Choice | None:
        """只凭本人知识、能力与动机决定；先明确答应，下一 tick 才尝试 GIVE。"""
        b = sit.beliefs
        for ob in sorted(b.obligations, key=lambda o: (o.since, o.request_ref or "")):
            if ob.kind != "request_item" or ob.state == "deferred" or ob.counterpart not in self._persons_here(b):
                continue
            def reply(social, why, ob=ob):
                choice = self._say(sit, ob.counterpart, social, why)
                return replace(choice, free=replace(choice.free, request_ref=ob.request_ref))

            hostile = b.attitude(ob.counterpart) < 0 or b.attitude(ob.beneficiary) < 0 or any(
                g.kind == GoalKind.HOSTILE and g.person in (ob.counterpart, ob.beneficiary)
                for g in sit.profile.goals if g.active(sit.now))
            protected = any(g.item == ob.item and g.kind in (GoalKind.PROTECT, GoalKind.ACQUIRE)
                            or (g.item == ob.item and g.kind == GoalKind.DELIVER and g.recipient != ob.counterpart)
                            for g in sit.profile.goals if g.active(sit.now))
            if hostile or protected:
                return reply(Social.REFUSE, "这项物品请求与我的态度或目标冲突，明确拒绝")
            can_give = b.location_of(ob.item) == b.owner and not self._status(b, b.owner, "subdued")
            useful = any(self._status(b, ob.beneficiary, status) and
                         (dict(b.sketch(ob.item).attrs).get("cures") == status)
                         for status in ("poisoned", "wounded")) if b.sketch(ob.item) else False
            willing = (ob.beneficiary in b.allies or ob.counterpart in b.allies or useful
                       or b.attitude(ob.counterpart) > 0 or b.trust.get(ob.counterpart, 0.6) >= 0.6)
            if not willing:
                return reply(Social.REFUSE, "不信任这项物品请求，明确拒绝")
            if not can_give:
                return reply(Social.EXPLAIN, "当前没有持有所求物品或无法递交，说明暂缓")
            if ob.state == "pending":
                return reply(Social.AGREE, "同意物品请求；答应尚未交付")
            give = self._pick(sit, "执行已答应的物品请求，尝试实际递交", Op.GIVE, ob.counterpart, ob.item)
            if give is not None:
                return replace(give, request_ref=ob.request_ref)
        return None

    def _wait(self, sit: Situation) -> Choice:
        """等待，并说明为什么：模仿学习要区分“合理地等”与“因为不知道而卡住”。"""
        goals = sit.profile.goals
        active = [g for g in goals if g.active(sit.now)]
        if not goals:
            return Choice(0, "没什么要做的，原地等待", "idle")
        if not active:
            return Choice(0, "时辰未到，先等着", "goal_inactive")
        reader = BeliefReader(sit.beliefs)
        status = [_REGISTRY.satisfied(reader, sit.agent, g, sit.profile.allies) for g in active]
        if all(s is True for s in status):
            return Choice(0, "一切如常，原地等待", "goal_done")
        if len(sit.candidates) <= 1:
            return Choice(0, "无事可为，只能等待", "no_candidate")
        if any(s is None for s in status) or any(self._unknown_target(sit, g) for g in active):
            return Choice(0, "不知从何下手，只好等待", "stuck_unknown")
        return Choice(0, "想不出办法，原地等待", "expert_no_action")

    @staticmethod
    def _unknown_target(sit: Situation, g: Goal) -> bool:
        b = sit.beliefs
        return any(x is not None and believed_place(b, x) is None for x in (g.item, g.person, g.recipient))

    # ------------------------------------------------------------
    #  回话：被当面搭话，按性情 × 态度 × 对方的言语行为回一句
    # ------------------------------------------------------------

    def _reply(self, sit: Situation) -> Choice | None:
        b, prof = sit.beliefs, sit.profile
        here = set(self._persons_here(b))
        pending = sorted((o for o in b.obligations if o.kind == "reply" and o.counterpart in here
                          and sit.now - o.since <= REPLY_TTL and not self._left_to_goals(sit, o.counterpart, o.social)),
                         key=lambda o: (o.counterpart != sit.player, o.since, o.counterpart))
        for ob in pending:
            who = ob.counterpart
            cue = next((c for c in reversed(b.cues) if c.frm == who and c.tick == ob.since and c.to == sit.agent), None)
            question = cue is not None and cue.op == Op.ASK
            if who != sit.player and not question and self._answers_me(sit, who, ob.since):
                continue            # 对方是在回我的话：一来一往到此为止，免得两个 NPC 没完没了
            attitude = b.attitude(who)
            if ob.social in HOSTILE_SOCIAL and prof.temper >= HOT and attitude <= -2 \
                    and random.Random(derive_seed("escalate", sit.agent, who, ob.since)).random() < prof.temper:
                strike = self._pick(sit, f"{self._name(b, who)}出言不逊，忍无可忍", Op.ATTACK, who)
                if strike is not None:
                    return strike
            act = reply_act(ob.social, temper=prof.temper, attitude=attitude, question=question, chatty=prof.chatty,
                            hostile=who in self._pursued(sit),
                            greeted=self._last_spoke(sit, who, (Social.GREET,)) is not None)
            return self._say(sit, who, act, f"回{self._name(b, who)}的话")
        return None

    def _answers_me(self, sit: Situation, who: str, since: int) -> bool:
        last = self._last_spoke(sit, who)
        return last is not None and last >= since - 1

    @staticmethod
    def _pursued(sit: Situation) -> set[str]:
        return {g.person for g in sit.profile.goals if g.kind == GoalKind.HOSTILE and g.person and g.active(sit.now)}

    def _left_to_goals(self, sit: Situation, who: str, social: Social | None) -> bool:
        """这句话由目标自己接：仇家说什么都交给先礼后兵——只在先礼后兵还要出手时（与 _hostile 同一判断：
        仇人已达了结条件或已被制住，气已出了，就自己回话）；守地时闯入者的顶撞交给守卫（顶撞即硬闯）。"""
        b = sit.beliefs
        if any(g.kind == GoalKind.HOSTILE and g.person == who and g.active(sit.now)
               and not self._status(b, who, g.until) and not self._status(b, who, "subdued")
               for g in sit.profile.goals):
            return True
        posts = {g.home for g in sit.profile.goals if g.kind == GoalKind.GUARD and g.home and g.active(sit.now)}
        return social in DEFIANT_SOCIAL and self._here(sit.beliefs) in posts and who not in sit.profile.allies

    # ------------------------------------------------------------
    #  回应提问：如实说出自己最相信的下落；不知道就说不知道
    # ------------------------------------------------------------

    def _answer_questions(self, sit: Situation) -> Choice | None:
        """欠着的问题（持久记录，不随经历缓冲滚掉）：问话的人在眼前、自己又知道答案，就如实相告；
        候选话题里没有这句答话（只谈关心的物品与认识的人），就趁热（REPLY_TTL 内）以 free 照实说出自己相信的命题——
        与“不知道”同一时限：只认下标的学习层把它当等待、问题勾销不了，时限保证示范者不会因此一直干等；
        不知道就趁热说一句“不知道”（不带命题的 EXPLAIN，说出口即勾销），而不是让问话的人干等。
        一个问题此刻答不出口，照旧往下看后面的问题。"""
        b = sit.beliefs
        here = set(self._persons_here(b))
        unknown = None
        for ob in b.obligations:
            if ob.kind != "answer" or ob.counterpart not in here or ob.topic is None:
                continue
            best = b.best(ob.topic.prop.subject, ob.topic.prop.predicate)
            if best is None:
                if unknown is None and sit.now - ob.since <= REPLY_TTL:
                    unknown = ob
                continue
            fact = Fact(best.prop, True)
            if not self._said(sit, ob.counterpart, fact):
                late = "（前番问过）" if sit.now - ob.since > 2 else ""
                why = f"{self._name(b, ob.counterpart)}问我{late}，如实相告"
                choice = self._pick(sit, why, Op.TELL, ob.counterpart, topic=fact)
                if choice is None and sit.now - ob.since <= REPLY_TTL:
                    choice = self._tell(sit, ob.counterpart, fact, why)
                if choice is not None:
                    return choice
        if unknown is not None and unknown.topic is not None:
            what = self._name(b, unknown.topic.prop.subject)
            return self._say(sit, unknown.counterpart, Social.EXPLAIN,
                             f"{self._name(b, unknown.counterpart)}问起{what}，我也不知道")
        return None

    # ------------------------------------------------------------
    #  闲谈：话多的人在主角身边会找话说
    # ------------------------------------------------------------

    def _chatter(self, sit: Situation) -> Choice | None:
        """确定性闸门：random(derive_seed(角色, tick)) < chatty。头一回开口是见礼，此后说笑；
        对主角（不知道谁是主角时：对任何人）开过口后 CHAT_COOLDOWN 个 tick 内不再找话说。仇家、心怀恶感的人、被制住的人不搭理。
        同处有别人刚开过口（QUIET 个 tick 内，凭自己听见看见的）就不插嘴；选定的话带 CHATTER 标签，交编排按 hush_chatter() 裁决。"""
        prof, b, me = sit.profile, sit.beliefs, sit.agent
        if prof.is_player or prof.chatty <= 0 or self._status(b, me, "subdued"):
            return None
        here_place = self._here(b)
        if any(ep.event.kind in (Op.TELL.value, Op.ASK.value) and ep.event.actor not in (None, me)
               and ep.event.place == here_place and sit.now - ep.tick <= QUIET for ep in b.episodes):
            return None
        foes = self._pursued(sit)
        here = [p for p in self._persons_here(b)
                if p not in foes and b.attitude(p) >= -1 and not self._status(b, p, "subdued")]
        partners = ([sit.player] if sit.player in here else []) if sit.player is not None else here
        last = self._last_spoke(sit, sit.player)
        if not partners or (last is not None and sit.now - last < CHAT_COOLDOWN):
            return None
        rng = random.Random(derive_seed("chatter", me, sit.now))
        if rng.random() >= prof.chatty:
            return None
        spoke = {p: self._last_spoke(sit, p) for p in partners}
        partner = min(partners, key=lambda p: (-1 if spoke[p] is None else spoke[p], p))
        social = Social.GREET if spoke[partner] is None else Social.JOKE if rng.random() < 0.3 else Social.REMARK
        return self._say(sit, partner, social, f"找{self._name(b, partner)}说几句话", CHATTER)

    # ------------------------------------------------------------
    #  目标
    # ------------------------------------------------------------

    def _goal_step(self, g: Goal) -> Callable[[Situation], Choice | None]:
        return {
            GoalKind.PROTECT: lambda s: self._protect(s, g),
            GoalKind.ACQUIRE: lambda s: self._acquire(s, g),
            GoalKind.DELIVER: lambda s: self._deliver(s, g),
            GoalKind.GUARD: lambda s: self._guard(s, g),
            GoalKind.HOSTILE: lambda s: self._hostile(s, g),
            GoalKind.ESCAPE: lambda s: self._escape(s, g),
            GoalKind.DEFEND: lambda s: self._defend(s, g),
        }[g.kind]

    def _protect(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        here = b.location_of(me)
        owners = b.subjects(Rel.OWNS.value, g.item)
        loc = b.location_of(g.item)
        name = self._name(b, g.item)
        if loc is not None and (loc == g.home or loc in owners):
            return None  # 一切如常
        if loc == me and g.home:
            home_place = believed_place(b, g.home)
            if here == home_place:
                return self._pick(sit, f"把{name}放回原处", Op.PUT, g.home, g.item, Manner.NORMAL)
            return self._go_towards(sit, home_place, f"带着{name}回去")
        if loc is not None:
            loc_kind = b.sketch(loc).kind if b.sketch(loc) else None
            if loc_kind == Kind.PERSON:
                return self._held_by_other(sit, g, loc, owners)
            if believed_place(b, loc) == here:
                return self._pick(sit, f"{name}被挪了位置，先收回来", Op.TAKE, g.item, manner=Manner.NORMAL)
            return self._go_towards(sit, believed_place(b, loc), f"去{self._name(b, loc)}找回{name}")
        # ---- 不知下落 ≠ 丢失：只有确知“不在原处”才开始找（盘问、搜身）；不知道就只去原处看一眼 ----
        home_belief = b.believed(Proposition.rel(g.item, Rel.AT, g.home)) if g.home else None
        if home_belief is None:
            if not self._home_searched(sit, g):
                return self._look_in_on(sit, g)
            # 亲手把原处仔细翻过、仍不见它：当作不在原处，才开始盘问与搜寻
        elif home_belief.holds:
            return None
        # ---- 下落不明：先问、再搜、再查看现场 ----
        suspects = [p for p, sk in sorted(b.entities.items())
                    if sk.kind == Kind.PERSON and p != me and p not in owners and b.location_of(p) == here]
        for s in suspects:
            if not self._did_recently(sit, Op.ASK, s):
                q = Fact(Proposition.rel(g.item, Rel.AT, None), True)
                return self._pick(sit, f"{name}不见了，先问问{self._name(b, s)}", Op.ASK, s, topic=q)
            if not self._did_recently(sit, Op.INSPECT, s):
                return self._pick(sit, f"问不出结果，只好搜{self._name(b, s)}的身", Op.INSPECT, s)
        if here and not self._did_recently(sit, Op.INSPECT, here):
            return self._pick(sit, f"仔细找找{name}", Op.INSPECT, here)
        return self._explore(sit, g.item)

    @staticmethod
    def _home_searched(sit: Situation, g: Goal) -> bool:
        b = sit.beliefs
        return g.home is not None and (g.home in b.searched or believed_place(b, g.home) in b.searched)

    def _look_in_on(self, sit: Situation, g: Goal) -> Choice | None:
        """守护之物从没亲眼见过：走到它该在的地方，仔细看一眼——核实，而不是怀疑任何人。"""
        if g.home is None:
            return None
        b = sit.beliefs
        where = believed_place(b, g.home)
        if where is None:
            return self._explore(sit, g.home)
        name = self._name(b, g.item or "")
        if where == b.location_of(sit.agent):
            choice = self._pick(sit, f"仔细看看{name}还在不在", Op.INSPECT, g.home) \
                or self._pick(sit, f"仔细看看{name}还在不在", Op.INSPECT, where)
        else:
            choice = self._go_towards(sit, where, f"去{self._name(b, where)}看看{name}还在不在")
        return Choice(choice.index, choice.rationale, "explore") if choice else None

    def _held_by_other(self, sit: Situation, g: Goal, holder: str, owners: tuple[str, ...]) -> Choice | None:
        """东西在别人身上：失主亲自去讨要；其他守护者去报告失主。说过的话不重复说。"""
        b, me = sit.beliefs, sit.agent
        here = b.location_of(me)
        name = self._name(b, g.item)
        if me in owners:
            holder_place = believed_place(b, holder)
            if holder_place is None:
                # 不知道那人在哪：向在场的人打听
                for p, sk in sorted(b.entities.items()):
                    if sk.kind == Kind.PERSON and p not in (me, holder) and b.location_of(p) == here \
                            and not self._did_recently(sit, Op.ASK, p):
                        q = Fact(Proposition.rel(holder, Rel.AT, None), True)
                        return self._pick(sit, f"打听{self._name(b, holder)}在哪里", Op.ASK, p, topic=q)
                return self._explore(sit, holder)       # 没人可问：凭自己的地图去找那人
            if holder_place != here:
                return self._go_towards(sit, holder_place, f"去找{self._name(b, holder)}讨回{name}")
            if not self._did_recently(sit, Op.ASK, holder):
                q = Fact(Proposition.rel(g.item, Rel.AT, None), True)
                return self._pick(sit, f"当面质问{self._name(b, holder)}", Op.ASK, holder, topic=q)
            return None
        fact = Fact(Proposition.rel(g.item, Rel.AT, holder), True)
        for owner in owners:
            if owner == me or self._said(sit, owner, fact):
                continue
            if b.location_of(owner) == here:
                return self._pick(sit, f"向失主报告{name}的下落", Op.TELL, owner, topic=fact)
            where = believed_place(b, owner)
            choice = self._go_towards(sit, where, f"去告诉失主{name}在谁那里") if where else self._explore(sit, owner)
            if choice is not None:
                return choice
        return None

    def _acquire(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        loc = self._whereabouts(sit, g.item)
        if loc == me:
            return None
        if loc is None or believed_place(b, loc) is None:
            return self._explore(sit, g.item)          # 不知道在哪：凭自己的地图去找，而不是原地干等
        if believed_place(b, loc) == b.location_of(me):
            return self._pick(sit, f"悄悄拿走{self._name(b, g.item)}", Op.TAKE, g.item, manner=Manner.CAREFUL)
        return self._go_towards(sit, believed_place(b, loc), f"去找{self._name(b, g.item)}")

    def _deliver(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        if b.location_of(g.item) != me:
            return self._acquire(sit, g)
        if g.recipient is None:
            return None
        if b.location_of(g.recipient) == b.location_of(me):
            return self._pick(sit, f"把{self._name(b, g.item)}交给{self._name(b, g.recipient)}", Op.GIVE,
                              g.recipient, g.item)
        where = believed_place(b, g.recipient)
        if where is None:
            return self._explore(sit, g.recipient)
        return self._go_towards(sit, where, f"去找{self._name(b, g.recipient)}")

    # ------------------------------------------------------------
    #  查探：守护范围内传来响动
    # ------------------------------------------------------------

    def _investigate_noise(self, sit: Situation) -> Choice | None:
        b = sit.beliefs
        guarded = {believed_place(b, g.home) for g in sit.profile.goals if g.kind == GoalKind.PROTECT and g.home}
        for ep in reversed(b.episodes):
            if ep.modality == Modality.SOUND and sit.now - ep.tick <= 3 and ep.event.place in guarded:
                choice = self._go_towards(sit, ep.event.place, f"{self._name(b, ep.event.place)}有动静，去看看")
                if choice is not None:
                    return choice
        return None
