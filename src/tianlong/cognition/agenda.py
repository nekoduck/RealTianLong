"""
[INPUT]: 依赖 core 的 Fact / Kind / Modality / Op / Outcome / Percept / Rel / Social
[OUTPUT]: 对外提供 Obligation（欠着别人的：被问到的问题 answer、被当面搭话 reply）、Said（对谁说过什么，含只有言语行为的闲话）、
          SocialCue（别人对我、或当着我的面做出的言语行为与姿态）、fold_agenda()、fold_social()、
          MAX_OBLIGATIONS / MAX_SAID / MAX_CUES / REPLY_TTL / ATTITUDE_RANGE / SOCIAL_ATTITUDE / SOFT_SOCIAL
[POS]: cognition 的持久任务状态与社交状态：短期经历缓冲（episodes，容量 12）会被环顾、响动挤掉，“有人问过我”“我已经告诉过他”不能跟着消失。
       这里把它们从感知折叠成独立的、有界的记录：被人问到 → 记一笔待答；被人当面搭话（不带命题的言语）→ 记一笔待回话；
       自己把答案说给了他 → 这一笔勾销，并记下“说过”；回了话（或以拳脚作答）→ 待回话勾销；说“不知道/不肯说”→ 待答也勾销。
       “说过”只对说的那一刻的认知有效：自己对那个槽位的认知后来变了（钥匙追回来了、又被偷了），或对方就同一件事又问了一遍，
       这一笔“说过”随即作废——变了的消息是新消息，再问一遍就是还想听；待回话过了 REPLY_TTL 还没回，时机已过即作废。
       社交状态（fold_social）：别人对我或当众的言语行为记为有界的 SocialCue（溢出时先丢别人之间的闲谈，冲着我或当众的留得更久）；
       对每个人的态度 ∈ ATTITUDE_RANGE 由看见/听见的言语行为与动手、救治、赠物确定性地增减——见礼只是客套，只把素不相识（0）暖成
       点头之交（+1），再多也不加分、抹不掉积怨；态度只存在于这个角色自己的心里；company 记着眼前每个人“自何时起一直在我身边”
       （只来自环顾），先礼后兵与守卫“一次闯入只动一次手”都以它为准；yielded 记着每个人最近一次冲着我或当众服软的时刻——
       “饶过他”是记在心里的事，不随 8 条的线索缓冲滚掉。
       与信念一样只来自感知，不读真相；容量有界且溢出时丢最旧的一条（线索先丢别人之间的闲谈；写明，不静默增长）
       物品请求义务区分待决定、已答应、暂缓和拒绝；回应绑定请求，实际 GIVE 才结束；暂缓只因相关已知状态变化唤醒。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace

from tianlong.core import Fact, Kind, Modality, Op, Outcome, Percept, Rel, Social

MAX_OBLIGATIONS = 16
MAX_SAID = 64
MAX_CUES = 8
REPLY_TTL = 3                 # 当面搭话多少个 tick 内不回就算错过了时机
ATTITUDE_RANGE = (-3, 3)
# 冲着我来的言语行为对态度的影响：辱骂最重；道谢、求情、赔罪、服软、安慰、见礼使其回暖
SOCIAL_ATTITUDE: Mapping[Social, int] = {
    Social.INSULT: -2, Social.TAUNT: -1, Social.THREATEN: -1, Social.CHALLENGE: -1, Social.COMMAND: -1,
    Social.THANK: 1, Social.PRAISE: 1, Social.APOLOGIZE: 1, Social.PLEAD: 1, Social.SUBMIT: 1, Social.COMFORT: 1,
    Social.GREET: 1,
}
COURTESY_CEILING = 1          # 见礼只是客套：只把素不相识（0）暖到点头之交（+1），再多的见礼不加分，也抹不掉积怨
SOFT_SOCIAL = frozenset({Social.SUBMIT, Social.APOLOGIZE, Social.PLEAD})    # 服软：记进 yielded
ATTACKED_ME, ATTACKED_ALLY, HELPED_ME = -2, -1, 2
_SPEECH = frozenset({Op.TELL.value, Op.ASK.value})
_SOCIAL_KINDS = frozenset({Op.TELL.value, Op.ASK.value, Op.WAIT.value, Op.REQUEST_ITEM.value})
_CLOSES_QUESTION = frozenset({Social.EXPLAIN, Social.REFUSE})              # “不知道”“不肯说”也是给了回音


@dataclass(frozen=True, slots=True)
class Obligation:
    kind: str                       # "answer"：有人问了我一个问题；"reply"：有人当面对我说了句不带命题的话
    counterpart: str                # 问话/搭话的人
    topic: Fact | None              # answer：问的命题（宾语为 None 的提问）；reply：None
    since: int
    social: Social | None = None    # reply：对方那句话的言语行为（回话按它、按性情与态度选）
    item: str | None = None
    beneficiary: str | None = None
    request_ref: str | None = None
    state: str = "pending"          # 物品请求：pending → accepted/deferred/refused；实际 GIVE 才结束


@dataclass(frozen=True, slots=True)
class Said:
    listener: str
    fact: Fact | None               # None：只有言语行为的闲话（回话、叫阵、搭话）
    tick: int
    social: Social | None = None


@dataclass(frozen=True, slots=True)
class SocialCue:
    """别人对我、或当着我的面做出的言语行为与姿态：修辞，不是事实；回话、先礼后兵、搭话据此判断对方的态度。"""

    frm: str
    op: Op                          # TELL / ASK / WAIT（姿态）
    social: Social | None
    tick: int
    utterance: str | None = None
    to: str | None = None           # 冲着谁：我、别人，或 None（当众的姿态）


def _answers(told: Fact, asked: Fact) -> bool:
    return told.prop.subject == asked.prop.subject and told.prop.predicate == asked.prop.predicate


def _item_requests(owner: str, obligations: tuple[Obligation, ...], p: Percept,
                   changed: Collection[tuple[str, str]]) -> tuple[Obligation, ...]:
    """听见、答应和交付分别折叠；同意的闲话须明确回应请求编号，不能当成物品已到手。"""
    awakened = []
    for o in obligations:
        related = {owner, o.counterpart, o.item, o.beneficiary}
        relevant = any(s in related and pred in (Rel.AT.value, "attr.subdued", "attr.poisoned", "attr.wounded")
                       for s, pred in changed)
        if o.kind == "request_item" and o.state == "deferred" and relevant:
            o = replace(o, state="pending")
        awakened.append(o)
    obligations = tuple(awakened)
    ev = p.event
    if ev is None or ev.outcome != Outcome.SUCCESS:
        return obligations
    if ev.kind == Op.REQUEST_ITEM.value and ev.actor and ev.target and ev.obj and ev.beneficiary and ev.request_ref:
        kind, counterpart = ("request_item", ev.actor) if p.modality == Modality.SPEECH and ev.target == owner \
                            else ("requested_item", ev.target) if p.modality == Modality.SELF and ev.actor == owner \
                            else (None, None)
        if kind:
            rest = tuple(o for o in obligations if not (o.kind == kind and o.counterpart == counterpart
                         and o.item == ev.obj and o.beneficiary == ev.beneficiary))
            return (*rest, Obligation(kind, counterpart, None, p.tick, ev.social, ev.obj,
                                      ev.beneficiary, ev.request_ref))[-MAX_OBLIGATIONS:]
    if ev.kind == Op.TELL.value and ev.request_ref:
        states = {Social.AGREE: "accepted", Social.REFUSE: "refused", Social.EXPLAIN: "deferred"}
        state = states.get(ev.social)
        result = []
        for o in obligations:
            mine = p.modality == Modality.SELF and o.kind == "request_item" and o.counterpart == ev.target
            heard = p.modality == Modality.SPEECH and ev.target == owner and o.kind == "requested_item" and o.counterpart == ev.actor
            if state and (mine or heard) and o.request_ref == ev.request_ref:
                if not (mine and state == "refused"):
                    result.append(replace(o, state=state))
            else:
                result.append(o)
        return tuple(result)
    if ev.kind == Op.GIVE.value and ev.obj:
        return tuple(o for o in obligations if not (
            o.item == ev.obj and ((p.modality == Modality.SELF and o.kind == "request_item" and o.counterpart == ev.target)
            or (p.modality == Modality.SIGHT and o.kind == "requested_item" and o.counterpart == ev.actor and ev.target == owner))
            and (ev.request_ref == o.request_ref or ev.request_ref is None)))
    return obligations


# ============================================================
#  承诺状态：待答、待回话、说过
# ============================================================


def fold_agenda(owner: str, obligations: tuple[Obligation, ...], said: tuple[Said, ...], p: Percept,
                changed: Collection[tuple[str, str]] = ()) -> tuple[tuple[Obligation, ...], tuple[Said, ...]]:
    """changed：本次修正里自己的认知真正变了值的槽位 (主语, 谓词)——关于它们的“说过”作废。"""
    if changed:
        said = tuple(s for s in said if s.fact is None or s.fact.prop.slot not in changed)
    obligations = _item_requests(owner, obligations, p, changed)
    obligations = tuple(o for o in obligations if o.kind != "reply" or p.tick - o.since <= REPLY_TTL)
    ev = p.event
    if ev is None:
        return obligations, said
    if p.modality == Modality.SPEECH and ev.kind in _SPEECH and ev.target == owner and ev.actor:
        if ev.topic is None:
            # 当面搭话：同一个人只记最新的一句
            rest = tuple(o for o in obligations if not (o.kind == "reply" and o.counterpart == ev.actor))
            obligations = (*rest, Obligation("reply", ev.actor, None, p.tick, ev.social))[-MAX_OBLIGATIONS:]
        elif ev.kind == Op.ASK.value:
            ob = Obligation("answer", ev.actor, ev.topic, p.tick)
            if not any(o.counterpart == ob.counterpart and o.topic == ob.topic for o in obligations):
                obligations = (*obligations, ob)[-MAX_OBLIGATIONS:]
            said = tuple(s for s in said
                         if not (s.listener == ev.actor and s.fact is not None and _answers(s.fact, ev.topic)))
    elif p.modality == Modality.SELF and ev.kind in _SPEECH and ev.outcome == Outcome.SUCCESS and ev.target:
        if ev.topic is None:
            said = (*(s for s in said if not (s.listener == ev.target and s.fact is None and s.social == ev.social)),
                    Said(ev.target, None, p.tick, ev.social))[-MAX_SAID:]
            closes = ("reply", "answer") if ev.social in _CLOSES_QUESTION else ("reply",)
            obligations = tuple(o for o in obligations if not (o.counterpart == ev.target and o.kind in closes))
        elif ev.kind == Op.TELL.value:
            said = (*(s for s in said if not (s.listener == ev.target and s.fact == ev.topic)),
                    Said(ev.target, ev.topic, p.tick))[-MAX_SAID:]
            obligations = tuple(o for o in obligations
                                if not (o.counterpart == ev.target and o.topic is not None
                                        and _answers(ev.topic, o.topic)))
    elif p.modality == Modality.SELF and ev.kind == Op.ATTACK.value and ev.target:
        # 以拳脚作答：这一句不必再回
        obligations = tuple(o for o in obligations if not (o.kind == "reply" and o.counterpart == ev.target))
    return obligations, said


# ============================================================
#  社交状态：线索、态度、眼前的人
# ============================================================


def _clamp(v: int) -> int:
    lo, hi = ATTITUDE_RANGE
    return max(lo, min(hi, v))


def attitude_delta(owner: str, allies: Collection[str], p: Percept,
                   attitudes: Mapping[str, int] | None = None) -> tuple[str, int] | None:
    """这条感知让我对谁的态度变了多少：只看冲着我（或我的自己人）来的、看得出是谁做的事。
    attitudes：眼下的态度——见礼只在素不相识时加分（COURTESY_CEILING），刷不出好感。"""
    ev = p.event
    if ev is None or not ev.actor or ev.actor == owner or p.modality not in (Modality.SPEECH, Modality.SIGHT):
        return None
    if ev.kind in _SOCIAL_KINDS and ev.target == owner and ev.social is not None:
        d = SOCIAL_ATTITUDE.get(ev.social, 0)
        if ev.social == Social.GREET and not 0 <= (attitudes or {}).get(ev.actor, 0) < COURTESY_CEILING:
            d = 0
    elif ev.kind == Op.ATTACK.value and ev.target == owner:
        d = ATTACKED_ME
    elif ev.kind == Op.ATTACK.value and ev.target in allies:
        d = ATTACKED_ALLY
    elif ev.kind in (Op.USE.value, Op.GIVE.value) and ev.target == owner and ev.outcome == Outcome.SUCCESS:
        d = HELPED_ME
    else:
        return None
    return (ev.actor, d) if d else None


def _company(owner: str, company: Mapping[str, int], p: Percept) -> Mapping[str, int] | None:
    """环顾时在我身边的人：一直在的保留“自何时起”，新来的记下此刻，走了的划掉。不是一次真正的环顾就不动。"""
    if p.modality != Modality.SCENE or not p.scopes:
        return None
    here = next((f.prop.value for f in p.facts
                 if f.holds and f.prop.subject == owner and f.prop.predicate == Rel.AT.value), None)
    if here is None:
        return None
    persons = {sk.id for sk in p.sketches if sk.kind == Kind.PERSON}
    present = sorted({f.prop.subject for f in p.facts if f.holds and f.prop.predicate == Rel.AT.value
                      and f.prop.value == here and f.prop.subject in persons and f.prop.subject != owner})
    return {x: min(company.get(x, p.tick), p.tick) for x in present}


def _bounded(owner: str, cues: tuple[SocialCue, ...]) -> tuple[SocialCue, ...]:
    """超出 MAX_CUES：先丢最旧的一条别人之间的闲谈（没人据它行事），没有才丢最旧的一条。"""
    while len(cues) > MAX_CUES:
        i = next((k for k, c in enumerate(cues) if c.to not in (owner, None)), 0)
        cues = cues[:i] + cues[i + 1:]
    return cues


def fold_social(owner: str, allies: Collection[str], cues: tuple[SocialCue, ...], attitudes: Mapping[str, int],
                company: Mapping[str, int], p: Percept, yielded: Mapping[str, int] | None = None
                ) -> tuple[tuple[SocialCue, ...], Mapping[str, int], Mapping[str, int], Mapping[str, int]]:
    """线索：冲着我、或当众（姿态；别人之间听得见的说话）的言语行为，有言语行为或原话才记。
    态度：按 attitude_delta 增减并截在 ATTITUDE_RANGE 内，归零即删（缺席 = 0）。
    yielded：冲着我或当众服软（SOFT_SOCIAL）的人 → 最近一次的时刻，与线索同时记下，但不随线索缓冲滚掉。"""
    ev = p.event
    yielded = yielded or {}
    if ev is not None and ev.actor and ev.actor != owner and ev.kind in _SOCIAL_KINDS \
            and p.modality in (Modality.SPEECH, Modality.SIGHT) and (ev.social is not None or ev.utterance) \
            and (ev.target in (owner, None) or p.modality == Modality.SPEECH):
        cues = _bounded(owner, (*cues, SocialCue(ev.actor, Op(ev.kind), ev.social, p.tick, ev.utterance, ev.target)))
        if ev.social in SOFT_SOCIAL and ev.target in (owner, None):
            yielded = {**yielded, ev.actor: max(p.tick, yielded.get(ev.actor, p.tick))}
    delta = attitude_delta(owner, allies, p, attitudes)
    if delta is not None:
        who, d = delta
        v = _clamp(attitudes.get(who, 0) + d)
        attitudes = {**{k: x for k, x in attitudes.items() if k != who}, **({who: v} if v else {})}
    moved = _company(owner, company, p)
    return cues, attitudes, (company if moved is None else moved), yielded
