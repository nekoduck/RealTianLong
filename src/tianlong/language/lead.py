"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome / Kind / derive_seed，language/templates 的 Names / REASONS / SOCIAL_VERBS / consequences
[OUTPUT]: 对外提供 lead_line(percepts, names, viewer) -> str：本回合“你这一步的结果”写成的一两句人话（没有可说的返回空串）、
          restates(sentence, percepts, names, viewer) -> bool：模型的一句是不是在换个说法复述先声（同一动作的动词 + 同一对象，
          或玩家原话/姿态原样再现）
[POS]: language 的先声：主持人之声的第一句不等模型。结算一完成，就把玩家自己这一步做成没有（去了哪、拿到没有、
       一掌打中没有、说了什么）用确定的句子交给玩家——首字延迟只剩“解释 + 结算”，而第一眼看到的正是最要紧的信息；
       声音模型再从下一句接着写旁人的反应与氛围。只取玩家自己的行动感知（SELF），措辞按 (tick, 操作, 对象) 派生的种子
       在几种说法里确定地轮换，失败必说原因、后果照实写出；等待不抢先（时辰与“时间悄悄过去”交给模型或模板）。
       模型被告知开头已写好，仍可能换个说法再讲一遍（“你屏住呼吸，从桌上拿起钥匙”）：restates() 供叙述者略过这样的句子。
       REQUEST_ITEM 的成功先声只写请求，失败先声解释原因，均不把物品写成已收到。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Sequence

from tianlong.core import Kind, Modality, Op, Outcome, PerceivedEvent, Percept, derive_seed
from tianlong.language.templates import REASONS, SOCIAL_VERBS, Names, consequences

MAX_SENTENCES = 3        # 一回合至多三步计划：每步一句

# 成功的说法（{t} 对象、{o} 工具/物件）：同一局面按种子轮换，免得回回一个腔调
_DONE: dict[Op, tuple[str, ...]] = {
    Op.MOVE: ("你来到{t}。", "你一路走到{t}。", "你到了{t}。"),
    Op.TAKE: ("你伸手拿起{t}。", "你把{t}拿在手里。", "你取过{t}。"),
    Op.PUT: ("你把{o}放在{t}。",),
    Op.GIVE: ("你把{o}递给{t}。", "你将{o}交到{t}手里。"),
    Op.UNLOCK: ("你用{o}打开了{t}的锁。",),
    Op.LOCK: ("你用{o}锁上了{t}。",),
    Op.ATTACK: ("你猛地向{t}出手。", "你一掌向{t}拍去。"),
    Op.STUDY: ("你埋头研读{t}。", "你捧着{t}细细参详。"),
}
_LOOK = {Kind.PLACE: ("你四下细细打量。", "你把{t}里里外外看了一遍。"),
         Kind.SURFACE: ("你凑近{t}细看。", "你仔细查看{t}。"),
         Kind.PERSON: ("你上前搜查{t}。",)}
_STUDY_NOTE = {"progress": "若有所悟，却还未能融会贯通。", "mastered": "豁然贯通！"}
# 失败的说法：先说想做什么，再说为什么不成
_TRY: dict[Op, str] = {
    Op.MOVE: "你想去{t}", Op.TAKE: "你伸手去拿{t}", Op.PUT: "你想把{o}放在{t}", Op.GIVE: "你想把{o}交给{t}",
    Op.UNLOCK: "你拿{o}去开{t}的锁", Op.LOCK: "你想用{o}锁上{t}", Op.INSPECT: "你想查看{t}", Op.ATTACK: "你向{t}出手",
    Op.STUDY: "你翻看{t}良久", Op.USE: "你把{o}用在{t}身上", Op.TELL: "你想对{t}说话", Op.ASK: "你想问{t}",
    Op.WAIT: "你想做点什么", Op.REQUEST_ITEM: "你想请求{t}把{o}交给自己",
}


def _name(names: Names, eid: str | None, viewer: str) -> str:
    if eid is None:
        return "那里"
    if eid == viewer:
        return "自己"
    sk = names.get(eid)
    return sk.name if sk else "那里"


def _pick(options: tuple[str, ...], ev: PerceivedEvent, tick: int) -> str:
    return options[derive_seed("lead", tick, ev.kind, ev.target, ev.obj) % len(options)]


def _say(ev: PerceivedEvent, t: str) -> str | None:
    if ev.utterance:
        return f"你对{t}道：“{ev.utterance}”" if ev.kind == Op.TELL.value else f"你问{t}：“{ev.utterance}”"
    if ev.social is not None:
        return "你" + SOCIAL_VERBS[ev.social].format(t=t) + "。"
    return None


def _sentence(p: Percept, names: Names, viewer: str) -> str | None:
    ev = p.event
    if ev is None or ev.kind == "noise":
        return None
    op = Op(ev.kind)
    t, o = _name(names, ev.target, viewer), _name(names, ev.obj, viewer)
    if ev.outcome == Outcome.REJECTED:
        return _TRY[op].format(t=t, o=o) + "，可这行不通。"
    if ev.outcome == Outcome.FAILURE:
        why = REASONS.get(ev.reason or "", "")
        if not why:
            return _TRY[op].format(t=t, o=o) + "，却没能成。"
        return _TRY[op].format(t=t, o=o) + ("，却" if why.startswith("被") else "，可") + why + "。"
    # ---- 成功 ----
    if op == Op.REQUEST_ITEM:
        if ev.utterance:
            return f"你对{t}道：“{ev.utterance}”"
        beneficiary = _name(names, ev.beneficiary, viewer)
        return f"你请求{t}把{o}交给自己，以便帮助{beneficiary}。"
    if op in (Op.TELL, Op.ASK):
        return _say(ev, t)
    if op == Op.WAIT:
        return f"你{ev.utterance}。" if ev.utterance else None          # 姿态照写；干等不抢先
    if op == Op.INSPECT:
        sk = names.get(ev.target) if ev.target else None
        text = _pick(_LOOK.get(sk.kind if sk else Kind.PLACE, _LOOK[Kind.PLACE]), ev, p.tick).format(t=t)
    elif op == Op.USE:
        text = f"你服下{o}。" if ev.target in (None, viewer) else f"你把{o}用在{t}身上。"
    else:
        text = _pick(_DONE[op], ev, p.tick).format(t=t, o=o)
    if op == Op.STUDY and ev.reason in _STUDY_NOTE:
        text = text[:-1] + "，" + _STUDY_NOTE[ev.reason]
    after = consequences(p, names, viewer, "你")
    if after:
        text = text[:-1] + "——" + "，".join(after) + "。"
    return text


def lead_line(percepts: Sequence[Percept], names: Names, viewer: str) -> str:
    """玩家本回合自己的行动（SELF 感知，按发生先后）写成的句子，至多 MAX_SENTENCES 句；没有可说的返回空串。"""
    out: list[str] = []
    for p in percepts:
        if p.modality != Modality.SELF or p.event is None or p.event.actor != viewer:
            continue
        s = _sentence(p, names, viewer)
        if s and s not in out:
            out.append(s)
        if len(out) >= MAX_SENTENCES:
            break
    return "".join(out)


# ============================================================
#  复述：模型换个说法把玩家这一步再讲一遍——同一动作的动词 + 同一对象，或原话/姿态原样再现
# ============================================================

_VERBS: dict[Op, tuple[str, ...]] = {
    Op.MOVE: ("来到", "走到", "到了", "走进", "进了", "抵达", "踏进", "步入", "回到", "进入"),
    Op.TAKE: ("拿起", "取过", "取下", "拿在", "抓起", "捡起", "拿过", "拾起", "握住", "接过", "拿到", "取了"),
    Op.PUT: ("放在", "放下", "搁在", "放回"),
    Op.GIVE: ("递给", "交给", "交到", "塞给", "递过"),
    Op.UNLOCK: ("打开", "开了", "开锁", "锁开"),
    Op.LOCK: ("锁上", "锁好"),
    Op.ATTACK: ("出手", "拍去", "打去", "击向", "刺向", "攻向", "一掌", "一拳", "挥掌", "出掌", "扑向"),
    Op.STUDY: ("研读", "翻看", "参详", "翻阅", "细读", "读了"),
    Op.INSPECT: ("打量", "查看", "细看", "环顾", "看了一遍", "端详", "搜查", "搜了"),
    Op.USE: ("服下", "用在", "敷在", "喂给"),
}


def restates(sentence: str, percepts: Sequence[Percept], names: Names, viewer: str) -> bool:
    """这一句是不是在复述玩家自己本回合的行动：原话或姿态原样再现，或者同一动作的动词与同一对象（名字）同句出现。
    查看当前地点而没点地名的（“你环顾四周，只见……”）不算——那一句多半在写所见，丢了可惜。"""
    for p in percepts:
        ev = p.event
        if p.modality != Modality.SELF or ev is None or ev.actor != viewer or ev.kind == "noise":
            continue
        if ev.utterance and len(ev.utterance) >= 2 and ev.utterance in sentence:
            return True
        op = Op(ev.kind)
        things = [n for n in (_name(names, ev.target, viewer), _name(names, ev.obj, viewer)) if n not in ("那里", "自己")]
        if any(v in sentence for v in _VERBS.get(op, ())) and any(n in sentence for n in things):
            return True
    return False
