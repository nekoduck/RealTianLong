"""
[INPUT]: 仅玩家 BeliefStore、玩家自己的目标、持久尝试记录、现有候选与语法契约；不读取 WorldState/NPC 私密目标
[OUTPUT]: build_choices()，以当前问题为先、枚举至多 24 个候选中的三项组合
[POS]: 玩家决策层。覆盖威胁、待回答、伤毒、进展和探索；保留路线与方式；多样性来自打算而非动作分类。
       同分按语义 ID 排序，不按展示文字排序；只描述尝试，不承诺内核尚未裁定的结果。
       已知对症物品在 NPC 身上才提出具体请求；答应后只提供等候实际交付，拿到后才推荐 USE。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from tianlong.cognition.beliefs import BeliefStore
from tianlong.cognition.candidates import Candidate, candidates
from tianlong.cognition.navigation import believed_place, route_to
from tianlong.core import FRIENDLY_SOCIAL, HOSTILE_SOCIAL, Fact, Kind, Manner, Op, Rel, Social
from tianlong.core.grammar import signature_error
from tianlong.core.profiles import Goal, GoalKind
from tianlong.language.parser import MoveKind, Parsed
from tianlong.runtime.choice_history import ChoiceHistory
from tianlong.runtime.choice_model import ChoiceSpec

log = logging.getLogger(__name__)
POOL_LIMIT = 24
RECENT = 5


@dataclass(frozen=True, slots=True)
class _Proposal:
    spec: ChoiceSpec
    relevance: int
    issue: str | None
    strategy: tuple[str, ...]


def known_attr(me: BeliefStore, eid: str | None, key: str):
    """未知保持未知，绝不向场景实体/世界查询。"""
    if eid is None:
        return None
    belief = me.best(eid, "attr." + key)
    if belief is not None:
        if belief.holds:
            return None if belief.prop.value == "none" else belief.prop.value
        if isinstance(belief.prop.value, bool):
            return not belief.prop.value
    negatives = [b for b in me.sorted_beliefs() if b.prop.subject == eid and b.prop.predicate == "attr." + key
                 and not b.holds and b.prop.value is True]
    if negatives:
        return False
    sketch = me.sketch(eid)
    return dict(sketch.attrs).get(key) if sketch else None


def _refs(c: Candidate) -> set[str]:
    refs = {x for x in (c.target, c.obj, c.beneficiary) if x}
    if c.topic:
        refs.add(c.topic.prop.subject)
        if c.topic.prop.predicate in {r.value for r in Rel} and isinstance(c.topic.prop.value, str):
            refs.add(c.topic.prop.value)
    return refs


def _threats(me: BeliefStore, people: set[str], friends: set[str]) -> set[str]:
    latest = {c.frm: c for c in me.cues if c.to in (me.owner, None, *friends)}
    hit = {ep.event.actor for ep in me.episodes if ep.event.kind == Op.ATTACK.value
           and ep.event.target in (me.owner, *friends) and ep.tick >= me.last_tick - RECENT}
    out = set()
    for who in people:
        cue = latest.get(who)
        if cue and cue.social in FRIENDLY_SOCIAL:
            continue
        if (who in hit or (cue and cue.social in HOSTILE_SOCIAL and cue.tick >= me.last_tick - RECENT)
                or me.attitudes.get(who, 0) < 0):
            out.add(who)
    return out


def build_choices(me: BeliefStore, limit: int = 3, *, goals: Sequence[Goal] = (),
                  history: ChoiceHistory | None = None) -> tuple[ChoiceSpec, ...]:
    if limit <= 0:
        return ()
    history = history or ChoiceHistory()
    player, here = me.owner, believed_place(me, me.owner)
    people = {eid for eid, sk in me.entities.items() if sk.kind == Kind.PERSON and eid != player
              and here is not None and believed_place(me, eid) == here}
    friends = set(me.allies) | {g.person for g in goals if g.kind == GoalKind.DEFEND and g.person}
    threats = _threats(me, people, friends)
    health = {who: {status for status in ("poisoned", "wounded") if known_attr(me, who, status) is True}
              for who in (player, *sorted(people))}
    health = {who: statuses for who, statuses in health.items() if statuses}
    urgent = {f"threat:{who}" for who in threats} | {f"care:{who}" for who in health}
    issues = urgent | {f"answer:{o.counterpart}" for o in me.obligations if o.counterpart in people}
    relevant = {ref for g in goals if g.active(me.last_tick) for ref in (g.item, g.person, g.recipient) if me.knows(ref)}
    goal_routes = {hop for g in goals if g.active(me.last_tick) and g.home and me.knows(g.home)
                   for hop in [route_to(me, g.home)] if hop}
    blocked_routes = {a.candidate.obj for a in history.attempts.values()
                      if a.candidate.op == Op.MOVE and a.reason == "door_locked"
                      and known_attr(me, a.candidate.obj, "locked") is True}
    needed_items = {a.candidate.target if a.candidate.op == Op.STUDY else a.candidate.obj
                    for a in history.attempts.values() if a.reason == "not_holding"}
    pool: dict[str, _Proposal] = {}

    def name(eid: str | None) -> str:
        sketch = me.sketch(eid) if eid else None
        return sketch.name if sketch else "自己"

    def add(c: Candidate, label: str, score: int, strategy: tuple[str, ...], issue: str | None = None,
            utterance: str | None = None, repeat: int = 1, evidence: tuple[str, ...] = ()) -> None:
        if any(not me.knows(ref) for ref in _refs(c)):
            return
        if signature_error(c.op, lambda eid: me.sketch(eid).kind if me.sketch(eid) else None,
                           c.target, c.obj, c.topic, c.beneficiary, c.request_ref) is not None:
            return
        if known_attr(me, player, "subdued") is True and c.op not in (Op.WAIT, Op.TELL, Op.ASK, Op.REQUEST_ITEM):
            return
        if history.blocked(me, c):
            return
        kind = MoveKind.SAY if c.op in (Op.TELL, Op.ASK, Op.REQUEST_ITEM) else MoveKind.ACT
        spec = ChoiceSpec.of(label, Parsed(c, utterance, kind=kind, repeat=repeat), c.target, evidence)
        proposed = _Proposal(spec, score, issue, strategy)
        prior = pool.get(spec.semantic_key)
        if prior is None or proposed.relevance > prior.relevance:
            pool[spec.semantic_key] = proposed

    # 待回答的话题来自持久义务，能说话的人来自玩家认知，不从 ATTACK 候选推断。
    for obligation in me.obligations:
        who = obligation.counterpart
        if who not in people:
            continue
        if obligation.kind in ("request_item", "requested_item"):
            if obligation.kind == "requested_item" and obligation.state in ("pending", "accepted"):
                add(Candidate(Op.WAIT), f"留在原处，等候{name(who)}实际递交{name(obligation.item)}", 96,
                    ("await-transfer", who), f"care:{obligation.beneficiary}", repeat=1,
                    evidence=(f"request:{obligation.request_ref}:{obligation.state}",))
            continue
        issue = f"threat:{who}" if who in threats else f"answer:{who}"
        if obligation.topic is not None:
            topic = obligation.topic.prop
            known = me.best(topic.subject, topic.predicate)
            if known and known.holds and me.knows(topic.subject) and me.knows(known.prop.value):
                words = f"{name(topic.subject)}在{name(known.prop.value)}。"
                add(Candidate(Op.TELL, who, topic=Fact(known.prop, True)),
                    f"把我所知的{name(topic.subject)}下落告诉{name(who)}", 82, ("answer", who), issue, words)
            elif me.knows(topic.subject):
                add(Candidate(Op.TELL, who, social=Social.EXPLAIN), f"说明我不知道{name(topic.subject)}的下落",
                    78, ("answer", who), issue, "这件事的下落，我并不知道。")
        if obligation.kind == "answer":
            add(Candidate(Op.TELL, who, social=Social.REFUSE), f"向{name(who)}明确表明不愿回应", 72,
                ("boundary", who), issue, "此事我不愿答复。")
        elif who not in threats and obligation.social in FRIENDLY_SOCIAL:
            add(Candidate(Op.TELL, who, social=Social.THANK), f"向{name(who)}致谢，回应善意", 64,
                ("acknowledge", who), issue, "多谢你的好意。")

    for who in sorted(people):
        if who in threats:
            issue, refs = f"threat:{who}", (f"known-conflict:{who}",)
            add(Candidate(Op.TELL, who, social=Social.EXPLAIN), f"向{name(who)}解释自己并无冒犯之意", 94,
                ("deescalate", who), issue, "在下并无冒犯之意，还请听我解释。", evidence=refs)
            add(Candidate(Op.TELL, who, social=Social.REFUSE), f"明确拒绝与{name(who)}争斗", 92,
                ("boundary", who), issue, "在下不愿与人争斗。", evidence=refs)
            add(Candidate(Op.TELL, who, social=Social.APOLOGIZE), f"向{name(who)}赔礼，尝试缓和冲突", 88,
                ("deescalate", who), issue, "方才多有得罪，还请见谅。", evidence=refs)
        elif not any(s.listener == who for s in me.said):
            add(Candidate(Op.TELL, who, social=Social.GREET), f"向{name(who)}见礼", 28,
                ("greet", who), utterance="有礼了。")
        if who in health:
            add(Candidate(Op.TELL, who, social=Social.COMFORT), f"向{name(who)}询问伤势并出言安慰", 70,
                ("comfort", who), f"care:{who}", "伤势可还撑得住？且莫惊慌。")
        if player in health and who not in threats:
            add(Candidate(Op.TELL, who, social=Social.PLEAD), f"向{name(who)}说明伤势，请求帮助", 76,
                ("seek-help", who), f"care:{player}", "在下身上不适，还请相助。")

    # 请求中的物品、持有者与对症用途都必须是玩家已知；不从 NPC 私有库存拿出解药。
    for item, sketch in sorted(me.entities.items()):
        if sketch.kind != Kind.ITEM or known_attr(me, item, "cures") is None:
            continue
        holder = me.location_of(item)
        if holder not in people:
            continue
        for patient in sorted({player, *friends} & health.keys()):
            if known_attr(me, item, "cures") not in health[patient]:
                continue
            if any(o.kind == "requested_item" and o.item == item and o.counterpart == holder and
                   o.beneficiary == patient and o.state in ("pending", "accepted") for o in me.obligations):
                continue
            whom = "我" if patient == player else name(patient)
            c = Candidate(Op.REQUEST_ITEM, holder, item, social=Social.PLEAD, beneficiary=patient)
            add(c, f"请求{name(holder)}把{name(item)}交给我，以便救助{whom}", 104,
                ("request-resource", holder, patient), f"care:{patient}",
                f"请把{name(item)}交给我，我想用它救助{whom}。", evidence=(f"known-holder:{item}:{holder}",))

    for c in candidates(me):
        target, obj = c.target, c.obj
        bonus = 12 if _refs(c) & relevant else 0
        if c.op == Op.WAIT:
            add(c, "暂且等候，留意眼前的动静", 4, ("wait",), repeat=5)
        elif c.op == Op.MOVE:
            if known_attr(me, obj, "locked") is True:
                continue  # 当前可行的取钥匙、开锁与其他路线仍会从候选池进入。
            issue = f"threat:{sorted(threats)[0]}" if threats else None
            prefix = "悄悄" if c.manner == Manner.CAREFUL else ""
            verb = "退往" if threats else "走向"
            score = (93 if threats else 48) + (14 if (target, obj) in goal_routes else 0)
            add(c, f"{prefix}经{name(obj)}{verb}{name(target)}", score - (2 if prefix else 0),
                ("leave",) if threats else ("route", target, obj), issue, evidence=(f"known-route:{obj}",))
        elif c.op == Op.ATTACK and target in threats:
            add(c, f"向{name(target)}动手，尝试反击", 81, ("retaliate", target), f"threat:{target}")
        elif c.op == Op.USE:
            cure = known_attr(me, obj, "cures")
            if cure and known_attr(me, target, cure) is True:
                whom = "自己" if target == player else name(target)
                label = (f"把{name(obj)}用在{whom}身上，尝试疗伤" if cure == "wounded"
                         else f"给{whom}使用{name(obj)}，尝试解毒")
                add(c, label, 110, ("treat", target, obj), f"care:{target}")
        elif c.op == Op.STUDY:
            taught = me.best(target, "attr.teaches")
            if taught is not None and taught.prop.value == "none":
                continue
            teaches = known_attr(me, target, "teaches")
            if not teaches and not any(w in name(target) for w in ("经", "秘籍", "书", "卷", "谱", "诀", "图")):
                continue
            if teaches and known_attr(me, player, teaches) is True:
                continue
            previous = history.latest(c) or next((ep.event for ep in reversed(me.episodes)
                             if ep.event.actor == player and ep.event.kind == Op.STUDY.value and ep.event.target == target), None)
            if previous and previous.reason in ("nothing_to_learn", "already_learned", "mastered") \
                    and (history.latest(c) is None or history.unchanged(me, c)):
                continue
            progress = previous is not None and previous.reason == "progress"
            score = 65 if progress else (42 if teaches else 20)
            add(c, f"{'继续' if progress else ''}研读{name(target)}，体会其中法门", score,
                ("study", target), f"study:{target}", evidence=(f"held:{target}",))
        elif c.op == Op.INSPECT:
            sk = me.sketch(target)
            if sk.kind == Kind.PERSON:
                if known_attr(me, target, "subdued") is True:
                    add(c, f"搜{name(target)}的身，查找随身之物", 12 + bonus, ("search-person", target))
            elif (target not in me.searched or (history.latest(c) and not history.unchanged(me, c))
                  or history.surface_changes.get(target, -1) > me.searched.get(target, me.last_tick)):
                renewed = target in me.searched
                add(c, "仔细查看周围，寻找新的线索" if target == here else f"仔细查探{name(target)}",
                    (74 if renewed else (52 if target == here and here not in me.surveyed else 38)) + bonus,
                    ("inspect", target), f"new-clue:{target}" if renewed else None)
        elif c.op == Op.TAKE:
            score = 40 + bonus + (15 if any(known_attr(me, target, "cures") in statuses for statuses in health.values()) else 0)
            prep = target in needed_items or any(b.prop.value in blocked_routes
                                                for b in me.positives(target, Rel.MATCHES.value))
            if prep:
                score += 40
            verb = "悄悄取走" if c.manner == Manner.CAREFUL else "拿起"
            add(c, f"{verb}{name(target)}", score - (1 if c.manner == Manner.CAREFUL else 0), ("acquire", target),
                f"prepare:{target}" if prep else None)
        elif c.op == Op.ASK and c.topic:
            subject = c.topic.prop.subject
            if believed_place(me, subject) is None:
                add(c, f"向{name(target)}询问{name(subject)}的下落", 44 + bonus,
                    ("information", subject), f"information:{subject}")
        elif c.op == Op.TELL and c.topic:
            subject = c.topic.prop.subject
            if subject in relevant and not any(s.listener == target and s.fact == c.topic for s in me.said):
                add(c, f"把我所知的{name(subject)}下落告诉{name(target)}", 32 + bonus,
                    ("inform", target, subject), utterance=f"{name(subject)}在{name(c.topic.prop.value)}。")
        elif c.op == Op.GIVE and any(g.kind == GoalKind.DELIVER and g.item == obj and g.recipient == target for g in goals):
            add(c, f"把{name(obj)}交给{name(target)}", 68, ("transfer", target, obj), f"deliver:{obj}")
        elif c.op == Op.UNLOCK and known_attr(me, target, "locked") is True:
            matched = any(b.prop.value == target for b in me.positives(obj, Rel.MATCHES.value))
            if matched or "钥匙" in name(obj):
                add(c, f"尝试用{name(obj)}打开{name(target)}的锁", 90 if target in blocked_routes else 60,
                    ("unlock", target, obj), f"route:{target}")
        elif c.op == Op.LOCK and threats and known_attr(me, target, "locked") is False:
            add(c, f"尝试用{name(obj)}锁上{name(target)}", 84, ("secure-route", target), f"threat:{sorted(threats)[0]}")
        elif c.op == Op.PUT and c.manner == Manner.CAREFUL and obj in relevant:
            add(c, f"把{name(obj)}藏到{name(target)}", 30, ("hide", target, obj))

    # 预算按具体打算留候选，不限制最终同一 Op 出现几次。
    ordered = sorted(pool.values(), key=lambda p: (-p.relevance, p.spec.semantic_key))
    bounded, axes = [], set()
    for p in ordered:
        if p.strategy not in axes:
            bounded.append(p)
            axes.add(p.strategy)
        if len(bounded) == POOL_LIMIT:
            break
    chosen_keys = {p.spec.semantic_key for p in bounded}
    bounded.extend(p for p in ordered if p.spec.semantic_key not in chosen_keys)
    bounded = bounded[:POOL_LIMIT]
    count = min(limit, 3, len(bounded))
    if count < min(limit, 3):
        log.warning("可执行选项不足：%s 项；保留真实选择，不用近义文案补位", count)
    if not count:
        return ()

    def value(combo):
        def cost(p):
            c = p.spec.parsed.candidate
            previous = history.latest(c)
            if p.issue in urgent or previous is None or previous.reason == "progress" or not history.unchanged(me, c):
                return 0
            return 35 if c.op == Op.ASK else 14
        return (sum(p.issue in urgent for p in combo), len({p.issue for p in combo if p.issue in issues}),
                len({p.strategy for p in combo}), sum(p.relevance - cost(p) for p in combo))

    combinations = itertools.combinations(sorted(bounded, key=lambda p: p.spec.semantic_key), count)
    picked = max(combinations, key=value)
    return tuple(p.spec for p in sorted(picked, key=lambda p: (-p.relevance, p.spec.semantic_key)))
