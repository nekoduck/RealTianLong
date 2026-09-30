"""
[INPUT]: 依赖 core 的 Op / Manner / Kind / Rel / Fact / Proposition / Intent / signature_error，cognition/beliefs 的 BeliefStore
[OUTPUT]: 对外提供 Candidate（结构化候选行动，可带言语行为 social——修辞，不进排序）、candidates()（从个人认知生成候选集）、budget()（行动族轮转配额截断）、FAMILIES、
          CANDIDATES_VERSION（候选规则语义版本，部署包据此拒绝规则已变的旧策略）
[POS]: cognition 的行动空间；候选对象只来自角色的认知图——按角色“以为”的世界剪枝是合理的，按真实世界剪枝则是泄密。
       策略（脚本/RL）与预测器都在这个候选集上工作。超出上限时按行动族轮转配额截断（等待、移动、言语、查看、物件、
       动手、施用、锁、研读各轮流取一个，族内与目标相关的在前）：物件组合再多也挤不掉交流、观察与等待
       v3 在原规范顺序末尾新增 REQUEST_ITEM；保留明确受益人，请求对象只从个人认知生成。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import Fact, Intent, Kind, Manner, Op, Proposition, Rel, Social, digest
from tianlong.core.grammar import signature_error

# 候选的规范顺序（下标即策略的动作编号）：WAIT 永居首位，其余按操作、再按对象
_PRIORITY = (Op.WAIT, Op.MOVE, Op.ATTACK, Op.USE, Op.TAKE, Op.PUT, Op.GIVE, Op.UNLOCK, Op.LOCK, Op.INSPECT, Op.STUDY,
             Op.TELL, Op.ASK, Op.REQUEST_ITEM)
_OP_ORDER = {op: i for i, op in enumerate(_PRIORITY)}
assert set(_OP_ORDER) == set(Op), "新增操作必须在规范顺序中登记"
# 截断配额：按行动族轮转取候选。族的顺序决定预算极紧时谁先入选——交流与观察排在组合爆炸的物件操作之前
FAMILIES: tuple[tuple[str, frozenset[Op]], ...] = (
    ("wait", frozenset({Op.WAIT})), ("move", frozenset({Op.MOVE})), ("speech", frozenset({Op.TELL, Op.ASK, Op.REQUEST_ITEM})),
    ("inspect", frozenset({Op.INSPECT})), ("handle", frozenset({Op.TAKE, Op.PUT, Op.GIVE})),
    ("combat", frozenset({Op.ATTACK})), ("care", frozenset({Op.USE})), ("locks", frozenset({Op.UNLOCK, Op.LOCK})),
    ("study", frozenset({Op.STUDY})),
)
assert set().union(*(ops for _, ops in FAMILIES)) == set(Op), "新增操作必须归入某个行动族"
# 候选规则的语义版本：生成规则（剪枝、话题、配额、顺序）一变就手动递增——候选下标是策略的动作编号，
# 规则变了，旧策略的输出就指向了别的行动；部署包据此拒绝
CANDIDATES_VERSION = "candidates-v3:" + digest(_PRIORITY, tuple((n, sorted(o.value for o in ops)) for n, ops in FAMILIES))
_WHILE_SUBDUED = frozenset({Op.WAIT, Op.TELL, Op.ASK, Op.REQUEST_ITEM})


@dataclass(frozen=True, slots=True)
class Candidate:
    op: Op
    target: str | None = None
    obj: str | None = None
    manner: Manner = Manner.NORMAL
    topic: Fact | None = None
    social: Social | None = None       # 言语/姿态的社交含义：修辞，不进候选排序（策略编号不变）
    beneficiary: str | None = None
    request_ref: str | None = None

    def to_intent(self, intent_id: str, actor: str, based_on: int, utterance: str | None = None) -> Intent:
        return Intent(intent_id, actor, self.op, self.target, self.obj, self.manner, self.topic, based_on, utterance,
                      self.social, self.beneficiary, self.request_ref)

    def sort_key(self) -> tuple:
        topic = self.topic.sort_key() if self.topic else ()
        return (_OP_ORDER[self.op], self.target or "", self.obj or "", self.manner.value, topic,
                self.beneficiary or "", self.request_ref or "")

    @staticmethod
    def of(it: Intent) -> Candidate:
        return Candidate(it.op, it.target, it.obj, it.manner, it.topic, it.social, it.beneficiary, it.request_ref)


def candidates(
    store: BeliefStore, interests: Iterable[str] | None = None, max_count: int | None = None
) -> tuple[Candidate, ...]:
    """角色此刻“想得到”的全部行动。interests 限定言语话题涉及的物品（默认：所有认识的物品）。"""
    interests = list(interests) if interests is not None else None
    me = store.owner
    here = store.location_of(me)

    def kind(eid: str) -> Kind | None:
        sk = store.sketch(eid)
        return sk.kind if sk else None

    def of_kind(k: Kind) -> list[str]:
        return sorted(e for e, sk in store.entities.items() if sk.kind == k and e != me)

    def is_here(eid: str) -> bool:
        loc = store.location_of(eid)
        if loc is None or here is None:
            return False
        return loc == here or (kind(loc) == Kind.SURFACE and store.location_of(loc) == here)

    places_here = [here] if here else []
    surfaces = [s for s in of_kind(Kind.SURFACE) if store.location_of(s) == here]
    persons = [p for p in of_kind(Kind.PERSON) if store.location_of(p) == here]
    held = [i for i in of_kind(Kind.ITEM) if store.location_of(i) == me]
    doors = [d for d in of_kind(Kind.DOOR) if here and store.holds(Proposition.rel(d, Rel.CONNECTS, here))]
    # 话题 = 关心的物品 + 认识的人（“钥匙在哪”“玩家在哪”都是值得说/问的）；连名字都不知道的无从谈起
    topics = sorted(t for t in set(interests if interests is not None else of_kind(Kind.ITEM)) | set(of_kind(Kind.PERSON))
                    if store.knows(t))

    out: list[Candidate] = [Candidate(Op.WAIT)]

    # ---- 移动：目的地 + 路线。经由认为存在的门（含自己发现的暗门），去认为相邻的地点；
    #      认为锁着也照样可以去推（记忆可能过时）；确知是单向且方向不对的才剪掉 ----
    for d in doors:
        oneway = next((b.prop.value for b in store.positives(d, "attr.oneway")), None)
        for b in store.positives(d, Rel.CONNECTS.value):
            if b.prop.value != here and (oneway is None or oneway == b.prop.value):
                out.append(Candidate(Op.MOVE, target=b.prop.value, obj=d))  # type: ignore[arg-type]
                out.append(Candidate(Op.MOVE, target=b.prop.value, obj=d, manner=Manner.CAREFUL))  # type: ignore[arg-type]

    # ---- 物件：拿认为在身边的（含认为已被制住者身上的），放/给/开锁用手里的 ----
    helpless = {p for p in persons if store.holds(Proposition.attr(p, "subdued", True))}
    for item in of_kind(Kind.ITEM):
        if item not in held and (is_here(item) or store.location_of(item) in helpless):
            out += [Candidate(Op.TAKE, item), Candidate(Op.TAKE, item, manner=Manner.CAREFUL)]
    for item in held:
        for dest in (*places_here, *surfaces):
            out += [Candidate(Op.PUT, dest, item), Candidate(Op.PUT, dest, item, Manner.CAREFUL)]
        out += [Candidate(Op.GIVE, p, item) for p in persons]
        for d in doors:
            if store.holds(Proposition.rel(item, Rel.MATCHES, d)) or not store.believed(
                Proposition.rel(item, Rel.MATCHES, d)
            ):
                # 认为配、或不知道配不配：都值得一试；确知不配才剪掉
                out += [Candidate(Op.UNLOCK, d, item), Candidate(Op.LOCK, d, item)]

    # ---- 查看 ----
    out += [Candidate(Op.INSPECT, t) for t in (*places_here, *surfaces, *persons)]

    # ---- 武斗、修习、施用：对在场者动手；研读手中之物；把手中之物用在在场者或自己身上 ----
    out += [Candidate(Op.ATTACK, p) for p in persons]
    out += [Candidate(Op.STUDY, i) for i in held]
    out += [Candidate(Op.USE, p, i) for i in held for p in (*persons, me)]

    # ---- 言语：说出自己相信的下落；询问不知下落的东西或人 ----
    for p in persons:
        for subject in topics:
            if subject == p:
                continue
            best = store.best(subject, Rel.AT.value)
            if best is not None:
                out.append(Candidate(Op.TELL, p, topic=Fact(best.prop, True)))
            # 以为知道也可以问：当面质问、求证，都是合理的言语行动
            out.append(Candidate(Op.ASK, p, topic=Fact(Proposition.rel(subject, Rel.AT, None), True)))
        # 只请求自己认识、认为在听者身上的物品；受益人限自己与已知自己人。
        beneficiaries = sorted({me, *(a for a in store.allies if a in persons)})
        for item in of_kind(Kind.ITEM):
            if store.location_of(item) == p:
                out.extend(Candidate(Op.REQUEST_ITEM, p, item, beneficiary=b) for b in beneficiaries)

    if store.holds(Proposition.attr(me, "subdued", True)):
        out = [c for c in out if c.op in _WHILE_SUBDUED]   # 自知穴道被制：只剩开口与等待
    valid = [c for c in out if signature_error(c.op, kind, c.target, c.obj, c.topic, c.beneficiary, c.request_ref) is None]
    ordered = [valid[0], *sorted(set(valid[1:]), key=Candidate.sort_key)]  # WAIT 永远在首位，截断时不丢
    return budget(ordered, max_count, interests)


def budget(cands, max_count: int | None, interests: Iterable[str] | None = None) -> tuple[Candidate, ...]:
    """按行动族轮转配额截到 max_count（None = 不截）；调用方可先拿全集、再截，以便报告截掉了多少。"""
    cands = list(cands)
    if max_count and len(cands) > max_count:
        cands = _budgeted(cands, max_count, set(interests or ()))
    return tuple(cands)


def _budgeted(ordered: list[Candidate], budget: int, focus: set[str]) -> list[Candidate]:
    """行动族轮转配额：每轮每族取一个（族内与目标相关者优先），直到用完预算；入选者保持规范顺序。"""
    def relevant(c: Candidate) -> bool:
        refs = {c.target, c.obj}
        if c.topic is not None:
            refs |= {c.topic.prop.subject, c.topic.prop.value if isinstance(c.topic.prop.value, str) else None}
        return bool(refs & focus)

    queues = [sorted((c for c in ordered if c.op in ops), key=lambda c: (not relevant(c), c.sort_key()))
              for _, ops in FAMILIES]
    chosen: set[Candidate] = set()
    while len(chosen) < budget and any(queues):
        for q in queues:
            if q and len(chosen) < budget:
                chosen.add(q.pop(0))
    return [c for c in ordered if c in chosen]
