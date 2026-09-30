"""
[INPUT]: 依赖 core 的 WorldState / Kind / Rel / Proposition / ATTRIBUTES / Access / AttrType / applies / true_value / reason_key，
         cognition/beliefs 的 BeliefStore
[OUTPUT]: 对外提供 ViewNode / ViewEdge / GraphView、world_view()（环境输入：真实机制变量）与 belief_view()（角色输入：只有已知值）、
          VIEW_RELS / EVENT_RELS / EVENT_KIND / ONEWAY_TO
[POS]: cognition 的统一图投影，也是 Schema v2 的“值层”：节点只携带**已知**的属性值（缺席 = 未知；不适用的属性从不出现），
       类型保留（数值不压成布尔、类别不压成有无）。两个入口面对同一张属性规格（core/attributes）：
       world_view 读 true_value——规则用到的就是模型看到的（内力、锋利、淬毒、难度、进度、点穴余时）；
       belief_view 只按获知途径读认知——外观要亲眼见过，状态要见过或听说，手感要拿过，内在只有本人知道。
       事件节点带着内容（操作、渠道、结果、原因、言语命题的谓词与极性），命题的主语与宾语以 ABOUT/ABOUT_VALUE 边连回实体；
       容纳者节点带着“多久以前看清过/翻查过”（个人记录，探索的依据）。
       learning 只接受 GraphView——角色入口在结构上拿不到 WorldState，隔离由类型边界保证
       物品请求以 FOR 边连接受益人；持久请求义务和回应状态在短期经历滚掉后仍进入个人视图。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import (
    ATTR_PREFIX,
    ATTRIBUTES,
    Access,
    AttrType,
    Kind,
    Proposition,
    Rel,
    Scalar,
    WorldState,
    applies,
    reason_key,
)
from tianlong.core.attributes import AttrSpec, true_value

# ============================================================
#  视图词汇
#  ONEWAY_TO：单向通道 → 它唯一通往的一端（世界里存为属性值，视图里是一条边，才能参与消息传递）
#  事件关系：OCCURRED_AT 地点 / BY 行动者 / ON 目标 / WITH 对象或路线 / ABOUT 命题主语 / ABOUT_VALUE 命题宾语
# ============================================================

ONEWAY_TO = "ONEWAY_TO"
VIEW_RELS = (*(r.value for r in Rel), ONEWAY_TO)
EVENT_RELS = ("OCCURRED_AT", "BY", "ON", "WITH", "ABOUT", "ABOUT_VALUE", "FOR")
EVENT_KIND = "event"


@dataclass(frozen=True, slots=True)
class ViewNode:
    id: str
    kind: str                                   # Kind 值，或 "event"
    attrs: tuple[tuple[str, Scalar], ...] = ()  # 已知属性值（bool / float / 类别字符串）；缺席 = 未知
    is_self: bool = False
    event: tuple[tuple[str, Scalar], ...] = ()  # 事件节点的内容：op / modality / outcome / reason / topic_pred / topic_holds / topic_query
    surveyed_age: int | None = None             # 角色视角：多久以前看清过它的直接内容（None = 从没看清过/全知视角）
    searched_age: int | None = None             # 角色视角：多久以前亲手仔细翻查过它

    def value(self, key: str) -> Scalar:
        return dict(self.attrs).get(key)

    def knows(self, key: str) -> bool:
        return any(k == key for k, _ in self.attrs)


@dataclass(frozen=True, slots=True)
class ViewEdge:
    src: str
    rel: str
    dst: str
    holds: bool = True
    confidence: float = 1.0
    age: int = 0
    hearsay: bool = False


@dataclass(frozen=True, slots=True)
class GraphView:
    owner: str | None   # None = 全知视角（仅供环境动态学习与评价）
    now: int
    nodes: tuple[ViewNode, ...]
    edges: tuple[ViewEdge, ...]

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)


# ============================================================
#  环境入口：实际世界 → 视图（环境动态模型的输入）
#  规则用到的每个机制变量都在这里：同一输入不再对应“必败”与“必胜”两种结局
# ============================================================


def world_view(s: WorldState, self_id: str | None = None) -> GraphView:
    nodes = tuple(
        ViewNode(
            e.id,
            e.kind.value,
            tuple((a.key, true_value(e, a.key, s.clock)) for a in ATTRIBUTES if applies(e.kind, a.key)),
            e.id == self_id,
        )
        for e in sorted(s.entities.values(), key=lambda e: e.id)
    )
    edges = [ViewEdge(r.src, r.type.value, r.dst) for r in s.sorted_relations()]
    edges += [ViewEdge(d.id, ONEWAY_TO, str(d.get("oneway")))
              for d in sorted(s.of_kind(Kind.DOOR), key=lambda d: d.id) if d.get("oneway") is not None]
    return GraphView(None, s.clock, nodes, tuple(edges))


# ============================================================
#  角色入口：个人认知 → 视图（角色决策与角色视角预测的输入）
#  - 节点只有“认识的实体” + 近期经历
#  - 属性按获知途径取值：不知道就缺席，绝不拿真相或 False 填空
#  - 边带着可信度、时效与极性：过时、传闻、否定都保留给模型
# ============================================================


_MISSING = object()


def _known_value(store: BeliefStore, eid: str, spec: AttrSpec, seen: bool, sketch_attrs: dict) -> object:
    if spec.access == Access.APPEARANCE:
        if not seen:
            return _MISSING                     # 只闻其名：外观未知，而不是“没有”
        raw = sketch_attrs.get(spec.key)
        if spec.type == AttrType.BOOL:
            return bool(raw)
        if spec.type == AttrType.CAT:
            return raw if raw in spec.categories else "none"
        return float(raw or 0.0)
    if spec.key == "oneway":
        return True if store.positives(eid, ATTR_PREFIX + "oneway") else _MISSING
    if spec.type == AttrType.BOOL:
        b = store.believed(Proposition.attr(eid, spec.key, True))
        return _MISSING if b is None else b.holds
    best = store.best(eid, ATTR_PREFIX + spec.key)
    if best is None:
        return _MISSING
    if spec.type == AttrType.NUM:
        return float(best.prop.value or 0.0)     # type: ignore[arg-type]
    return best.prop.value if best.prop.value in spec.categories else "none"


def belief_view(store: BeliefStore, now: int) -> GraphView:
    nodes: list[ViewNode] = []
    for eid in sorted(store.entities):
        sk = store.entities[eid]
        shown = dict(sk.attrs)
        attrs = []
        for spec in ATTRIBUTES:
            if not applies(sk.kind, spec.key):
                continue
            v = _known_value(store, eid, spec, sk.seen, shown)
            if v is not _MISSING:
                attrs.append((spec.key, v))
        sv, sr = store.surveyed.get(eid), store.searched.get(eid)
        nodes.append(ViewNode(eid, sk.kind.value, tuple(attrs), eid == store.owner,  # type: ignore[arg-type]
                              surveyed_age=None if sv is None else max(0, now - sv),
                              searched_age=None if sr is None else max(0, now - sr)))

    edges: list[ViewEdge] = []
    for b in store.sorted_beliefs():
        p = b.prop
        if p.is_attr:
            if p.attr_key == "oneway" and b.holds and isinstance(p.value, str) and store.knows(p.value) \
                    and store.knows(p.subject):
                edges.append(ViewEdge(p.subject, ONEWAY_TO, p.value, True, b.confidence, now - b.learned_at, b.hearsay))
            continue
        if not (store.knows(p.subject) and isinstance(p.value, str) and store.knows(p.value)):
            continue
        edges.append(ViewEdge(p.subject, p.predicate, p.value, b.holds, b.confidence, now - b.learned_at, b.hearsay))

    for i, ep in enumerate(store.episodes):
        eid = f"episode:{i}"
        ev = ep.event
        content: list[tuple[str, Scalar]] = [("op", ev.kind), ("modality", ep.modality.value)]
        if ev.outcome is not None:
            content.append(("outcome", ev.outcome.value))
        if ev.reason is not None:
            content.append(("reason", reason_key(ev.reason)))
        topic_ids: tuple[str | None, str | None] = (None, None)
        if ev.topic is not None:
            tp = ev.topic.prop
            content += [("topic_pred", tp.predicate), ("topic_holds", ev.topic.holds),
                        ("topic_query", tp.value is None)]
            topic_ids = (tp.subject, tp.value if isinstance(tp.value, str) else None)
        nodes.append(ViewNode(eid, EVENT_KIND, (), False, tuple(content)))
        age = now - ep.tick
        ends = (ev.place, ev.actor, ev.target, ev.obj, *topic_ids, ev.beneficiary)
        for rel, other in zip(EVENT_RELS, ends, strict=True):
            if other and store.knows(other):
                edges.append(ViewEdge(eid, rel, other, True, 1.0, age, ep.informant is not None))

    # 请求义务在经历滚掉后仍进入策略观测。它只含这个角色听见/说出的请求与回应，不读任何库存真相。
    for i, ob in enumerate(store.obligations):
        if ob.kind not in ("request_item", "requested_item"):
            continue
        eid = f"request:{i}"
        nodes.append(ViewNode(eid, EVENT_KIND, event=(("op", "request_item"), ("request_state", ob.state))))
        actor, listener = (ob.counterpart, store.owner) if ob.kind == "request_item" else (store.owner, ob.counterpart)
        for rel, other in (("BY", actor), ("ON", listener), ("WITH", ob.item), ("FOR", ob.beneficiary)):
            if other and store.knows(other):
                edges.append(ViewEdge(eid, rel, other, age=max(0, now-ob.since)))

    return GraphView(store.owner, now, tuple(nodes), tuple(edges))
