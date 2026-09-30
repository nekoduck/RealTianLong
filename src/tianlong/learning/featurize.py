"""
[INPUT]: 依赖 numpy，learning/schema 的列布局与词表，cognition 的 GraphView / Candidate，core 的 Scalar / reason_key
[OUTPUT]: 对外提供 GraphTensors、featurize()、ActionCode、encode_action()、bool_tri()（布尔属性三态读回）、num_known()（数值属性读回），
          并再导出 schema 的维度常量（F_NODE / F_EDGE / REL_VOCAB / N_OPS / N_MANNERS / OP_INDEX / MANNER_INDEX / HOLDER_KINDS / LOCATED_KINDS）
[POS]: learning 的输入编码器：只接受 GraphView——角色入口的张量在构造上就拿不到世界真相。
       按 schema 的列布局把“已知值”写进节点列，未知留零且 known=0，不适用 known=-1；
       关系类型（含极性与方向）编码进边特征，与可信度/时效/传闻一起交给支持 edge_dim 的卷积；
       numpy 是中立格式：动态模型转成 PyG Data，RL 环境把它填充成定长观测
       编码请求受益人和义务状态，未知受益人指针为 -1。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tianlong.cognition import Candidate, GraphView
from tianlong.core import REASONS, Kind, applies
from tianlong.core.attributes import AttrType
from tianlong.learning.schema import (
    AGE_SCALE,
    ATTR_BLOCK,
    ATTR_BLOCKS,
    EV_MODALITY,
    EV_OP,
    EV_OUTCOME,
    EV_REASON,
    EV_REQUEST_STATE,
    EV_TOPIC,
    EV_TOPIC_HOLDS,
    EV_TOPIC_QUERY,
    EVENT_OPS,
    F_EDGE,
    F_NODE,
    MANNER_INDEX,
    MODALITIES,
    N_MANNERS,
    N_OPS,
    NODE_KINDS,
    OP_INDEX,
    OUTCOMES,
    REL_INDEX,
    REL_VOCAB,
    REQUEST_STATES,
    SELF_COL,
    SURVEY,
    TOPIC_INDEX,
    AttrBlock,
)

__all__ = [
    "F_EDGE", "F_NODE", "HOLDER_KINDS", "LOCATED_KINDS", "MANNER_INDEX", "N_MANNERS", "N_OPS", "OP_INDEX", "REL_VOCAB",
    "ActionCode", "GraphTensors", "bool_tri", "encode_action", "featurize", "num_known",
]

HOLDER_KINDS = (Kind.PLACE.value, Kind.SURFACE.value, Kind.PERSON.value)
LOCATED_KINDS = (Kind.PERSON.value, Kind.ITEM.value, Kind.SURFACE.value)
_REASON_INDEX = {r: i for i, r in enumerate(REASONS)}
_KINDS = {k.value: k for k in Kind}


@dataclass(frozen=True)
class GraphTensors:
    node_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    x: np.ndarray            # float32 [N, F_NODE]
    edge_index: np.ndarray   # int64   [2, E]
    edge_attr: np.ndarray    # float32 [E, F_EDGE]

    def index_of(self, eid: str | None) -> int:
        if eid is None:
            return -1
        try:
            return self.node_ids.index(eid)
        except ValueError:
            return -1

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)


# ============================================================
#  属性块：值 + known（1 已知 / 0 未知 / -1 不适用）
# ============================================================


def _write_attr(row: np.ndarray, b: AttrBlock, value: object) -> None:
    row[b.known] = 1.0
    if b.type == AttrType.BOOL:
        row[b.start] = 1.0 if value else -1.0
    elif b.type == AttrType.NUM:
        row[b.start] = float(np.clip(float(value) / b.scale, -1.0, 1.0))  # type: ignore[arg-type]
    elif value in b.categories:
        row[b.start + b.categories.index(value)] = 1.0          # type: ignore[arg-type]


def featurize(view: GraphView) -> GraphTensors:
    ids = view.node_ids()
    index = {eid: i for i, eid in enumerate(ids)}
    x = np.zeros((len(ids), F_NODE), dtype=np.float32)
    for i, n in enumerate(view.nodes):
        row = x[i]
        row[NODE_KINDS.index(n.kind)] = 1.0
        row[SELF_COL] = 1.0 if n.is_self else 0.0
        kind = _KINDS.get(n.kind)
        known = dict(n.attrs)
        for b in ATTR_BLOCKS:
            if kind is None or not applies(kind, b.key):
                row[b.known] = -1.0
            elif b.key in known:
                _write_attr(row, b, known[b.key])
        if n.event:
            _write_event(row, dict(n.event))
        for j, age in enumerate((n.surveyed_age, n.searched_age)):
            if age is not None:
                row[SURVEY.start + 2 * j] = 1.0
                row[SURVEY.start + 2 * j + 1] = max(0.05, 1.0 - min(age, AGE_SCALE) / AGE_SCALE)

    src, dst, attr = [], [], []
    for e in view.edges:
        if e.src not in index or e.dst not in index:
            continue
        for rev in (False, True):
            feat = np.zeros(F_EDGE, dtype=np.float32)
            feat[REL_INDEX[(e.rel, not e.holds, rev)]] = 1.0
            feat[-3] = e.confidence
            feat[-2] = min(e.age, AGE_SCALE) / AGE_SCALE
            feat[-1] = 1.0 if e.hearsay else 0.0
            a, b = (index[e.dst], index[e.src]) if rev else (index[e.src], index[e.dst])
            src.append(a)
            dst.append(b)
            attr.append(feat)
    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.stack(attr) if attr else np.zeros((0, F_EDGE), dtype=np.float32)
    return GraphTensors(ids, tuple(n.kind for n in view.nodes), x, edge_index, edge_attr)


def _one_hot(row: np.ndarray, span: slice, vocab: tuple[str, ...], value: object) -> None:
    if value in vocab:
        row[span.start + vocab.index(value)] = 1.0     # type: ignore[arg-type]


def _write_event(row: np.ndarray, ev: dict) -> None:
    _one_hot(row, EV_OP, EVENT_OPS, ev.get("op"))
    _one_hot(row, EV_MODALITY, MODALITIES, ev.get("modality"))
    _one_hot(row, EV_OUTCOME, OUTCOMES, ev.get("outcome"))
    _one_hot(row, EV_REQUEST_STATE, REQUEST_STATES, ev.get("request_state"))
    if ev.get("reason") in _REASON_INDEX:
        row[EV_REASON.start + _REASON_INDEX[ev["reason"]]] = 1.0
    if ev.get("topic_pred") in TOPIC_INDEX:
        row[EV_TOPIC.start + TOPIC_INDEX[ev["topic_pred"]]] = 1.0
        row[EV_TOPIC_HOLDS] = 1.0 if ev.get("topic_holds") else -1.0
        row[EV_TOPIC_QUERY] = 1.0 if ev.get("topic_query") else 0.0


# ============================================================
#  读回：动态属性的预测目标直接从节点列取，保证“输入怎么编码、标签就怎么定义”
# ============================================================


def bool_tri(x: np.ndarray, key: str) -> np.ndarray:
    """[N] 布尔属性三态：-1 否 / 0 未知或不适用 / 1 是。"""
    b = ATTR_BLOCK[key]
    return np.where(x[:, b.known] > 0.5, x[:, b.start], 0.0).astype(np.float32)


def num_known(x: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    """[N] 数值属性（已缩放）与是否已知。"""
    b = ATTR_BLOCK[key]
    known = x[:, b.known] > 0.5
    return np.where(known, x[:, b.start], 0.0).astype(np.float32), known


# ============================================================
#  行动编码：操作 + 方式 + 目标/对象（MOVE 为路线门）/行动者 + 言语命题（谓词、主语、宾语、极性、是否提问）
#  下标为 -1 表示无；言语命题的主语/宾语也是图中的节点——换一个话题、换一个说法，就是另一个行动
# ============================================================


@dataclass(frozen=True, slots=True)
class ActionCode:
    op: int
    manner: int
    target: int
    obj: int
    actor: int
    topic_pred: int = -1
    topic_subj: int = -1
    topic_val: int = -1
    topic_holds: float = 0.0
    topic_query: float = 0.0
    beneficiary: int = -1


def encode_action(g: GraphTensors, actor: str, cand: Candidate) -> ActionCode:
    pred, subj, val, holds, query = -1, -1, -1, 0.0, 0.0
    if cand.topic is not None:
        p = cand.topic.prop
        pred = TOPIC_INDEX.get(p.predicate, -1)
        subj = g.index_of(p.subject)
        val = g.index_of(p.value) if isinstance(p.value, str) else -1
        holds = 1.0 if cand.topic.holds else -1.0
        query = 1.0 if p.value is None else 0.0
    return ActionCode(OP_INDEX[cand.op], MANNER_INDEX[cand.manner], g.index_of(cand.target), g.index_of(cand.obj),
                      g.index_of(actor), pred, subj, val, holds, query, g.index_of(cand.beneficiary))
