"""
[INPUT]: 依赖 numpy / torch / torch_geometric 的 Data，learning/featurize 的 GraphTensors / featurize / encode_action / bool_tri / num_known，
         learning/schema 的 DYN_BOOL / DYN_NUM / ATTR_BLOCK / OBS_GAIN_CAP，cognition 的 world_view / belief_view / BeliefStore / Candidate，
         core 的 WorldState / Rel / Proposition
[OUTPUT]: 对外提供 Sample、env_sample()（环境动态样本）、agent_sample()（角色视角样本）、
          agent_queries() / agent_query()（推理输入：同一决策的候选共用一张认知图）、
          DynData、to_data()、shared_graph_batch()（一张图 × 多个候选的批，与通用拼接张量逐项相同）、
          UNKNOWN / GONE / NEW（容纳者的三个“空”类）
[POS]: learning 的监督信号定义（StateDelta）。两类样本刻意分开构造：
       环境样本 = 真实状态（含全部机制变量）+ 行动 → 真实的下一状态；
       角色样本 = 个人认知 + 自己的行动 → 结算后“我会相信什么”。
       “不知道”被拆成三件事分别建模：仍不知道（UNKNOWN）、确知不在原处而去向不明（GONE）、认识了新实体（discover）；
       动态属性按类型给标签：布尔三态、数值（值 + 是否已知）——修习进度、学成的技能、被吸走的内力都是预测目标。
       标签只来自内核实际执行的结果
       受益人行动指针有有无掩码，PyG 批处理时与其他节点指针一起偏移。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from tianlong.cognition import BeliefStore, Candidate, belief_view, world_view
from tianlong.core import Proposition, Rel, WorldState
from tianlong.learning.featurize import (
    HOLDER_KINDS,
    LOCATED_KINDS,
    ActionCode,
    GraphTensors,
    bool_tri,
    encode_action,
    featurize,
    num_known,
)
from tianlong.learning.schema import ATTR_BLOCK, DYN_BOOL, DYN_NUM, OBS_GAIN_CAP

UNKNOWN = -1   # 容纳者未知
GONE = -2      # 确知不在原以为的地方，去向不明
NEW = -3       # 确知在一个此前不认识的容纳者那里（指针指不到这一刻的图里）


@dataclass(frozen=True)
class Sample:
    graph: GraphTensors
    action: ActionCode
    success: float
    located: np.ndarray        # [M] 需要预测位置的节点
    holder_now: np.ndarray     # [M] 当前（认为的）容纳者下标；UNKNOWN / GONE
    holder_next: np.ndarray    # [M] 下一刻（认为的）容纳者下标；UNKNOWN / GONE
    bool_now: np.ndarray       # [N, B] 动态布尔属性三态 -1/0/1
    bool_next: np.ndarray      # [N, B]
    bool_mask: np.ndarray      # [N, B] 属性对该节点适用
    num_now: np.ndarray        # [N, K] 动态数值属性（已缩放）
    num_now_known: np.ndarray  # [N, K]
    num_next: np.ndarray       # [N, K]
    num_next_known: np.ndarray  # [N, K]
    num_mask: np.ndarray       # [N, K]
    discover: float            # 下一刻认识了新实体（角色视角）
    obs_gain: float            # 不属于本行动直接效果的信念变化条数（角色视角，截断到 OBS_GAIN_CAP）
    view: str                  # "env" | "agent"


# ============================================================
#  从节点列取动态属性（输入怎么编码，标签就怎么定义）
# ============================================================


def _dyn(g: GraphTensors) -> tuple[np.ndarray, ...]:
    n = g.num_nodes
    b = np.stack([bool_tri(g.x, k) for k in DYN_BOOL], axis=1) if n else np.zeros((0, len(DYN_BOOL)), np.float32)
    b_mask = np.stack([g.x[:, ATTR_BLOCK[k].known] >= 0 for k in DYN_BOOL], axis=1) if n \
        else np.zeros((0, len(DYN_BOOL)), bool)
    if n:
        pairs = [num_known(g.x, k) for k in DYN_NUM]
        v = np.stack([p[0] for p in pairs], axis=1)
        known = np.stack([p[1] for p in pairs], axis=1)
        v_mask = np.stack([g.x[:, ATTR_BLOCK[k].known] >= 0 for k in DYN_NUM], axis=1)
    else:
        v, known, v_mask = (np.zeros((0, len(DYN_NUM)), np.float32), np.zeros((0, len(DYN_NUM)), bool),
                            np.zeros((0, len(DYN_NUM)), bool))
    return b, b_mask, v, known, v_mask


def _aligned(g: GraphTensors, g_next: GraphTensors) -> tuple[np.ndarray, ...]:
    """把下一刻的动态属性按实体 ID 对齐到这一刻的节点顺序（角色视角里下一刻可能多出新认识的实体）。"""
    b, _, v, known, _ = _dyn(g_next)
    pos = {eid: i for i, eid in enumerate(g_next.node_ids)}
    rows = [pos.get(eid, -1) for eid in g.node_ids]
    b_now, _, v_now, k_now, _ = _dyn(g)
    b_out, v_out, k_out = b_now.copy(), v_now.copy(), k_now.copy()
    for i, j in enumerate(rows):
        if j >= 0 and not g.node_ids[i].startswith("episode:"):
            b_out[i], v_out[i], k_out[i] = b[j], v[j], known[j]
    return b_out, v_out, k_out


def _located(g: GraphTensors) -> np.ndarray:
    return np.array([i for i, k in enumerate(g.kinds) if k in LOCATED_KINDS], dtype=np.int64)


def _holder_index(g: GraphTensors, eid: str | None) -> int:
    i = g.index_of(eid)
    return i if i >= 0 and g.kinds[i] in HOLDER_KINDS else UNKNOWN


def env_sample(before: WorldState, after: WorldState, actor: str, cand: Candidate, success: bool) -> Sample:
    g = featurize(world_view(before, actor))
    g_next = featurize(world_view(after, actor))
    assert g.node_ids == g_next.node_ids, "实体集合在一步之内不变"
    located = _located(g)
    now = np.array([_holder_index(g, before.target(g.node_ids[i], Rel.AT)) for i in located], dtype=np.int64)
    nxt = np.array([_holder_index(g, after.target(g.node_ids[i], Rel.AT)) for i in located], dtype=np.int64)
    b, b_mask, v, known, v_mask = _dyn(g)
    b2, _, v2, known2, _ = _dyn(g_next)
    return Sample(g, encode_action(g, actor, cand), float(success), located, now, nxt,
                  b, b2, b_mask, v, known, v2, known2, v_mask, 0.0, 0.0, "env")


def _believed_holder(g: GraphTensors, store: BeliefStore, eid: str) -> int:
    return _holder_index(g, store.location_of(eid))


def _next_holder(g: GraphTensors, before: BeliefStore, after: BeliefStore, eid: str) -> int:
    idx = _believed_holder(g, after, eid)
    if idx != UNKNOWN:
        return idx
    if after.location_of(eid) is not None:
        return NEW                                # 知道在哪，只是那个容纳者这一刻还不在图里
    was = before.location_of(eid)
    if was is not None:
        b = after.believed(Proposition.rel(eid, Rel.AT, was))
        if b is not None and not b.holds:
            return GONE                           # 确知不在原处，去向不明
    return UNKNOWN


def agent_queries(store: BeliefStore, now: int, actor: str, cands: Sequence[Candidate]) -> list[Sample]:
    """推理用样本：只有输入（认知 + 候选行动），标签位填“保持现状”，不参与训练。
    同一次决策的所有候选看的是同一张认知图：图与动态属性只构造一次，各候选只换行动编码。"""
    g = featurize(belief_view(store, now))
    located = _located(g)
    cur = np.array([_believed_holder(g, store, g.node_ids[i]) for i in located], dtype=np.int64)
    b, b_mask, v, known, v_mask = _dyn(g)
    return [Sample(g, encode_action(g, actor, c), 0.0, located, cur, cur, b, b, b_mask, v, known, v, known, v_mask,
                   0.0, 0.0, "agent") for c in cands]


def agent_query(store: BeliefStore, now: int, actor: str, cand: Candidate) -> Sample:
    return agent_queries(store, now, actor, (cand,))[0]


def agent_sample(
    before: BeliefStore, after: BeliefStore, now: int, actor: str, cand: Candidate, success: bool,
    obs_gain: int = 0,
) -> Sample:
    g = featurize(belief_view(before, now))
    g_next = featurize(belief_view(after, now + 1))
    located = _located(g)
    cur = np.array([_believed_holder(g, before, g.node_ids[i]) for i in located], dtype=np.int64)
    nxt = np.array([_next_holder(g, before, after, g.node_ids[i]) for i in located], dtype=np.int64)
    b, b_mask, v, known, v_mask = _dyn(g)
    b2, v2, known2 = _aligned(g, g_next)
    discovered = {e for e in g_next.node_ids if not e.startswith("episode:")} - set(g.node_ids)
    return Sample(g, encode_action(g, actor, cand), float(success), located, cur, nxt,
                  b, b2, b_mask, v, known, v2, known2, v_mask, float(bool(discovered)),
                  float(min(obs_gain, OBS_GAIN_CAP)), "agent")


# ============================================================
#  PyG 批处理：指针类字段随批次偏移，“空”类用独立掩码表示（负数不能参与偏移）
# ============================================================

_INC_KEYS = frozenset({"located", "holder_now_idx", "holder_next_idx", "act_target", "act_obj", "act_actor",
                       "act_topic_subj", "act_topic_val", "act_beneficiary"})


class DynData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in _INC_KEYS:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def _ptr(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.as_tensor(values, dtype=torch.long)
    return v.clamp(min=0), v < 0


def _action_fields(a: ActionCode) -> dict[str, torch.Tensor]:
    """一个候选行动的编码：指针类字段钳到 0 并配“有无”掩码（负数不能参与批内偏移）。"""
    tgt, no_tgt = _ptr(np.array([a.target]))
    obj, no_obj = _ptr(np.array([a.obj]))
    act, no_act = _ptr(np.array([a.actor]))
    subj, no_subj = _ptr(np.array([a.topic_subj]))
    val, no_val = _ptr(np.array([a.topic_val]))
    bene, no_bene = _ptr(np.array([a.beneficiary]))
    return dict(
        act_op=torch.tensor([a.op]), act_manner=torch.tensor([a.manner]),
        act_target=tgt, act_has_target=~no_tgt, act_obj=obj, act_has_obj=~no_obj, act_actor=act, act_has_actor=~no_act,
        act_topic_pred=torch.tensor([a.topic_pred + 1]),          # 0 = 无命题
        act_topic_subj=subj, act_has_topic_subj=~no_subj, act_topic_val=val, act_has_topic_val=~no_val,
        act_topic_flags=torch.tensor([[a.topic_holds, a.topic_query]], dtype=torch.float32),
        act_beneficiary=bene, act_has_beneficiary=~no_bene,
    )


def to_data(s: Sample) -> DynData:
    g = s.graph
    now_idx, _ = _ptr(s.holder_now)
    next_idx, _ = _ptr(s.holder_next)
    return DynData(
        x=torch.as_tensor(g.x), edge_index=torch.as_tensor(g.edge_index), edge_attr=torch.as_tensor(g.edge_attr),
        is_holder=torch.tensor([k in HOLDER_KINDS for k in g.kinds], dtype=torch.bool),
        **_action_fields(s.action),
        success=torch.tensor([s.success], dtype=torch.float32),
        located=torch.as_tensor(s.located, dtype=torch.long),
        holder_now_idx=now_idx, holder_now_kind=torch.as_tensor(np.minimum(s.holder_now, 0), dtype=torch.long),
        holder_next_idx=next_idx, holder_next_kind=torch.as_tensor(np.minimum(s.holder_next, 0), dtype=torch.long),
        bool_now=torch.as_tensor(s.bool_now, dtype=torch.long) + 1,     # -1/0/1 → 0/1/2
        bool_next=torch.as_tensor(s.bool_next, dtype=torch.long) + 1,
        bool_mask=torch.as_tensor(s.bool_mask, dtype=torch.bool),
        num_now=torch.as_tensor(s.num_now), num_now_known=torch.as_tensor(s.num_now_known, dtype=torch.bool),
        num_next=torch.as_tensor(s.num_next), num_next_known=torch.as_tensor(s.num_next_known, dtype=torch.bool),
        num_mask=torch.as_tensor(s.num_mask, dtype=torch.bool),
        discover=torch.tensor([s.discover], dtype=torch.float32),
        obs_gain=torch.tensor([s.obs_gain], dtype=torch.float32),
        is_agent=torch.tensor([s.view == "agent"]),
        num_nodes=g.num_nodes,
    )


def shared_graph_batch(samples: Sequence[Sample]) -> DynData:
    """一张认知图 × 多个候选行动的批（推理热路径）：与 Batch.from_data_list([to_data(s) ...]) 张量逐项相同，
    但图只转换一次、按块复制，不走逐样本的通用拼接。samples 必须共享同一张图（agent_queries 的产物）。"""
    base = to_data(samples[0])
    b, n = len(samples), base.num_nodes
    acts = [_action_fields(s.action) for s in samples]
    shift = torch.arange(b) * n
    out: dict[str, torch.Tensor] = {}
    for key, v in base:
        if not isinstance(v, torch.Tensor):
            continue
        if key in acts[0]:
            t = torch.cat([f[key] for f in acts])
            out[key] = t + shift if key in _INC_KEYS else t
        elif key == "edge_index":
            out[key] = v.repeat(1, b) + shift.repeat_interleave(v.size(1))
        elif key in _INC_KEYS:
            out[key] = v.repeat(b) + shift.repeat_interleave(v.size(0))
        else:
            out[key] = v.repeat(b, *([1] * (v.dim() - 1)))
    data = DynData(**out, num_nodes=b * n)
    data.batch = torch.arange(b).repeat_interleave(n)
    data.ptr = torch.cat([shift, torch.tensor([b * n])])
    return data
