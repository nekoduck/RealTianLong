"""
[INPUT]: 依赖 torch，torch_geometric 的 TransformerConv / global_mean_pool / global_max_pool / to_dense_batch，
         learning/schema 的维度常量与预测目标规格
[OUTPUT]: 对外提供 RelationalEncoder（带边特征的关系图编码器）、DynamicsModel（行动条件化的后果预测器）、DynamicsOutput、loss_terms()
[POS]: learning 的 GNN 本体。编码器用 TransformerConv(edge_dim)：关系类型/极性/方向与可信度/时效/传闻都在边特征里，
       RGCN 类卷积会丢掉这些认知语义。动态模型回答“这个行动之后会怎样”，覆盖范围由 schema.TARGETS 声明：
       成败、可定位节点的下一容纳者（含 UNKNOWN / GONE / NEW 三个空类）、动态布尔属性三态、动态数值属性（值 + 是否已知）、
       是否认识新实体、有效新观察数。行动条件化包含言语命题（谓词、主语、宾语、极性、提问）。
       位置与已知性各带可学习的“惯性”项：大多数事实不变，模型只需学会何时改变。
       关系编码在行动条件化之前、与行动无关：forward(encoded=...) 让“一张认知图 × 多个候选”只编码一次
       行动条件化含第六个角色：请求受益人。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from tianlong.learning.schema import (
    DYN_BOOL,
    DYN_NUM,
    F_EDGE,
    F_NODE,
    N_MANNERS,
    N_OPS,
    N_TOPICS,
    OBS_GAIN_CAP,
)


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class RelationalEncoder(nn.Module):
    """x, edge_index, edge_attr → 节点表示。残差 + LayerNorm，层数决定推理跨越几跳（人→地点→门→地点 需要 3 跳）。"""

    def __init__(self, hidden: int = 64, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.inp = nn.Linear(F_NODE, hidden)
        self.convs = nn.ModuleList(
            TransformerConv(hidden, hidden // heads, heads=heads, edge_dim=F_EDGE) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.inp(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))
        return h


class ActionEncoder(nn.Module):
    """ActionCode → 行动向量 + 逐节点角色标记（目标/对象/行动者/命题主语/命题宾语/受益人）。策略网络与动态模型共用同一种编码。"""

    ROLES = 6

    def __init__(self, d: int) -> None:
        super().__init__()
        self.op_emb = nn.Embedding(N_OPS, 16)
        self.manner_emb = nn.Embedding(N_MANNERS, 8)
        self.topic_emb = nn.Embedding(N_TOPICS + 1, 8)            # 0 = 无命题
        self.mix = _mlp(16 + 8 + 8 + 2 + self.ROLES * d, d, d)

    def forward(self, h: Tensor, op: Tensor, manner: Tensor, topic: Tensor, flags: Tensor,
                refs: list[tuple[Tensor, Tensor]]) -> Tensor:
        picked = [h[idx] * has.unsqueeze(-1).float() for idx, has in refs]
        return self.mix(torch.cat([self.op_emb(op), self.manner_emb(manner), self.topic_emb(topic), flags, *picked], -1))


@dataclass
class DynamicsOutput:
    success: Tensor        # [B] logit
    holder: Tensor         # [M, Nmax + 3] logit，末三类 = UNKNOWN、GONE、NEW
    holder_now: Tensor     # [M] 当前容纳者类别（由输入推出，供惯性基线与指标使用）
    holder_next: Tensor    # [M] 监督目标类别
    attr: Tensor           # [N, len(DYN_BOOL), 3] logit
    num: Tensor            # [N, len(DYN_NUM)] 缩放后的值
    num_known: Tensor      # [N, len(DYN_NUM)] logit
    discover: Tensor       # [B] logit
    obs_gain: Tensor       # [B] 期望条数（0 ~ OBS_GAIN_CAP）


class DynamicsModel(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        d = hidden
        self.encoder = RelationalEncoder(d, layers, heads)
        self.action = ActionEncoder(d)
        self.condition = _mlp(2 * d + ActionEncoder.ROLES, d, d)   # 行动条件化：节点知道自己在行动里扮演什么
        self.post = nn.ModuleList(TransformerConv(d, d // heads, heads=heads, edge_dim=F_EDGE) for _ in range(2))
        self.post_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(2))
        self.success_head = _mlp(3 * d, d, 1)
        self.query = _mlp(2 * d, d, d)
        self.key = nn.Linear(d, d)
        self.null_head = _mlp(2 * d, d, 3)                           # UNKNOWN / GONE / NEW
        self.inertia = nn.Parameter(torch.tensor(3.0))               # 事实倾向于保持不变
        self.attr_head = _mlp(2 * d, d, len(DYN_BOOL) * 3)
        self.num_head = _mlp(2 * d, d, len(DYN_NUM) * 2)
        self.known_inertia = nn.Parameter(torch.tensor(3.0))
        self.discover_head = _mlp(3 * d, d, 1)
        self.gain_head = _mlp(3 * d, d, 1)

    def forward(self, data, encoded: Tensor | None = None) -> DynamicsOutput:
        """encoded：批内各图的关系编码（编码器不看行动）。同一张认知图配多个候选行动时，
        调用方可以只编码一次再逐图复制传进来——与逐图重复编码是同一个计算。"""
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, ei, ea) if encoded is None else encoded
        n, b = h.size(0), int(data.act_op.numel())
        refs = [(data.act_target, data.act_has_target), (data.act_obj, data.act_has_obj),
                (data.act_actor, data.act_has_actor), (data.act_topic_subj, data.act_has_topic_subj),
                (data.act_topic_val, data.act_has_topic_val), (data.act_beneficiary, data.act_has_beneficiary)]
        a = self.action(h, data.act_op, data.act_manner, data.act_topic_pred, data.act_topic_flags, refs)   # [B, d]

        flags = torch.zeros(n, ActionEncoder.ROLES, device=h.device)
        for col, (idx, has) in enumerate(refs):
            flags[idx[has], col] = 1.0
        h = h + self.condition(torch.cat([h, a[batch], flags], dim=-1))
        for conv, norm in zip(self.post, self.post_norms, strict=True):
            h = norm(h + torch.relu(conv(h, ei, ea)))

        pooled = torch.cat([global_mean_pool(h, batch, b), global_max_pool(h, batch, b), a], dim=-1)
        success = self.success_head(pooled).squeeze(-1)

        # ---- 位置指针：located 节点 i 对同图内每个容纳者 j 打分 + UNKNOWN / GONE ----
        dense, mask = to_dense_batch(h, batch, batch_size=b)            # [B, Nmax, d]
        holder_ok, _ = to_dense_batch(data.is_holder, batch, batch_size=b, fill_value=False)
        offset = torch.zeros(b, dtype=torch.long, device=h.device)
        counts = torch.bincount(batch, minlength=b)
        offset[1:] = torch.cumsum(counts, 0)[:-1]
        loc = data.located
        g = batch[loc]
        q = self.query(torch.cat([h[loc], a[g]], dim=-1))              # [M, d]
        k = self.key(dense[g])                                          # [M, Nmax, d]
        scores = (k @ q.unsqueeze(-1)).squeeze(-1) / q.size(-1) ** 0.5
        nmax = scores.size(1)

        def cls(idx: Tensor, kind: Tensor) -> Tensor:
            return torch.where(kind == 0, idx - offset[g], nmax - 1 - kind)       # -1/-2/-3 → nmax/nmax+1/nmax+2

        now_cls, next_cls = cls(data.holder_now_idx, data.holder_now_kind), cls(data.holder_next_idx,
                                                                                  data.holder_next_kind)
        current = F.one_hot(now_cls, nmax + 3).float()
        scores = scores.masked_fill(~(mask[g] & holder_ok[g]), float("-inf"))
        holder = torch.cat([scores, self.null_head(torch.cat([h[loc], a[g]], dim=-1))], dim=-1)
        holder = holder + self.inertia * current

        ha = torch.cat([h, a[batch]], dim=-1)
        attr = self.attr_head(ha).view(n, len(DYN_BOOL), 3)
        num_out = self.num_head(ha).view(n, len(DYN_NUM), 2)
        num = data.num_now + num_out[..., 0]                             # 残差：大多数数值不变
        known = num_out[..., 1] + self.known_inertia * (2 * data.num_now_known.float() - 1)
        discover = self.discover_head(pooled).squeeze(-1)
        gain = F.softplus(self.gain_head(pooled).squeeze(-1)).clamp(max=OBS_GAIN_CAP)
        return DynamicsOutput(success, holder, now_cls, next_cls, attr, num, known, discover, gain)


def loss_terms(out: DynamicsOutput, data) -> dict[str, Tensor]:
    """每个预测目标一项损失；角色视角才有的目标（发现、观察增益、已知性变化）只在角色样本上计。"""
    zero = out.success.sum() * 0
    terms = {"success": F.binary_cross_entropy_with_logits(out.success, data.success)}
    terms["holder"] = F.cross_entropy(out.holder, out.holder_next) if out.holder.numel() else zero
    m = data.bool_mask & ((data.bool_now != 1) | (data.bool_next != 1))
    terms["attr"] = F.cross_entropy(out.attr[m], data.bool_next[m]) if m.any() else zero
    vm = data.num_mask & data.num_next_known
    terms["num"] = F.mse_loss(out.num[vm], data.num_next[vm]) if vm.any() else zero
    agent_nodes = data.is_agent[data.batch].unsqueeze(-1) & data.num_mask
    terms["known"] = F.binary_cross_entropy_with_logits(out.num_known[agent_nodes],
                                                        data.num_next_known[agent_nodes].float()) \
        if agent_nodes.any() else zero
    ag = data.is_agent
    terms["discover"] = F.binary_cross_entropy_with_logits(out.discover[ag], data.discover[ag]) if ag.any() else zero
    terms["obs_gain"] = F.smooth_l1_loss(out.obs_gain[ag] / OBS_GAIN_CAP, data.obs_gain[ag] / OBS_GAIN_CAP) \
        if ag.any() else zero
    return terms
