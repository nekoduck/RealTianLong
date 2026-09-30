"""
[INPUT]: 依赖 torch，ray.rllib 的 TorchRLModule / ValueFunctionAPI / Columns，learning/model 的 RelationalEncoder，
         learning/schema 的维度常量，learning/rl/observation 的 ROLES / GOAL_FIELDS / GOAL_PTRS / CAND_FLOAT，agents/predictors 的 PRED_FIELDS，
         memory/view 的 MEMORY_FIELDS
[OUTPUT]: 对外提供 CandidateScoringModule（RLlib 新 API 栈的策略/价值网络）、GraphPolicyNet（纯 torch 核心，供模仿学习与游戏内推理复用）
[POS]: learning/rl 的策略网络：定长观测 → 还原为稀疏批图 → 与动态模型同构的关系编码器（节点再叠加角色标记与长期记忆摘要）→
       目标槽位编码（目标特征 + 所指实体的节点表示，按掩码汇总）→ 逐候选打分（指针式策略：操作、方式、言语谓词、
       目标/对象/命题主语/命题宾语的节点表示、命题极性、冻结世界模型的预测）；掩码外的空位 logit = -inf；
       价值头只看认知池化与目标汇总
       候选打分的第五个实体指针是请求受益人，与新观测布局一致。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from torch import Tensor, nn

from tianlong.agents.predictors import PRED_FIELDS
from tianlong.learning.model import RelationalEncoder
from tianlong.learning.rl.observation import CAND_FLOAT, GOAL_FIELDS, GOAL_PTRS, ROLES
from tianlong.learning.schema import N_MANNERS, N_OPS, N_TOPICS
from tianlong.memory.view import MEMORY_FIELDS

NEG_INF = -1e9


def _gather(h: Tensor, idx: Tensor) -> Tensor:
    """h [B, N, d]，idx [B, K]（-1 = 无）→ [B, K, d]，无指向处为零向量。"""
    got = torch.gather(h, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, h.size(-1)))
    return got * (idx >= 0).unsqueeze(-1).float()


class GraphPolicyNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        d = hidden
        self.encoder = RelationalEncoder(d, layers)
        self.role = nn.Linear(len(ROLES), d)
        self.memory = nn.Linear(len(MEMORY_FIELDS), d)          # 长期记忆摘要注入节点表示
        self.goal = nn.Sequential(nn.Linear(len(GOAL_FIELDS) + len(GOAL_PTRS) * d, d), nn.GELU(), nn.Linear(d, d))
        self.op_emb = nn.Embedding(N_OPS, 16)
        self.manner_emb = nn.Embedding(N_MANNERS, 8)
        self.topic_emb = nn.Embedding(N_TOPICS + 1, 8)          # 0 = 无命题
        cand_in = 16 + 8 + 8 + 5 * d + len(CAND_FLOAT) + len(PRED_FIELDS) + 2 * d + d
        self.score = nn.Sequential(nn.Linear(cand_in, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(2 * d + d, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """定长观测 → [B, N, d] 节点表示与 [B, 2d] 图池化。填充节点不参与消息传递与池化。"""
        x, node_mask = obs["x"], obs["node_mask"] > 0.5
        b, n, _ = x.shape
        ei = obs["edge_index"].long()                               # [B, 2, E]
        emask = obs["edge_mask"] > 0.5                              # [B, E]
        offset = (torch.arange(b, device=x.device) * n).view(b, 1, 1)
        flat_ei = (ei + offset).permute(1, 0, 2).reshape(2, -1)[:, emask.reshape(-1)]
        flat_ea = obs["edge_attr"].reshape(b * obs["edge_attr"].shape[1], -1)[emask.reshape(-1)]
        h = self.encoder(x.reshape(b * n, -1), flat_ei, flat_ea).view(b, n, -1)
        h = (h + self.role(obs["role"]) + self.memory(obs["memory"])) * node_mask.unsqueeze(-1)
        denom = node_mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(1) / denom
        maxed = h.masked_fill(~node_mask.unsqueeze(-1), NEG_INF).max(1).values
        maxed = torch.where(node_mask.any(1, keepdim=True), maxed, torch.zeros_like(maxed))
        return h, torch.cat([mean, maxed], -1)

    def goals(self, h: Tensor, obs: dict[str, Tensor]) -> Tensor:
        """目标槽位 → [B, d]：每个槽位 = 目标特征 + 所指实体的节点表示，按掩码平均。"""
        b, g, p = obs["goal_ptr"].shape
        ptr = obs["goal_ptr"].long().reshape(b, g * p)
        refs = _gather(h, ptr).reshape(b, g, p * h.size(-1))
        slots = self.goal(torch.cat([obs["goal"], refs], -1))                  # [B, G, d]
        m = obs["goal_mask"].unsqueeze(-1)
        return (slots * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        h, pooled = self.encode(obs)
        goal = self.goals(h, obs)
        b, a = obs["cand"].shape[:2]
        cand = obs["cand"].long()
        op, manner, topic = cand[..., 0].clamp(min=0), cand[..., 1].clamp(min=0), (cand[..., 4] + 1).clamp(min=0)
        refs = [_gather(h, cand[..., j]) for j in (2, 3, 5, 6, 7)]          # 目标、对象、命题主语/宾语、受益人
        feats = torch.cat([
            self.op_emb(op), self.manner_emb(manner), self.topic_emb(topic), *refs, obs["cand_flag"],
            obs["cand_pred"], pooled.unsqueeze(1).expand(-1, a, -1), goal.unsqueeze(1).expand(-1, a, -1),
        ], -1)
        logits = self.score(feats).squeeze(-1)
        logits = logits.masked_fill(obs["action_mask"] < 0.5, NEG_INF)
        value = self.value(torch.cat([pooled, goal], -1)).squeeze(-1)
        return logits, value


class CandidateScoringModule(TorchRLModule, ValueFunctionAPI):
    """model_config: {"hidden": 64, "layers": 2}"""

    def setup(self) -> None:
        cfg = self.model_config or {}
        self.net = GraphPolicyNet(int(cfg.get("hidden", 64)), int(cfg.get("layers", 2)))

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        logits, _ = self.net(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: logits}

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> Tensor:
        _, value = self.net(batch[Columns.OBS])
        return value
