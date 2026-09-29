"""
[INPUT]: 依赖 torch，learning/model 的 DynamicsModel，learning/samples 的 agent_queries / shared_graph_batch，
         learning/schema 的 check_schema / DYN_BOOL，agents/predictors 的 Prediction / BranchValuer / entropy，
         cognition 的 BeliefStore / Candidate，core 的 Fact / Proposition / Rel，core/profiles 的 Profile
[OUTPUT]: 对外提供 GNNPredictor（OutcomePredictor 协议的 GNN 实现）、GAIN_SCALE
[POS]: learning 与 agents 的接缝：训练好的角色视角动态模型以“预测器”身份接入 LangGraph 决策流程，
       替换 HeuristicPredictor 而不改策略与图。输入只有角色认知；输出保留概率——预测不是事实。
       加载时同时核对规格指纹与视角：全知（env）模型不能冒充角色的主观预测；成败头按校准世界拟合的温度缩放。
       预期获知来自“有效新观察数”头（扣除行动本身的直接效果），不是位置变化概率之和；
       目标进展与风险：把模型预测的下一刻认知（最可能的容纳者与身体状态）当作假想分支，用同一套目标语义求势能之差
       （与启发式共用 BranchValuer：碰不到目标读集的预测事实不假想、相同的只算一次，与逐候选 branch_value 逐位相同），
       再加上“自己下一刻受伤/中毒/被制”的预测概率作为风险。这是孤立行动效果模型：旁人同时行动不在预测之内。
       一次决策的所有候选共用同一张认知图：构图与关系编码各只做一次，只有行动条件化之后的部分逐候选计算
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from tianlong.agents.predictors import BranchValuer, Prediction, entropy
from tianlong.cognition import BeliefStore, Candidate
from tianlong.core import Fact, Proposition, Rel
from tianlong.core.profiles import Profile
from tianlong.learning.model import DynamicsModel, DynamicsOutput
from tianlong.learning.samples import Sample, agent_queries, shared_graph_batch
from tianlong.learning.schema import DYN_BOOL, check_schema

GAIN_SCALE = 5.0    # 期望有效新观察数 → [0, 1] 的尺度：五条以上视为“收获很大”
_HARM = tuple(DYN_BOOL.index(k) for k in ("wounded", "poisoned", "subdued"))


class GNNPredictor:
    def __init__(self, model: DynamicsModel, temperature: float = 1.0) -> None:
        self.model = model.eval()
        self.temperature = temperature      # 校准世界上拟合的成败头温度

    @classmethod
    def load(cls, path: str | Path) -> GNNPredictor:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check_schema(ckpt, path, view="agent")
        model = DynamicsModel(ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, float(ckpt.get("success_temperature", 1.0)))

    @torch.no_grad()
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = (),
        profile: Profile | None = None,
    ) -> list[Prediction]:
        if not cands:
            return []
        samples = agent_queries(store, now, store.owner, cands)       # 同一张认知图，只换行动编码
        batch = shared_graph_batch(samples)
        n, e = samples[0].graph.num_nodes, samples[0].graph.edge_index.shape[1]          # 批里的第一份图
        encoded = self.model.encoder(batch.x[:n], batch.edge_index[:, :e], batch.edge_attr[:e])
        encoded = encoded.repeat(len(samples), 1)                   # 关系编码与行动无关：只编码一次
        out = self.model(batch, encoded=encoded)
        success = torch.sigmoid(out.success / self.temperature)
        gain = (out.obs_gain / GAIN_SCALE).clamp(0, 1)
        facts = _predicted_facts(out, samples)
        harm = _self_harm(out, batch, samples, store.owner)
        branches = BranchValuer(store, now, profile)
        preds = []
        for i in range(len(cands)):
            p = float(success[i])
            delta = branches(facts[i])
            preds.append(Prediction(p, float(gain[i]), "gnn", min(1.0, max(0.0, delta)),
                                    min(1.0, max(harm[i], -delta)), entropy(p)))
        return preds


def _offsets(batch) -> list[int]:
    counts = torch.bincount(batch.batch, minlength=int(batch.act_op.numel())).tolist()
    return [sum(counts[:i]) for i in range(len(counts))]


def _predicted_facts(out: DynamicsOutput, samples: Sequence[Sample]) -> list[list[Fact]]:
    """模型认为最可能的下一刻：容纳者换了的写成新的 AT，确知不在原处的写成原处的否定。"""
    facts: list[list[Fact]] = [[] for _ in samples]
    pred = out.holder.argmax(-1).tolist()
    nmax = out.holder.size(1) - 3           # 指针类是图内的局部下标；末三类为 UNKNOWN、GONE、NEW
    row = 0
    for g, s in enumerate(samples):
        for k, node in enumerate(s.located.tolist()):
            j = row + k
            if pred[j] == int(out.holder_now[j]):
                continue
            eid = s.graph.node_ids[node]
            if pred[j] < nmax and pred[j] < s.graph.num_nodes:
                facts[g].append(Fact(Proposition.rel(eid, Rel.AT, s.graph.node_ids[pred[j]])))
            elif pred[j] == nmax + 1 and int(out.holder_now[j]) < nmax:
                facts[g].append(Fact(Proposition.rel(eid, Rel.AT, s.graph.node_ids[int(out.holder_now[j])]), False))
        row += len(s.located)
    return facts


def _self_harm(out: DynamicsOutput, batch, samples: Sequence[Sample], me: str) -> list[float]:
    """自己下一刻变成受伤/中毒/被制（而此刻不是）的预测概率，取最大者。"""
    probs = torch.softmax(out.attr, -1)[..., 2]            # [N, B] “是”的概率
    offs = _offsets(batch)
    harm = []
    for g, s in enumerate(samples):
        i = s.graph.index_of(me)
        if i < 0:
            harm.append(0.0)
            continue
        n = offs[g] + i
        now = s.bool_now[i]
        harm.append(max((float(probs[n, j]) for j in _HARM if now[j] <= 0), default=0.0))
    return harm
