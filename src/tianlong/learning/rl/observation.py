"""
[INPUT]: 依赖 numpy（gymnasium 只在 observation_space() 里按需导入），core 的 digest，learning/featurize 的 featurize / encode_action / GraphTensors，learning/schema 的维度常量，
         cognition 的 BeliefStore / Candidate / belief_view，cognition/goals 的 BeliefReader，core/goals 的 GoalRegistry / GoalMode，
         agents/predictors 的 Prediction / PRED_FIELDS，core/profiles 的 Profile / GoalKind / Goal，memory/view 的 MemoryView
[OUTPUT]: 对外提供 ObsSpec、CropReport、Observation、observation_space()、build_observation()、encode_goals()、
          GOAL_KINDS / GOAL_FIELDS / GOAL_PTRS / ROLES / CAND_INT / CAND_FLOAT、OBS_VERSION（观测布局与语义的指纹，部署包据此拒绝旧策略）、
          ABLATIONS / ablate()（消融 = 某一类输入整列置零：训练期、测试期、上线时同一个定义）
[POS]: learning/rl 的观测契约：个人认知图 + 候选集（完整行动编码，含言语命题）+ 冻结世界模型的预测 + 目标槽位 + 长期记忆摘要
       （逐节点：被经历提起几次、此人的说法被亲眼证实/证伪几次、最近一次相关经历距今多久——来自 memory/view，与线上同一定义）。
       目标槽位带着指向图中实体的指针（寻仇的是谁、护的是谁、东西送给谁）、权重、是否已激活与距激活的时间、了结条件、
       一次性/持续语义、以及角色自己以为的达成状态与进展（BeliefReader，不知道就是 0）——换一个仇人，输入就不同。
       裁剪契约：超预算时先保证自身、目标所指、以及每个保留候选的全部引用都在图里，其余节点按到这些必要节点的跳数入选；
       引用放不下的候选整个剔除并计入 CropReport，绝不留下“候选有效、指针却是 -1”的悬空引用。
       RL 环境与游戏内 LearnedPolicy 共用它——训练时看到什么，上线时就看到什么；掩码只屏蔽空位，从不按真相屏蔽行动
       候选编码和裁剪都保留受益人引用；观测布局指纹随字段改变。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tianlong.agents.predictors import PRED_FIELDS, Prediction
from tianlong.cognition import BeliefStore, Candidate, belief_view
from tianlong.cognition.goals import BeliefReader
from tianlong.core import digest
from tianlong.core.goals import GoalMode, GoalRegistry
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.learning.featurize import GraphTensors, encode_action, featurize
from tianlong.learning.schema import F_EDGE, F_NODE, N_OPS, N_TOPICS
from tianlong.memory.view import COUNT_SCALE, MEMORY_FIELDS, RECENCY_SCALE, MemoryView

if TYPE_CHECKING:
    import gymnasium as gym

GOAL_KINDS = tuple(GoalKind)
GOAL_PTRS = ("item", "home", "recipient", "person")
UNTIL = ("wounded", "subdued")
GOAL_FIELDS = (*(f"kind:{k.value}" for k in GOAL_KINDS), "weight", "active", "wait", *(f"until:{u}" for u in UNTIL),
               "maintain", "status", "potential")
ROLES = ("self", "goal_item", "goal_place", "goal_recipient", "goal_person", "ally")
CAND_INT = ("op", "manner", "target", "obj", "topic_pred", "topic_subj", "topic_val", "beneficiary")
CAND_FLOAT = ("topic_holds", "topic_query")
WAIT_SCALE = 240.0     # 距目标激活的时间：四个时辰以上视为同等遥远
# 观测布局与语义的指纹：列一变（目标槽位、角色标记、候选编码、预测列、记忆列、尺度），旧策略的输入就换了含义——
# 形状相同时 load_state_dict 不会报错，只能靠它拒绝；改动编码语义而列名不变时手动递增前缀
# 消融：某一类输入整列置零。训练期消融的策略从头到尾没见过这类输入——名字写进检查点，上线时照样置零
ABLATIONS: dict[str, tuple[str, ...]] = {"predictions": ("cand_pred",), "memory": ("memory",)}


def ablate(obs: dict[str, np.ndarray], names: Iterable[str]) -> dict[str, np.ndarray]:
    for name in names:
        if name not in ABLATIONS:
            raise ValueError(f"未知的消融 {name!r}：只有 {sorted(ABLATIONS)}")
        for key in ABLATIONS[name]:
            obs[key][...] = 0.0
    return obs


OBS_VERSION = digest("obs-v2", tuple(k.value for k in GOAL_KINDS), GOAL_PTRS, UNTIL, GOAL_FIELDS, ROLES, CAND_INT,
                     CAND_FLOAT, WAIT_SCALE, COUNT_SCALE, RECENCY_SCALE, PRED_FIELDS, MEMORY_FIELDS)

_REGISTRY = GoalRegistry()


@dataclass(frozen=True)
class ObsSpec:
    max_nodes: int = 48
    max_edges: int = 320
    max_cands: int = 48
    max_goals: int = 4


@dataclass(frozen=True)
class CropReport:
    nodes: int
    nodes_kept: int
    edges: int
    edges_kept: int
    cands: int
    cands_kept: int
    required: int          # 必须保留的节点数（自身 + 目标所指 + 保留候选的引用）

    @property
    def cropped(self) -> bool:
        return self.nodes_kept < self.nodes or self.edges_kept < self.edges or self.cands_kept < self.cands


@dataclass(frozen=True)
class Observation:
    obs: dict[str, np.ndarray]
    candidates: tuple[Candidate, ...]      # 与动作编号一一对应：被裁掉的候选不在其中
    predictions: tuple[Prediction, ...]
    report: CropReport


def observation_space(spec: ObsSpec) -> gym.spaces.Dict:
    import gymnasium as gym
    n, e, a, g = spec.max_nodes, spec.max_edges, spec.max_cands, spec.max_goals
    f32 = np.float32
    return gym.spaces.Dict({
        "x": gym.spaces.Box(-1.0, 1.0, (n, F_NODE), f32),
        "node_mask": gym.spaces.Box(0.0, 1.0, (n,), f32),
        "role": gym.spaces.Box(0.0, 1.0, (n, len(ROLES)), f32),
        "memory": gym.spaces.Box(0.0, 1.0, (n, len(MEMORY_FIELDS)), f32),
        "edge_index": gym.spaces.Box(0.0, float(n - 1), (2, e), f32),
        "edge_attr": gym.spaces.Box(0.0, 1.0, (e, F_EDGE), f32),
        "edge_mask": gym.spaces.Box(0.0, 1.0, (e,), f32),
        "goal": gym.spaces.Box(-1.0, 1.0, (g, len(GOAL_FIELDS)), f32),
        "goal_ptr": gym.spaces.Box(-1.0, float(n - 1), (g, len(GOAL_PTRS)), f32),
        "goal_mask": gym.spaces.Box(0.0, 1.0, (g,), f32),
        "cand": gym.spaces.Box(-1.0, float(max(n, N_OPS, N_TOPICS)), (a, len(CAND_INT)), f32),
        "cand_flag": gym.spaces.Box(-1.0, 1.0, (a, len(CAND_FLOAT)), f32),
        "cand_pred": gym.spaces.Box(0.0, 1.0, (a, len(PRED_FIELDS)), f32),
        "action_mask": gym.spaces.Box(0.0, 1.0, (a,), f32),
    })


# ============================================================
#  目标槽位：同一套目标语义（core/goals）在角色自己的认知上求值
# ============================================================


def _goal_refs(g: Goal) -> tuple[str | None, ...]:
    return tuple(getattr(g, f) for f in GOAL_PTRS)


def encode_goals(store: BeliefStore, now: int, profile: Profile, index: dict[str, int], max_goals: int,
                 registry: GoalRegistry = _REGISTRY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = np.zeros((max_goals, len(GOAL_FIELDS)), np.float32)
    ptrs = np.full((max_goals, len(GOAL_PTRS)), -1.0, np.float32)
    mask = np.zeros(max_goals, np.float32)
    reader = BeliefReader(store)
    col = {k: i for i, k in enumerate(GOAL_FIELDS)}
    for slot, g in enumerate(profile.goals[:max_goals]):
        row = feats[slot]
        row[col[f"kind:{g.kind.value}"]] = 1.0
        row[col["weight"]] = float(np.clip(g.weight, -1.0, 1.0))
        row[col["active"]] = 1.0 if g.active(now) else 0.0
        if g.not_before is not None and not g.active(now):
            row[col["wait"]] = min((g.not_before - now) / WAIT_SCALE, 1.0)
        if g.kind == GoalKind.HOSTILE and g.until in UNTIL:
            row[col[f"until:{g.until}"]] = 1.0
        row[col["maintain"]] = 1.0 if registry.mode(g) == GoalMode.MAINTAIN else 0.0
        sat = registry.satisfied(reader, store.owner, g, profile.allies)
        row[col["status"]] = 0.0 if sat is None else (1.0 if sat else -1.0)
        row[col["potential"]] = float(np.clip(registry.potential(reader, store.owner, g, profile.allies), 0.0, 1.0))
        for j, ref in enumerate(_goal_refs(g)):
            ptrs[slot, j] = index.get(ref, -1) if ref else -1
        mask[slot] = 1.0
    return feats, ptrs, mask


# ============================================================
#  裁剪：必要节点优先，其余按跳数；候选的引用放不下就整个剔除
# ============================================================


def _cand_refs(c: Candidate) -> tuple[str, ...]:
    refs = [c.target, c.obj, c.beneficiary]
    if c.topic is not None:
        refs.append(c.topic.prop.subject)
        if isinstance(c.topic.prop.value, str):
            refs.append(c.topic.prop.value)
    return tuple(r for r in refs if r)


def _select(g: GraphTensors, store: BeliefStore, profile: Profile, cands: Sequence[Candidate], spec: ObsSpec
            ) -> tuple[list[int], list[int], frozenset[int]]:
    """返回 (保留节点的原下标, 保留候选的下标, 必要节点的原下标)。"""
    index = {eid: i for i, eid in enumerate(g.node_ids)}
    kept: list[int] = []
    seen: set[int] = set()

    def admit(ids: Sequence[str], strict: bool = True) -> bool:
        if strict and any(e not in index for e in ids):
            return False                       # 引用了图里没有的实体：整个候选剔除，绝不留 -1 指针
        new = [index[e] for e in dict.fromkeys(ids) if e in index and index[e] not in seen]
        if len(kept) + len(new) > spec.max_nodes:
            return False
        kept.extend(new)
        seen.update(new)
        return True

    admit([store.owner])
    admit([r for gl in profile.goals[:spec.max_goals] for r in _goal_refs(gl) if r], strict=False)
    cand_kept: list[int] = []
    for i, c in enumerate(cands[:spec.max_cands]):
        if admit(_cand_refs(c)):
            cand_kept.append(i)
    required = frozenset(kept)

    # ---- 其余节点：按到必要节点的跳数（无向），同跳按原顺序 ----
    adj: dict[int, list[int]] = {}
    for a, b in g.edge_index.T.tolist():
        adj.setdefault(a, []).append(b)
    frontier = deque(sorted(seen))
    dist = {i: 0 for i in seen}
    order: list[tuple[int, int]] = []
    while frontier:
        cur = frontier.popleft()
        for nxt in sorted(adj.get(cur, ())):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                order.append((dist[nxt], nxt))
                frontier.append(nxt)
    rest = [i for _, i in sorted(order)] + [i for i in range(g.num_nodes) if i not in dist]
    for i in rest:
        if len(kept) >= spec.max_nodes:
            break
        kept.append(i)
        seen.add(i)
    return sorted(kept), cand_kept, required


def build_observation(
    store: BeliefStore, now: int, profile: Profile, cands: Sequence[Candidate], preds: Sequence[Prediction],
    spec: ObsSpec, memory: MemoryView | None = None, offered: int | None = None,
) -> Observation:
    """offered：行动族配额截断之前的候选总数（给出时裁剪报告把配额截掉的也算进去）。"""
    g = featurize(belief_view(store, now))
    nodes, cand_idx, required = _select(g, store, profile, cands, spec)
    remap = {old: new for new, old in enumerate(nodes)}
    n = len(nodes)

    x = np.zeros((spec.max_nodes, F_NODE), np.float32)
    x[:n] = g.x[nodes]
    node_mask = np.zeros(spec.max_nodes, np.float32)
    node_mask[:n] = 1.0
    ids = tuple(g.node_ids[i] for i in nodes)
    index = {eid: i for i, eid in enumerate(ids)}

    # ---- 边：两端都保留的边；超预算时先保留触及必要节点的（正反两条成对取舍）----
    pairs = []
    ei = g.edge_index
    for k in range(0, ei.shape[1], 2):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a in remap and b in remap:
            pairs.append((k, a not in required and b not in required))
    pairs.sort(key=lambda p: (p[1], p[0]))
    pairs = pairs[: spec.max_edges // 2]
    cols = sorted(c for k, _ in pairs for c in (k, k + 1))
    edge_index = np.zeros((2, spec.max_edges), np.float32)
    edge_attr = np.zeros((spec.max_edges, F_EDGE), np.float32)
    edge_mask = np.zeros(spec.max_edges, np.float32)
    m = len(cols)
    if m:
        edge_index[:, :m] = np.vectorize(remap.get)(ei[:, cols])
        edge_attr[:m] = g.edge_attr[cols]
        edge_mask[:m] = 1.0

    # ---- 角色标记与目标槽位 ----
    role = np.zeros((spec.max_nodes, len(ROLES)), np.float32)
    if store.owner in index:
        role[index[store.owner], 0] = 1.0
    for gl in profile.goals:
        if not gl.active(now):
            continue                 # 未激活的目标只在槽位里（带距激活时间），不把它的对象标成“眼下要办的事”
        for col, ref in zip(("goal_item", "goal_place", "goal_recipient", "goal_person"), _goal_refs(gl), strict=True):
            if ref in index:
                role[index[ref], ROLES.index(col)] = 1.0
    for ally in profile.allies:
        if ally in index:
            role[index[ally], ROLES.index("ally")] = 1.0
    goal, goal_ptr, goal_mask = encode_goals(store, now, profile, index, spec.max_goals)
    mem = np.zeros((spec.max_nodes, len(MEMORY_FIELDS)), np.float32)
    if memory is not None:
        for eid, i in index.items():
            mem[i] = memory.features(eid, now)

    # ---- 候选：完整行动编码 ----
    kept_cands = tuple(cands[i] for i in cand_idx)
    kept_preds = tuple(preds[i] if i < len(preds) else Prediction(0.0, 0.0) for i in cand_idx)
    cand = np.full((spec.max_cands, len(CAND_INT)), -1.0, np.float32)
    flag = np.zeros((spec.max_cands, len(CAND_FLOAT)), np.float32)
    pred = np.zeros((spec.max_cands, len(PRED_FIELDS)), np.float32)
    mask = np.zeros(spec.max_cands, np.float32)
    local = GraphTensors(ids, tuple(g.kinds[i] for i in nodes), x[:n], np.zeros((2, 0), np.int64),
                         np.zeros((0, F_EDGE), np.float32))
    for i, (c, p) in enumerate(zip(kept_cands, kept_preds, strict=True)):
        code = encode_action(local, store.owner, c)
        cand[i] = (code.op, code.manner, code.target, code.obj, code.topic_pred, code.topic_subj, code.topic_val, code.beneficiary)
        flag[i] = (code.topic_holds, code.topic_query)
        pred[i] = np.clip([getattr(p, f) for f in PRED_FIELDS], 0.0, 1.0)
        mask[i] = 1.0
    report = CropReport(g.num_nodes, n, g.edge_index.shape[1], m, max(len(cands), offered or 0), len(kept_cands),
                        len(required))
    obs = {"x": x, "node_mask": node_mask, "role": role, "memory": mem, "edge_index": edge_index, "edge_attr": edge_attr,
           "edge_mask": edge_mask, "goal": goal, "goal_ptr": goal_ptr, "goal_mask": goal_mask, "cand": cand,
           "cand_flag": flag, "cand_pred": pred, "action_mask": mask}
    return Observation(obs, kept_cands, kept_preds, report)
