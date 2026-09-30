"""
[INPUT]: 依赖 torch，learning/rl 的 GraphPolicyNet / ObsSpec / build_observation / ablate，learning/schema 的 check_schema，
         agents/policies 的 Choice / Situation，core 的 Op
[OUTPUT]: 对外提供 LearnedPolicy（Policy 协议的神经网络实现）
[POS]: learning/rl 与 agents 的接缝：训练好的策略以“策略”身份接入 LangGraph 决策图，替换 ScriptedPolicy 而不改图。
       游玩时运行的是训练好的网络，不在每次玩家输入后临时重新训练；加载时核对规格指纹，并按训练时的 ObsSpec 看世界；
       训练期消融过的策略（从没见过预测或记忆列）上线时照样置零——评测里看到的行为，就是游戏里的行为。
       选中等待时标 "idle"（与脚本示范者同一口径），套在外面的驱力层（agents/drives 的 Driven）据此在它闲着时试 IDLE 驱力
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tianlong.agents.policies import Choice, Situation
from tianlong.core import Op
from tianlong.learning.rl.module import GraphPolicyNet
from tianlong.learning.rl.observation import ObsSpec, ablate, build_observation
from tianlong.learning.schema import check_schema


class LearnedPolicy:
    def __init__(self, net: GraphPolicyNet, spec: ObsSpec | None = None, ablations: tuple[str, ...] = ()) -> None:
        self.net = net.eval()
        self.spec = spec or ObsSpec()
        self.ablations = tuple(ablations)

    @classmethod
    def load(cls, path: str | Path) -> LearnedPolicy:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check_schema(ckpt, path, view="policy")
        net = GraphPolicyNet(ckpt["config"]["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        legacy = ("predictions",) if ckpt["config"].get("ablate_predictions") else ()
        return cls(net, ObsSpec(**ckpt["obs_spec"]) if "obs_spec" in ckpt else None, tuple(ckpt.get("ablate", legacy)))

    @torch.no_grad()
    def choose(self, sit: Situation) -> Choice:
        ob = build_observation(sit.beliefs, sit.now, sit.profile, sit.candidates, sit.predictions, self.spec, sit.memory)
        ablate(ob.obs, self.ablations)
        batch = {k: torch.as_tensor(np.expand_dims(v, 0)) for k, v in ob.obs.items()}
        logits, _ = self.net(batch)
        probs = torch.softmax(logits[0], -1)
        idx = int(probs.argmax())
        if idx >= len(ob.candidates):
            return _tagged(sit, 0, "策略网络无有效选择，等待")
        # 观测里的动作编号对应裁剪后保留的候选，映射回决策图给的候选集
        return _tagged(sit, sit.candidates.index(ob.candidates[idx]), f"策略网络选择（置信 {float(probs[idx]):.2f}）")


def _tagged(sit: Situation, index: int, why: str) -> Choice:
    """选中等待就标 "idle"：只是标签，下标与训练都不受影响。"""
    waits = bool(sit.candidates) and sit.candidates[index].op == Op.WAIT
    return Choice(index, why, "idle" if waits else "")
