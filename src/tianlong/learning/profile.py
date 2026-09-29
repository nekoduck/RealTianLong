"""
[INPUT]: 依赖 time，learning/task 的 TaskConfig，kernel 的 Kernel，cognition 的 BeliefStore / candidates / belief_view，core 的 Fact / Profile，
         agents 的 HeuristicPredictor / BranchValuer / branch_value / direct_effects，
         learning 的 featurize / rl.observation（torch 可用时另测 GNN 预测与策略前向）
[OUTPUT]: 对外提供 profile()（逐阶段平均耗时，毫秒）、main()（python -m tianlong.learning.profile --worlds N --device auto）
[POS]: learning 的剖析工具：先量再改。GPU 只帮得上学习器与网络前向；世界生成、内核结算、认知折叠、图构造、候选生成与编码
       都在 Python/CPU 上——放大训练之前先看清时间花在哪一段，而不是假设“上了 GPU 就快了”。报告写明设备。
       假想分支的微基准：同一批候选的直接效果，逐候选走定义（branch_value_definition，每个候选完整假想一次）与
       一次决策共用的快路径（branch_value_shared，BranchValuer）各计一次时——两者逐位相同由 tests/test_predictors 保证
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from tianlong.agents.predictors import BranchValuer, HeuristicPredictor, branch_value, direct_effects
from tianlong.cognition import BeliefStore, belief_view, candidates
from tianlong.core import Fact, make_id
from tianlong.core.profiles import Profile
from tianlong.kernel import Kernel
from tianlong.learning.task import TaskConfig


class _Timer:
    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def time(self, name: str, fn, *a, **kw):
        t = time.perf_counter()
        out = fn(*a, **kw)
        self.total[name] += time.perf_counter() - t
        self.count[name] += 1
        return out

    def report(self) -> dict[str, dict]:
        return {k: {"mean_ms": round(1000 * v / self.count[k], 3), "calls": self.count[k], "total_s": round(v, 3)}
                for k, v in sorted(self.total.items(), key=lambda kv: -kv[1])}


def _per_candidate(store: BeliefStore, now: int, prof: Profile, effects: list[list[Fact]]) -> list[float]:
    return [branch_value(store, now, prof, f) for f in effects]


def _shared(store: BeliefStore, now: int, prof: Profile, effects: list[list[Fact]]) -> list[float]:
    value = BranchValuer(store, now, prof)
    return [value(f) for f in effects]


def profile(worlds: int = 20, steps: int = 12, device: str = "auto", task: TaskConfig | None = None) -> dict:
    task = task or TaskConfig()
    t = _Timer()
    kernel, heur = Kernel(), HeuristicPredictor()
    gnn = net = None
    dev = "cpu"
    try:
        import torch
        from torch_geometric.data import Batch  # noqa: F401

        from tianlong.learning.featurize import featurize
        from tianlong.learning.model import DynamicsModel
        from tianlong.learning.predictor import GNNPredictor
        from tianlong.learning.rl.module import GraphPolicyNet
        from tianlong.learning.rl.observation import ObsSpec, build_observation
        dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        gnn = GNNPredictor(DynamicsModel(64))
        net = GraphPolicyNet(64).to(dev).eval()
    except ImportError:
        featurize = None  # type: ignore[assignment]
    for w in range(worlds):
        sc = t.time("scenario", task.scenario, 7_000 + w)
        state = sc.state
        stores = {a: BeliefStore(a).revise_all(sc.priors.get(a, ()))[0] for a in sc.profiles}
        for _ in range(steps):
            intents = []
            for a, prof in sc.profiles.items():
                cands = t.time("candidates", candidates, stores[a], prof.interests(), 48)
                preds = t.time("predict_heuristic", heur.predict, stores[a], state.clock, cands, prof.interests(),
                               profile=prof)
                effects = [direct_effects(stores[a], c) for c in cands]
                t.time("branch_value_definition", _per_candidate, stores[a], state.clock, prof, effects)
                t.time("branch_value_shared", _shared, stores[a], state.clock, prof, effects)
                if featurize is not None:
                    t.time("belief_view+featurize", lambda s=stores[a], c=state.clock: featurize(belief_view(s, c)))
                    ob = t.time("build_observation", build_observation, stores[a], state.clock, prof, cands, preds,
                                ObsSpec())
                    t.time("predict_gnn_cpu", gnn.predict, stores[a], state.clock, cands, prof.interests(),
                           profile=prof)
                    with torch.no_grad():
                        batch = {k: torch.as_tensor(v[None]).to(dev) for k, v in ob.obs.items()}
                        t.time(f"policy_forward_{dev}", net, batch)
                intents.append(cands[0].to_intent(make_id("p", w, a, state.version), a, state.version))
            r = t.time("kernel_step", kernel.step, state, intents)
            for o in r.observations:
                stores[o.observer] = t.time("belief_revise", lambda s=stores[o.observer], p=o.percept: s.revise(p)[0])
            state = r.state
    return {"device": dev, "worlds": worlds, "steps": steps, "task": task.to_dict(), "stages": t.report()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.profile", description="逐阶段剖析数据生成与决策的耗时")
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = profile(a.worlds, a.steps, a.device)
    text = json.dumps(rep, indent=2, ensure_ascii=False)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
