"""
[INPUT]: 依赖 tianlong.learning 的 bundle / provenance / train / task / profile / schema，learning/rl 的 train / module / observation，
         agents/predictors 的 HeuristicPredictor
[OUTPUT]: 部署与放大训练的验收：C02（部署包记下训练时的全部语义版本与文件哈希——任一语义版本、文件内容、策略配套的预测器不符都被拒绝，
          预测器与策略跨版本混搭在打包时即被拒绝）、GNN 断点续训与不中断逐位相同且换配置的断点被拒绝、PPO 断点配置核对、
          布尔 CLI 参数严格解析、剖析工具冒烟
[POS]: tests 的部署层；与 test_learning（检查点规格/视角，C01）互补——那里管“这是不是同一套特征”，这里管“这是不是同一套语义”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("gymnasium")

from tianlong.learning.bundle import BUNDLE_FILE, file_sha256, load_bundle, write_bundle  # noqa: E402
from tianlong.learning.model import DynamicsModel  # noqa: E402
from tianlong.learning.provenance import compat_signature, run_manifest  # noqa: E402
from tianlong.learning.schema import FEATURES_VERSION, SCHEMA, StaleModel  # noqa: E402
from tianlong.learning.task import TaskConfig, arg_type  # noqa: E402
from tianlong.learning.train import TrainConfig, save_checkpoint, train_dynamics  # noqa: E402

# ============================================================
#  夹具：极小的预测器与策略检查点（结构与训练 CLI 写出的一致）
# ============================================================


def _predictor(path, hidden=16):
    cfg = TrainConfig(view="agent", hidden=hidden)
    save_checkpoint(DynamicsModel(hidden), cfg, {"manifest": run_manifest("dynamics_agent", asdict(cfg), cfg.task())},
                    path)
    return path


def _policy(path, predictor_sha=None, hidden=16):
    from tianlong.learning.rl.module import GraphPolicyNet
    from tianlong.learning.rl.observation import ObsSpec
    task = TaskConfig()
    torch.save({"state_dict": GraphPolicyNet(hidden).state_dict(), "config": {"hidden": hidden}, "schema": SCHEMA,
                "predictor_sha256": predictor_sha, "features_version": FEATURES_VERSION, "view": "policy",
                "obs_spec": asdict(ObsSpec()), "task": task.to_dict(),
                "manifest": run_manifest("policy", {"hidden": hidden}, task)}, path)
    return path


def _edit(root, fn):
    p = root / BUNDLE_FILE
    b = json.loads(p.read_text())
    fn(b)
    p.write_text(json.dumps(b))


# ============================================================
#  C02：部署包的完整兼容
# ============================================================


def test_c02_bundle_roundtrip_pairs_policy_with_the_predictor_it_was_trained_with(tmp_path):
    from tianlong.learning.predictor import GNNPredictor
    from tianlong.learning.rl.policy import LearnedPolicy
    (tmp_path / "gnn").mkdir()
    pred = _predictor(tmp_path / "gnn" / "dynamics_agent.pt")
    pol = _policy(tmp_path / "policy.pt", predictor_sha=file_sha256(pred))
    write_bundle(tmp_path, pred, pol)
    b = json.loads((tmp_path / BUNDLE_FILE).read_text())
    assert b["compat"] == compat_signature()
    assert b["files"]["predictor"]["path"] == "gnn/dynamics_agent.pt"       # 相对路径：整个目录可搬
    assert b["files"]["policy"]["predictor"] == "gnn"
    loaded = load_bundle(tmp_path)
    assert isinstance(loaded.policy, LearnedPolicy) and isinstance(loaded.predictor, GNNPredictor)
    # 只要预测器也可以；不要策略时不加载策略
    only = load_bundle(tmp_path, want_predictor=True, want_policy=False)
    assert only.policy is None and isinstance(only.predictor, GNNPredictor)


def test_c02_heuristic_trained_policy_gets_the_heuristic_predictor(tmp_path):
    from tianlong.agents.predictors import HeuristicPredictor
    pol = _policy(tmp_path / "policy.pt", predictor_sha=None)
    write_bundle(tmp_path, policy=pol)
    loaded = load_bundle(tmp_path, want_predictor=False, want_policy=True)
    assert isinstance(loaded.predictor, HeuristicPredictor)     # 训练时用启发式，上线也必须用它


@pytest.mark.parametrize("key", sorted(compat_signature()))
def test_c02_any_semantic_version_mismatch_is_refused_and_named(tmp_path, key):
    pred = _predictor(tmp_path / "dynamics_agent.pt")
    write_bundle(tmp_path, pred)
    _edit(tmp_path, lambda b: b["compat"].__setitem__(key, "old"))
    with pytest.raises(StaleModel, match=key):
        load_bundle(tmp_path)


def test_c02_missing_version_key_counts_as_mismatch(tmp_path):
    write_bundle(tmp_path, _predictor(tmp_path / "dynamics_agent.pt"))
    _edit(tmp_path, lambda b: b["compat"].pop("observation"))
    with pytest.raises(StaleModel, match="observation"):
        load_bundle(tmp_path)


def test_c02_tampered_file_or_unknown_bundle_format_is_refused(tmp_path):
    pred = _predictor(tmp_path / "dynamics_agent.pt")
    write_bundle(tmp_path, pred)
    with open(pred, "ab") as f:
        f.write(b"\0")
    with pytest.raises(StaleModel, match="哈希"):
        load_bundle(tmp_path)
    write_bundle(tmp_path, _predictor(tmp_path / "dynamics_agent.pt"))
    _edit(tmp_path, lambda b: b.__setitem__("bundle_version", "bundle-v0"))
    with pytest.raises(StaleModel, match="格式"):
        load_bundle(tmp_path)


def test_c02_policy_without_its_gnn_predictor_cannot_be_bundled(tmp_path):
    pred = _predictor(tmp_path / "a.pt")
    other = _predictor(tmp_path / "b.pt", hidden=8)          # 另一个文件：哈希不同
    pol = _policy(tmp_path / "policy.pt", predictor_sha=file_sha256(pred))
    with pytest.raises(StaleModel, match="预测器"):
        write_bundle(tmp_path, None, pol)
    with pytest.raises(StaleModel, match="预测器"):
        write_bundle(tmp_path, other, pol)


def test_c02_bundle_claiming_gnn_without_predictor_file_is_refused(tmp_path):
    pred = _predictor(tmp_path / "dynamics_agent.pt")
    write_bundle(tmp_path, pred, _policy(tmp_path / "policy.pt", predictor_sha=file_sha256(pred)))
    _edit(tmp_path, lambda b: b["files"].pop("predictor"))
    with pytest.raises(StaleModel, match="没有带上预测器"):
        load_bundle(tmp_path)


def test_c02_versions_come_from_training_time_not_bundling_time(tmp_path):
    """检查点训练时的版本与当前代码不同：打包即拒绝（不能在新代码上打包就把旧模型“洗白”）。"""
    pred = tmp_path / "dynamics_agent.pt"
    cfg = TrainConfig(view="agent", hidden=16)
    manifest = run_manifest("dynamics_agent", asdict(cfg), cfg.task())
    manifest["versions"]["reward"] = "reward-v1"
    save_checkpoint(DynamicsModel(16), cfg, {"manifest": manifest}, pred)
    with pytest.raises(StaleModel, match="reward"):
        write_bundle(tmp_path, pred)
    legacy = tmp_path / "legacy.pt"                           # 没有 manifest 的旧检查点：版本全缺，一律对不上
    save_checkpoint(DynamicsModel(16), cfg, {}, legacy)
    with pytest.raises(StaleModel):
        write_bundle(tmp_path, legacy)


def test_c02_predictor_and_policy_from_different_versions_are_refused(tmp_path):
    pred = _predictor(tmp_path / "dynamics_agent.pt")
    pol = tmp_path / "policy.pt"
    _policy(pol, predictor_sha=file_sha256(pred))
    ck = torch.load(pol, weights_only=True)
    ck["manifest"]["versions"]["candidates"] = "candidates-v1"
    torch.save(ck, pol)
    with pytest.raises(StaleModel, match="candidates"):
        write_bundle(tmp_path, pred, pol)


def test_c02_env_view_model_cannot_be_bundled_as_predictor(tmp_path):
    cfg = TrainConfig(view="env", hidden=16)
    save_checkpoint(DynamicsModel(16), cfg, {"manifest": run_manifest("dynamics_env", asdict(cfg), cfg.task())},
                    tmp_path / "dynamics_env.pt")
    with pytest.raises(StaleModel, match="角色视角"):
        write_bundle(tmp_path, tmp_path / "dynamics_env.pt")


def test_bundle_files_must_live_inside_the_bundle_dir(tmp_path):
    (tmp_path / "pkg").mkdir()
    pred = _predictor(tmp_path / "dynamics_agent.pt")
    with pytest.raises(ValueError, match="不在部署包目录"):
        write_bundle(tmp_path / "pkg", pred)


def test_cli_refuses_missing_or_stale_bundle_with_one_line(tmp_path, capsys):
    from tianlong.runtime.cli import main
    assert main(["--predictor", "gnn", "--artifacts", str(tmp_path)]) == 2
    assert "bundle.json" in capsys.readouterr().out
    write_bundle(tmp_path, _predictor(tmp_path / "dynamics_agent.pt"))
    _edit(tmp_path, lambda b: b["compat"].__setitem__("kernel", "kernel-v0"))
    assert main(["--predictor", "gnn", "--artifacts", str(tmp_path)]) == 2
    assert "kernel" in capsys.readouterr().out


# ============================================================
#  断点续训
# ============================================================

_TINY = TrainConfig(view="agent", worlds=6, steps=4, epochs=2, batch=16, hidden=16, seed=3, device="cpu")


class _Interrupt(Exception):
    pass


def _stop_at_epoch(n):
    def log(msg):
        if msg.startswith(f"[epoch {n}]"):
            raise _Interrupt
    return log


def test_gnn_resume_is_bitwise_identical_to_an_uninterrupted_run(tmp_path):
    full, m_full = train_dynamics(_TINY, log=lambda *_: None, state_path=tmp_path / "a.resume.pt")
    state = tmp_path / "b.resume.pt"
    with pytest.raises(_Interrupt):             # 第 2 轮训练完、存断点之前断掉：断点停在第 1 轮
        train_dynamics(_TINY, log=_stop_at_epoch(2), state_path=state)
    assert torch.load(state, weights_only=True)["epoch"] == 1
    resumed, m_res = train_dynamics(_TINY, log=lambda *_: None, state_path=state, resume=True)
    assert m_res["model_selection"]["resumed_from_epoch"] == 1
    assert m_res["model_selection"]["best_epoch"] == m_full["model_selection"]["best_epoch"]
    assert [h["calib_loss"] for h in m_res["model_selection"]["history"]] == \
        [h["calib_loss"] for h in m_full["model_selection"]["history"]]
    for k, v in full.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k
    assert m_res["success_brier"] == m_full["success_brier"]


def test_gnn_model_selection_uses_calibration_worlds_only(tmp_path):
    _, m = train_dynamics(_TINY, log=lambda *_: None)
    sel = m["model_selection"]
    assert "test never used" in sel["criterion"]
    losses = [h["calib_loss"] for h in sel["history"]]
    assert sel["best_calib_loss"] == pytest.approx(min(losses), abs=1e-5)       # 历史按 5 位小数记录
    assert losses[sel["best_epoch"] - 1] == min(losses)


def test_gnn_resume_refuses_a_checkpoint_from_another_config(tmp_path):
    state = tmp_path / "s.resume.pt"
    with pytest.raises(_Interrupt):
        train_dynamics(_TINY, log=_stop_at_epoch(2), state_path=state)
    with pytest.raises(ValueError, match="lr"):
        train_dynamics(replace(_TINY, lr=1e-2), log=lambda *_: None, state_path=state, resume=True)


def test_ppo_resume_checks_config_and_allows_only_harmless_changes(tmp_path):
    pytest.importorskip("ray.rllib")
    from tianlong.learning.rl.module import GraphPolicyNet
    from tianlong.learning.rl.train import RLConfig, _resume_state, train_ppo
    cfg = RLConfig(hidden=16, ppo_iterations=2)
    latest = tmp_path / "latest.pt"
    net = GraphPolicyNet(16)
    torch.save({"state_dict": net.state_dict(), "iteration": 2, "config": asdict(cfg)}, latest)
    with pytest.raises(ValueError, match="lr"):
        _resume_state(latest, replace(cfg, lr=1.0))
    _resume_state(latest, replace(cfg, ppo_iterations=5, eval_episodes=7, env_runners=3, gpus=1.0))
    # 断点已到目标轮数：直接返回断点里的权重，不启动 ray
    got = train_ppo(cfg, None, log=lambda *_: None, latest=latest, resume=True)
    for k, v in net.state_dict().items():
        assert torch.equal(v, got.state_dict()[k])


def test_resume_flag_does_not_enter_run_id():
    """续训与不中断是同一次实验：--resume 是 CLI 开关而不是配置字段。"""
    from tianlong.learning.train import TrainConfig as T
    assert "resume" not in asdict(T())
    pytest.importorskip("ray.rllib")
    from tianlong.learning.rl.train import RLConfig
    assert "resume" not in asdict(RLConfig())


# ============================================================
#  CLI 与剖析
# ============================================================


@pytest.mark.parametrize("raw,want", [("true", True), ("False", False), ("1", True), ("no", False), ("YES", True)])
def test_bool_cli_args_are_parsed_strictly(raw, want):
    assert arg_type(False)(raw) is want


def test_bool_cli_args_reject_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        arg_type(True)("flase")
    assert arg_type(3)("4") == 4 and arg_type(0.5)("0.25") == 0.25 and arg_type("x")("y") == "y"


def test_profile_reports_every_stage_and_the_device():
    from tianlong.learning.profile import profile
    rep = profile(worlds=1, steps=2, device="cpu")
    assert rep["device"] == "cpu"
    assert {"scenario", "candidates", "predict_heuristic", "branch_value_definition", "branch_value_shared", "kernel_step",
            "belief_revise", "belief_view+featurize", "build_observation", "predict_gnn_cpu",
            "policy_forward_cpu"} <= set(rep["stages"])
    assert all(v["calls"] > 0 and v["mean_ms"] >= 0 for v in rep["stages"].values())
