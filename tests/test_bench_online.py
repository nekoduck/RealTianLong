"""
[INPUT]: 依赖 scripts/bench_online.py（按文件加载，连带 bench_gm / bench_rival），scripts/bench_probes.json 与 bench_probes_duanyu.json，
         tianlong.language.llm 的 ScriptedLLM
[OUTPUT]: 在线代理协议（plan §8.4）的验收：会话键（本引擎带不进盲评的整局，对照组加 B: 前缀、没有额外整局）、运行目录按世界与种子分开
          而旧版种子 7 仍是 run1–6 的布局、--world 与探针文件的 variant 不符即报错；对照组逐次作答——rival_key 停在第一个没答的调用上交出
          圣经与“完整对话 + 新输入”，答全了就回放出同样的对话，step 在命令行上写下 DONE；assemble 按新探针拼出报告（本引擎的 extras、
          对照组取 B:* 的回复、面板与新指标）；对照组的回复绝不借别的运行的录制（普通人版或 B:* 跑了一半即报出没跑完的键，
          只有旧版种子 7 一个 B:* 都没跑时才取 rival_answers.json）
[POS]: tests 的在线代理评测工具验收；只跑离线、快速的路径（真代理作答由外部编排）。起会话的用例缺 LangGraph / Qdrant 时跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_online", ROOT / "scripts" / "bench_online.py")
online = importlib.util.module_from_spec(_spec)
sys.modules["bench_online"] = online
_spec.loader.exec_module(online)


class Halt(BaseException):            # 对照组的 turn 会吞下 Exception（照记失败）：停下须越过它
    pass


def _raise(pending):
    raise Halt(pending)


def _tiny(tmp_path: Path) -> Path:
    full = online.load_probes(path=online.PROBE_FILES["wuliang"])
    tiny = {"variant": "wuliang", "playthrough": full["playthrough"][:2], "playthrough_hall": full["playthrough_hall"][:2],
            **{g: full[g][:1] for g in online.GROUPS}}
    path = tmp_path / "probes.json"
    path.write_text(json.dumps(tiny, ensure_ascii=False), "utf-8")
    return path


def test_keys_and_layout(tmp_path):
    probes = online.load_probes(3)
    keys, rival = online.session_keys(probes), online.session_keys(probes, rival=True)
    assert keys[:3] == ["playthrough", "playthrough_hall", "playthrough_flee"] and len(keys) == 3 + 20 + 12 + 3
    assert rival[0] == "B:playthrough" and not any("playthrough_" in k for k in rival) and len(rival) == len(keys) - 2
    assert online.run_dir(tmp_path, "wuliang-duanyu", 7) == tmp_path                   # run1–6 的布局原样可读
    assert online.online_dir(tmp_path, "wuliang", 11) == tmp_path / "wuliang-s11" / "online"
    assert online._file(tmp_path, "B:gaslight:g01", "done.json").name == "B__gaslight__g01.done.json"
    legacy = online.load_probes(path=online.PROBE_FILES["wuliang-duanyu"])
    assert online.session_keys(legacy)[:2] == ["playthrough", "gaslight:g01"]              # 旧版没有额外整局
    with pytest.raises(SystemExit):
        online.main(["keys", str(tmp_path), "--world", "wuliang", "--probes", str(online.PROBE_FILES["wuliang-duanyu"])])


def test_rival_answers_one_call_at_a_time(tmp_path):
    """对照组的每回合由隔离代理看着“圣经 + 完整对话 + 新输入”作答：停在第一个没答的调用上，答全了就回放出同样的对话。"""
    probes = online.load_probes(1, _tiny(tmp_path))
    with pytest.raises(Halt) as got:
        online.rival_key("B:playthrough", ["你环顾四周，满堂寂静。"], probes, 11, on_missing=_raise)
    pending = got.value.args[0]
    assert pending["key"] == "B:playthrough" and pending["call"] == 1 and "玩家扮演阿顺" in pending["system"]
    assert "主持人：你环顾四周，满堂寂静。" in pending["prompt"]
    assert pending["prompt"].endswith(f"玩家：{probes['playthrough'][1]}\n主持人：")
    rec = online.rival_key("B:playthrough", ["一", "二"], probes, 11, on_missing=_raise)
    assert rec == {"replies": ["一", "二"], "calls": 2}
    g = probes["sycophancy"][0]
    assert online.rival_key(f"B:sycophancy:{g['id']}", ["甲"], probes, 11)["calls"] == 1 + len(g.get("setup", ()))
    od = online.online_dir(tmp_path, "wuliang", 11)
    od.mkdir(parents=True)
    online._file(od, "B:playthrough", "answers.json").write_text(json.dumps(["一", "二"]), "utf-8")
    argv = ["step", str(tmp_path), "B:playthrough", "--seed", "11", "--probes", str(_tiny(tmp_path))]
    assert online.main(argv) == 0
    assert json.loads(online._file(od, "B:playthrough", "done.json").read_text("utf-8"))["replies"] == ["一", "二"]


def test_assemble_with_rival_sessions(tmp_path):
    """全部 DONE 之后 assemble 按新探针拼报告：本引擎的额外整局进 extras，对照组的回复取 B:* 的 DONE，新指标两边都在。"""
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    probes = online.load_probes(1, _tiny(tmp_path))
    od = online.online_dir(tmp_path, "wuliang", 7)
    od.mkdir(parents=True)
    for key in online.session_keys(probes):
        rec = online.run_key(key, [], {}, probes, 7, on_missing=lambda p: "")      # 叙述答空：模板照样交付
        online._file(od, key, "done.json").write_text(json.dumps(rec, ensure_ascii=False, default=str), "utf-8")
    for key in online.session_keys(probes, rival=True):
        rec = online.rival_key(key, [], probes, 7, on_missing=lambda p: f"主持人答第 {p['call']} 次。")
        online._file(od, key, "done.json").write_text(json.dumps(rec, ensure_ascii=False), "utf-8")
    result = online.assemble(tmp_path, probes, 7)
    assert set(result["engine"]["extras"]) == {"playthrough_hall"} and len(result["engine"]["probes"]) == 3
    assert result["baseline"]["playthrough"][1]["narration"] == "主持人答第 1 次。"
    rd = online.run_dir(tmp_path, "wuliang", 7)
    assert (rd / "panel.json").exists() and (rd / "report" / "report.md").exists()
    ids = {m["id"] for m in result["metrics"]["engine"]} & {m["id"] for m in result["metrics"]["baseline"]}
    assert {"B1", "E1", "NAME", "SEAM", "FPD", "R5", "F3", "TOK", "END"} <= ids


def test_rival_answers_never_borrow_another_runs_recording(tmp_path):
    """共用一个 DIR 时：普通人版的对照组没跑完就报出没跑完的键，绝不拿 DIR/rival_answers.json（run6 旧版的录制）顶替；
    旧版种子 7 也只在一个 B:* 都没跑时才取录制；B:* 全部 DONE 就用它们。"""
    probes = online.load_probes(1, _tiny(tmp_path))
    (tmp_path / "rival_answers.json").write_text(json.dumps({"playthrough": ["（run6 旧版：段誉的回复）"]}), "utf-8")
    rd = online.run_dir(tmp_path, "wuliang", 11)
    od = rd / "online"
    od.mkdir(parents=True)
    keys = online.session_keys(probes, rival=True)
    online._file(od, keys[0], "done.json").write_text(json.dumps({"replies": ["甲", "乙"]}), "utf-8")
    with pytest.raises(SystemExit, match="没跑完"):
        online._rival_answers(rd, od, probes, legacy=False)
    with pytest.raises(SystemExit, match="没跑完"):                    # 旧版也一样：跑了一半不混用
        online._rival_answers(tmp_path, od, probes, legacy=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):                                     # 普通人版一个都没跑：同样不借录制
        online._rival_answers(tmp_path, empty, probes, legacy=False)
    assert online._rival_answers(tmp_path, empty, probes, legacy=True) == {"playthrough": ["（run6 旧版：段誉的回复）"]}
    for k in keys[1:]:
        online._file(od, k, "done.json").write_text(json.dumps({"replies": [k]}), "utf-8")
    got = online._rival_answers(rd, od, probes, legacy=False)
    assert got["playthrough"] == ["甲", "乙"] and set(got) == {k.removeprefix(online.RIVAL) for k in keys}
