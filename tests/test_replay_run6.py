"""
[INPUT]: 依赖 scripts/replay_drops.py（按文件加载，连带同目录的 bench_online 与 bench_gm；load_inputs 与命令行 --inputs 同一个读法），
         tests/data/playthrough_duanyu.json
         （run6 那一局玩家的输入：旧版评测整局游玩）与 tests/data/run6 的录制答案（playthrough.answers.json：叙述按调用序号；
         interp_answers.json：解释器按玩家原文）
[OUTPUT]: run6 重放验收：在当前代码上重放 run6 的整局游玩，37 次叙述调用恰好用完 37 条答案、走到澜沧江畔的结局，
          叙述闸门恰好丢掉 RUN6_DROPS 这五句（M1 的误杀基线：月下舞剑的“长剑”、蒲团绣字、“嗒”、打定主意捱到天黑、回忆里的玉像）；
          命令行的 --inputs 确实取代了探针文件里的整局游玩（不重放、不慢）
[POS]: tests 的闸门精度基线（重放为 slow：默认不跑）。M1 修误杀时改的是这里的期望——每修掉一类误杀，就从 RUN6_DROPS 里划掉对应的句子，
       直到出口条件“重放 run6 误杀 0 句”；丢句变多即是回归。出口条件的命令（replay_drops --inputs 指向同一份输入文件）
       与本测试重放的是同一份输入
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("data") / "run6"
INPUTS = Path(__file__).with_name("data") / "playthrough_duanyu.json"

# run6 在当前代码上被叙述闸门丢掉的句子（按交付先后）：全是误杀，M1 逐条修掉
RUN6_DROPS = [
    "你打定主意要在这湖边捱到天黑，便抱膝坐着，看湖面上的光一寸寸挪动。",                        # puppet：复述自己的意图
    "那面玉璧上，忽然隐隐现出一个人影，衣袂飘飘，手中似握着长剑，竟像有仙人在壁上翩然舞剑。",    # entity：比喻里的“长剑”
    "不知哪根藤萝上的露水，“嗒”的一声，滴在了你手背上。",                                        # puppet：拟声不是话
    "蒲团面子上绣着两行细字，年深日久，丝线已有些褪色，凑近了才辨认得出：“既入此室，叩首千遍，自有所得。”"
    "字下那道朽裂的口子不大，两卷帛书就嵌在里头，一卷“北冥神功”，一卷“凌波微步”，都还好端端地躺在原处，"
    "帛面上落着薄薄一层灰。",                                                                   # puppet：绣着的字
    "剑湖宫里的喝彩与冷笑，崖下的月色与玉像，此刻都远了，只剩这一江流水，浩浩荡荡，不舍昼夜地向南奔去。",  # entity：回忆里的玉像
]


def _replay_drops():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    spec = importlib.util.spec_from_file_location("replay_drops", ROOT / "scripts" / "replay_drops.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_drops"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_replay_run6_pins_the_gate_drops():
    rd = _replay_drops()
    given = json.loads((DATA / "playthrough.answers.json").read_text("utf-8"))
    interp = rd.bench_online.load_interp(DATA / "interp_answers.json")
    inputs = rd.load_inputs(INPUTS)                             # 与评测探针文件脱钩：探针改版不动基线
    rec, records = rd.replay(given, interp, probes={"playthrough": inputs})
    assert rec["calls"] == len(given) == 37, "重放与录制一一对应：每条答案恰好答一次叙述调用"
    assert len(rec["playthrough"]) == 35 and rec["ending"] and rec["final_place"] == "澜沧江畔", rec["errors"]
    assert not rec["errors"]
    assert not [t["error"] for t in rec["playthrough"] if t["error"]]
    assert rd.dropped(records) == RUN6_DROPS


def test_inputs_option_replaces_the_probe_playthrough(monkeypatch, tmp_path):
    """命令行 --inputs：整局游玩的输入取自给定文件（与本文件的重放同一份），坏文件照 argparse 的口径报错退出。"""
    rd = _replay_drops()
    seen = {}

    def fake(given, interp, key, seed, probes):
        seen.update(key=key, probes=probes)
        return {"calls": len(given)}, []

    monkeypatch.setattr(rd, "replay", fake)
    base = [str(DATA), "--answers", str(DATA / "playthrough.answers.json")]
    assert rd.main([*base, "--inputs", str(INPUTS)]) == 0
    assert seen["key"] == "playthrough" and seen["probes"]["playthrough"] == json.loads(INPUTS.read_text("utf-8"))
    bad = tmp_path / "bad.json"
    for text in ('{"playthrough": []}', "[]", '["环顾四周", 3]', "not json"):
        bad.write_text(text, "utf-8")
        with pytest.raises(SystemExit):
            rd.main([*base, "--inputs", str(bad)])
