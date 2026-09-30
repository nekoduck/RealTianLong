"""
[INPUT]: 依赖 scripts/bench_player.py（按文件加载，连带 bench_gm / bench_rival / bench_metrics），bench_rival 的脚本模型（scripted_respond /
         scripted_gm）作各角色的替身，tianlong.language.interpret 的 _FIELDS（解释器 schema 的字段）
[OUTPUT]: 玩家代理模式（plan §8.4）的全流程：ScriptedLLM 替身逐次作答（玩家、叙述、解释、对照组），A（本引擎，回合数封顶）与 B
          （纯模型主持人，回复里出现结局标题即停）各走一局；玩家的提示词只有本方看得见的正文与固定的玩家设定；答案录进 DIR/player/<side>/，
          step 按录下的答案从头重放出同一局并写下 game.json；缺一条玩家答案就停在 player 上、交出完整提示词，answer 以 stdin 补上后走完；
          超过 30 字的玩家输入拒收；panel 按种子打乱 A/B 并给出两边的结局与 A 的 B1、E1
[POS]: tests 的评测工具验收（玩家代理模式）；只用替身、不接任何模型；起会话，缺 LangGraph / Qdrant 时跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")
_spec = importlib.util.spec_from_file_location("bench_player", ROOT / "scripts" / "bench_player.py")
bp = importlib.util.module_from_spec(_spec)
sys.modules["bench_player"] = bp
_spec.loader.exec_module(bp)

from bench_rival import scripted_gm, scripted_respond  # noqa: E402

from tianlong.language.interpret import _FIELDS  # noqa: E402

LINES = ("环顾四周", "问马五爷这是怎么回事", "等一会儿", "对龚光杰说：爷息怒")


class Halt(BaseException):              # 叙述调用在读流线程里：停下须越过会话与评测的 except Exception
    pass


def _answer(role: str, n: int, system: str, prompt: str) -> str:
    """替身：玩家按 LINES 轮着说，叙述与解释用脚本模型，对照组第 3 次回复里带上结局标题。"""
    if role == "player":
        assert system == bp.PLAYER_SYSTEM and prompt.startswith("以下是你到目前为止看到的全部正文")
        return LINES[n % len(LINES)]
    if role == "interp":
        return scripted_respond(prompt, system, {"properties": dict.fromkeys(_FIELDS)})
    if role == "rival":
        return scripted_gm(prompt, system, None) + ("东方发白。第一幕终 · 天亮了" if n == 2 else "")
    return scripted_respond(prompt, system, None)


def _stand_in(d: Path, side: str):
    """像 answer 那样把替身的答案录进文件（重放时逐次读回）。"""
    def ask(role, n, system, prompt):
        text = _answer(role, n, system, prompt)
        bp.record(d, side, role, text)
        return text
    return ask


def test_player_mode_full_flow(tmp_path, monkeypatch):
    live = {side: bp.game(side, "wuliang", 7, _stand_in(tmp_path, side), turns=3) for side in ("A", "B")}
    a, b = live["A"], live["B"]
    assert len(a["turns"]) == 3 and a["ending"] is None and a["calls"]["player"] == 3 and a["calls"]["narrator"] >= 4
    assert len(b["turns"]) == 3 and b["ending"] == "dawn" and b["calls"] == {"player": 3, "rival": 3}
    assert [t["text"] for t in a["turns"]] == list(LINES[:3]) and a["opening"] and "阿顺" in b["opening"]
    # ---- step：按录下的答案从头重放出同一局（缺答即停：停下时写 pending.json、打印提示词、退出进程——这里把退出换成异常） ----
    monkeypatch.setattr(bp.os, "_exit", lambda code: (_ for _ in ()).throw(Halt()))
    for side in ("A", "B"):
        assert bp.main(["step", str(tmp_path), side, "--turns", "3"]) == 0
        replay = json.loads((tmp_path / "player" / side / "game.json").read_text("utf-8"))
        assert [t["narration"] for t in replay["turns"]] == [t["narration"] for t in live[side]["turns"]]
    # ---- 多走一回合：先停在 player 上（完整提示词只有本方正文），answer 以 stdin 补上，逐个角色答到 DONE ----
    roles = []
    for _ in range(12):
        try:
            bp.main(["step", str(tmp_path), "A", "--turns", "4"])
            break
        except Halt:
            pending = json.loads((tmp_path / "player" / "A" / "pending.json").read_text("utf-8"))
            if not roles:
                assert (pending["role"], pending["n"]) == ("player", 3) and pending["system"] == bp.PLAYER_SYSTEM
                assert f"你：{LINES[2]}" in pending["prompt"] and a["turns"][2]["narration"] in pending["prompt"]
            roles.append(pending["role"])
            monkeypatch.setattr(sys, "stdin", io.StringIO(_answer(**{k: pending[k] for k in ("role", "n", "system", "prompt")})))
            assert bp.main(["answer", str(tmp_path), "A"]) == 0
    else:
        pytest.fail(f"12 次作答还没走完：{roles}")
    done = json.loads((tmp_path / "player" / "A" / "game.json").read_text("utf-8"))
    assert roles[0] == "player" and "narrator" in roles and [t["text"] for t in done["turns"]] == list(LINES)
    with pytest.raises(ValueError):
        bp.record(tmp_path, "A", "player", "我" * (bp.MAX_CHARS + 1))
    summary = bp.panel(tmp_path, 7)
    assert set(summary["panel_key"].values()) == {"engine", "baseline"} and summary["baseline"]["ending"] == "dawn"
    assert summary["engine"]["turns"] == 4 and summary["engine"]["B1"].endswith("/11")
    assert (tmp_path / "player" / "panel.json").exists()
