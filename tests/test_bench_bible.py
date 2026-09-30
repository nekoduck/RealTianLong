"""
[INPUT]: 依赖 scripts/bench_rival.py（按文件加载，不连带 bench_gm）的 world_bible，tianlong.runtime.gm 的 PLAYER_GOALS，
         tianlong.scenarios 的 build_wuliang
[OUTPUT]: 对照组世界圣经的玩家行验收：玩家的目标用玩家自己的口吻、逐条取自主持层场外问答的那张 PLAYER_GOALS（两处同一口径），
          不含 ESCAPE 的 NPC 语义“撞见外人便灭口”，标签照旧是“目标：”（对照组的系统提示除去灭口别无改动）
[POS]: tests 的评测对照组守护（设计 M0 test_bible_player_row）；单独成文件，已有的 test_bench_gm 一行不改
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tianlong.runtime.gm import PLAYER_GOALS
from tianlong.scenarios import build_wuliang

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_rival", ROOT / "scripts" / "bench_rival.py")
rival = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rival)


def test_bible_player_row():
    """玩家的目标用玩家自己的口吻：ESCAPE 的 NPC 语义“撞见外人便灭口”不能写成玩家的打算。"""
    sc = build_wuliang()
    rows = [r for r in rival.world_bible(sc).splitlines() if r.startswith("- 段誉（玩家")]
    assert len(rows) == 1, rows
    row = rows[0]
    assert "灭口" not in row and "设法去往澜沧江畔" in row and "护着钟灵" in row, row
    names = {e: sc.state.entity(e).name for e in ("lancang", "zhongling")}
    phrases = [PLAYER_GOALS[g.kind].format(item="", home=names.get(g.home, ""), recipient="", person=names.get(g.person, ""))
               for g in sc.profiles[sc.player].goals]
    assert "目标：" + "；".join(phrases) in row, row          # 逐条取自场外问答的那张表，标签照旧
