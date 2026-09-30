"""
[INPUT]: 依赖 scripts/bench_rival.py（按文件加载，不连带 bench_gm）的 world_bible / pair_note / pairwise_prompt，
         tianlong.runtime.gm 的 PLAYER_GOALS，tianlong.core 的 digest，tianlong.scenarios 的 build_wuliang / build_wuliang_commoner
[OUTPUT]: 对照组世界圣经验收：玩家行的目标用玩家自己的口吻、逐条取自主持层场外问答的那张 PLAYER_GOALS（两处同一口径），
          不含 ESCAPE 的 NPC 语义“撞见外人便灭口”，标签照旧是“目标：”；旧版圣经全文逐字不变（摘要钉住）、盲评不加须知；
          普通人版（M4，plan §8.2）圣经含“玩家扮演阿顺”（普通人、不会武功）、每条 Drive.gloss、每句驱力台词、
          由时刻与驱力时间窗推出的时间表（幽会、搜人、月出、换岗、天亮）、外貌称呼规则、三种结局及变体，玩家行没有“灭口”，
          盲评须知写明“玩家不是段誉”不算错
[POS]: tests 的评测对照组守护（设计 M0 test_bible_player_row 与 M4 的 world_bible(commoner)）；单独成文件
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tianlong.core import digest
from tianlong.runtime.gm import PLAYER_GOALS
from tianlong.scenarios import build_wuliang, build_wuliang_commoner

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


# 旧版圣经的全文摘要：M4 起圣经按场景分出普通人版的附加段，旧版（段誉作玩家）逐字不变（run1–6 的对照组看到的就是这一份）
LEGACY_BIBLE = "6af2bb2e230a13c6"


def test_legacy_bible_unchanged():
    assert digest(rival.world_bible(build_wuliang(7)))[:16] == LEGACY_BIBLE
    assert rival.pair_note(build_wuliang(7)) == ""                   # 旧版盲评提示词不加须知


def test_bible_commoner():
    """普通人版圣经由场景生成（plan §8.2）：玩家扮演阿顺（普通人、不会武功），每条 Drive.gloss 与每句驱力台词都在，
    时间表（幽会、搜人、月出、换岗、天亮）从时刻与驱力时间窗推出，外貌称呼规则，三种结局及变体；玩家行里没有“灭口”。"""
    sc = build_wuliang_commoner(7)
    bible = rival.world_bible(sc)
    assert "玩家扮演阿顺（普通人，" in bible and "不会武功" in bible.splitlines()[0]
    for a, drives in sc.drives.items():
        for d in drives:
            assert d.gloss in bible, (a, d.key)
            for line in (d.line, *d.lines):
                assert not line or line in bible, (a, d.key, line)
    table = bible.split("【时间表】", 1)[1].split("\n\n", 1)[0]
    for t in ("18:05–19:20", "18:40 起", "19:40：月出", "20:00–22:00", "次日 05:00：天亮"):
        assert t in table, t
    assert all(f"“{e}”" in bible for e in sc.epithets.values()) and "外貌称呼" in bible
    for e in sc.endings:
        assert e.title in bible and all(v.label in bible for v in e.variants)
    rows = [r for r in bible.splitlines() if r.startswith("- 阿顺（玩家")]
    assert len(rows) == 1 and "灭口" not in rows[0] and "目标：" in rows[0], rows
    assert "不是段誉" in rival.pair_note(sc) and "“玩家不是段誉”不算错" in rival.pairwise_prompt([], [], rival.pair_note(sc))
