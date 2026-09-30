"""
[INPUT]: 依赖 tianlong.language.lead 的 lead_line，tianlong.runtime.session 的 GameSession，tianlong.scenarios 的 build_wuliang
[OUTPUT]: 先声验收：玩家自己这一步写成确定的一两句人话——拿到了、到了哪、说了什么、失败说明原因、动手写出后果；模型迟迟不出字时先声到点顶上；
          只说玩家自己的行动，旁人的言行一概不抢（留给声音模型）；干等不抢先；同一回合同一句（确定性）
[POS]: tests 的先声：证伪“第一句要等模型”“先声替 NPC 说话”“失败被写成成功”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.core import Modality, Outcome  # noqa: E402
from tianlong.language.lead import lead_line  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402


@pytest.fixture()
def play():
    s = GameSession(build_wuliang(7))
    s.intro()
    seen: dict[str, tuple] = {}
    real = s.narrator.narrate_scene

    def spy(viewer, percepts, *a, **k):
        seen["p"] = tuple(percepts)
        return real(viewer, percepts, *a, **k)

    s.narrator.narrate_scene = spy

    def turn(text: str) -> tuple[str, tuple]:
        s.turn(text)
        percepts = seen.pop("p", ())
        return lead_line(percepts, s.beliefs(s.player).entities, s.player), percepts

    return s, turn


def test_success_speech_and_failure_are_told_plainly(play):
    s, turn = play
    lead, _ = turn("拿起兵器架上的长剑")
    assert lead.startswith("你") and "长剑" in lead and lead.endswith("。")
    lead, _ = turn("对钟灵说：多谢姑娘")
    assert lead == "你对钟灵道：“多谢姑娘。”", "玩家没打标点，只补句号不动字"
    lead, percepts = turn("研读易经")
    mine = [p for p in percepts if p.modality == Modality.SELF and p.event and p.event.actor == s.player]
    assert mine and mine[0].event.outcome == Outcome.FAILURE
    assert "易经" in lead and "看不出什么门道" in lead           # 失败必说原因，不写成成功
    lead, _ = turn("去后院")
    assert "后院" in lead


def test_lead_never_speaks_for_others_and_waiting_does_not_jump_ahead(play):
    s, turn = play
    lead, percepts = turn("等一会儿")
    assert lead == ""
    others = {p.event.actor for p in percepts if p.event and p.event.actor not in (None, s.player)}
    names = s.beliefs(s.player).entities
    lead, percepts = turn("向钟灵赔罪")
    for who in {p.event.actor for p in percepts if p.event and p.event.actor not in (None, s.player)} | others:
        sk = names.get(who)
        assert sk is None or not lead.startswith(sk.name)      # 旁人的言行留给声音模型
    assert lead.startswith("你")


def test_same_percepts_same_lead(play):
    s, turn = play
    lead, percepts = turn("环顾四周")
    names = s.beliefs(s.player).entities
    assert lead and lead_line(percepts, names, s.player) == lead


def test_first_text_does_not_wait_for_a_slow_model():
    """接进会话：模型慢吞吞（首字 1.5 秒），先声到点（这里调成 0.3 秒）就顶上，玩家不必干等；模型的文字随后照常交付。"""
    from tianlong.language.llm import ScriptedLLM

    voice = ScriptedLLM(lambda p, s, sc: "兵器架旁的弟子们面面相觑。", first_delay=1.5)
    s = GameSession(build_wuliang(7), llm=voice, fast_llm=ScriptedLLM(lambda p, s, sc: ""))
    s.narrator.lead_after = 0.3
    s.intro()
    got: list[str] = []
    r = s.turn("拿起兵器架上的长剑", on_text=got.append)
    assert got and got[0].startswith("你") and "长剑" in got[0]
    assert r.first_text_ms is not None and r.first_text_ms < 1000, r.first_text_ms
    assert "面面相觑" in r.narration


def test_kowtow_reads_as_a_kowtow_and_discoveries_are_grouped():
    """原著路线的伏地叩拜：先声写的是磕头而不是“四下打量”；翻出的两卷帛书按藏处归成一句，再翻一遍不算又“发现”。"""
    from tianlong.language.lead import _BOW
    from tianlong.language.templates import consequences

    s = GameSession(build_wuliang(7))
    s.intro()
    for text in ("去后院", "去后山", "去后山崖顶", "跳下断崖", "等到天黑", "等到天黑", "等到天黑", "等到天黑",
                 "等到天黑", "等到天黑", "等到天黑", "等到天黑", "等到天黑", "等到天黑", "等到天黑"):
        if s.beliefs(s.player).location_of(s.player) == "jianhu" and "入夜" in s.clock() or s.turn(text) is None:
            break
    seen: dict[str, tuple] = {}
    real = s.narrator.narrate_scene

    def spy(viewer, percepts, *a, **k):
        seen["p"] = tuple(percepts)
        return real(viewer, percepts, *a, **k)

    s.narrator.narrate_scene = spy
    for text in ("查看无量玉璧", "钻进石缝", "进石门", "向玉像磕头"):
        s.turn(text)
    names = s.beliefs(s.player).entities
    lead = lead_line(seen["p"], names, s.player)
    if not lead:
        pytest.skip("本局没走到琅嬛福地（原著路线由 test_wuliang 另行验收）")
    assert any(b[:-1] in lead for b in _BOW), lead
    found = [c for p in seen["p"] for c in consequences(p, names, s.player, "你") if c.startswith("发现")]
    assert all(c.count("发现") == 1 for c in found) and len(found) <= 1, found
    again = s.turn("查看蒲团")
    assert "帛卷" in again.narration and "发现" not in again.narration, "已知下落的东西再翻出来，不算又“发现”一回"
