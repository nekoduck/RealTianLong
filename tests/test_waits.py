"""
[INPUT]: 依赖 language/waits 的 wait_length，runtime/session 的 GameSession（模板模式），scenarios 的 build_wuliang_commoner / build_wuliang，
         scenarios/tianlong/drives_c 的 MOONRISE，core 的 at，tests/test_commoner 的 stage（把人挪到某处、把时钟拨到某刻重新建档）
[OUTPUT]: 等待验收：“等到月亮出来/月上”认作月出、“等到天亮/天明/等下去”认作天亮、“天黑/月亮”照旧入夜；
          19:30 在剑湖畔“等到月亮出来”恰停在 MOONRISE 且认得出月下玉璧的看点、正文交付的正是月下那段描写（runtime/staging.lore_at）；“等下去”在玩家感知到看点的那个 tick 停下，
          一次性的看点只让等待停一回；场景没有月出这个时刻（旧版）时，月亮照旧按入夜算
[POS]: tests 的等待层：时刻是场景的时钟事实（Scenario.moments），等待只读玩家自己的感知决定何时停
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.core import at
from tianlong.language.waits import wait_length
from tianlong.scenarios.tianlong.drives_c import MOONRISE


def test_wait_length_knows_moonrise_dawn_and_nightfall():
    assert wait_length("一直等到月亮出来") == (1, "moon") and wait_length("等到月上") == (1, "moon")
    assert wait_length("等到天亮") == (1, "dawn") and wait_length("等到天明再说") == (1, "dawn")
    assert wait_length("就这么等下去") == (1, "dawn")
    assert wait_length("等到天黑") == (1, "night") and wait_length("不理会旁人，继续等到天黑") == (1, "night")
    assert wait_length("看着月亮发呆") == (1, "night"), "只说月亮、没说等到它出来：照旧按入夜（旧版的说法不变）"
    assert wait_length("等一炷香") == (30, None) and wait_length("等5分钟") == (5, None)


def _at_lake(clock: int):
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    from tianlong.scenarios import build_wuliang_commoner

    from .test_commoner import stage

    return GameSession(stage(build_wuliang_commoner(7), {"ashun": "jianhu"}, clock), pipeline=False)


def test_waiting_for_the_moon_stops_at_moonrise():
    s = _at_lake(at(1, 19, 30))
    r = s.turn("等到月亮出来")
    assert s.authority.head().clock == MOONRISE, "月出是场景的时钟事实：停在 19:40"
    assert r.parsed.until == "moon" and "moon" in r.beats


def test_moonrise_shows_the_moonlit_wall():
    """月出那一回合交付的正是月下玉璧（舞剑人影）的描写，不是入夜时“看不出什么异样”的那段：看点与正文不相矛盾。"""
    from tianlong.scenarios.tianlong.lore_commoner import LORE_C
    s = _at_lake(at(1, 19, 30))
    assert LORE_C["yubi@night"] in s.intro(), "月出之前：夜色里黑沉沉的玉璧"
    r = s.turn("等到月亮出来")
    assert "moon" in r.beats and LORE_C["yubi@moon"] in r.narration and LORE_C["yubi@night"] not in r.narration
    assert LORE_C["yubi@moon"] not in s.turn("等待").narration, "月下玉璧也只在初见时描写"


def test_long_wait_stops_at_a_perceived_beat_only_once():
    s = _at_lake(at(1, 19, 30))
    r = s.turn("等下去")
    assert s.authority.head().clock == MOONRISE and "moon" in r.beats, "等到天亮也在月下玉璧那一刻停下"
    s.turn("等下去")
    assert s.authority.head().clock > MOONRISE + 1, "一次性的看点只让等待停一回：月亮不会再“升”一次"
    assert "moon" in s.session_state()["staged"]


def test_legacy_world_without_moonrise_waits_until_night():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    from tianlong.scenarios import build_wuliang

    s = GameSession(build_wuliang(7), pipeline=False)
    parsed = s.interpreter.interpret("一直等到月亮出来", s.beliefs("duanyu"))
    assert parsed.until == "moon" and s._ticks_for(parsed, at(1, 18, 0)) == 60, "旧版没有月出：照旧等到入夜"
    dawn = s.interpreter.interpret("等到天亮", s.beliefs("duanyu"))
    assert s._ticks_for(dawn, at(1, 18, 0)) == 240, "没有天亮时刻按卯时算，仍受 MAX_WAIT 限制"
