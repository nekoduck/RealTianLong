"""
[INPUT]: 依赖 runtime/endings 的 ended / title_for / chronicle / names_for / CHRONICLE_HEAD，scenarios 的 build_wuliang_commoner / build_wuliang，
         core 的 at / relocate / SetAttr / Op / Outcome / Rel，runtime/session 的 GameSession（模板模式），
         tests/test_commoner 的 _session / _until（会话与推进时钟）
[OUTPUT]: 结局验收：到澜沧江畔是 river、到山脚是 downhill、时钟到第二日 05:00 是 dawn（地点结局优先）；river 的三种变体
          （身负奇功 / 与段公子同行 / 怀揣帛卷）与 dawn 的身在何处按落幕那一刻的真相加在标题后；旧版只有澜沧江畔、标题不变；
          终章纪事“那一夜你没看见的事”只写真相里发生过、玩家当时不在场的事——段誉跳崖、学成、以药救人、龚光杰作罢都对得上事件日志，
          玩家亲眼看见的开场不进纪事，没人学成就不写“参透”；外貌称呼（没被引介的人不叫名字）
[POS]: tests 的落幕层：结局与纪事都由世界真相确定地生成，纪事只是真相的江湖口吻
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Op, Outcome, Rel, SetAttr, at, relocate
from tianlong.runtime.endings import CHRONICLE_HEAD, chronicle, ended, names_for, title_for
from tianlong.scenarios import build_wuliang, build_wuliang_commoner

from .test_commoner import _events, _session, _until


def _moved(st, who: str, place: str):
    return st.apply(relocate(who, st.target(who, Rel.AT), place))


def test_endings_by_place_and_by_clock():
    sc = build_wuliang_commoner(7)
    st = sc.state
    assert ended(sc, st, "ashun") is None
    assert ended(sc, _moved(st, "ashun", "lancang"), "ashun").key == "river"
    assert ended(sc, _moved(st, "ashun", "shanjiao"), "ashun").key == "downhill"
    dawn = st.stamp(st.version, at(2, 5, 0))
    assert ended(sc, dawn, "ashun").key == "dawn" and ended(sc, st.stamp(st.version, at(2, 4, 59)), "ashun") is None
    assert ended(sc, _moved(dawn, "ashun", "shanjiao"), "ashun").key == "downhill", "天亮时刚好下了山：地点结局优先"
    legacy = build_wuliang(7)
    assert [e.key for e in legacy.endings] == ["river"] and ended(legacy, legacy.state.stamp(0, at(2, 6, 0)), "duanyu") is None


def test_river_variants_and_dawn_whereabouts():
    sc = build_wuliang_commoner(7)
    river = next(e for e in sc.endings if e.key == "river")
    st = _moved(sc.state, "ashun", "lancang")
    assert title_for(river, st, "ashun") == "第一幕终 · 澜沧江畔"
    skilled = st.apply((SetAttr("ashun", "evasion", None, True),))
    assert title_for(river, skilled, "ashun") == "第一幕终 · 澜沧江畔 · 身负奇功"
    together = _moved(st, "duanyu", "lancang")
    assert title_for(river, together, "ashun") == "第一幕终 · 澜沧江畔 · 与段公子同行"
    scroll = together.apply(relocate("scroll_lb", "putuan", "ashun"))
    assert title_for(river, scroll, "ashun") == "第一幕终 · 澜沧江畔 · 与段公子同行、怀揣帛卷"
    dawn = next(e for e in sc.endings if e.key == "dawn")
    assert title_for(dawn, _moved(sc.state, "ashun", "jianhu"), "ashun") == "第一幕终 · 天亮了 · 困在崖底"
    assert title_for(dawn, sc.state, "ashun") == "第一幕终 · 天亮了 · 身在剑湖宫"


def test_chronicle_tells_only_what_truly_happened_out_of_sight():
    s = _session()
    _until(s, at(1, 20, 5))
    sc, st, events = s.scenario, s.authority.head(), s.store.events(s.ref)
    text = chronicle(sc, st, events, names_for(sc, "ashun", st))
    lines = text.splitlines()
    assert lines[0] == CHRONICLE_HEAD and len(lines) > 3
    leap = [e for e in _events(s, "duanyu", Op.MOVE) if e.intent.obj == "d_cliff" and e.outcome == Outcome.SUCCESS]
    mastered = [e for e in _events(s, "duanyu", Op.STUDY) if e.reason == "mastered"]
    assert leap and any("断崖" in x and "只去不回" in x for x in lines), "段誉跳崖：真相里有，纪事里就有"
    assert mastered and any("参透" in x for x in lines)
    assert any("撂下一句狠话" in x for x in lines), "龚光杰作罢"
    assert not any("解药" in x for x in lines), "开场的以药换人玩家亲眼看见了：不是“没看见的事”"
    assert "钟灵" not in text and "青衫少女" in text, "没被引介的人用外貌称呼"
    assert s.epilogue().count(CHRONICLE_HEAD) == 1 and "真相揭晓" not in s.epilogue(), "有纪事角色就用纪事代替真相表"


def test_chronicle_never_invents_a_mastery():
    s = _session()
    _until(s, at(1, 19, 0))
    sc, st = s.scenario, s.authority.head()
    text = chronicle(sc, st, s.store.events(s.ref), names_for(sc, "ashun", st))
    assert "参透" not in text and "救了" not in text, "19:00 之前没人学成、玩家看见了以药救人：一概不写"
