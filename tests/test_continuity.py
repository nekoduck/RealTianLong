"""
[INPUT]: 依赖 tianlong.runtime.continuity 的 continuity，tianlong.runtime.gm 的 salient，tianlong.language 的 Narrator / SceneBrief /
         ScriptedLLM / RenderStatus，tianlong.core 的 Op / Social / Modality / PerceivedEvent / Percept，
         tests/test_session_gm 的小世界（session / Script / HERO / OUT）
[OUTPUT]: 前后照应验收（对照评审挑出的“前后接不上”）：玩家挨了打，此后每回合的叙述上下文都带着“你受了伤”，说到它不算状态升级；
          以为还在大殿的人出现在眼前，上下文写明“你原以为……此刻却在眼前”，他随后开口时带上他自己一路走来的经历；
          一路相随的同伴没跟来就说一声；抵达结局那一回合收在余韵上；闲扯不打断等待而当面招呼照旧打断；
          无事发生时有模型就写眼前的光景与身边的人，没有模型才是“时间悄悄过去”
[POS]: tests 的前后照应：证伪“人被点了穴，下一幕毫发无伤地走来却没人觉得奇怪”“挨了一掌此后再没人提”
       “等到天黑被闲聊打断得原地打转”“到了结局还问要不要回头”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.core import Modality, Op, Outcome, PerceivedEvent, Percept, Social  # noqa: E402
from tianlong.language.llm import ScriptedLLM  # noqa: E402
from tianlong.language.narrator import Narrator  # noqa: E402
from tianlong.language.render import RenderStatus  # noqa: E402
from tianlong.language.scene import SceneBrief  # noqa: E402
from tianlong.runtime import gm  # noqa: E402
from tianlong.runtime.continuity import continuity, lately  # noqa: E402

from .test_session_gm import HERO, OUT, T0, Script, session  # noqa: E402


def test_the_players_wound_stays_in_the_brief_and_may_be_mentioned():
    s = session({"gong": Script(plan={T0: (Op.ATTACK, HERO)})})
    s.turn("等一会")
    for text in ("等一会", "去后院"):
        r = s.turn(text)
        if r.brief is not None and r.brief.condition:
            break
    assert r.brief is not None and r.brief.condition and "受了伤" in r.brief.condition
    assert "wounded" in r.brief.statuses and any(a == "wounded" for _, a in r.brief.afflicted)


def test_someone_believed_elsewhere_turning_up_is_noted_and_they_can_tell_how_they_came():
    ling = Script(plan={T0 + 1: (Op.MOVE, "yard")}, reply=True)
    s = session({"ling": ling})
    s.turn("去后院")
    r = s.turn("等一会")
    notes = " ".join(r.brief.notes) if r.brief else ""
    assert "你原以为钟灵" in notes and "还在大殿" in notes, notes
    r = s.turn("钟灵，你怎么也来了")
    lines = [vl for vl in r.brief.lines if vl.speaker == "ling"] if r.brief else []
    assert lines and "后院" in lines[0].lately, "刚来到眼前的人开口时，带着他自己一路走来的经历"


def test_a_companion_left_behind_is_noted():
    s = session()
    before = s.beliefs(HERO)
    s.turn("去后院", request_id="walk")
    env = s.store.request(s.ref, "walk")
    ctx = continuity(env, s.beliefs(HERO), before, friends=("ling",))
    assert "钟灵没有跟来" in ctx.notes
    assert ctx.present[0] == "后院" and "钟灵" not in ctx.present


def test_reaching_an_ending_closes_on_an_afterglow():
    s = session(endings=(OUT,))
    s.turn("去后院")
    r = s.turn("去山道")
    assert r.ending is not None and r.brief is not None and r.brief.closing


def test_idle_chatter_does_not_cut_a_wait_but_a_greeting_does():
    here = "hall"

    def said(social):
        ev = PerceivedEvent(Op.TELL.value, here, "ling", HERO, social=social, outcome=Outcome.SUCCESS)
        return [Percept(T0, Modality.SPEECH, ev, informant="ling")]
    assert not gm.salient(said(Social.REMARK), HERO, (), here)
    assert not gm.salient(said(Social.JOKE), HERO, (), here)
    assert gm.salient(said(Social.GREET), HERO, (), here)


def test_nothing_happening_still_gets_a_few_lines_about_the_scene():
    llm = ScriptedLLM(lambda p, s, sc: "暮色四合，钟灵坐在石上，双脚一荡一荡。")
    brief = SceneBrief(present=("后院", "钟灵"))
    r = Narrator(llm).narrate_scene(HERO, (), {}, brief=brief)
    assert r.status == RenderStatus.LLM and "钟灵" in r.text
    assert "玩家以为自己此刻在：后院，身边有钟灵" in llm.prompts[-1][1]
    quiet = Narrator(None).narrate_scene(HERO, (), {}, brief=brief)
    assert quiet.text.startswith("时间悄悄过去")


def test_lately_tells_only_the_way_here_up_to_the_moment_he_speaks():
    """lately 只给一路走到此地的那几步（到他开口那一刻为止），不给他之后的行踪，也不给他对别人的动手。"""
    ling = Script(plan={T0 + 3: (Op.MOVE, "yard"), T0 + 4: (Op.MOVE, "road")}, busy="rack")
    s = session({"ling": ling})
    for text in ("去后院", "去山道", "等一会"):
        s.turn(text)
    mind = s.beliefs("ling")
    assert lately(mind, "ling", mind.last_tick) == "我来到后院；我来到山道"
    assert lately(mind, "ling", T0 + 3) == "我来到后院", "他开口之后才发生的事，不会跑进他的台词"
    assert lately(mind, "ling", T0 + 2) == ""


def test_a_recovery_in_the_same_room_is_not_an_arrival():
    """同一处的人从被制到能动：说“他已能动了”，不说“此刻却在眼前”。"""
    s = session()
    me = s.beliefs(HERO)
    from tianlong.core import Proposition
    before = replace(me, beliefs={**me.beliefs, Proposition.attr("ling", "subdued", True):
                                  replace(next(iter(me.beliefs.values())), prop=Proposition.attr("ling", "subdued", True))})
    s.turn("环顾四周", request_id="look")
    env = s.store.request(s.ref, "look")
    ctx = continuity(env, s.beliefs(HERO), before)
    assert any("钟灵动弹不得——此刻他已能动了" in n for n in ctx.notes), ctx.notes
    assert not any("此刻却在眼前" in n for n in ctx.notes) and "ling" not in ctx.newcomers
