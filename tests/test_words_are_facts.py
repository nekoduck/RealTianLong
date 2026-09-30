"""
[INPUT]: 依赖 runtime/session 的 GameSession（模板模式；叙述者换成 ScriptedLLM），language/llm 的 ScriptedLLM，
         language/quotes 的 check_quotes，language/render 的 RenderPlan，language/scene 的 SceneBrief / VoiceLine，
         language/voice_prompt 的 scene_prompt / render_voice，scenarios/tianlong/drives_c 的 DRIVES_C，tests/test_commoner 的 _session
[OUTPUT]: plan §7 M2 test_words_are_facts（原则 2 话即事实）：钟灵的驱力台词（挂在施用解药上）原样出现在玩家的
          PerceivedEvent.utterance 与在场 NPC 的经历里；brief 里是 said=True 的台词（act 是施用这一下）；模型给她编了另一套词 →
          该句被丢、违规为 fidelity；截取连续一段则放行；模板回退原样印出这句台词；提示词把它归入“照录的原话”一节；
          只有录入原话的说话者才查一致性（模板措辞的台词照旧可以改写）
[POS]: tests 的话即事实：NPC 说出口的原话是随意图落库的事实（“X 说了 Y”），叙述者只照录、不另编
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.language.llm import ScriptedLLM
from tianlong.language.quotes import check_quotes
from tianlong.language.render import RenderPlan
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.language.voice_prompt import render_voice, scene_prompt
from tianlong.scenarios.tianlong.drives_c import DRIVES_C

from .test_commoner import _session

LINE = next(d.line for d in DRIVES_C["zhongling"] if d.key == "bargain")      # “解药在这儿。你们不再为难这书呆子，我便救他。”


def _bargain(script: str | None = None):
    """被动玩家原地等待，直到钟灵拿解药换人（驱力 bargain：原话挂在 USE 上）的那一回合。"""
    s = _session()
    s.intro()
    if script is not None:
        s.narrator.llm = ScriptedLLM(lambda p, sy, sc: script)
        s.narrator.lead_after = None
    for _ in range(8):
        r = s.turn("等待")
        if any(vl.speaker == "zhongling" and vl.said for vl in r.brief.lines):
            return s, r
    raise AssertionError("钟灵没有拿解药换人")


def test_the_drive_line_is_what_everyone_present_perceived():
    s, r = _bargain()
    ev = next(e for e in s.store.events(s.ref) if e.actor == "zhongling" and e.intent.utterance == LINE)
    assert ev.op.value == "use", "驱力台词挂在任何操作上：这一句挂在施用解药上"
    mine = [o.percept.event for o in r.settlement.observations_of("ashun") if o.source_event == ev.id]
    assert mine and all(e.utterance == LINE for e in mine), "玩家的 PerceivedEvent.utterance 就是原话"
    assert any(ep.event.utterance == LINE for ep in s.beliefs("ashun").episodes if ep.event is not None)
    for npc in ("zuozimu", "mawude"):
        assert any(ep.event is not None and ep.event.utterance == LINE for ep in s.beliefs(npc).episodes), \
            f"在场的 {npc} 也记住了这句原话"
    vl = next(vl for vl in r.brief.lines if vl.speaker == "zhongling")
    assert vl.said and vl.template == LINE and vl.act.startswith("对龚光杰用了") and vl.knows == ""


def test_the_template_prints_the_line_verbatim():
    _, r = _bargain()
    assert LINE in r.narration and f"“{LINE}”" in r.narration


def test_an_invented_line_is_dropped_as_infidelity():
    _, r = _bargain("梁上的青衫少女道：“这瓶药你拿去，咱们两清。”")
    assert "两清" not in r.narration
    assert any(v.kind == "fidelity" for v in r.render.violations), r.render.violations


def test_a_contiguous_excerpt_passes():
    _, r = _bargain("梁上的青衫少女把瓷瓶一扬，道：“解药在这儿。”")
    assert "“解药在这儿。”" in r.narration
    assert not any(v.kind == "fidelity" for v in r.render.violations), r.render.violations


def _plan() -> RenderPlan:
    return RenderPlan("ashun", (), frozenset({"钟灵", "龚光杰"}), frozenset(), frozenset(), frozenset(), (), (),
                      people=(("你", "阿顺"), ("钟灵", "钟灵"), ("龚光杰", "龚光杰")), viewer_name="阿顺")


def _line(said: bool) -> VoiceLine:
    return VoiceLine("zhongling", "钟灵", "龚光杰", "use", None, None, LINE, act="对龚光杰用了解药", said=said)


def test_fidelity_is_checked_only_for_recorded_words():
    """录入的原话照录（改写一半以上即 fidelity）；模板措辞的台词照旧可以换个说法。"""
    text = "钟灵道：“你们别再欺负他，这药就给你。”"
    kinds = {v.kind for v in check_quotes(text, SceneBrief((_line(True),)), _plan())}
    assert "fidelity" in kinds
    assert "fidelity" not in {v.kind for v in check_quotes(text, SceneBrief((_line(False),)), _plan())}
    close = "钟灵道：“解药在这儿，你们不再为难这书呆子，我就救他。”"            # 换了一两个字：字二元组过六成
    assert "fidelity" not in {v.kind for v in check_quotes(close, SceneBrief((_line(True),)), _plan())}


def test_the_prompt_files_recorded_words_under_verbatim_and_the_template_keeps_them():
    brief = SceneBrief((_line(True),))
    prompt = scene_prompt(brief, "", [], [], [], _plan(), frozenset())
    assert "照录的原话" in prompt and f"钟灵（对龚光杰用了解药）对龚光杰：“{LINE}”" in prompt
    assert "要说出口的话" not in prompt, "只有录入原话时不另列“要说出口的话”"
    assert render_voice(_line(True)).endswith(f"：“{LINE}”") and render_voice(_line(True)).startswith("钟灵")
