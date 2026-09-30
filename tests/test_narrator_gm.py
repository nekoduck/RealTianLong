"""
[INPUT]: 依赖 tianlong.language 的 narrator / render / quotes / scene / llm（ScriptedLLM），tianlong.runtime.authority（真实内核产出的玩家感知），
         tianlong.scenarios 的 build_wuliang
[OUTPUT]: 主持人之声验收：流式逐句过闸门且按序、尽早交付；点名清单外实体的句子被丢而其余照常流出；替玩家开口（引语与念头）、
          NPC 台词点名许可之外的人、凭空多出的说话者、台词里的状态升级各被拦下；丢满两句或一句未过即补模板；模型不可用（含中途失败）
          保留已交付的并补模板；模板把 NPC 言语写成带言语行为的台词；提示词带最近正文与台词要素且没有钟点数字；首句交付早于整段完成；
          分句器处理引号（含错配的收引号）、省略号、较长的后置归属与流的边界；合法的道谢、挑衅与如实的位置说法不被误伤；
          漏掉的台词与内核结果补上模板行、传闻说成事实在流出前就丢、只看见的耳语不算开口、场景秘密被拦、写够长即停、回退的模板行以句号收尾
[POS]: tests 的主持层叙述；证伪“流式叙述会把没过闸门的句子交给玩家”“主持人替玩家说话”“NPC 说出他不该知道的名字”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools
import re
import time

import pytest

from tianlong.core import Fact, Intent, Manner, Op, Proposition, Rel, Social
from tianlong.language.lead import lead_line
from tianlong.language.llm import LLMUnavailable, ScriptedLLM
from tianlong.language.narrator import MAX_CHARS, SOCIAL_PHRASES, Narrator, render_voice
from tianlong.language.quotes import check_quotes
from tianlong.language.render import RenderStatus, build_plan, check, sentence_ends
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_wuliang

_ids = itertools.count()
SC = build_wuliang()
KNOWN = frozenset(e.name for e in SC.state.entities.values()) | {a for al in SC.aliases.values() for a in al}
CLOCK = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}|第\s*\d+\s*[日天]")

TAUNT = "你笑什么？下场比划比划！"
GONG = VoiceLine("gongguangjie", "龚光杰", "你", Op.TELL.value, Social.CHALLENGE, None, TAUNT,
                 voice="骄横好胜，说话夹枪带棒", knows="无量剑东宗每五年与西宗比剑一次",
                 may_name=frozenset({"龚光杰", "段誉", "左子穆", "左掌门", "剑湖宫大殿", "大殿"}),
                 answering="段誉方才忍不住笑出声来")
LING = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.GREET, None, None,
                 voice="娇憨顽皮", may_name=frozenset({"钟灵", "段誉", "龚光杰"}))

G1 = "大殿里的喧笑声忽地一静。"
G2 = "龚光杰霍地站起，冷笑道：“你笑什么？有胆便下场来比划比划！”"
G3 = "满堂目光都落在你身上，等你答话。"
B_ENTITY = "琅嬛福地的深处隐隐传来水声。"
B_PUPPET = "你连忙摆手道：“我这就走。”"
B_NAME = "龚光杰又喝道：“钟灵，你少管闲事！”"
BRIEF = SceneBrief(lines=(GONG,))


@pytest.fixture(scope="module")
def view():
    """龚光杰当众向段誉叫阵（不带命题的自由言语）：段誉本回合的感知与名称表。"""
    auth = WorldAuthority.found(InMemoryWorldStore(), SC)
    v = auth.head().version
    s = auth.settle([Intent(f"gm{next(_ids)}", "gongguangjie", Op.TELL, "duanyu", None, Manner.NORMAL, None, v, TAUNT,
                            Social.CHALLENGE)])
    return [o.percept for o in s.observations_of("duanyu")], auth.store.beliefs(auth.ref, "duanyu").entities


def _narrator(llm=None) -> Narrator:
    return Narrator(llm, SC.setting, SC.lore, SC.style, SC.aliases)


def _run(view, llm, brief=BRIEF, **kw):
    percepts, names = view
    got: list[str] = []
    r = _narrator(llm).narrate_scene("duanyu", percepts, names, brief=brief, show_scene=True, known=KNOWN,
                                     on_text=got.append, **kw)
    assert r.text == "".join(got), "Rendered.text 恒等于交付给玩家的全部文字"
    return r, got


def _script(text: str, **kw) -> ScriptedLLM:
    return ScriptedLLM(lambda prompt, system, schema: text, **kw)


def _template(view, brief=BRIEF, **kw) -> str:
    percepts, names = view
    return _narrator().narrate_scene("duanyu", percepts, names, brief=brief, show_scene=True, known=KNOWN, **kw).text


def _prose(text: str) -> str:
    """回退给玩家的模板行读起来是句子：没有句末标点或收引号的补“。”。"""
    return "\n".join(x if x.endswith(("。", "！", "？", "”", "」")) else x + "。" for x in text.splitlines())


# ============================================================
#  模板：NPC 的言语写成带言语行为的台词
# ============================================================


def test_template_voices_npc_lines_with_social_phrasing(view):
    text = _template(view, SceneBrief(lines=(GONG, LING)))
    challenge = {f"龚光杰{p.format(to='你')}：“{TAUNT}”" for p in SOCIAL_PHRASES[Social.CHALLENGE]}
    greet = {f"钟灵{p.format(to='你')}。" for p in SOCIAL_PHRASES[Social.GREET]}
    rows = text.splitlines()
    assert challenge & set(rows) and greet & set(rows), text
    assert "听见龚光杰对你说" not in text, "清单里的那句言语被台词取代，不重复"
    assert "你看到：" in text, "其余事实照旧"
    assert text == _template(view, SceneBrief(lines=(GONG, LING))), "相同输入 → 相同文字"
    variants = {render_voice(LING, salt=f"上一回合{i}") for i in range(12)}
    assert len(variants) >= 2 and variants <= greet, "措辞随上下文轮换，但只在言语行为的措辞表里选"


def test_template_without_llm_is_delivered_through_the_sink(view):
    r, got = _run(view, None)
    assert r.status == RenderStatus.TEMPLATE and got == [r.text]


# ============================================================
#  流式：逐句过闸门，通过即交付
# ============================================================


def test_streaming_delivers_passing_sentences_in_order(view):
    r, got = _run(view, _script(G1 + G2 + G3, chunk=4))
    assert got == [G1, G2, G3]
    assert r.status == RenderStatus.LLM and r.violations == ()


def test_sentence_naming_unlisted_entity_is_dropped_and_rest_streams(view):
    r, got = _run(view, _script(G1 + B_ENTITY + G2, chunk=5))
    assert got == [G1, G2]
    assert r.status == RenderStatus.LLM and {v.kind for v in r.violations} == {"entity"} and r.dropped == 1


def test_puppeting_quote_is_dropped(view):
    r, got = _run(view, _script(G1 + B_PUPPET + G2))
    assert got == [G1, G2] and {v.kind for v in r.violations} == {"puppet"}, "玩家什么也没说，主持人不能替他开口"
    said = SceneBrief(lines=(GONG,), player_line="我这就走")
    r, got = _run(view, _script(G1 + B_PUPPET + G2), said)
    assert got == [G1, B_PUPPET, G2] and r.violations == (), "玩家自己的原话可以照引"


def test_npc_quote_naming_outside_may_name_is_dropped(view):
    assert check(B_NAME, build_plan("duanyu", view[0], view[1], True, aliases=SC.aliases), KNOWN) == (), \
        "钟灵就在殿上，叙述闸门放行——拦下它的是台词闸门"
    r, got = _run(view, _script(G1 + B_NAME + G2))
    assert got == [G1, G2]
    assert [v for v in r.violations if v.kind == "quote_entity"][0].detail == "龚光杰:钟灵"


def test_two_drops_fall_back_to_template_after_streamed_prose(view):
    r, got = _run(view, _script(G1 + B_ENTITY + B_PUPPET + G3))
    assert r.status == RenderStatus.GATED_FALLBACK
    assert got[0] == G1 and G3 not in r.text, "丢满两句即停，不再读流"
    tail = _prose(_template(view))
    assert r.text == G1 + "\n" + tail, "已交付过正文的，换行后只补清单与台词（读起来是句子）"
    assert {v.kind for v in r.violations} == {"entity", "puppet"}


def test_nothing_accepted_falls_back_to_full_template(view):
    r, got = _run(view, _script(B_ENTITY))
    assert r.status == RenderStatus.GATED_FALLBACK and r.text == _prose(_template(view)) and got == [r.text]
    r, _ = _run(view, _script("   "))
    assert r.status == RenderStatus.GATED_FALLBACK and [v.kind for v in r.violations] == ["empty"]


def test_llm_unavailable_falls_back_to_template(view):
    def boom(prompt, system, schema):
        raise LLMUnavailable("offline")
    r, got = _run(view, ScriptedLLM(boom))
    assert r.status == RenderStatus.LLM_UNAVAILABLE and r.text == _prose(_template(view)) and got == [r.text]


class _Flaky:
    """流到一半断线：先吐出完整的一句和半句，然后失败。"""
    model = "flaky"

    def stream(self, prompt, *, system=None, max_tokens=None):
        yield G1
        yield G2[:6]
        raise LLMUnavailable("connection reset")


def test_llm_failing_mid_stream_keeps_delivered_and_appends_template(view):
    r, got = _run(view, _Flaky())
    assert r.status == RenderStatus.LLM_UNAVAILABLE
    assert got[0] == G1 and r.text == G1 + "\n" + _prose(_template(view)), "半句话不交付"


class _GenerateOnly:
    model = "plain"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt, *, system=None, schema=None, temperature=None):
        return self.text


def test_generate_only_client_is_gated_sentence_by_sentence(view):
    r, got = _run(view, _GenerateOnly(G1 + B_ENTITY + G2))
    assert got == [G1, G2] and r.status == RenderStatus.LLM


def test_voice_lines_are_sourced_even_without_a_matching_percept(view):
    """会话交来的台词本身就是出处：说话者即便不在本回合的感知里，也可以被点名、被写成对白。"""
    percepts, names = view
    comfort = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.COMFORT, None, "书呆子，别怕他！",
                        may_name=frozenset({"钟灵", "段誉", "龚光杰"}))
    brief = SceneBrief(lines=(GONG, comfort))
    got: list[str] = []
    line = "钟灵在梁上笑道：“书呆子，别怕他！”"
    r = _narrator(_script(G2 + line)).narrate_scene("duanyu", percepts, names, brief=brief, known=KNOWN,
                                                     on_text=got.append)
    assert got == [G2, line] and r.status == RenderStatus.LLM, r.violations
    plain = _narrator().narrate_scene("duanyu", percepts, names, brief=brief, known=KNOWN).text
    assert plain.splitlines()[-1] == render_voice(comfort), "清单里对不上的台词补在最后"


@pytest.mark.parametrize("clock", ["此时已是第1日 18:27。", "眼看已近19点20分。", "此刻已是晚上七点二十分。", "不觉已是戌时七点整。"])
def test_clock_label_in_prose_is_dropped(view, clock):
    r, got = _run(view, _script(G1 + clock + G2))
    assert got == [G1, G2] and {v.kind for v in r.violations} == {"clock"}


def test_everyday_words_with_dian_are_not_clocks(view):
    text = "你心里有一点点发怵，差一点笑出声来，一点半点也不敢怠慢。"
    r, got = _run(view, _script(G1 + text + G2))
    assert got == [G1, text, G2] and r.violations == ()


def test_stream_stops_once_the_passage_is_long_enough(view):
    r, got = _run(view, _script(G2 + "大殿里又静了下来。" * 160, chunk=20))
    assert r.status == RenderStatus.LLM and MAX_CHARS <= len(r.text) < MAX_CHARS + 20, "啰嗦的模型也有个头"


# ============================================================
#  覆盖：台词与内核结果没讲到，补上模板行，状态照实记下
# ============================================================


def _settle(*intents, auth=None):
    """在无量山上（默认全新的一局）结算一个 tick，返回段誉的感知与名称表。"""
    auth = auth or WorldAuthority.found(InMemoryWorldStore(), SC)
    v = auth.head().version
    s = auth.settle([Intent(f"gm{next(_ids)}", *x[:5], x[5] if len(x) > 5 else None, v, *x[6:]) for x in intents])
    return [o.percept for o in s.observations_of("duanyu")], auth.store.beliefs(auth.ref, "duanyu").entities


@pytest.mark.parametrize("stream", [G1 + G3, G1 + "龚光杰霍地站起，冷笑道：“你笑什么？钟灵那丫头也护不了你！”" + G3])
def test_npc_line_left_out_or_dropped_is_appended(view, stream):
    r, got = _run(view, _script(stream))
    assert got[:2] == [G1, G3] and got[-1] == "\n" + render_voice(GONG), "漏掉（或被丢）的台词补在最后"
    assert r.status == RenderStatus.GATED_FALLBACK and "omitted" in {v.kind for v in r.violations}
    r, got = _run(view, _script(G1 + G2 + G3))
    assert got == [G1, G2, G3] and r.status == RenderStatus.LLM, "讲到了就不补"


def _lead_run(view, llm, lead_after):
    percepts, names = view
    got: list[str] = []
    r = Narrator(llm, SC.setting, SC.lore, SC.style, SC.aliases, lead_after=lead_after).narrate_scene(
        "duanyu", percepts, names, brief=SceneBrief(), show_scene=True, known=KNOWN, on_text=got.append)
    assert r.text == "".join(got)
    return r, got


def test_instant_lead_is_told_first_and_never_restated():
    """lead_after=0：先声立即交付，模型被告知开头已写好；漏讲不必补，换个说法复述的那句悄悄略过，夹带错的照样丢句。"""
    view = _settle(("duanyu", Op.ATTACK, "gongguangjie", None, Manner.NORMAL))
    lead = lead_line(view[0], view[1], "duanyu")
    assert lead.startswith("你") and "龚光杰" in lead and "挡了开去" in lead, "落空照实写出，而且写明原因"
    calm = "满堂目光都落在你身上，谁也没有作声。"
    for stream in (calm, "你一掌拍向龚光杰，却被他挡了开去。" + calm):
        r, got = _lead_run(view, _script(stream), 0)
        assert got == [lead, calm] and r.status == RenderStatus.LLM and r.dropped == 0
    r, got = _lead_run(view, _script("你猛地向龚光杰出手，他侧身一让，你便被震得受了伤。" + calm), 0)
    assert got == [lead, calm] and r.dropped == 1, "伤落错了人：照样丢句记账"
    prompt = _script(calm)
    _lead_run(view, prompt, 0)
    sent = prompt.prompts[-1][1]
    assert lead in sent and "不要复述" in sent, "模型知道开头已经写好"
    assert "但没有成功（被对方挡了开去）" not in sent, "先声讲过的那一行不再列给模型"


def test_late_lead_only_steps_in_when_the_model_is_slow():
    """默认：模型快，先声不出场，模型照常铺陈这一步（悬念留给模型）；模型迟迟不交付第一句，先声到点顶上、自成一段，
    模型此后几乎逐字重复它的句子略过，其余照常；模型不知道先声的存在。"""
    view = _settle(("duanyu", Op.ATTACK, "gongguangjie", None, Manner.NORMAL))
    lead = lead_line(view[0], view[1], "duanyu")
    drama = "你一掌拍向龚光杰，却被他挡了开去。满堂目光都落在你身上。"
    fast = _script(drama)
    r, got = _lead_run(view, fast, 1.0)
    assert lead not in r.text and r.text.startswith("你一掌拍向龚光杰") and r.status == RenderStatus.LLM
    assert "开头一句已经写好" not in fast.prompts[-1][1]
    t0 = time.perf_counter()
    r, got = _lead_run(view, _script(lead + "满堂目光都落在你身上。", first_delay=0.6), 0.1)
    assert got[0] == lead + "\n" and time.perf_counter() - t0 < 2
    assert r.text == lead + "\n" + "满堂目光都落在你身上。", "逐字复述的那句略过"
    r, got = _lead_run(view, _script(drama, first_delay=0.6), 0.1)
    assert got[0] == lead + "\n" and "你一掌拍向龚光杰" in r.text, "换个说法的铺陈照常交付：模型本不知道先声"


def test_lead_is_never_used_without_a_model_or_when_disabled():
    view = _settle(("duanyu", Op.ATTACK, "gongguangjie", None, Manner.NORMAL))
    lead = lead_line(view[0], view[1], "duanyu")
    r, got = _run(view, None, SceneBrief())
    assert r.status == RenderStatus.TEMPLATE and lead not in r.text, "没有模型时的模板照旧"
    r, got = _lead_run(view, _script("满堂目光都落在你身上。", first_delay=0.3), None)
    assert lead not in r.text


def test_fallback_template_lines_read_as_sentences(view):
    r, _ = _run(view, _script(B_ENTITY))
    rows = r.text.splitlines()
    assert rows[0] == render_voice(GONG), "以收引号收尾的台词不再补标点"
    assert rows[1].startswith("你看到：") and rows[1].endswith("长剑在兵器架上。"), "清单行补上句号"
    assert _template(view).splitlines()[1].endswith("长剑在兵器架上"), "没有模型时的模板照旧"


def test_fallback_tail_skips_voiced_lines_and_tells_the_time_in_words(view):
    r, got = _run(view, _script(G1 + G2 + B_ENTITY + B_PUPPET))
    assert r.status == RenderStatus.GATED_FALLBACK and r.text.count("比划比划") == 1, "已经说出口的台词不再重复"
    r, got = _run(view, _script(G1 + B_ENTITY + B_PUPPET), lapse="第1日 19:00")
    assert "（不觉已是入夜戌时）" in r.text and not CLOCK.search(r.text), "补上的时辰也用文字"
    r, got = _run(view, _script("不觉天色已晚，已是入夜戌时。" + B_ENTITY + B_PUPPET), lapse="第1日 19:00")
    assert "不觉已是" not in r.text, "正文交代过时辰就不再补"


@pytest.fixture(scope="module")
def rumor():
    """马五德告诉段誉“干光豪在剑湖宫后院”（其实干光豪就在大殿）：段誉只闻其说。"""
    told = Fact(Proposition.rel("ganguanghao", Rel.AT, "houyuan"))
    view = _settle(("mawude", Op.TELL, "duanyu", None, Manner.NORMAL, told, None, Social.EXPLAIN))
    line = VoiceLine("mawude", "马五德", "你", Op.TELL.value, Social.EXPLAIN, "干光豪在剑湖宫后院", None,
                     may_name=frozenset({"马五德", "段誉", "干光豪", "剑湖宫后院", "后院"}))
    return view, SceneBrief(lines=(line,))


@pytest.mark.parametrize("fact", ["干光豪此刻正在后院。", "马五德朝后院努了努嘴。干光豪此刻就在后院。"])
def test_hearsay_stated_as_fact_is_dropped_before_it_is_shown(rumor, fact):
    view, brief = rumor
    after = "你心头一动，望向通往后院的回廊。"
    r, got = _run(view, _script(fact + after), brief)
    assert not any("此刻正在后院" in g or "此刻就在后院" in g for g in got), "传闻说成事实：流出去之前就丢"
    assert after in got and "hearsay" in {v.kind for v in r.violations}
    assert got[-1].startswith("\n马五德") and "干光豪在剑湖宫后院" in got[-1], "补上带归属的那句话"


def test_attributed_hearsay_still_streams(rumor):
    view, brief = rumor
    said = "马五德捋须道：“干光豪在剑湖宫后院。”"
    r, got = _run(view, _script(said + "你心头一动，干光豪去后院做什么？"), brief)
    assert got[0] == said and len(got) == 2 and r.status == RenderStatus.LLM


def test_unheard_whisper_gets_no_words_and_keeps_its_own_line():
    auth = WorldAuthority.found(InMemoryWorldStore(), SC)
    whisper = ("gongguangjie", Op.TELL, "ganguanghao", None, Manner.CAREFUL, None, "待会儿看我收拾这书呆子", Social.REMARK)
    percepts, seen = _settle(whisper, auth=auth)
    plan = build_plan("duanyu", percepts, seen, aliases=SC.aliases)
    assert "龚光杰" not in plan.speakers, "只看见在耳语、没听见内容的人不算开口"
    found = check_quotes("龚光杰凑到干光豪耳边，低声道：“今夜就收拾这书呆子。”", SceneBrief(), plan, KNOWN)
    assert ("voice", "龚光杰") in {(v.kind, v.detail) for v in found}, "没听见的话不许替他编出来"
    heard, names = _settle(("gongguangjie", Op.TELL, "duanyu", None, Manner.NORMAL, None, TAUNT, Social.CHALLENGE), auth=auth)
    text = _narrator().narrate_scene("duanyu", [*percepts, *heard], names, brief=BRIEF, known=KNOWN).text
    assert "低声说了些什么" in text and text.count(TAUNT) == 1, "耳语那一行留着，台词只取代听见的那句"


def test_scenario_secrets_are_not_spilled_by_the_narrator(view):
    lovers = "干光豪与葛光佩暗暗交换了一个眼色，似已约好入夜后一同私奔，去投神农帮。"
    got: list[str] = []
    narrator = Narrator(_script(G2 + lovers + G3), SC.setting, SC.lore, SC.style, SC.aliases, SC.secrets)
    r = narrator.narrate_scene("duanyu", view[0], view[1], brief=BRIEF, show_scene=True, known=KNOWN, on_text=got.append)
    assert got == [G2, G3] and {v.kind for v in r.violations} == {"secret"}


# ============================================================
#  提示词：最近正文、台词要素、没有钟点数字
# ============================================================


def test_prompt_carries_recent_passages_and_voice_lines(view):
    llm = _script(G1)
    old = "最早那一回合的正文，不该再出现。"
    long = "你沿着回廊慢慢走去，" + "檐下灯笼摇晃，" * 60 + "（不觉已是第1日 18:27）末尾一句。"
    brief = SceneBrief(lines=(GONG,), recent=(old, "第二段正文。", "第三段正文。", long), player_line="在下只是觉得好笑",
                       stakes="龚光杰等着你答话")
    _run(view, llm, brief, command="拱手赔笑", lapse="第1日 19:00")
    system, prompt = llm.prompts[0]
    assert old not in prompt and "第二段正文。" in prompt and "第三段正文。" in prompt and "末尾一句。" in prompt
    passage = next(line for line in prompt.splitlines() if line.startswith("【3】"))
    assert len(passage) <= 310, "每段正文只留末尾约 300 字"
    for part in (TAUNT, "叫阵", GONG.voice, GONG.knows, GONG.answering, "左子穆", "闲话", "在下只是觉得好笑",
                 "拱手赔笑", "龚光杰等着你答话"):
        assert part in prompt, part
    assert "听见龚光杰对你说" not in prompt, "要说的话不在事实清单里重复一遍"
    assert "戌时" in prompt, "时辰换成文字"
    assert not CLOCK.search(system) and not CLOCK.search(prompt), "提示词里没有钟点数字"
    for rule in ("你", "80~250", "某某道：“……”", "可点名", "不替玩家", "钩子", "钟点", "按兵不动"):
        assert rule in system, rule
    assert SC.style in system


def test_legacy_prompt_still_starts_with_player_input(view):
    llm = _script(G1)
    percepts, names = view
    _narrator(llm).narrate_rendered("duanyu", percepts, names, command="飞上房梁", known=KNOWN)
    assert llm.prompts[0][1].startswith("玩家的输入：飞上房梁") and "听见龚光杰对你说" in llm.prompts[0][1]


# ============================================================
#  延迟：第一句一完整就交付
# ============================================================


def test_first_sentence_is_delivered_before_the_stream_ends(view):
    llm = _script(G1 + G2 + G3 + G3.replace("答话", "开口"), first_delay=0.05, chars_per_second=150, chunk=3)
    stamps: list[float] = []
    percepts, names = view
    t0 = time.perf_counter()
    r = _narrator(llm).narrate_scene("duanyu", percepts, names, brief=SceneBrief(lines=(GONG,)), show_scene=True,
                                     known=KNOWN, on_text=lambda s: stamps.append(time.perf_counter() - t0))
    total = time.perf_counter() - t0
    assert r.status == RenderStatus.LLM and len(stamps) == 4
    assert stamps[0] < 0.6 * total, (stamps, total)
    assert stamps == sorted(stamps)


# ============================================================
#  分句器
# ============================================================


@pytest.mark.parametrize("text, ends", [
    ("你好。他来了！", [3, 7]),
    ("龚光杰道：“你笑什么？下场来！”说着拔出长剑。", [23]),
    ("“你笑什么？”龚光杰喝道。满堂哗然。", [13, 18]),
    ("龚光杰道：“下场来！”满堂目光都落在你身上。", [11, 22]),
    ("满堂哗然……“好！”", [6, 10]),
    ("他问：“真的？！”\n你一怔", [9]),
    ("龚光杰道：“下场来！\"钟灵一笑。满堂哗然。", [11, 16, 21]),               # 全角开、半角收：照样收
    ("“告辞！”你一边拱手一边连声赔笑道。满堂哗然。", [18, 23]),              # 引语之后较长的归属小句
])
def test_sentence_ends(text, ends):
    assert sentence_ends(text) == ends


def test_mismatched_closing_quote_does_not_swallow_the_rest(view):
    text = "龚光杰冷笑道：“你笑什么？有胆便下场来！\"钟灵在梁上咯咯一笑。满堂目光都落在你身上。"
    r, got = _run(view, _script(text, chunk=5))
    assert got == ["龚光杰冷笑道：“你笑什么？有胆便下场来！\"", "钟灵在梁上咯咯一笑。", "满堂目光都落在你身上。"]
    assert r.status == RenderStatus.LLM and r.violations == ()


def test_sentence_ends_waits_for_lookahead_in_a_stream():
    assert sentence_ends("你好。", final=False) == [], "后面也许还有省略号或收引号"
    assert sentence_ends("你好。", final=True) == [3]
    assert sentence_ends("龚光杰道：“好！”", final=False) == [], "收引号之后也许还接着“龚光杰喝道”"
    assert sentence_ends("龚光杰道：“好！”你", final=False) == [], "后面这一小句还没写完"
    assert sentence_ends("龚光杰道：“好！”满堂哗然，", final=False) == [9], "那一小句不是归属：引语自成一句"
    assert sentence_ends("“好！”钟灵笑道，", final=False) == [], "带言说动词的小句是归属，句子继续"
    assert sentence_ends("“好！”钟灵笑道：“", final=False) == [4], "以冒号收尾的是下一段引语的引子"
    assert sentence_ends("龚光杰道：“好！”\n", final=False) == [9]


# ============================================================
#  台词闸门：合法对白放行，越界的拦下
# ============================================================


@pytest.fixture(scope="module")
def plan(view):
    return build_plan("duanyu", view[0], view[1], True, aliases=SC.aliases)


MA = VoiceLine("mawude", "马五德", "你", Op.TELL.value, Social.EXPLAIN, "长剑在兵器架上", None,
               may_name=frozenset({"马五德", "段誉"}))
THANKS = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.THANK, None, "多谢你啦",
                   may_name=frozenset({"钟灵", "段誉"}))


@pytest.mark.parametrize("text, lines", [
    ("钟灵拍手笑道：“书呆子，多谢你啦！”", (THANKS,)),
    ("龚光杰斜眼冷笑道：“就凭你，也配在剑湖宫大殿撒野？”", (GONG,)),
    ("“你笑什么？”龚光杰喝道。", (GONG,)),
    ("马五德捋须道：“长剑就搁在兵器架上。”", (MA,)),
    ("钟灵拨了拨兵器架上的长剑，冲龚光杰做了个鬼脸，笑道：“多谢你啦。”", (THANKS, GONG)),
    ("龚光杰喝道：“下场来！”“你笑什么？”", (GONG,)),
    ("龚光杰收剑入鞘，目光直直落在你脸上，冷笑道：“你笑什么？”", (GONG,)),
    ("龚光杰见你不答，喝道：“你笑什么？”", (GONG,)),
    ("你听见梁上的钟灵笑道：“多谢你啦。”", (THANKS,)),
    ("你身旁的钟灵笑道：“多谢你啦。”", (THANKS,)),
    ("是去是留，由你决定。", (GONG,)),
    ("你打算如何应对？此事须你自己拿定主意。", (GONG,)),
    ("龚光杰霍地站起。“你笑什么？”", (GONG,)),                        # 句首引语沿用上一句的主语：正是说话的人
    ("龚光杰拍了拍你的肩膀，笑道：“你笑什么？”", (GONG,)),               # 宾语位置的“你”不是说话者
    ("龚光杰冷笑着说你笑什么。", (GONG,)),                             # 转述：说话者自己的话照样放行
    ("你心中想必有些发怵。", (GONG,)),                                 # “想必”是揣测，不是替玩家起念头
    ("说时迟，那时快，龚光杰已抢到你面前。", (GONG,)),                  # 成语里的“说”不是转述
    ("龚光杰二话不说，拔剑出鞘。", (GONG,)),
])
def test_legitimate_npc_dialogue_passes(plan, text, lines):
    brief = SceneBrief(lines=lines)
    assert check_quotes(text, brief, plan, KNOWN) == ()
    assert check(text, plan, KNOWN) == ()


@pytest.mark.parametrize("text, kind, detail", [
    ("龚光杰指着你道：“钟灵护不了你。”", "quote_entity", "龚光杰:钟灵"),
    ("你指着龚光杰道：“钟灵护不了你。”", "puppet", "钟灵护不了你。"),
    ("你心中一动。“我这就走。”", "puppet", "我这就走。"),
    ("片刻之后你开口道：“我这就走。”", "puppet", "我这就走。"),
    ("龚光杰的师父左子穆道：“够了。”", "voice", "左子穆"),
    ("你当即打定主意，转身便走。", "puppet", "你当即打定主意"),
    ("左子穆沉声道：“够了。”", "voice", "左子穆"),
    ("他冷笑道：“左子穆也保不住你。”", "quote_entity", "?:左子穆"),
    ("龚光杰冷笑道：“你已中了毒。”", "quote_status", "龚光杰:poisoned:中了毒"),
    ("龚光杰喝道：马五德也救不了你。", "quote_entity", "龚光杰:马五德"),
    # 句首引语紧跟在谁的动作之后，就是谁说的：没开口的人不许凭空多一句
    ("左子穆脸色一沉。“光杰，退下！”", "voice", "左子穆"),
    ("干光豪看了葛光佩一眼。“今晚子时，老地方。”", "voice", "干光豪"),
    ("马五德打了个哈哈。“两位且慢动手。”", "voice", "马五德"),
    # 替玩家开口：状语之后的“你”、较长的后置归属、转述，与更多起念头的说法
    ("情急之下你脱口而出：“龚兄息怒，小弟绝无此意！”", "puppet", "龚兄息怒，小弟绝无此意！"),
    ("此刻你忙道：“龚兄息怒，小弟这就告辞！”", "puppet", "龚兄息怒，小弟这就告辞！"),
    ("慌乱中你连声道：“得罪了！”", "puppet", "得罪了！"),
    ("“龚兄息怒，小弟这就告辞！”你一边拱手一边连声赔笑道。", "puppet", "龚兄息怒，小弟这就告辞！"),
    ("你赔笑说小弟这就告辞。", "puppet", "小弟这就告辞"),
    ("你连忙拱手向龚光杰赔罪，说自己只是一时失笑，绝无冒犯之意。", "puppet", "自己只是一时失笑，绝无冒犯之意"),
    ("你在心中暗暗打定了主意，今日绝不下场。", "puppet", "你在心中暗暗打定了主意"),
    ("你心里想着，这龚光杰好生无礼。", "puppet", "你心里想"),
    ("你横下心来，转身便走。", "puppet", "你横下心"),
    # 无引号的转述同样是台词：没开口的人不许说，开口的人不许越界点名
    ("干光豪低声说今夜要带葛光佩走。", "voice", "干光豪"),
    ("龚光杰冷笑着说你怀里那卷易经也救不了你。", "quote_entity", "龚光杰:易经"),
    ("龚光杰冷笑着说钟灵那小丫头也护不了你。", "quote_entity", "龚光杰:钟灵"),
])
def test_quote_gate_rejects(plan, text, kind, detail):
    found = check_quotes(text, SceneBrief(lines=(GONG, LING)), plan, KNOWN)
    assert (kind, detail) in {(v.kind, v.detail) for v in found}, found


def test_players_own_words_may_be_quoted(plan):
    brief = SceneBrief(lines=(GONG,), player_line="在下段誉，只是觉得好笑")
    for text in ("你拱手道：“在下段誉，只是觉得好笑。”", "“只是觉得好笑，”你说。"):
        assert check_quotes(text, brief, plan, KNOWN) == (), text
    assert check_quotes("你道：“我这就走。”", SceneBrief(lines=(GONG,)), plan, KNOWN, command="说我这就走") == ()


def test_sourced_inscription_is_not_anyones_line(view):
    looks = [SC.lore["d_stonedoor"]]
    plan = build_plan("duanyu", view[0], view[1], True, looks, aliases=SC.aliases)
    text = "你抬头望去，石门上刻着“琅嬛福地”四个大字。"
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN) == ()


def test_since_only_checks_new_quotes(plan):
    text = B_PUPPET + G2
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN)
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN, since=len(B_PUPPET)) == ()
