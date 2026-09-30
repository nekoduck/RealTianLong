"""
[INPUT]: 依赖 language/voice_prompt 的 scene_prompt / REPEAT / SAID_SHOWN，language/quotes 的 said_by，language/render 的 RenderPlan，
         language/scene 的 SceneBrief / Sky / VoiceLine，runtime/talk 的 said / SAID_KEEP，runtime/gm 的 _people，runtime/names 的 forms
[OUTPUT]: M3 台词不重复与天色的提示词验收（设计 §7 M3 的 test_said_before）：本回合开口的人最近说过的原话进提示词
          “他最近说过的话（别重复）”，至多 SAID_SHOWN 句，与这回合要说的原话三字片段重合 ≥REPEAT 的加注“换个说法”、
          不重合的只列不注；本回合不开口的人不列，said_before 为空时提示词与旧版逐字相同；sky 给了才有“天色”一节；
          系统规则 8 不许编造征兆；台词账本只记确凿归到本回合有台词的人名下的引语（转述、拟声、归属不明、玩家自己的话都不记；
          引子里认不出的人不往前记到上一个说话者名下），
          每人最近 SAID_KEEP 句；会话交来的开口者别称 brief.people 同样认得（“灵儿拍手笑道：”），gm._people 只给玩家叫得出名字的人
          带名的别称、两人共用的别称谁也不给（M3 语言侧遗留 3 接上）
[POS]: tests 的台词账本与提示词规格。只依赖核心，零依赖 CI 同跑；会话里账本随运行态落库、读档恢复由 test_continuity 验
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

from tianlong.core import Op
from tianlong.language.render import RenderPlan
from tianlong.language.scene import MOON_NONE, SceneBrief, Sky, VoiceLine
from tianlong.language.voice_prompt import SAID_SHOWN, scene_prompt, system_prompt
from tianlong.runtime.talk import SAID_KEEP, said

PLAN = RenderPlan("duanyu", ("钟灵跟你说笑",), frozenset({"段誉", "钟灵"}), frozenset(), frozenset(), frozenset(), (), (),
                  viewer_name="段誉", people=(("段誉", "段誉"), ("钟灵", "钟灵")))
LING = VoiceLine("ling", "钟灵", "你", Op.TELL.value, None, None, "我家住万劫谷，这回是偷偷溜出来玩的")
GONG = VoiceLine("gong", "龚光杰", "你", Op.TELL.value, None, None, "你笑什么？")
HEADER = "钟灵最近说过的话（别重复）"
MARK = "（与这回合要说的几乎一样：换个说法）"


def _prompt(brief: SceneBrief) -> str:
    return scene_prompt(brief, "", [], ["钟灵跟你说笑"], [], PLAN, frozenset())


def test_said_before_marks_the_line_about_to_be_repeated():
    brief = SceneBrief(lines=(LING,), said_before=(("钟灵", ("书呆子，你也掉下来啦？", "我家住万劫谷，是偷偷溜出来玩的！")),))
    prompt = _prompt(brief)
    assert HEADER in prompt
    assert f"- “我家住万劫谷，是偷偷溜出来玩的！”{MARK}" in prompt, "与这回合的原话重合过六成：加注"
    assert "- “书呆子，你也掉下来啦？”\n" in prompt, "不重合的只列不注"
    assert prompt.index(HEADER) > prompt.index("要说出口的话"), "紧跟在要说出口的话之后"


def test_said_before_lists_only_the_last_few_of_todays_speakers():
    old = tuple(f"第{i}句闲话说些别的" for i in range(6))
    brief = SceneBrief(lines=(GONG,), said_before=(("钟灵", old), ("龚光杰", old)))
    prompt = _prompt(brief)
    assert HEADER not in prompt, "钟灵本回合不开口：不列"
    assert "龚光杰最近说过的话（别重复）" in prompt and MARK not in prompt
    assert [f"第{i}句" in prompt for i in range(6)] == [False] * (6 - SAID_SHOWN) + [True] * SAID_SHOWN, "只列最近几句"


def test_empty_said_before_and_sky_leave_the_prompt_unchanged():
    brief = SceneBrief(lines=(LING,))
    assert _prompt(brief) == _prompt(replace(brief, said_before=(("龚光杰", ("你笑什么？",)),))), "不开口的人：逐字不变"
    assert "天色" not in _prompt(brief)
    sky = Sky("酉时将尽，天色昏暗，月亮还没出来", night=False, moon=MOON_NONE)
    assert "：酉时将尽，天色昏暗，月亮还没出来" in _prompt(replace(brief, sky=sky)), "sky.text 原样"


def test_rule_eight_forbids_invented_omens():
    assert "不编造征兆、突然的静默、来历不明的影子；钩子只落在本回合给出的事实、台词、看点或眼前可做的事上。" in system_prompt()


def test_ledger_keeps_only_sure_lines_of_todays_speakers():
    brief = SceneBrief(lines=(LING, GONG))
    text = ("钟灵笑道：“我家住万劫谷。”龚光杰冷笑道：“你笑什么？”“还不快滚！”"
            "“嗒”的一声，露水落下。你拱手道：“在下段誉。”马五德说今日天气不错。")
    ledger = said({}, text, brief)
    assert ledger == {"ling": ["我家住万劫谷。"], "gong": ["你笑什么？", "还不快滚！"]}, ledger
    assert said(ledger, "她低声道：“别出声。”", brief) == ledger, "归属不明（代词）不记"
    for i in range(SAID_KEEP + 2):
        ledger = said(ledger, f"钟灵道：“第{i}句。”", brief)
    assert ledger["ling"] == [f"第{i}句。" for i in range(2, SAID_KEEP + 2)], "每人只留最近几句（旧→新）"
    assert said(ledger, "钟灵道：“多一句。”", None) == ledger, "没有叙述上下文不记"


def test_ledger_never_hands_an_unrecognised_speakers_line_to_someone_else():
    """引子里的人认不出（别称“左掌门/灵儿”、玩家本名、本回合没台词的人、无名的声音），归属就往前找到了上一个说话者：
    账本只收本名就在这段引语自己引子里的（或紧接着他上一段记下的引语说），宁可少记也不记错人。"""
    brief = SceneBrief(lines=(GONG,))
    for other in ("左掌门沉声喝道：“光杰，退下！”", "灵儿拍手笑道：“我家住万劫谷。”", "段誉拱手道：“在下段誉。”",
                  "钟灵拍手道：“好玩。”", "一个清脆的声音道：“好玩。”"):
        assert said({}, "龚光杰冷笑道：“你笑什么？”" + other, brief) == {"gong": ["你笑什么？"]}, other
    assert said({}, "“你笑什么？”龚光杰冷笑道。", brief) == {"gong": ["你笑什么？"]}, "句首引语紧跟的“某某道”照记"


def test_ledger_knows_the_speakers_forms_the_session_hands_over():
    """M3 语言侧遗留（open issue 3）接上：会话经 build_brief 交来开口者的别称（brief.people，相识账本过滤过），
    “灵儿拍手笑道：”记到钟灵名下；不是本回合开口的人的别称照旧不认。"""
    brief = SceneBrief(lines=(LING,), people=(("灵儿", "钟灵"), ("左掌门", "左子穆")))
    assert said({}, "灵儿拍手笑道：“我家住万劫谷。”", brief) == {"ling": ["我家住万劫谷。"]}
    assert said({}, "龚光杰冷笑道：“你笑什么？”左掌门沉声喝道：“光杰，退下！”", replace(brief, lines=(GONG,))) == {
        "gong": ["你笑什么？"]}, "左子穆本回合没开口：他的别称不认"


def test_session_forms_hide_names_the_player_cannot_call():
    """gm._people：玩家叫不出名字的人只给不带名的别称（“青衫少女”），带名的（钟姑娘、灵儿）不给；两个开口的人共用的别称谁也不给。"""
    from tianlong.runtime.gm import _people
    from tianlong.runtime.names import forms
    from tianlong.scenarios import build_wuliang_commoner
    sc = build_wuliang_commoner(7)
    zl = replace(LING, speaker="zhongling", speaker_name="梁上的青衫少女")
    people = dict(_people((zl,), sc, {"zhongling": forms(sc, "zhongling")}))
    assert people.get("青衫少女") == "梁上的青衫少女" and not {"钟姑娘", "灵儿", "钟灵"} & set(people)
    named = replace(zl, speaker_name="钟灵")
    known = dict(_people((named,), sc, {}))
    assert known.get("钟姑娘") == "钟灵" and known.get("灵儿") == "钟灵"
    twin = replace(named, speaker_name="钟二")               # 同一套别称归到两个开口的人：谁也不给
    assert "钟姑娘" not in dict(_people((named, twin), sc, {}))
