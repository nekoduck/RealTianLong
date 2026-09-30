"""
[INPUT]: 依赖 runtime/names 的相识账本（Acquaintance / initial / masked / lookup / gate_aliases / may_name / veiled / early_line），
         runtime/session 的 GameSession（模板模式；叙述者换成 ScriptedLLM），runtime/gm 的 leaked，language/llm 的 ScriptedLLM，
         language/gate 的 violations，language/quotes 的 introduces，language/scene 的 SceneBrief / VoiceLine，
         language/render 的 build_plan，persistence 的 InMemoryWorldStore，scenarios 的 build_wuliang / build_wuliang_commoner，
         tests/test_commoner 的 _session（缺 LangGraph / Qdrant 即跳过）
[OUTPUT]: plan §7 M2 test_names：引介之前含“钟灵”的叙述句被丢、brief 里是“梁上的青衫少女”；她录入的“我叫钟灵”之后，
          玩家与在场 NPC 的账本都有了她、名字放行；存档再读档账本不变（JSON 往返不变）；玩家抢先打出真名时解析从宽、只提示一次
          （那句提示过得了闸门）；展示用副本永不落库；没有外貌称呼的旧版逐字不变；
          对抗审查回归：NPC 教不会他自己叫不出的名字（模板答话展示成外貌称呼）、问句里复述的真名不算引介、问外貌称呼答来历、
          照着外貌称呼对人说话解析得到、gm.leaked 抓得到“钟姑娘/葛姑娘”、只有“姓”字引子之后的单姓才算自报、NPC 叫不出阿顺
[POS]: tests 的称呼与相识：名字要有人道出才算知道——叙述者、解释器、场外问答都只看得见玩家叫得出的名字
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json

from tianlong.cognition import BeliefStore
from tianlong.language.gate import violations
from tianlong.language.llm import ScriptedLLM
from tianlong.language.quotes import introduces
from tianlong.language.render import build_plan
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime import gm, names
from tianlong.scenarios import build_wuliang, build_wuliang_commoner

from .test_commoner import _session

THANKS = "对那姑娘说：多谢姑娘替我家公子出头"


def _scripted(s, text: str) -> None:
    """叙述者换成写死一稿的模型（先声不上场：只看模型这一稿过不过闸门）。"""
    s.narrator.llm = ScriptedLLM(lambda p, sy, sc: text)
    s.narrator.lead_after = None


def _introduced(s) -> None:
    """17:44 向梁上的少女道谢：她（驱力 introduce）自报“我叫钟灵”——等到玩家的账本里有了她为止。"""
    for _ in range(4):
        s.turn("等待")
    s.turn(THANKS)
    for _ in range(3):
        if "zhongling" in s._acq.of("ashun"):
            return
        s.turn("等待")
    raise AssertionError("钟灵没有自报姓名")


# ============================================================
#  账本本身（不需要会话）
# ============================================================


def test_initial_ledger_and_json_round_trip():
    sc = build_wuliang_commoner(7)
    acq = names.initial(sc)
    assert "zhongling" not in acq.of("ashun") and {"duanyu", "mawude", "gongguangjie"} <= acq.of("ashun")
    assert "ashun" in acq.of("duanyu"), "先验里彼此见过礼"
    again = names.Acquaintance.from_state(json.loads(json.dumps(acq.to_state())))
    assert again == acq and again.to_state() == acq.to_state()
    early = names.Acquaintance(acq.known, frozenset({"zhongling"}))
    assert names.Acquaintance.from_state(json.loads(json.dumps(early.to_state()))) == early


def test_masked_view_renames_only_the_unintroduced_and_legacy_is_untouched():
    sc = build_wuliang_commoner(7)
    me = BeliefStore("ashun").revise_all(sc.priors["ashun"])[0]
    view = names.masked(me, names.initial(sc), sc)
    assert view.sketch("zhongling").name == "梁上的青衫少女" and view.sketch("duanyu").name == "段誉"
    assert me.sketch("zhongling").name == "钟灵", "展示用副本：认知本身不动"
    assert set(view.entities) == set(me.entities) and view.beliefs == me.beliefs
    aliases = names.gate_aliases(names.initial(sc), "ashun", sc)
    assert "钟灵" in aliases["zhongling@name"] and "钟姑娘" in aliases["zhongling@name"]
    assert not {"钟灵", "钟姑娘", "灵儿"} & set(aliases["zhongling"])
    assert "钟灵" in names.gate_aliases(names.initial(sc), "ashun", sc, said="打量钟灵")["zhongling"], "玩家自己打出的不算泄露"
    assert "钟灵" in names.lookup(sc)["zhongling"] and "梁上的青衫少女" in names.lookup(sc)["zhongling"], "解析从宽"
    # 旧版没有外貌称呼：展示、闸门、解析、可点名逐字不变
    old = build_wuliang(7)
    mine = BeliefStore("duanyu").revise_all(old.priors["duanyu"])[0]
    acq = names.initial(old)
    assert names.masked(mine, acq, old) is mine and names.gate_aliases(acq, "duanyu", old) == old.gate_aliases
    assert names.lookup(old) == {k: tuple(v) for k, v in old.aliases.items()}
    npc = BeliefStore("zhongling").revise_all(old.priors["zhongling"])[0]
    assert names.may_name("zhongling", npc, acq, old) == frozenset(
        n for eid, sk in npc.entities.items() for n in (sk.name, *old.aliases.get(eid, ())))


def test_npcs_name_only_whom_they_know():
    sc = build_wuliang_commoner(7)
    acq = names.initial(sc)
    mind = BeliefStore("gongguangjie").revise_all(sc.priors["gongguangjie"])[0]
    may = names.may_name("gongguangjie", mind, acq, sc)
    assert "钟灵" not in may and "钟姑娘" not in may and "青衫少女" in may, "没人道出她的名字：只能用不带名的称呼"
    assert "钟灵" in names.may_name("gongguangjie", mind, acq.learn({"gongguangjie": {"zhongling"}}), sc)
    own = BeliefStore("zhongling").revise_all(sc.priors["zhongling"])[0]
    assert "钟灵" in names.may_name("zhongling", own, acq, sc), "自己的名字自己知道"


# ============================================================
#  会话（模板模式 + 写死一稿的叙述者）
# ============================================================


def test_before_the_introduction_her_name_is_dropped_and_the_brief_shows_her_look():
    s = _session()
    s.intro()
    _scripted(s, "钟灵在梁上嗑着瓜子。梁上的青衫少女朝你们这边瞥了一眼。")
    r = s.turn("抬头看看梁上那少女")
    assert "梁上的青衫少女" in r.brief.present and not any("钟灵" in x for x in r.brief.present)
    assert ("entity", "钟灵") in {(v.kind, v.detail) for v in r.render.violations}, "引介之前点她的名即违规"
    assert "钟灵" not in r.narration and "梁上的青衫少女朝你们这边瞥了一眼" in r.narration
    assert s.beliefs("ashun").sketch("zhongling").name == "钟灵", "外貌称呼只在展示用副本里，从不落库"


def test_after_she_says_her_name_everyone_present_knows_it_and_it_passes():
    s = _session()
    s.intro()
    _introduced(s)
    heard = {e.id for e in s.store.events(s.ref) if e.actor == "zhongling" and "我叫钟灵" in (e.intent.utterance or "")}
    assert heard, "自报姓名是录入的原话"
    witnesses = {a for a in s.scenario.npcs if a != "zhongling" and "zhongling" in s._acq.of(a)}
    assert {"zuozimu", "mawude"} <= witnesses, "在场的 NPC 也听见了"
    assert "zhongling" in s.session_state()["names"]["known"]["ashun"], "随世界同一事务落库"
    _scripted(s, "钟灵朝你眨了眨眼。")
    r = s.turn("等待")
    assert "钟灵朝你眨了眨眼" in r.narration and not any(v.detail == "钟灵" for v in r.render.violations)


def test_the_ledger_survives_save_and_load():
    _session()                                                   # 缺 LangGraph / Qdrant 即跳过，再导入会话
    from tianlong.runtime.session import GameSession

    store = InMemoryWorldStore()
    s = GameSession(build_wuliang_commoner(7), store=store, pipeline=False)
    s.intro()
    _introduced(s)
    s.turn("等待")                                               # 交付正文里学到的随下一次提交落库
    again = GameSession(build_wuliang_commoner(7), store=store, pipeline=False)
    assert again._acq == s._acq and again.session_state()["names"] == s.session_state()["names"]
    assert again._view().sketch("zhongling").name == "钟灵"


def test_typing_her_true_name_first_is_understood_and_remarked_on_once():
    s = _session()
    s.intro()
    first = s.turn("打量钟灵")
    assert first.parsed.kind.value == "gesture" and "钟灵" in (first.parsed.utterance or ""), "解释从宽：照常解析"
    remark = first.narration.splitlines()[-1]
    assert remark.startswith("梁上的青衫少女") and "这个名字" in remark, "只一次地附一句（外貌称呼、不带引号）"
    assert s.session_state()["names"]["early"] == ["zhongling"]
    assert "zhongling" not in s._acq.of("ashun"), "叫得出不等于有人引介：她仍是外貌称呼"
    again = s.turn("打量钟灵")
    assert "这个名字" not in again.narration, "只提示一次"
    thanks = s.turn("对钟灵说：多谢姑娘")
    assert thanks.parsed.candidate.target == "zhongling" and "这个名字" not in thanks.narration


def test_the_early_remark_passes_the_gate():
    sc = build_wuliang_commoner(7)
    me = names.masked(BeliefStore("ashun").revise_all(sc.priors["ashun"])[0], names.initial(sc), sc)
    plan = build_plan("ashun", sc.priors["ashun"][-1:], me.entities, aliases=names.gate_aliases(names.initial(sc),
                                                                                                "ashun", sc))
    known = frozenset(e.name for e in sc.state.entities.values()) | {sk.name for sk in me.entities.values()}
    for command in ("打量钟灵", "对钟灵说：多谢", "钟灵，你好", "看看钟灵", "问钟灵"):
        line = names.early_line(["zhongling"], command, sc).strip()
        assert line.startswith("梁上的青衫少女") and not violations(line, line, "", plan, SceneBrief(), known, command)


# ============================================================
#  对抗审查的回归：名字只能出自叫得出它的人之口
# ============================================================


def test_an_npc_cannot_teach_a_name_he_never_heard():
    """马五德没听人道出过钟灵的名字：他的模板答话取自草图、带着真名，展示时换成外貌称呼，听见的人也学不会。"""
    s = _session()
    s.intro()
    s.turn("环顾四周")
    s.turn("问马五德梁上那少女是谁")
    r = s.turn("问马五德那姑娘在哪")
    assert "zhongling" not in s._acq.of("mawude")
    assert "zhongling" not in s._acq.of("duanyu"), "马五德的听众没从他嘴里学到她的名字"
    assert "zhongling" not in s._acq.of("ashun") and "钟灵" not in r.narration
    assert "梁上的青衫少女在剑湖宫大殿" in r.narration, "他指给你看的是那个人，不是那个名字"
    assert not any("钟灵" in x for x in s.belief_lines())


def test_asking_by_her_true_name_is_not_an_introduction():
    """玩家问句里打出的真名被先声复述，不算有人道出；问外貌称呼一样答来历（说话者叫得出他的名字才答）。"""
    s = _session()
    s.intro()
    r = s.turn("问左子穆：干光豪是谁？")
    assert "ganguanghao" not in s._acq.of("ashun") and "这个名字" in r.narration
    assert "高个子的东宗弟子" in s.turn("环顾四周").narration.splitlines()[-1]
    s2 = _session()
    s2.intro()
    r2 = s2.turn("问左子穆：高个子的东宗弟子是谁？")
    about = next(vl.about for vl in r2.brief.lines if vl.speaker == "zuozimu")
    assert about.startswith("干光豪："), "外貌称呼也算问到了他"


def test_the_player_can_address_people_by_the_epithets_he_sees():
    s = _session()
    s.intro()
    me = s._view()
    for text, eid in (("对梁上的青衫少女说：多谢", "zhongling"), ("向梁上的青衫少女道谢", "zhongling"),
                      ("跟梁上的青衫少女说：多谢", "zhongling"), ("对青衫少女说：多谢", "zhongling"),
                      ("对钟姑娘说：多谢", "zhongling"), ("对葛姑娘说：你好", "geguangpei"),
                      ("对高个子的东宗弟子说：你好", "ganguanghao"), ("对清秀的西宗女弟子说：你好", "geguangpei")):
        p = s._parse(text, me)
        assert p.candidate is not None and p.candidate.target == eid, (text, p.clarification)


def test_named_forms_of_the_unintroduced_are_leaks_even_when_a_known_alias_is_inside():
    sc = build_wuliang_commoner(7)
    acq = names.initial(sc)
    me = names.masked(BeliefStore("ashun").revise_all(sc.priors["ashun"])[0], acq, sc)
    veiled = names.veiled(acq, "ashun", sc)
    for text, name in (("梁上那位是钟姑娘，别招惹她。", "钟姑娘"), ("葛姑娘低着头。", "葛姑娘"), ("钟灵坐在梁上。", "钟灵")):
        assert name in gm.leaked(text, me, sc, veiled=veiled), text
    assert not gm.leaked("钟姑娘坐在梁上。", me, sc, said="钟姑娘是谁", veiled=veiled), "玩家自己说出的不算"


def test_a_bare_surname_counts_only_after_a_surname_lead():
    gan, zhong = ("干光豪", "干师兄"), ("钟灵", "钟姑娘", "灵儿")
    for words in ("你当我是干什么吃的？", "看什么看？我是干这行的", "叫我干啥？"):
        assert not introduces(words, gan), words
    assert not introduces("我是钟家的丫头", zhong) and not introduces("在下司马", ("司空玄", "司空帮主"))
    assert introduces("我叫钟灵，你呢？", zhong) and introduces("本姑娘姓钟", zhong) and introduces("在下姓干", gan)
    sc = build_wuliang_commoner(7)
    acq = names.initial(sc)
    vl = VoiceLine("ganguanghao", "高个子的东宗弟子", "你", "tell", None, None, "看什么看？还不去干你的活！", said=True)
    text = "高个子的东宗弟子喝道：“看什么看？你当我是干什么吃的？还不去干你的活！”"
    new, _ = names.learn_delivered(acq, "ashun", text, SceneBrief((vl,)), "等待", sc)
    assert "ganguanghao" not in new.of("ashun")


def test_in_the_commoner_world_npcs_cannot_name_anyone_unintroduced():
    sc = build_wuliang_commoner(7)
    acq = names.initial(sc)
    mind = BeliefStore("zhongling").revise_all(sc.priors["zhongling"])[0]
    may = names.may_name("zhongling", mind, acq, sc)
    assert "ashun" not in acq.of("zhongling") and "阿顺" not in may and "段誉" not in may and "书呆子" in may
    assert "阿顺" in names.may_name("zhongling", mind, acq.learn({"zhongling": {"ashun"}}), sc)
