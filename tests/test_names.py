"""
[INPUT]: 依赖 runtime/names 的相识账本（Acquaintance / initial / masked / lookup / gate_aliases / may_name / veiled / early_line），
         runtime/session 的 GameSession（模板模式；叙述者换成 ScriptedLLM），language/llm 的 ScriptedLLM，language/gate 的 violations，
         language/render 的 build_plan，persistence 的 InMemoryWorldStore，scenarios 的 build_wuliang / build_wuliang_commoner，
         tests/test_commoner 的 _session（缺 LangGraph / Qdrant 即跳过）
[OUTPUT]: plan §7 M2 test_names：引介之前含“钟灵”的叙述句被丢、brief 里是“梁上的青衫少女”；她录入的“我叫钟灵”之后，
          玩家与在场 NPC 的账本都有了她、名字放行；存档再读档账本不变（JSON 往返不变）；玩家抢先打出真名时解析从宽、只提示一次
          （那句提示过得了闸门）；展示用副本永不落库；没有外貌称呼的旧版逐字不变
[POS]: tests 的称呼与相识：名字要有人道出才算知道——叙述者、解释器、场外问答都只看得见玩家叫得出的名字
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json

from tianlong.cognition import BeliefStore
from tianlong.language.gate import violations
from tianlong.language.llm import ScriptedLLM
from tianlong.language.render import build_plan
from tianlong.language.scene import SceneBrief
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime import names
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
