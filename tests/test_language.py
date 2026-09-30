"""
[INPUT]: 依赖 tianlong.language 的 parser / narrator / speaker / llm / templates，tianlong.cognition 的 BeliefStore
[OUTPUT]: 语言层测试：规则解析表、LLM 只在规则失败时调用、LLM 引用陌生实体被拒、叙述只含玩家感知、时辰先于所见、玩家原话只作意图、失败回退、磁盘缓存
[POS]: tests 的语言层；验证“LLM 负责开放语义与文字表达，但永远不裁定事实、不越过认知边界”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json

import pytest

from tianlong.cognition import BeliefStore, Candidate
from tianlong.core import Fact, Manner, Op, Proposition, Rel
from tianlong.core.profiles import Profile
from tianlong.language.llm import CachedLLM, LLMUnavailable
from tianlong.language.narrator import Narrator
from tianlong.language.parser import IntentParser, rule_parse
from tianlong.language.speaker import LLMSpeaker
from tianlong.scenarios import build_warehouse


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str | None = None, fail: bool = False) -> None:
        self.reply, self.fail, self.calls = reply, fail, []

    def generate(self, prompt, *, system=None, schema=None, temperature=0.4):
        self.calls.append(prompt)
        if self.fail:
            raise LLMUnavailable("boom")
        return self.reply or ""


@pytest.fixture
def player_store():
    sc = build_warehouse()
    return BeliefStore("player").revise_all(sc.priors["player"])[0]


@pytest.mark.parametrize("text, expected", [
    ("拿走桌上的钥匙", Candidate(Op.TAKE, "key")),
    ("悄悄拿钥匙", Candidate(Op.TAKE, "key", manner=Manner.CAREFUL)),
    ("把桌上那把钥匙揣进兜里", Candidate(Op.TAKE, "key")),
    ("去仓库入口", Candidate(Op.MOVE, "entrance", "door_main")),
    ("走向仓库大门", Candidate(Op.MOVE, "entrance", "door_main")),
    ("查看桌面", Candidate(Op.INSPECT, "table")),
    ("看看四周", Candidate(Op.INSPECT, "warehouse")),
    ("等待", Candidate(Op.WAIT)),
])
def test_rule_parse_table(player_store, text, expected):
    assert rule_parse(text, player_store).candidate == expected


def test_rule_parse_speech_and_negation(player_store):
    store, _ = player_store.revise_all([])
    # 让玩家认识守卫：一条“看见守卫在仓库”的感知
    from tianlong.core import EntitySketch, Kind, Modality, Percept
    guard = EntitySketch("guard", Kind.PERSON, "守卫")
    harbor = EntitySketch("harbor", Kind.PLACE, "港口")
    store, _ = store.revise(Percept(0, Modality.SIGHT, None,
                                    (Fact(Proposition.rel("guard", Rel.AT, "warehouse")),), (), (guard, harbor)))
    tell = rule_parse("告诉守卫钥匙在港口", store).candidate
    assert tell == Candidate(Op.TELL, "guard", topic=Fact(Proposition.rel("key", Rel.AT, "harbor"), True))
    lie = rule_parse("告诉守卫钥匙不在桌面", store).candidate
    assert lie.topic == Fact(Proposition.rel("key", Rel.AT, "table"), False)
    ask = rule_parse("问守卫钥匙在哪", store).candidate
    assert ask == Candidate(Op.ASK, "guard", topic=Fact(Proposition.rel("key", Rel.AT, None), True))


def test_unknown_entities_cannot_be_referenced(player_store):
    # 玩家不认识账簿：规则解析与 LLM 解析都不能指向它
    assert rule_parse("拿账簿", player_store).candidate is None
    llm = FakeLLM(json.dumps({"mode": "immediate", "actor": "player", "op": "take", "target": "ledger", "obj": None,
                              "manner": "normal",
                              "topic_subject": None, "topic_value": None, "topic_holds": True, "clarification": ""}))
    parsed = IntentParser(llm).parse("拿账簿", player_store)
    assert parsed.candidate is None and llm.calls


def test_llm_consulted_only_when_rules_fail(player_store):
    llm = FakeLLM(json.dumps({"mode": "immediate", "actor": "player", "op": "take", "target": "key", "obj": None,
                              "manner": "careful",
                              "topic_subject": None, "topic_value": None, "topic_holds": True, "clarification": ""}))
    p = IntentParser(llm)
    assert p.parse("拿钥匙", player_store).source == "rules" and not llm.calls
    got = p.parse("那串叮当响的东西我想顺走", player_store)
    assert got.source == "llm" and got.candidate == Candidate(Op.TAKE, "key", manner=Manner.CAREFUL)
    assert "ledger" not in llm.calls[0] and "账簿" not in llm.calls[0], "LLM 只看得到玩家认识的实体"


def test_llm_failure_falls_back(player_store):
    parsed = IntentParser(FakeLLM(fail=True)).parse("那串东西我想顺走", player_store)
    assert parsed.candidate is None and parsed.clarification


def test_narrator_sees_only_player_percepts(authority, act):
    act(("player", Op.TAKE, "key"))
    r = act(("guard", Op.MOVE, "warehouse", "door_main"))
    llm = FakeLLM("守卫走了进来。")
    store = authority.store.beliefs(authority.ref, "player")
    text = Narrator(llm).narrate("player", [o.percept for o in r.observations_of("player")], store.entities)
    assert text == "守卫走了进来。"
    prompt = llm.calls[0]
    assert "守卫经仓库大门来到仓库" in prompt and "账簿" not in prompt and "船长" not in prompt
    fallback = Narrator(FakeLLM(fail=True)).narrate("player", [o.percept for o in r.observations_of("player")],
                                                   store.entities)
    assert "看见守卫经仓库大门来到仓库" in fallback and "我" not in fallback



def test_narrator_orders_lapse_first_and_command_is_only_intent(authority, act):
    """等待之后先交代时辰再讲所见；玩家原话进 prompt 但被标明只是意图。"""
    r = act(("guard", Op.MOVE, "warehouse", "door_main"))
    store = authority.store.beliefs(authority.ref, "player")
    percepts = [o.percept for o in r.observations_of("player")]
    plain = Narrator().narrate("player", percepts, store.entities, lapse="第1日 19:00")
    assert plain.splitlines()[0] == "（不觉已是第1日 19:00）"
    assert Narrator().narrate("player", [], store.entities, lapse="第1日 19:00").endswith("（不觉已是第1日 19:00）")
    llm = FakeLLM("……")
    Narrator(llm).narrate("player", percepts, store.entities, command="飞上房梁")
    assert llm.calls[0].startswith("玩家的输入：飞上房梁") and "守卫经仓库大门来到仓库" in llm.calls[0]

def test_speaker_keeps_proposition_on_failure():
    names = build_warehouse()
    from tianlong.kernel.perception import sketches_for
    sk = {s.id: s for s in sketches_for(names.state, ["key", "player", "table"])}
    cand = Candidate(Op.TELL, "player", topic=Fact(Proposition.rel("key", Rel.AT, "table")))
    line = LLMSpeaker(FakeLLM(fail=True)).utter(Profile("guard", "guard", "守卫"), cand, sk)
    assert line == "钥匙在桌面上"


def test_cached_llm_hits_disk(tmp_path):
    inner = FakeLLM("好")
    cached = CachedLLM(inner, tmp_path)
    assert cached.generate("你好") == cached.generate("你好") == "好"
    assert len(inner.calls) == 1
