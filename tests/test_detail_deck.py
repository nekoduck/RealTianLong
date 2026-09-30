"""
[INPUT]: 依赖 runtime/session 的 GameSession（缺 LangGraph/Qdrant 跳过），language/llm 的 ScriptedLLM，scenarios 的 build_wuliang_commoner，
         scenarios/tianlong/stagecraft 的 DETAILS_C，tests/test_commoner 的 stage（把阿顺挪到后院重新建档）
[OUTPUT]: 细节卡组验收（plan §7 M3 test_detail_deck）：在后院连续三次“四下打量”，提示词里给出三条不同的细节（卡组的先后）；
          一幕之内不重复——存档再读档接着给第四条，卡组给完就不再给；模板模式把细节照印在所见之后；只有查看才给，走到那里不给
[POS]: tests 的细节卡组层：同一处再看，给下一条还没给过的；账本（会话运行态 facets）随下一次提交落库、读档恢复
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.language.llm import ScriptedLLM
from tianlong.scenarios import build_wuliang_commoner
from tianlong.scenarios.tianlong.stagecraft import DETAILS_C

from .test_commoner import stage

DECK = DETAILS_C["houyuan"]
SECTION = "这回多看出的一处细节"


def _yard(llm=None, store=None):
    pytest.importorskip("langgraph")                # 会话要 LangGraph：核心零依赖环境先跳过，再导入
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    sc = build_wuliang_commoner(7)
    return GameSession(stage(sc, {"ashun": "houyuan"}, sc.state.clock), store=store, llm=llm, pipeline=False)


def _detail(prompt: str) -> str | None:
    part = next((p for p in prompt.split("\n\n") if p.startswith(SECTION)), None)
    return part.split("：", 1)[1] if part else None


def test_three_looks_three_details_never_repeated():
    llm = ScriptedLLM(lambda *_: "")
    s = _yard(llm)
    s.intro()
    given = []
    for _ in range(3):
        s.turn("四下打量一番")
        given.append(_detail(llm.prompts[-1][1]))
    assert given == list(DECK[:3]), "三条不同的细节，按卡组的先后"
    assert s.session_state()["facets"] == sorted(DECK[:3])
    s.turn("等待")                                  # 账本是派生数据，与谈资账本同一契约：随下一次提交落库
    again = _yard(ScriptedLLM(lambda *_: ""), store=s.store)            # 读档：账本随运行态恢复
    again.turn("四下打量一番")
    assert _detail(again.llm.prompts[-1][1]) == DECK[3], "一幕之内不重复：读档之后接着给下一条"
    again.turn("四下打量一番")
    assert _detail(again.llm.prompts[-1][1]) is None, "卡组给完就不再给"


def test_template_prints_the_detail_and_only_looking_gives_one():
    s = _yard()
    s.intro()
    assert f"（{DECK[0]}）" in s.turn("四下打量一番").narration, "模板模式：细节照印在所见之后"
    r = s.turn("去后山")
    assert r.brief.details == () and not any(d in r.narration for ds in DETAILS_C.values() for d in ds), "走到那里不给细节"
