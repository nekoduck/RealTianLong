"""
[INPUT]: 依赖 tianlong.language 的 sheet / narrator / render / scene / llm（ScriptedLLM），tianlong.runtime.authority（真实内核产出的玩家感知），
         tianlong.scenarios 的 build_wuliang；scripts/replay_drops.py（按文件加载，连带 bench_online）与 tests/data/run6、
         tests/data/playthrough_duanyu.json（run6 前三回合的重放）
[OUTPUT]: 节目单验收：run6 第 3 回合的感知排成 [玩家问话, 马五德回答, 钟灵→龚光杰（合并）, 左子穆→钟灵, 格挡]，
          每对（施动者, 目标）只有一节交手、同一对的两下合成一行、吆喝挂在对应的交手上，交给模型的事实清单按节目单排；
          离开玩家所在地的人必讲，模型漏写就补一句“某某转身往某处去了”，讲到了不补，玩家跟着一起走开的不算；
          玩家换了地方，原处见到的事排在“你来到……”之前；来到眼前的人排在他动手之前，先走后回的回来排在走之后、走不再必讲
[POS]: tests 的节目单（language/sheet）；证伪“交手按发生先后一行一行堆给模型、回答被挤到后面”“人在正文里凭空消失”
       “原处的交手读来像发生在新地方”“先写他动手、再写他来到”“补句替还在眼前的人说转身走了”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

from tianlong.core import Intent, Manner, Op
from tianlong.language import narrator as narrator_mod
from tianlong.language.llm import ScriptedLLM
from tianlong.language.narrator import Narrator
from tianlong.language.render import RenderStatus, build_plan
from tianlong.language.scene import SceneBrief
from tianlong.language.sheet import _LEAVE, ANSWER, CLASH, COME, MOVE, SELF, compose, scene_rows
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_wuliang

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("data")
SC = build_wuliang()
KNOWN = frozenset(e.name for e in SC.state.entities.values()) | {a for al in SC.aliases.values() for a in al}
_ids = itertools.count()


# ============================================================
#  回答在前：run6 第 3 回合（问马五德“这位龚师兄是什么来头”）
# ============================================================


def _replay_drops():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    spec = importlib.util.spec_from_file_location("replay_drops", ROOT / "scripts" / "replay_drops.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_drops"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_answer_first(monkeypatch):
    """那一回合的感知按发生先后是：钟灵一下（中毒）→ 你问马五德 → 马五德招呼 → 钟灵又一下（受伤）→ 左子穆打钟灵 →
    龚光杰打钟灵被化解 → 马五德回话。节目单：你的问话 → 马五德的回答 → 钟灵→龚光杰（两下合一）→ 左子穆→钟灵 → 格挡。"""
    rd = _replay_drops()
    sheets: list[tuple] = []
    real = narrator_mod.compose

    def spy(rows, brief):
        sheets.append(real(rows, brief))
        return sheets[-1]

    monkeypatch.setattr(narrator_mod, "compose", spy)
    given = json.loads((DATA / "run6" / "playthrough.answers.json").read_text("utf-8"))
    interp = rd.bench_online.load_interp(DATA / "run6" / "interp_answers.json")
    inputs = rd.load_inputs(DATA / "playthrough_duanyu.json")[:3]
    dump: list[dict] = []
    rd.bench_online.run_key("playthrough", given, interp, {"playthrough": inputs}, dump=dump)
    sheet = next(s for s in sheets if s and s[0].rows[0].text.startswith("你问马五德"))
    assert [s.kind for s in sheet] == [SELF, ANSWER, CLASH, CLASH, CLASH], [s.lines for s in sheet]
    assert sheet[0].lines == ("你问马五德：“这位龚师兄是什么来头？”",)
    assert sheet[1].pair == ("马五德", "你") and any(vl.answering for vl in sheet[1].voices), "回话那人冲你说的几句合成一节"
    clashes = [s.pair for s in sheet if s.kind == CLASH]
    assert clashes == [("钟灵", "龚光杰"), ("左子穆", "钟灵"), ("龚光杰", "钟灵")]
    assert len(set(clashes)) == len(clashes), "每对（施动者, 目标）只有一节交手"
    assert sheet[2].lines == ("看见钟灵向龚光杰连出两下——龚光杰中了毒，又受了伤",), "同一对的两下合成一行"
    assert "没有成功" in sheet[4].lines[0], "格挡：龚光杰那一下被化解"
    assert [vl.speaker_name for vl in sheet[2].shouts] == ["钟灵"] and [vl.speaker_name for vl in sheet[3].shouts] == ["左子穆"]
    prompt = dump[3]["prompt"]
    facts = prompt.split("本回合玩家感知到的事实：\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert facts == [x for s in sheet for x in s.facts], "交给模型的事实清单按节目单排"
    assert facts[0].startswith("你问马五德") and facts[1] == sheet[2].lines[0]


# ============================================================
#  进出：离开玩家所在地的人必讲
# ============================================================


def _seq(*ticks):
    """依次结算几刻（每刻一组意图）：玩家这几刻的全部感知，与他此刻认得的名字。"""
    auth = WorldAuthority.found(InMemoryWorldStore(), SC)
    percepts = []
    for intents in ticks:
        v = auth.head().version
        s = auth.settle([Intent(f"sh{next(_ids)}", *x, None, v) for x in intents])
        percepts += [o.percept for o in s.observations_of("duanyu")]
    return percepts, auth.store.beliefs(auth.ref, "duanyu").entities


def _settle(*intents):
    return _seq(intents)


def _script(text: str) -> ScriptedLLM:
    return ScriptedLLM(lambda prompt, system, schema: text)


def test_departure_must():
    """干光豪经宫门走了：这一行必讲；模型只写了殿里的光景，就补一句人话（“干光豪转身往无量山山道去了。”）；
    写到了他就不补；玩家自己跟着走开了（同一刻也挪了地方），那就不是离开玩家所在之处。"""
    leave = ("ganguanghao", Op.MOVE, "shandao", "d_gate", Manner.NORMAL)
    percepts, names = _settle(("duanyu", Op.WAIT, None, None, Manner.NORMAL), leave)
    plan = build_plan("duanyu", percepts, names)
    rows = scene_rows(plan, "duanyu", percepts, names, SceneBrief(), "", SC.gate_aliases)
    gone = [r for r in rows if r.text.startswith("看见干光豪走向")]
    assert gone and gone[0].must, "离开玩家所在地的人列为必讲"
    assert [s.kind for s in compose(rows, SceneBrief()) if gone[0] in s.rows] == [MOVE]
    voice = Narrator(_script("殿中一时静了下来，檐下风铃叮当。"), SC.setting, SC.lore, SC.style, SC.gate_aliases,
                     lead_after=None)
    r = voice.narrate_scene("duanyu", percepts, names, brief=SceneBrief(), known=KNOWN)
    said = {x.format(a="干光豪", t="无量山山道") for x in _LEAVE}
    assert r.status == RenderStatus.GATED_FALLBACK and any(r.text.endswith(x) for x in said), r.text
    assert "omitted" in {v.kind for v in r.violations} and "看见" not in r.text, "补句是人话，不是原始事实行"
    told = "干光豪一言不发，转身出了剑湖宫宫门。"
    r = Narrator(_script(told), SC.setting, SC.lore, SC.style, SC.gate_aliases, lead_after=None).narrate_scene(
        "duanyu", percepts, names, brief=SceneBrief(), known=KNOWN)
    assert r.text == told and r.status == RenderStatus.LLM, r.violations
    percepts, names = _settle(("duanyu", Op.MOVE, "shandao", "d_gate", Manner.NORMAL), leave)
    rows = scene_rows(build_plan("duanyu", percepts, names), "duanyu", percepts, names, SceneBrief(), "", SC.gate_aliases)
    together = [r for r in rows if r.text.startswith("看见干光豪走向")]
    assert together and not together[0].must, "一起走开的：不是离开玩家所在之处"


# ============================================================
#  先后：换了地方分段、来到在动手之前、先走后回不算离开
# ============================================================

WAIT = ("duanyu", Op.WAIT, None, None, Manner.NORMAL)
OUT = ("duanyu", Op.MOVE, "shandao", "d_gate", Manner.NORMAL)


def _sheet(*ticks):
    percepts, names = _seq(*ticks)
    rows = scene_rows(build_plan("duanyu", percepts, names), "duanyu", percepts, names, SceneBrief(), "", SC.gate_aliases)
    return rows, [x for s in compose(rows, SceneBrief()) for x in s.lines]


def test_left_behind_first():
    """玩家换了地方：在原处见到的事（大殿里的交手、大殿的所见）排在“你来到无量山山道”之前——
    节目单要是把玩家这一步整个提到最前，模型读来就像钟灵在山道上出手。"""
    _, told = _sheet([OUT, ("zhongling", Op.ATTACK, "gongguangjie", None, Manner.NORMAL)], [WAIT])
    assert told.index(next(x for x in told if "钟灵" in x)) < told.index("你来到无量山山道"), told
    _, told = _sheet([("duanyu", Op.TAKE, "sword", None, Manner.NORMAL)], [OUT])
    assert told[0] == "你拿起长剑" and told.index(next(x for x in told if "干光豪" in x)) < told.index("你来到无量山山道"), told


def test_arrive_before_act():
    """来到眼前的人排在他动手之前；先走后回的人，回来排在走之后，走的那一行也不再必讲（补句不替还在眼前的人说“转身走了”）。"""
    rows, told = _sheet([WAIT, ("shennong", Op.MOVE, "hall", "d_gate", Manner.NORMAL)],
                        [WAIT, ("shennong", Op.ATTACK, "duanyu", None, Manner.NORMAL)])
    assert [s.kind for s in compose(rows, SceneBrief())] == [COME, CLASH] and "来到" in told[0], told
    rows, told = _sheet([WAIT, ("ganguanghao", Op.MOVE, "shandao", "d_gate", Manner.NORMAL)],
                        [WAIT, ("ganguanghao", Op.MOVE, "hall", "d_gate", Manner.NORMAL)])
    assert told[0].startswith("看见干光豪走向") and "来到" in told[1], told
    assert not any(r.must for r in rows), "先走后回：人还在眼前，不必补“某某转身走了”"
