"""
[INPUT]: 依赖 scripts/bench_metrics.py 与 scripts/bench_latency.py（按文件加载），tianlong.core 的 Op / Rel，tianlong.language.llm 的 ScriptedLLM，
         tianlong.scenarios 的 build_wuliang_commoner
[OUTPUT]: M4 新指标（plan §8.3）在手造的回合记录上逐个算得出、算得对：B1（看点并集对 11 个）、E1（只数推进回合）、NAME（账本不可得即“不可得”，
          可得时数越过账本的名字）、SEAM（最后一句不在模型原始回复里；整段模板不算缝）、FPD（标注语料的 FP 条目全部放行）、
          R5（当面搭话同回合有回应）、F3（同一人引语 3-gram 重合 ≥0.6，按“某某道：“……””抽）、TOK（首 → 末 与中位）、END（结局与回合数）；
          两边一览的编号齐全；Metered 记下提示词字数与原始回复且不改变模型的回答、没有流式接口就不假装有；ledger() 认得几种账本写法；
          observe() 在假会话上给出 idle / name_hits / seam / addressed / answered / tok；
          延迟模拟：抽样确定（同一次序同一值）、快模型落在 0.4–0.6 s、叙述模型中位约 2.2 s 且 p95 约 5.3 s、提示词超过 TOK_REF 首 token 变长、
          SimLLM 按缩放睡眠且记下未缩放的秒数
[POS]: tests 的评测指标单元测试：只依赖核心（不起会话），核心零依赖 CI 同跑
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tianlong.core import Op, Rel
from tianlong.language.llm import ScriptedLLM
from tianlong.scenarios import build_wuliang_commoner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bm = _load("bench_metrics")
lat = _load("bench_latency")
SC = build_wuliang_commoner(7)
KEYS = tuple(dict.fromkeys(b.key for b in SC.beats))


def _t(**kw):
    return {"text": "环顾四周", "narration": "", "advanced": True, "beats": [], "idle": False, **kw}


# ============================================================
#  指标：手造记录
# ============================================================


def test_b1_and_e1():
    turns = [_t(beats=["challenge", "beating"]), _t(beats=["marten"], idle=False), _t(idle=True),
             _t(advanced=False, idle=None), _t(beats=["challenge"])]
    b1 = bm.b1(turns, KEYS)
    assert b1["display"] == "3/11" and b1["pass"] is False and b1["value"] == ["challenge", "beating", "marten"]
    assert bm.b1(turns + [_t(beats=list(KEYS[3:8]))], KEYS)["pass"] is True
    assert bm.b1(turns, ())["pass"] is None                              # 旧版没有看点：不适用
    e1 = bm.e1(turns)
    assert e1["display"] == "1/4 = 25%" and e1["pass"] is False           # 不推进的回合不进分母
    assert bm.e1([_t()] * 10)["pass"] is True and bm.e1([])["pass"] is None


def test_name_unavailable_or_counted():
    assert bm.names([_t(name_hits=None)])["display"] == "不可得"
    got = bm.names([_t(name_hits=[]), _t(name_hits=["钟灵"]), _t(name_hits=["钟灵", "干光豪"])])
    assert got["value"] == 3 and got["pass"] is False and "钟灵、干光豪" in got["note"]
    assert bm.names([_t(name_hits=[])])["pass"] is True


def test_seam():
    assert bm.seam_of("你四下看了看。段誉跟你说笑。", "你四下看了看。") is True          # 模板行挂在模型最后一句之后
    assert bm.seam_of("你四下看了看。段誉跟你说笑。钟灵笑了。", "你四下看了看。钟灵笑了。") is False   # 补句在钩子之前
    assert bm.seam_of("时间悄悄过去。", "全被闸门丢了的一段话。") is None                 # 模型一句没交付：归 G1
    assert bm.seam_of("你看了看。", "") is None
    turns = [_t(seam=True, patched=True), _t(seam=False, patched=True), _t(seam=None)]
    m = bm.seam(turns)
    assert m["value"] == 1 and m["n"] == 2 and "补句 2 回合" in m["note"]
    assert bm.seam([_t(seam=None)])["pass"] is None


def test_fpd_on_the_labelled_corpus():
    m = bm.fpd()
    assert m["value"] == 0 and m["pass"] is True and m["display"] == "0/28"


def test_r5():
    turns = [_t(addressed="mawude", answered=True), _t(addressed="duanyu", answered=False), _t(addressed=None),
             _t(addressed="zhongling", answered=True)]
    m = bm.r5(turns)
    assert m["display"] == "2/3 = 67%" and m["pass"] is False
    assert bm.r5([_t(addressed="mawude", answered=True)] * 7)["pass"] is True and bm.r5([])["pass"] is None


def test_quotes_and_f3():
    forms = bm.speaker_forms(SC)
    assert forms["梁上的青衫少女"] == "zhongling" and forms["钟灵"] == "zhongling" and "姑娘" not in forms
    text = "钟灵拍手笑道：“书呆子你快看！石壁上有仙人！”马五爷叹道：“和气生财。”你道：“好。”"
    assert bm.quotes(text, forms) == [("zhongling", "书呆子你快看！石壁上有仙人！"), ("mawude", "和气生财。")]
    play = [text, "钟灵又道：“书呆子你快看，石壁上有仙人！”", "钟灵道：“我叫钟灵。”", "马五德道：“哈哈，龚老弟消消气。”"]
    hit, n = bm.repeats(play, forms)
    assert (hit, n) == (1, 5)                                              # 钟灵的第二句与第一句几乎一样
    m = bm.f3(play, forms)
    assert m["display"] == "1/5 = 20%" and m["pass"] is False
    assert bm.f3(play[2:], forms)["pass"] is True


def test_tok_and_end():
    m = bm.tok([_t(tok=900), _t(tok=None), _t(tok=950), _t(tok=880)], "本引擎")
    assert m["value"] == {"p50": 900, "first": 900, "last": 880, "max": 950} and m["pass"] is None
    assert bm.tok([_t()], "x")["display"].startswith("不适用")
    rec = {"playthrough": [_t()] * 37, "ending": "river"}
    extras = {"playthrough_hall": {"playthrough": [_t()] * 23, "ending": "dawn"},
              "playthrough_flee": {"playthrough": [_t()] * 5, "ending": None}}
    m = bm.end(rec, extras)
    assert m["display"] == "river（第 37 回合）" and m["value"] == {"ending": "river", "turns": 37}
    assert "playthrough_hall：dawn（第 23 回合）" in m["note"] and "playthrough_flee：未落幕（5 回合）" in m["note"]


def test_both_sides_list_every_new_metric():
    play = [_t(beats=["challenge"], narration="钟灵道：“你好。”", tok=800, seam=False, name_hits=[],
               addressed="zhongling", answered=True)]
    engine = {"playthrough": play, "probes": [], "ending": None,
              "extras": {"playthrough_hall": {"playthrough": [_t(idle=True)], "ending": "dawn"}}}
    ids = [m["id"] for m in bm.engine_extra(engine, SC)]
    assert ids == ["B1", "E1", "NAME", "SEAM", "FPD", "R5", "F3", "TOK", "END"]
    e = {m["id"]: m for m in bm.engine_extra(engine, SC)}
    assert "playthrough_hall 0/11" in e["B1"]["note"] and "playthrough_hall 1/1 = 100%" in e["E1"]["note"]
    base = {"playthrough": [{"text": "问马五爷这是怎么回事", "narration": "马五爷叹道：“这可难办。”", "tok": 4000},
                            {"text": "等待", "narration": "天亮了。第一幕终 · 天亮了", "tok": 4500}], "probes": []}
    b = {m["id"]: m for m in bm.baseline_extra(base, SC)}
    assert list(b) == ids and b["R5"]["display"] == "1/1 = 100%" and b["END"]["value"] == {"ending": "dawn", "turns": 2}
    assert b["TOK"]["value"]["last"] == 4500 and b["NAME"]["display"] == "未测量"


# ============================================================
#  逐回合记录：Metered、账本、observe
# ============================================================


def test_metered_records_calls_without_changing_answers():
    inner = ScriptedLLM(lambda p, s, sc: "你看了看。" if sc is None else "{}", "fake")
    m = bm.Metered(inner)
    assert m.model == "fake" and "".join(m.stream("问", system="系统")) == "你看了看。"
    assert m.generate("解释", system="S", schema={"type": "object"}) == "{}"
    assert m.calls == [{"chars": 3, "raw": "你看了看。", "structured": False}, {"chars": 3, "raw": "{}", "structured": True}]

    class GenerateOnly:
        def generate(self, prompt, **kw):
            return "好"
    assert bm.Metered(GenerateOnly()).stream is None                        # 没有流式接口就不假装有


def test_ledger_forms():
    def s(state, acq=None):
        return SimpleNamespace(player="ashun", session_state=lambda: state, **({"_acq": acq} if acq is not None else {}))
    assert bm.ledger(s({"scheduler": {}})) is None
    assert bm.ledger(s({"names": {"ashun": ["duanyu", "mawude"], "duanyu": ["ashun"]}})) == {"duanyu", "mawude"}
    assert bm.ledger(s({"acquaintance": ["zhongling"]})) == {"zhongling"}
    assert bm.ledger(s({}, {"ashun": {"gongguangjie"}})) == {"gongguangjie"}


def test_observe_on_a_fake_session():
    head = SC.state
    ep = SimpleNamespace(tick=head.clock, event=SimpleNamespace(actor="mawude"))
    me = SimpleNamespace(episodes=(ep,), entities={"ashun": 1, "mawude": 1})
    calls = [{"chars": 999, "raw": "旧的", "structured": False}]
    session = SimpleNamespace(player="ashun", scenario=SC, llm=SimpleNamespace(calls=calls),
                              authority=SimpleNamespace(head=lambda: head), beliefs=lambda a: me,
                              session_state=lambda: {"names": {"ashun": ["mawude", "duanyu", "gongguangjie"]}})
    snap = bm.snapshot(session)
    assert snap["mark"] == 1 and snap["known"] == frozenset({"ashun", "mawude"})
    calls += [{"chars": 1600, "raw": "你问马五德。马五德答道：“那是东宗的龚光杰。”", "structured": False},
              {"chars": 500, "raw": "{}", "structured": True}]
    ask = SimpleNamespace(actor="ashun", op=Op.ASK, intent=SimpleNamespace(target="mawude"))
    line = SimpleNamespace(speaker="mawude", speaker_name="马五德", answering="这是谁", listener_name="你", template="")
    report = SimpleNamespace(beats=(), advanced=True, events=(ask,), brief=SimpleNamespace(lines=(line,)),
                             narration="你问马五德。马五德答道：“那是东宗的龚光杰。”钟灵在梁上笑。\n马五德向你解释。",
                             render=SimpleNamespace(violations=(SimpleNamespace(kind="omitted"),)))
    got = bm.observe(session, report, snap, "问马五爷这是谁")
    assert got["idle"] is False and got["beats"] == []                      # 马五德这一回合有动作：不算空转
    assert got["name_hits"] == ["钟灵"] and got["names"] == ["duanyu", "gongguangjie", "mawude"]
    assert got["seam"] is True and got["patched"] is True                  # 模型之后追加的“马五德向你解释。”
    assert got["addressed"] == "mawude" and got["answered"] is True and got["tok"] == 1000
    assert head.target("mawude", Rel.AT) == head.target("ashun", Rel.AT)


# ============================================================
#  延迟模拟
# ============================================================


def test_latency_draws_follow_the_public_distribution():
    assert lat.first_token("voice", 3, 100) == lat.first_token("voice", 3, 100)     # 确定性
    fast = [lat.first_token("fast", i, 100) for i in range(500)]
    assert all(0.4 <= x <= 0.6 for x in fast)
    voice = sorted(lat.first_token("voice", i, 100) for i in range(4000))
    assert voice[2000] == pytest.approx(lat.VOICE_P50, rel=0.08) and voice[3800] == pytest.approx(lat.VOICE_P95, rel=0.12)
    assert lat.first_token("rival", 5, 20000) - lat.first_token("rival", 5, 100) == pytest.approx(17000 / lat.PREFILL_TPS)


def test_sim_llm_sleeps_scaled_and_keeps_raw_draws():
    llm = lat.SimLLM(lambda p, s, sc: "你好", "sim", "voice", scale=0.01)
    t0 = time.perf_counter()
    assert "".join(llm.stream("问", system="S")) == "你好" and llm.generate("再问") == "你好"
    took = time.perf_counter() - t0
    assert len(llm.drawn) == 2 and llm.drawn[0] == round(lat.first_token("voice", 0, 3 / 1.6), 3)
    assert took == pytest.approx(sum(llm.drawn) * 0.01 + 4 / lat.CPS * 0.01, abs=0.05)
    assert set(lat.sim_llms(0.5)) == {"voice", "fast", "baseline", "judge"}
