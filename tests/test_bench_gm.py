"""
[INPUT]: 依赖 scripts/bench_gm.py（按文件加载，连带同目录的 bench_rival）与 scripts/bench_probes.json，tianlong.scenarios 的 build_wuliang，
         tianlong.language.llm 的 ScriptedLLM，tianlong.language.parser 的 MoveKind / Parsed
[OUTPUT]: 主持层评测工具的测试：探针文件结构与实体核对、指标函数（分位数、4-gram 重合、R4 归类、C2/R1 真相判定、词法启发式、指标汇总、首字计时的三种来源）、
          --llm none 三条探针的端到端（写出 bench.json 与 report.md）、scripted 全链路（对照组 + 评审 + 盲评）、
          纯模型主持人保留完整对话并量出首字与总耗时、世界圣经全知、评审 JSON 解析容忍坏输入、脚本解释器兼容新旧两种 schema
[POS]: tests 的评测工具验收；只跑离线、快速的路径（真模型与全量探针由命令行手动跑）。会话相关用例缺 LangGraph / Qdrant 时跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tianlong.language.llm import ScriptedLLM
from tianlong.language.parser import MoveKind, Parsed
from tianlong.scenarios import build_wuliang

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_gm", ROOT / "scripts" / "bench_gm.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_gm"] = bench
_spec.loader.exec_module(bench)


def _needs_session() -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")


def _tiny(tmp_path: Path) -> Path:
    """三条探针（每组一条）+ 两回合游玩：端到端只求快。"""
    full = bench.load_probes()
    tiny = {"playthrough": ["环顾四周", "叹了口气"],
            "open_actions": [full["open_actions"][0]],
            "gaslight": [full["gaslight"][0]],
            "sycophancy": [full["sycophancy"][0]]}
    path = tmp_path / "probes.json"
    path.write_text(json.dumps(tiny, ensure_ascii=False), "utf-8")
    return path


# ============================================================
#  探针文件
# ============================================================


def test_probes_file_schema_and_entities():
    data = bench.load_probes()
    assert bench.validate_probes(data, build_wuliang(7)) == []
    assert 25 <= len(data["playthrough"]) <= 40
    assert [len(data[g]) for g in bench.GROUPS] == [60, 15, 10]
    texts = {p["text"] for g in bench.GROUPS for p in data[g]}
    for example in ("拱手向众人赔个不是", "给钟灵讲个笑话", "装疯卖傻", "问马五德神农帮是什么来头", "我躲到钟灵身后",
                    "叹了口气", "大声吟诗", "我从怀里掏出北冥神功", "我施展凌波微步", "我把解药给龚光杰", "我一招制服左子穆"):
        assert example in texts
    # 需要铺垫的探针写明了 setup 与必须达成的前提
    cliff = next(p for p in data["sycophancy"] if p["claim"] == {"at": "yading"})
    assert cliff["setup"] and cliff["require"] == {"at": "jianhu"}


def test_validate_probes_rejects_malformed():
    bad = {"playthrough": [],
           "open_actions": [{"id": "o1"}, {"id": "o2", "text": "叹气", "expect": ["dance"]}],
           "gaslight": [{"id": "g1", "text": "我掏出金牌", "claim": {"owns": "x"}, "assert": ["金牌"], "note": "无"},
                        {"id": "g1", "text": "重复", "claim": {"holds": "nobody"}, "assert": ["x"], "note": "无"}],
           "sycophancy": [{"id": "s1", "text": "我飞走", "claim": {"skill": "flying"}}]}
    errs = "\n".join(bench.validate_probes(bad, build_wuliang(7)))
    for needle in ("playthrough", "缺 id 或 text", "expect", "不合格式", "重复的 id", "不认识 'nobody'",
                   "不认识 'flying'", "须有 claim、assert 与 note"):
        assert needle in errs


# ============================================================
#  指标函数（合成数据）
# ============================================================


def test_percentile_linear_interpolation():
    assert bench.percentile([], 50) is None
    assert bench.percentile([5.0], 95) == 5.0
    assert bench.percentile([4, 1, 3, 2], 50) == 2.5
    assert bench.percentile(range(1, 101), 95) == pytest.approx(95.05)
    assert bench.percentile([10, 20], 100) == 20


def test_four_gram_overlap():
    a = "龚光杰冷笑一声，拔剑出鞘。"
    assert bench.overlap(a, a) == 1.0
    assert bench.overlap("春眠不觉晓", "处处闻啼鸟") == 0.0
    assert bench.overlap("龚光杰冷笑", "龚光杰，冷笑！") == 1.0          # 标点与空白不计
    assert bench.overlap("abc", "abc") == 0.0                          # 不足 4 字没有 4-gram
    # 后一段的 4 个 4-gram 里只有“钟灵拍手”出现在前一段
    assert bench.overlap("钟灵拍手笑道", "钟灵拍手叫好呀") == pytest.approx(1 / 4)


def test_r4_classification():
    ask = Parsed(None, kind=MoveKind.ASK_GM, question="我该怎么办")
    assert bench.classify(SimpleNamespace(advanced=True, parsed=Parsed(None))) == "advanced"
    assert bench.classify(SimpleNamespace(advanced=False, parsed=ask)) == "ask_gm"
    assert bench.classify(SimpleNamespace(advanced=False, kind=MoveKind.META, parsed=Parsed(None))) == "meta"
    assert bench.classify(SimpleNamespace(advanced=False, parsed=Parsed(None, clarification="你想做什么？"))) == "unclear"
    assert bench.classify(SimpleNamespace(advanced=False, kind="unclear", parsed=ask)) == "unclear"   # 回合上的 kind 优先
    assert bench.classify(None) == "error"


def test_claim_met_reads_world_truth():
    sc = build_wuliang(7)
    st, me = sc.state, sc.player
    assert bench.claim_met(st, me, {"holds": "yijing"})
    assert not bench.claim_met(st, me, {"holds": "scroll_bm"})
    assert bench.claim_met(st, me, {"gave": "antidote", "to": "zhongling"})
    assert bench.claim_met(st, me, {"at": "hall"}) and not bench.claim_met(st, me, {"at": "lancang"})
    assert not bench.claim_met(st, me, {"skill": ["absorb", "evasion"]})
    assert not bench.claim_met(st, me, {"subdued": ["zuozimu"]}) and not bench.claim_met(st, me, {"wounded": ["zuozimu"]})
    assert not bench.claim_met(st, me, {})
    ev = lambda actor, op, outcome: SimpleNamespace(actor=actor, op=op, outcome=outcome)  # noqa: E731
    from tianlong.core import Op, Outcome
    assert bench.player_acted([ev(me, Op.TAKE, Outcome.SUCCESS)], me)
    assert not bench.player_acted([ev(me, Op.TELL, Outcome.SUCCESS), ev(me, Op.ATTACK, Outcome.FAILURE),
                                   ev("zhongling", Op.ATTACK, Outcome.SUCCESS)], me)


def test_lexical_heuristics():
    assert bench.asserts_claim("你从怀里掏出北冥神功，众人大惊。", ["北冥神功"])
    assert not bench.asserts_claim("你身上并没有北冥神功帛卷。", ["北冥神功"])
    assert not bench.asserts_claim("你问钟灵：“钟灵，你昨天答应过我的事呢”", ["答应"], "钟灵，你昨天答应过我的事呢")
    assert bench.asserts_claim("你走到无量玉璧前，已到剑湖畔。", ["剑湖畔"])            # “无量”不算否定
    assert bench.asserts_claim("你点中穴道，他登时动弹不得。", ["动弹不得"])            # 关键词自带的“不”也不算
    assert bench.puppet_hits("你笑道：“在下告辞。”", "拱手")
    assert not bench.puppet_hits("你说道：“在下失礼了”", "对龚光杰说：在下失礼了")
    assert not bench.puppet_hits("钟灵对你说：“呆子。”龚光杰瞪着你道：“哼。”", "")
    assert bench.puppet_hits("你决定先离开这里。", "")


def _turn(text, narration, label="advanced", **kw):
    return {"group": "g", "text": text, "narration": narration, "label": label, "first_ms": 100.0, "total_ms": 200.0,
            "streamed": False, "error": None, "c1": 0, "violations": [], "render": None, "dropped": None,
            "npc_present": False, "npc_spoke": False, **kw}


def test_engine_metrics_on_synthetic_records():
    probe = lambda pid, group, turn, **kw: {"id": pid, "group": group, "text": turn["text"], "setup_ok": True,  # noqa: E731
                                            "c1_setup": 0, "turn": turn, **kw}
    engine = {
        "playthrough": [_turn("环顾四周", "你看到龚光杰站在殿中。", npc_present=True, npc_spoke=True),
                        _turn("等待", "你看到龚光杰站在殿中。", npc_present=True, first_ms=9000.0, total_ms=9000.0),
                        _turn("叹气", "你笑道：“我走了。”", render="llm", violations=["puppet"], c1=1)],
        "probes": [probe("o1", "open_actions", _turn("拱手", "你拱手。")),
                   probe("o2", "open_actions", _turn("我该怎么办", "先离开大殿。", label="ask_gm")),
                   probe("o3", "open_actions", _turn("装疯", "没听懂。", label="unclear")),
                   probe("g1", "gaslight", _turn("掏出神功", "你身上并没有。"), gained=False, asserted=False),
                   probe("s1", "sycophancy", _turn("爬回崖顶", "你爬回崖顶。"), gained=True, asserted=True),
                   probe("s2", "sycophancy", _turn("爬回崖顶", "……"), gained=True, setup_ok=False)],
        "errors": [],
    }
    m = {x["id"]: x for x in bench.engine_metrics(engine)}
    assert m["R4"]["display"] == "2/3 = 67%" and m["R4"]["pass"] is False
    assert m["C1"]["value"] == 1 and m["C1"]["pass"] is False
    assert m["C2"]["pass"] is True
    assert m["R1"]["display"] == "1/1 = 100%" and "setup 未达成 1 条" in m["R1"]["note"]      # setup 没达成的不计入
    assert m["L1"]["value"] == {"p50_ms": 100.0, "p95_ms": pytest.approx(5440.0)} and m["L1"]["pass"] is False
    assert m["L2"]["value"]["p50_ms"] == 200.0 and m["L2"]["pass"] is True    # 9 回合里一回合 9 s：p95 插值 5.44 s
    assert m["N1"]["display"] == "1/2 = 50%" and m["N1"]["pass"] is True
    assert m["F2"]["value"] == pytest.approx((1.0 + 0.0) / 2)
    assert m["P1"]["value"] == 1 and m["P1"]["pass"] is False
    assert m["G1"]["value"] == pytest.approx(1 / 2) and "近似" in m["G1"]["note"]            # 1 条违规 vs 1 句交付
    assert m["C3"]["pass"] is None


def test_timed_turn_reads_first_text_from_session():
    import time

    class Streaming:                                   # 会话支持 on_text：首字取第一次回调
        def turn(self, text, request_id=None, on_text=None):
            time.sleep(0.03)
            on_text("你")
            time.sleep(0.03)
            return SimpleNamespace(advanced=True, narration="你好")

    class Reported:                                    # 会话自己量首字
        def turn(self, text, request_id=None):
            return SimpleNamespace(advanced=True, narration="你好", first_text_ms=12.5)

    class Plain:                                       # 老会话：首字 = 整回合
        def turn(self, text, request_id=None):
            return SimpleNamespace(advanced=True, narration="你好")

    class Broken:
        def turn(self, text, request_id=None):
            raise RuntimeError("会话炸了")

    _, first, total, streamed, err = bench.timed_turn(Streaming(), "x")
    assert streamed and err is None and 25 <= first < total
    assert bench.timed_turn(Reported(), "x")[1] == 12.5 and bench.timed_turn(Reported(), "x")[3]
    _, first, total, streamed, _ = bench.timed_turn(Plain(), "x")
    assert first == total and not streamed
    report, _, _, _, err = bench.timed_turn(Broken(), "x")
    assert report is None and err == "RuntimeError: 会话炸了"


# ============================================================
#  评审解析：坏 JSON 判无效而不是瞎猜
# ============================================================


def test_parse_verdict_tolerates_malformed_json():
    ok = bench.parse_verdict('```json\n{"accepted": true, "reason": "他接受了"}\n```', {"accepted": "bool"})
    assert ok == {"ok": True, "accepted": True, "reason": "他接受了"}
    assert bench.parse_verdict('判定如下：{"accepted": false, "reason": "拒绝"} 以上。', {"accepted": "bool"})["accepted"] is False
    assert bench.parse_verdict("判定：{accepted: false}", {"accepted": "bool"}) == {"ok": True, "accepted": False,
                                                                               "reason": None}
    for bad in ("完全不是 JSON", "", None, "[1, 2]", '{"accepted": "maybe"}', '{"reason": "缺字段"}', '{"accepted": tru'):
        got = bench.parse_verdict(bad, {"accepted": "bool"})
        assert got["ok"] is False and "accepted" not in got
    fields = dict.fromkeys(bench.PAIR_KEYS, "ab")
    pair = bench.parse_verdict('{"coherent": "a", "reasonable": "tie", "fun": "B", "reasons": {"fun": "更好看"}}', fields)
    assert pair["ok"] and (pair["coherent"], pair["reasonable"], pair["fun"]) == ("A", "tie", "B")
    assert bench.parse_verdict('{"coherent": "A", "reasonable": "C", "fun": "B"}', fields)["ok"] is False


def test_pairwise_verdict_maps_back_through_seeded_order():
    engine = {"playthrough": [_turn("环顾四周", "引擎的叙述")], "probes": []}
    baseline = {"playthrough": [_turn("环顾四周", "对照组的叙述")], "probes": []}
    judge = ScriptedLLM(lambda p, s, sc: '{"coherent": "A", "reasonable": "B", "fun": "tie", "reasons": {}}')
    for seed in (1, 2, 3, 4):
        a_is_engine = bench.pairwise_order(seed)
        assert bench.pairwise_order(seed) == a_is_engine                     # 同一种子，顺序不变
        got = bench.run_judges(judge, {}, engine, baseline, seed)
        assert got["engine_is"] == ("A" if a_is_engine else "B")
        assert got["coherent"] == ("engine" if a_is_engine else "baseline") and got["fun"] == "tie"
        prompt = judge.prompts[-1][1]
        assert (prompt.index("引擎的叙述") < prompt.index("对照组的叙述")) == a_is_engine
    assert {bench.pairwise_order(s) for s in range(20)} == {True, False}     # 真的会换边


# ============================================================
#  对照组：纯模型主持人
# ============================================================


def test_pure_llm_gm_keeps_transcript_and_measures_latency():
    reply = "你环顾四周。龚光杰冷笑道：“看什么看？”"
    llm = ScriptedLLM(lambda p, s, sc: reply, first_delay=0.05, chunk=4)
    gm = bench.PureLLMGM(llm, build_wuliang(7))
    r1 = gm.turn("环顾四周")
    r2 = gm.turn("问马五德神农帮是什么来头")
    assert gm.transcript == [("环顾四周", reply), ("问马五德神农帮是什么来头", reply)]
    system, prompt = llm.prompts[-1]
    assert system == gm.system
    assert "玩家：环顾四周" in prompt and reply in prompt and prompt.endswith("玩家：问马五德神农帮是什么来头\n主持人：")
    for r in (r1, r2):
        assert r["streamed"] and r["error"] is None and r["narration"] == reply
        assert r["first_ms"] >= 50 and r["total_ms"] >= r["first_ms"]


def test_pure_llm_gm_without_stream_and_on_failure():
    class GenerateOnly:
        model = "fake"

        def generate(self, prompt, *, system=None, schema=None, temperature=None, max_tokens=None):
            return "你等了一会儿。"
    r = bench.PureLLMGM(GenerateOnly(), build_wuliang(7)).turn("等待")
    assert r["narration"] == "你等了一会儿。" and not r["streamed"] and r["first_ms"] == r["total_ms"]

    def broken(prompt, system, schema):
        raise RuntimeError("网络断了")
    gm = bench.PureLLMGM(ScriptedLLM(broken), build_wuliang(7))
    r = gm.turn("等待")
    assert r["error"].startswith("RuntimeError") and gm.transcript == [("等待", "")]


def test_world_bible_is_omniscient():
    bible = bench.world_bible(build_wuliang(7))
    for needle in ("不替玩家", "停在钩子上", "声口各异", "连贯",                   # Jenova 的公开主持规矩
                   "只能从后山崖顶到剑湖畔", "只在夜里月光下显现",                   # 单向与夜现路线
                   "私奔", "闪电貂：在钟灵身上", "北冥神功帛卷：在蒲团上", "藏着",     # 秘密与物品所在
                   "与段誉为敌", "19:20 之后才动身", "龚光杰（东宗弟子）"):         # 目标与时间闸门
        assert needle in bible, needle


# ============================================================
#  --llm scripted 的脚本解释器：新旧两种 schema 都给出合乎 schema 的 JSON
# ============================================================


def test_scripted_interpreter_fits_both_schemas():
    from bench_rival import scripted_respond
    new = {"properties": dict.fromkeys(("kind", "mode", "actor", "steps", "listener", "speech", "line", "social",
                                        "topic_subject", "topic_value", "topic_holds", "missing", "reply"))}
    table = "你是段誉（duanyu）。\nzhongling|钟灵/钟姑娘|人|剑湖宫大殿\nhall|剑湖宫大殿|地点|\n"
    got = json.loads(scripted_respond(table + "玩家输入：给钟灵讲个笑话", None, new))
    assert set(got) == set(new["properties"])
    assert (got["kind"], got["listener"], got["social"]) == ("say", "zhongling", "joke")
    assert json.loads(scripted_respond(table + "玩家输入：我从怀里掏出北冥神功", None, new))["missing"] == "北冥神功"
    assert json.loads(scripted_respond(table + "玩家输入：叹了口气", None, new))["kind"] == "gesture"
    old = {"properties": dict.fromkeys(("mode", "actor", "op", "target", "obj", "manner", "topic_subject",
                                        "topic_value", "topic_holds", "clarification"))}
    legacy = "你是 duanyu。你认识的实体：\n- zhongling：钟灵（person，你认为在 hall）\n\n"
    got = json.loads(scripted_respond(legacy + "玩家输入：给钟灵讲个笑话", None, old))
    assert set(got) == set(old["properties"]) and (got["op"], got["target"]) == ("tell", "zhongling")


# ============================================================
#  端到端（需要会话：LangGraph + Qdrant）
# ============================================================


def test_end_to_end_template_three_probes(tmp_path):
    _needs_session()
    out = tmp_path / "out"
    assert bench.main(["--out", str(out), "--llm", "none", "--probes", str(_tiny(tmp_path))]) == 0
    data = json.loads((out / "bench.json").read_text("utf-8"))
    report = (out / "report.md").read_text("utf-8")
    assert data["meta"]["mode"] == "none" and data["meta"]["errors"] == 0
    assert len(data["engine"]["playthrough"]) == 2 and len(data["engine"]["probes"]) == 3
    assert all("turn" in p for p in data["engine"]["probes"])
    metrics = {m["id"]: m for m in data["metrics"]["engine"]}
    assert {"L1", "L2", "R4", "C1", "C2", "R1", "G1", "N1", "F2", "P1"} <= set(metrics)
    assert metrics["C1"]["value"] == 0 and metrics["C2"]["value"] == 0 and metrics["R1"]["value"] == 0
    assert metrics["G1"]["pass"] is None                                     # 模板模式没有模型叙述
    assert data["baseline"] is None and "模板模式" in report
    assert "| C2 | 瞎编前提被接受 | 0 |" in report and ("PASS" in report or "FAIL" in report)


def test_end_to_end_scripted_with_baseline_and_judge(tmp_path):
    _needs_session()
    result = bench.run(tmp_path / "out", llm="scripted", baseline=True, judge=True, probes_path=_tiny(tmp_path),
                       latency_scale=0.0, echo=lambda s: None)
    report = (tmp_path / "out" / "report.md").read_text("utf-8")
    assert "ScriptedLLM" in report and "不是真实模型" in report and "纯模型主持人" in report
    assert result["meta"]["mode"] == "scripted" and result["meta"]["voice"] == "scripted-voice"
    assert result["pairwise"]["ok"] and result["pairwise"]["engine_is"] in ("A", "B")
    judged = [p for side in ("engine", "baseline") for p in result[side]["probes"] if p["group"] != "open_actions"]
    assert judged and all(p["judge"]["ok"] for p in judged)
    ids = {m["id"] for m in result["metrics"]["baseline"]}
    assert {"C2j", "R1j", "F2", "P1"} <= ids
    assert len(result["baseline"]["playthrough"]) == 2
