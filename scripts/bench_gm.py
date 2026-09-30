"""
[INPUT]: 依赖 tianlong.runtime.session 的 GameSession（懒加载：核心零依赖套件只用得到指标函数），tianlong.scenarios 的 SCENARIOS / Scenario，
         tianlong.kernel 的 violations 与 kernel/space 的 is_subdued（真相读者），tianlong.language.llm 的 llm_from_env / fast_llm_from_env / GeminiClient（GEMINI_JUDGE_MODEL 另选评审型号），
         tianlong.core 的 Op / Outcome / Rel / Modality / SKILLS / digest，同目录 bench_rival（对照组、评审、脚本模型）、bench_metrics（新指标与逐回合记录）、
         bench_latency（延迟模拟），scripts/bench_probes.json（v2 普通人版；旧版 bench_probes_duanyu.json）
[OUTPUT]: 命令行 python scripts/bench_gm.py --out DIR [--llm auto|scripted|none] [--baseline] [--judge] [--turns N] [--probes FILE] [--latency-sim]，
          也是可导入的库：load_probes / validate_probes / world_of / scenario_of，percentile / ngrams / overlap / kind_of / classify / asserts_claim /
          puppet_hits，claim_met / player_acted，timed_turn / engine_turn / play / run_engine，run_baseline / run_judges，engine_metrics / baseline_metrics，
          write_report，run / main；并再导出 bench_rival 的 world_bible / PureLLMGM / premise_prompt / pairwise_prompt / parse_verdict / pairwise_order / scripted_llms
[POS]: scripts 的主持层评测（设计 §7 与“评测细则”，M4 起加 plan §8.3 的 B1 / E1 / NAME / SEAM / FPD / R5 / F3 / TOK / END）。不属于引擎本体：
       只把 GameSession 当黑盒一回合一回合地跑，按 TurnReport 与世界真相计分（走到结局时连同终章一并记下、交给整局盲评）；探针文件的 variant
       决定场景（没写的是旧版：--probes scripts/bench_probes_duanyu.json 即 run1–6 的评测），不进盲评的整局（留守型、夜遁型）只量 END / E1 / B1。
       会话新加的字段（on_text、first_text_ms、kind、ending、beats、render.violations / dropped、相识账本）一律 getattr 取、缺了就退化。
       每条探针从全新会话出发、先走 setup，异常逐条记下、绝不中断整轮。对照组没有内核可查，C2/R1 只能交给独立评审（--judge，严格 JSON）或词法启发式。
       --llm scripted 的报告开头注明它不是真模型；--latency-sim 按公开分布模拟延迟、报告标注“模拟”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tianlong.core import SKILLS, Modality, Op, Outcome, Rel, digest
from tianlong.kernel import violations as kernel_violations
from tianlong.kernel.space import is_subdued
from tianlong.language.llm import GeminiClient, fast_llm_from_env, llm_from_env
from tianlong.scenarios import SCENARIOS, Scenario

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 bench_rival
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_latency import sim_llms  # noqa: E402
from bench_metrics import Metered, baseline_extra, engine_extra, observe, snapshot, tokens  # noqa: E402
from bench_metrics import metric as _metric  # noqa: E402
from bench_metrics import rate as _rate  # noqa: E402
from bench_rival import (  # noqa: E402
    PAIR_KEYS,
    PAIR_SCHEMA,
    PREMISE_SCHEMA,
    PureLLMGM,
    ask_judge,
    pair_note,
    pairwise_order,
    pairwise_prompt,
    parse_verdict,
    premise_prompt,
    scripted_llms,
    world_bible,
)

__all__ = ["PureLLMGM", "pairwise_order", "pairwise_prompt", "parse_verdict", "premise_prompt", "scripted_llms",
           "world_bible", "load_probes", "validate_probes", "world_of", "scenario_of", "percentile", "ngrams", "overlap",
           "kind_of", "classify", "asserts_claim", "puppet_hits", "claim_met", "player_acted", "timed_turn", "engine_turn",
           "play", "run_engine", "run_baseline", "run_judges", "engine_metrics", "baseline_metrics", "write_report", "run",
           "main"]

PROBES = Path(__file__).with_name("bench_probes.json")           # v2：普通人版（variant "wuliang"）
LEGACY = "wuliang-duanyu"                                        # 没写 variant 的探针（v1 与钉住的输入）是旧版
GROUPS = ("open_actions", "gaslight", "sycophancy")
EXTRAS = ("playthrough_hall", "playthrough_flee")                # 不进盲评的整局：只量 END、E1、B1 的分布
CLAIM_KEYS = frozenset({"holds", "skill", "at", "gave", "to", "subdued", "wounded"})
KINDS = ("act", "say", "gesture", "ask_gm", "meta", "unclear")
CAUGHT = frozenset({"advanced", "ask_gm", "meta"})        # R4：推进了世界，或当作问主持人/元指令作答
Echo = Callable[[str], None]

# ============================================================
#  探针文件
# ============================================================


def load_probes(path: Path | str = PROBES) -> dict[str, Any]:
    return json.loads(Path(path).read_text("utf-8"))


def world_of(probes: Mapping[str, Any]) -> str:
    """探针文件的 variant 决定用哪个场景（SCENARIOS 注册表）；没写的是旧版。"""
    return str(probes.get("variant") or LEGACY)


def scenario_of(probes: Mapping[str, Any], seed: int) -> Scenario:
    return SCENARIOS[world_of(probes)](seed)


def _as_list(v: Any) -> list[Any]:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def validate_probes(data: Mapping[str, Any], scenario: Scenario | None = None) -> list[str]:
    """结构校验；给了场景就再核对 claim/require 里的实体 ID 与技能名。返回错误说明（空 = 合格）。"""
    errs: list[str] = []
    if world_of(data) not in SCENARIOS:
        errs.append(f"variant 不认识 {world_of(data)!r}")
    for key in ("playthrough", *(k for k in EXTRAS if k in data)):
        pt = data.get(key)
        if not isinstance(pt, list) or not pt or not all(isinstance(x, str) and x.strip() for x in pt):
            errs.append(f"{key} 须是非空字符串列表")
    ents = scenario.state.entities if scenario is not None else None
    seen: set[str] = set()
    for g in GROUPS:
        items = data.get(g)
        if not isinstance(items, list):
            errs.append(f"{g} 须是列表")
            continue
        for p in items:
            pid = p.get("id") if isinstance(p, dict) else None
            if not pid or not isinstance(p.get("text"), str) or not p["text"].strip():
                errs.append(f"{g}: 缺 id 或 text: {p!r}")
                continue
            if pid in seen:
                errs.append(f"重复的 id {pid}")
            seen.add(pid)
            if not isinstance(p.get("setup", []), list) or not all(isinstance(s, str) for s in p.get("setup", [])):
                errs.append(f"{pid}: setup 须是字符串列表")
            for key in ("claim", "require"):
                c = p.get(key)
                if c is None:
                    continue
                if not isinstance(c, dict) or set(c) - CLAIM_KEYS or ("to" in c) != ("gave" in c):
                    errs.append(f"{pid}: {key} 不合格式 {c!r}")
                    continue
                for k, v in c.items():
                    for x in _as_list(v):
                        known = x in SKILLS if k == "skill" else (ents is None or x in ents)
                        if not isinstance(x, str) or not known:
                            errs.append(f"{pid}: {key}.{k} 不认识 {x!r}")
            if g != "open_actions" and ("claim" not in p or not p.get("assert") or not p.get("note")):
                errs.append(f"{pid}: {g} 探针须有 claim、assert 与 note")
            if not set(p.get("expect", ())) <= set(KINDS):
                errs.append(f"{pid}: expect 只能取 {KINDS}")
    return errs


# ============================================================
#  指标函数：纯函数，合成数据即可测
# ============================================================


def percentile(values: Iterable[float], q: float) -> float | None:
    """线性插值分位数（与 numpy 默认一致）；空序列返回 None。"""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


_NON_TEXT = re.compile(r"[\W_]+")


def ngrams(text: str, n: int = 4) -> set[str]:
    """字符 n-gram（去掉标点与空白）。"""
    t = _NON_TEXT.sub("", text or "")
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def overlap(prev: str, cur: str, n: int = 4) -> float:
    """后一段的 n-gram 里有多少已出现在前一段（F2 的重复度）。"""
    a, b = ngrams(prev, n), ngrams(cur, n)
    return len(a & b) / len(b) if a and b else 0.0


def kind_of(report: Any) -> str | None:
    k = getattr(report, "kind", None)
    if k is None:
        k = getattr(getattr(report, "parsed", None), "kind", None)
    return None if k is None else str(getattr(k, "value", k))


def classify(report: Any) -> str:
    """R4 归类：advanced / ask_gm / meta 算接住，其余（追问、听不懂）算 unclear，没有回合结果算 error。"""
    if report is None:
        return "error"
    if getattr(report, "advanced", False):
        return "advanced"
    kind = kind_of(report)
    return kind if kind in ("ask_gm", "meta") else "unclear"


_QUOTE = re.compile(r"[“「\"]([^”」\"]*)[”」\"]")
_SENTENCES = re.compile(r"[。！？!?\n]")
_NEG = ("没", "不", "未", "无", "非", "休", "哪", "何来", "却", "但", "别", "岂")
_NOT_NEG = ("无量", "不禁", "不由", "不觉", "不住", "不已", "毫不", "不慌不忙")


def _bare(text: str) -> str:
    return _NON_TEXT.sub("", text or "")


def strip_player_quotes(text: str, player_text: str) -> str:
    """去掉引号里玩家自己的原话：叙述照引玩家说的话，不算叙述者在断言。"""
    said = _bare(player_text)

    def cut(m: re.Match[str]) -> str:
        q = _bare(m.group(1))
        return "" if q and said and (q in said or said in q) else m.group(0)
    return _QUOTE.sub(cut, text or "")


def asserts_claim(narration: str, keywords: Sequence[str], player_text: str = "") -> bool:
    """词法启发式：去掉玩家原话后，某一句里出现声称之物、且没有否定词——当作叙述者断言玩家得手。"""
    for s in _SENTENCES.split(strip_player_quotes(narration, player_text)):
        if any(k in s for k in keywords):
            plain = s
            for w in (*keywords, *_NOT_NEG):          # 关键词自带的“不”（动弹不得）不算否定
                plain = plain.replace(w, "")
            if not any(n in plain for n in _NEG):
                return True
    return False


_CLAUSE = r"(?:^|(?<=[，。！？；：、\s”」]))你"
_PUPPET_SAY = re.compile(_CLAUSE + r"[^“”「」。！？\n]{0,8}?(?:说|道|问|答|喊|叫|嚷|吼)[：:，,]?\s*[“「]([^”」]*)[”」]")
_PUPPET_MIND = re.compile(_CLAUSE + r"[^。！？\n]{0,4}?(?:心想|暗想|心道|暗道|决定|打定主意|下定决心|拿定主意)")


def puppet_hits(narration: str, player_text: str = "") -> list[str]:
    """P1 词法启发式：归到“你”名下却不是玩家原话的引语，以及“你决定/你心想”这类替玩家拿主意的说法。"""
    said = _bare(player_text)
    hits = [m.group(0) for m in _PUPPET_SAY.finditer(narration or "") if _bare(m.group(1)) not in said]
    return hits + [m.group(0) for m in _PUPPET_MIND.finditer(narration or "")]


def _sentence_count(text: str) -> int:
    return sum(1 for s in _SENTENCES.split(text or "") if _bare(s))        # 句末剩下的收引号不算一句


# ============================================================
#  真相读者：声称之物是否在世界里成立
# ============================================================


def claim_met(st: Any, player: str, claim: Mapping[str, Any] | None) -> bool:
    if not claim:
        return False

    def at(e: str | None) -> str | None:
        return st.target(e, Rel.AT) if e and st.has_entity(e) else None
    if "holds" in claim:
        return at(claim["holds"]) == player
    if "gave" in claim:
        return at(claim["gave"]) == claim.get("to")
    if "at" in claim:
        return at(player) == claim["at"]
    if "skill" in claim:
        return any(bool(st.attr(player, s, False)) for s in _as_list(claim["skill"]))
    if "subdued" in claim:
        return any(st.has_entity(p) and is_subdued(st, p) for p in _as_list(claim["subdued"]))
    if "wounded" in claim:
        return any(st.has_entity(p) and bool(st.attr(p, "wounded", False)) for p in _as_list(claim["wounded"]))
    return False


def player_acted(events: Iterable[Any], player: str) -> bool:
    """玩家本回合有成功的非言语、非等待行动（得手须归到玩家名下，NPC 之间打斗不算）。"""
    return any(e.actor == player and e.outcome == Outcome.SUCCESS and e.op not in (Op.WAIT, Op.TELL, Op.ASK)
               for e in events)


# ============================================================
#  本引擎：每回合挂钟计时；首字取 on_text 回调，其次 TurnReport.first_text_ms，再次等于整回合
# ============================================================


def _session(scenario: Scenario, voice: Any, fast: Any) -> Any:
    from tianlong.runtime.session import GameSession
    params = inspect.signature(GameSession.__init__).parameters
    kw: dict[str, Any] = {"llm": Metered(voice) if voice is not None else None}      # 记下叙述调用：TOK 与 SEAM
    extra = next((n for n in ("fast_llm", "fast") if n in params), None)
    if extra and fast is not None:
        kw[extra] = fast
    s = GameSession(scenario, **kw)
    scale = getattr(voice, "lead_scale", None)                   # 延迟模拟缩放了时间：先声的时限同比缩放
    if scale is not None and getattr(s.narrator, "lead_after", None):
        s.narrator.lead_after *= scale
    return s


def timed_turn(session: Any, text: str) -> tuple[Any, float, float, bool, str | None]:
    """→ (TurnReport 或 None, 首字 ms, 整回合 ms, 首字是否真的量到, 异常)。"""
    t0 = time.perf_counter()
    first: list[float] = []

    def sink(piece: str) -> None:
        if piece and not first:
            first.append((time.perf_counter() - t0) * 1000)
    report, err = None, None
    try:
        if "on_text" in inspect.signature(session.turn).parameters:
            report = session.turn(text, on_text=sink)
        else:
            if hasattr(session, "on_text") and not inspect.ismethod(session.on_text):   # 回调挂在会话属性上的写法
                session.on_text = sink
            report = session.turn(text)
    except Exception as e:  # noqa: BLE001 —— 评测逐条记下异常，绝不中断整轮
        err = f"{type(e).__name__}: {e}"
    total = (time.perf_counter() - t0) * 1000
    measured = first[0] if first else getattr(report, "first_text_ms", None)
    return report, (measured if measured is not None else total), total, measured is not None, err


def _present(st: Any, player: str, npcs: Sequence[str]) -> bool:
    here = st.target(player, Rel.AT)
    return any(st.has_entity(n) and st.target(n, Rel.AT) == here for n in npcs)


def _spoke(report: Any, player: str, npcs: frozenset[str], places: set[str | None]) -> bool:
    if any(e.actor in npcs and e.op in (Op.TELL, Op.ASK) and e.place in places for e in report.events):
        return True
    if any(getattr(x, "lines", ()) for x in (report, getattr(report, "brief", None))) or getattr(report, "voice_lines", ()):
        return True                                 # 会话交出了要替 NPC 说出口的台词
    settlement = getattr(report, "settlement", None)
    heard = settlement.observations_of(player) if settlement is not None else ()
    return any(o.percept.modality == Modality.SPEECH and o.percept.informant in npcs for o in heard)


def _status(render: Any) -> str | None:
    status = getattr(render, "status", None)
    return None if status is None else str(getattr(status, "value", status))


def engine_turn(session: Any, text: str, group: str, probe: str | None = None) -> tuple[dict, Any, Any, Any]:
    """跑一回合并记下计分所需的一切 → (记录, TurnReport, 回合前真相, 回合后真相)。"""
    player, npcs = session.player, session.scenario.npcs
    snap = snapshot(session)
    before = snap["head"]
    report, first, total, streamed, err = timed_turn(session, text)
    after = session.authority.head()
    rec: dict[str, Any] = {"group": group, "probe": probe, "text": text, "first_ms": round(first, 1),
                           "total_ms": round(total, 1), "streamed": streamed, "error": err,
                           "c1": len(kernel_violations(after)) + (1 if err and err.startswith("InvariantViolation") else 0)}
    if report is None:
        rec.update(label="error", kind=None, advanced=False, narration="", violations=[], npc_present=False,
                   npc_spoke=False, render=None, dropped=None, ending=None)
        return rec, None, before, after
    render = getattr(report, "render", None)
    ending = getattr(report, "ending", None)
    places = {before.target(player, Rel.AT), after.target(player, Rel.AT)}
    rec.update(label=classify(report), kind=kind_of(report), advanced=bool(report.advanced),
               narration=report.narration or "", render=_status(render),
               violations=[getattr(v, "kind", str(v)) for v in (getattr(render, "violations", None) or ())],
               dropped=getattr(render, "dropped", None),
               npc_present=_present(before, player, npcs) or _present(after, player, npcs),
               npc_spoke=_spoke(report, player, frozenset(npcs), places),
               ending=None if ending is None else str(getattr(ending, "key", ending)), **observe(session, report, snap, text))
    return rec, report, before, after


def _engine_probe(p: Mapping[str, Any], group: str, seed: int, voice: Any, fast: Any, world: str = LEGACY) -> dict[str, Any]:
    rec: dict[str, Any] = {"id": p["id"], "group": group, "text": p["text"], "expect": list(p.get("expect", ())),
                           "setup_ok": True, "setup": [], "c1_setup": 0}
    try:
        s = _session(SCENARIOS[world](seed), voice, fast)
        s.intro()
        for text in p.get("setup", ()):
            srec, *_ = engine_turn(s, text, "setup", p["id"])
            rec["setup"].append({"text": text, "advanced": srec["advanced"], "error": srec["error"]})
            rec["c1_setup"] += srec["c1"]
        if p.get("require") and not claim_met(s.authority.head(), s.player, p["require"]):
            rec["setup_ok"] = False
        turn, report, before, after = engine_turn(s, p["text"], group, p["id"])
        rec["turn"] = turn
        if "claim" in p:
            claim = p["claim"]
            rec["gained"] = bool(report is not None and claim_met(after, s.player, claim)
                                 and not claim_met(before, s.player, claim) and player_acted(report.events, s.player))
            rec["asserted"] = asserts_claim(turn["narration"], p.get("assert", ()), p["text"])
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _echo_turn(echo: Echo, rec: Mapping[str, Any]) -> None:
    echo(f"[{rec['group']}] {rec['text']} → {rec.get('label')} {rec['total_ms']:.0f} ms"
         + (f"  !! {rec['error']}" if rec.get("error") else ""))


def play(scenario: Scenario, inputs: Sequence[str], voice: Any, fast: Any, group: str = "playthrough",
         echo: Echo = print, before_epilogue: Callable[[], None] | None = None) -> dict[str, Any]:
    """一整局：开场、逐句推进，落幕即记下终章并停下。"""
    out: dict[str, Any] = {"intro": "", "playthrough": [], "errors": [], "ending": None, "final_place": None}
    try:
        s = _session(scenario, voice, fast)
        out["intro"] = s.intro()
        for text in inputs:
            rec, *_ = engine_turn(s, text, group)
            out["playthrough"].append(rec)
            _echo_turn(echo, rec)
            if rec.get("ending"):
                out["ending"] = rec["ending"]
                if before_epilogue is not None:
                    before_epilogue()
                out["epilogue"] = s.epilogue()      # 玩家在网页与命令行里落幕时读到的终章（收束 + 真相揭晓）
                break                               # 落幕：本幕到此为止
        head = s.authority.head()
        out["final_place"] = head.entity(head.target(s.player, Rel.AT)).name   # 整局走到了哪里（真相）
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{group}: {type(e).__name__}: {e}")
    return out


def run_engine(probes: Mapping[str, Any], seed: int = 7, voice: Any = None, fast: Any = None, turns: int | None = None,
               limit: int | None = None, groups: Sequence[str] = GROUPS, echo: Echo = print) -> dict[str, Any]:
    world = world_of(probes)
    out = dict(play(SCENARIOS[world](seed), probes["playthrough"][:turns], voice, fast, echo=echo), probes=[])
    extras = {k: play(SCENARIOS[world](seed), probes[k][:turns], voice, fast, k, echo) for k in EXTRAS if k in probes}
    if extras:
        out["extras"] = extras
        out["errors"] += [e for x in extras.values() for e in x["errors"]]
    for g in groups:
        for p in probes.get(g, [])[:limit]:
            rec = _engine_probe(p, g, seed, voice, fast, world)
            out["probes"].append(rec)
            if "turn" in rec:
                _echo_turn(echo, rec["turn"])
            else:
                echo(f"[{g}] {p['text']}  !! {rec.get('error')}")
    return out


# ============================================================
#  对照组与评审的编排：模型侧（PureLLMGM、评审提示词与解析、脚本模型）在 bench_rival
# ============================================================


_ASK_BACK = re.compile(r"你(?:想|要|打算|是想|准备)[^。！\n]{0,24}[？?]")


def _npc_forms(scenario: Scenario) -> list[str]:
    st = scenario.state
    return [f for a in scenario.npcs for f in (st.entity(a).name, *scenario.aliases.get(a, ())) if len(f) >= 2]


def _gm_record(rec: dict[str, Any], group: str, probe: str | None, forms: Sequence[str]) -> dict[str, Any]:
    reply = rec["narration"]
    asked_back = _ASK_BACK.search(_QUOTE.sub("", reply))          # NPC 台词里的“你想怎样？”不算主持人反问
    label = "error" if rec["error"] or not reply else ("unclear" if asked_back else "advanced")
    spoke = any(re.search(re.escape(f) + r"[^“”。！？\n]{0,12}[“「]", reply) for f in forms)
    rec.update(group=group, probe=probe, label=label, npc_present=True, npc_spoke=spoke)
    return rec


def run_baseline(probes: Mapping[str, Any], llm: Any, seed: int = 7, turns: int | None = None, limit: int | None = None,
                 groups: Sequence[str] = GROUPS, echo: Echo = print) -> dict[str, Any]:
    scenario = scenario_of(probes, seed)
    forms = _npc_forms(scenario)
    gm = PureLLMGM(llm, scenario)

    def turn(text: str, group: str, probe: str | None) -> dict[str, Any]:
        tok = tokens(len(gm.system) + len(gm.prompt(text)))       # 圣经 + 完整对话 + 新输入
        return dict(_gm_record(gm.turn(text), group, probe, forms), tok=tok)
    out: dict[str, Any] = {"system": gm.system, "playthrough": [], "probes": []}
    for text in probes["playthrough"][:turns]:
        rec = turn(text, "playthrough", None)
        out["playthrough"].append(rec)
        _echo_turn(echo, rec)
    for g in groups:
        for p in probes.get(g, [])[:limit]:
            gm = PureLLMGM(llm, scenario)
            for text in p.get("setup", ()):
                gm.turn(text)
            rec = turn(p["text"], g, p["id"])
            prec: dict[str, Any] = {"id": p["id"], "group": g, "text": p["text"], "setup_ok": True, "turn": rec}
            if "claim" in p:
                prec["asserted"] = asserts_claim(rec["narration"], p.get("assert", ()), p["text"])
            out["probes"].append(prec)
            _echo_turn(echo, rec)
    return out


def run_judges(llm: Any, probes: Mapping[str, Any], engine: dict, baseline: dict | None, seed: int) -> dict | None:
    """逐条判定两边的 gaslight / sycophancy 回复（结果写回探针记录的 judge），有对照组时再盲评整局游玩。"""
    index = {p["id"]: p for g in GROUPS for p in probes.get(g, [])}
    for side in (engine, baseline):
        for rec in (side or {}).get("probes", ()):
            if rec["group"] in ("gaslight", "sycophancy") and "turn" in rec and rec.get("setup_ok", True):
                rec["judge"] = ask_judge(llm, premise_prompt(index[rec["id"]], rec["group"], rec["turn"]["narration"]),
                                         PREMISE_SCHEMA, {"accepted": "bool"})
    if baseline is None:
        return None
    ours = [(r["text"], r["narration"]) for r in engine["playthrough"]]
    if engine.get("epilogue"):
        ours.append(("（落幕）", engine["epilogue"]))           # 玩家落幕时读到的终章同样交给评审
    theirs = [(r["text"], r["narration"]) for r in baseline["playthrough"]]
    engine_is_a = pairwise_order(seed)
    note = pair_note(scenario_of(probes, seed)) if probes else ""
    verdict = ask_judge(llm, pairwise_prompt(*((ours, theirs) if engine_is_a else (theirs, ours)), note), PAIR_SCHEMA,
                        dict.fromkeys(PAIR_KEYS, "ab"))
    side = {"A": "engine" if engine_is_a else "baseline", "B": "baseline" if engine_is_a else "engine", "tie": "tie"}
    verdict["engine_is"] = "A" if engine_is_a else "B"
    for k in PAIR_KEYS:
        if k in verdict:
            verdict[k] = side[verdict[k]]
    return verdict


# ============================================================
#  汇总：门槛取自设计 §7（R1 见“评测细则”）
# ============================================================


def _latency(mid: str, turns: Sequence[Mapping], key: str, lim50: float, lim95: float) -> dict[str, Any]:
    xs = [t[key] for t in turns if t.get(key) is not None and not t.get("error")]
    p50, p95 = percentile(xs, 50), percentile(xs, 95)
    name = "首字延迟（回车到第一段文字）" if key == "first_ms" else "整回合完成"
    th = f"p50 ≤ {lim50 / 1000:g} s，p95 ≤ {lim95 / 1000:g} s"
    if p50 is None or p95 is None:
        return _metric(mid, name, th, "无数据", None, 0)
    note = f"{sum(1 for t in turns if t.get('streamed'))}/{len(xs)} 回合量到流式首字，其余首字 = 整回合" if key == "first_ms" else ""
    return _metric(mid, name, th, f"p50 {p50 / 1000:.2f} s / p95 {p95 / 1000:.2f} s", p50 <= lim50 and p95 <= lim95,
                   len(xs), note, {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1)})


def _judged(probes: Sequence[Mapping], group: str, mid: str, name: str) -> dict[str, Any] | None:
    verdicts = [p["judge"] for p in probes if p["group"] == group and p.get("judge", {}).get("ok")]
    if not any("judge" in p for p in probes if p["group"] == group):
        return None
    hit = sum(1 for v in verdicts if v["accepted"])
    return _metric(mid, name + "（评审）", "0", _rate(hit, len(verdicts)), hit == 0 if verdicts else None, len(verdicts),
                   "独立评审调用、严格 JSON；无效判定不计入")


def _common(turns: Sequence[Mapping], playthrough: Sequence[Mapping]) -> dict[str, dict[str, Any]]:
    """两边同口径的：L1、L2、F2、P1。"""
    narr = [t["narration"] for t in playthrough]
    pairs = [overlap(a, b) for a, b in zip(narr, narr[1:], strict=False) if a and b]
    mean = sum(pairs) / len(pairs) if pairs else None
    hits = [h for t in turns for h in puppet_hits(t.get("narration", ""), t["text"])]
    return {
        "L1": _latency("L1", turns, "first_ms", 1500, 3000),
        "L2": _latency("L2", turns, "total_ms", 3500, 6000),
        "F2": _metric("F2", "相邻回合的重复（4-gram 重合度）", "≤ 0.15",
                      "无数据" if mean is None else f"均值 {mean:.3f} / 最大 {max(pairs):.3f}",
                      None if mean is None else mean <= 0.15, len(pairs), "只算整局游玩的相邻两段", mean),
        "P1": _metric("P1", "替玩家说话或做决定", "0", str(len(hits)), not hits, len(turns),
                      "词法启发式：归到“你”名下却不是玩家原话的引语、“你决定/你心想”" + (f"；例：{hits[0]}" if hits else ""),
                      len(hits)),
    }


def engine_metrics(engine: Mapping[str, Any], scenario: Scenario | None = None) -> list[dict[str, Any]]:
    """门槛指标（设计 §7）；给了场景再接上 plan §8.3 的新指标（bench_metrics.engine_extra）。"""
    probes = engine["probes"]
    turns = list(engine["playthrough"]) + [p["turn"] for p in probes if "turn" in p]
    common = _common(turns, engine["playthrough"])
    opens = [p for p in probes if p["group"] == "open_actions"]
    caught = [p for p in opens if p.get("turn", {}).get("label") in CAUGHT]
    fits = [p for p in caught if not p.get("expect") or p["turn"].get("kind") in p.get("expect", ())]
    c1 = sum(t.get("c1", 0) for t in turns) + sum(p.get("c1_setup", 0) for p in probes)
    out = [common["L1"], common["L2"],
           _metric("R4", "花样输入在剧情内被接住", "≥ 90%", _rate(len(caught), len(opens)),
                   len(caught) / len(opens) >= 0.9 if opens else None, len(opens),
                   f"接住 = 推进了世界或当作问主持人作答；类别与 expect 吻合 {len(fits)}/{len(opens)}（参考）"),
           _metric("C1", "内核不变量违例", "0", str(c1), c1 == 0, len(turns), "每回合后对世界头版本跑 kernel.violations()", c1)]
    for mid, group, name in (("C2", "gaslight", "瞎编前提被接受"), ("R1", "sycophancy", "不可能的企图得逞")):
        valid = [p for p in probes if p["group"] == group and p.get("setup_ok") and "turn" in p]
        hit = sum(1 for p in valid if p.get("gained"))
        lex = sum(1 for p in valid if p.get("asserted"))
        skipped = sum(1 for p in probes if p["group"] == group and not p.get("setup_ok"))
        out.append(_metric(mid, name, "0", _rate(hit, len(valid)), hit == 0 if valid else None, len(valid),
                           f"按世界真相判定；叙述断言得手（词法启发式）{lex} 条" + (f"；setup 未达成 {skipped} 条不计" if skipped else ""),
                           hit))
        judged = _judged(probes, group, mid + "j", name)
        if judged:
            out.append(judged)
    out.append(_metric("C3", "叙述泄露玩家没感知到的事", "0", "未自动测量", None, 0, "结构保证：叙述只读玩家感知，再过闸门"))
    llm_turns = [t for t in turns if t.get("render") in ("llm", "gated_fallback")]
    exact = all(t.get("dropped") is not None for t in llm_turns)
    dropped = sum(t["dropped"] if t.get("dropped") is not None else sum(1 for v in t.get("violations", ()) if v != "empty")
                  for t in llm_turns)
    sentences = sum(_sentence_count(t["narration"]) for t in llm_turns)
    ratio = dropped / (dropped + sentences) if llm_turns and dropped + sentences else None
    out.append(_metric("G1", "叙述句子被闸门丢弃的比例", "≤ 5%", "不适用（没有模型叙述）" if ratio is None else f"{ratio:.1%}",
                       None if ratio is None else ratio <= 0.05, len(llm_turns),
                       "" if exact or not llm_turns else "会话未暴露丢弃句数：以违规条数近似", ratio))
    present = [t for t in turns if t.get("npc_present")                  # 追问、场外问答不推进时间：NPC 无从开口
               and t.get("advanced", t.get("label") == "advanced")]
    spoke = sum(1 for t in present if t.get("npc_spoke"))
    out.append(_metric("N1", "有 NPC 在场的回合里 NPC 开口", "≥ 50%", _rate(spoke, len(present)),
                       spoke / len(present) >= 0.5 if present else None, len(present),
                       "只数推进了时间的回合；在场按真相；开口 = NPC 的言语事件、玩家听到的 NPC 言语或会话给出的台词"))
    return out + [common["F2"], common["P1"]] + (engine_extra(engine, scenario) if scenario is not None else [])


def baseline_metrics(baseline: Mapping[str, Any], scenario: Scenario | None = None) -> list[dict[str, Any]]:
    probes = baseline["probes"]
    turns = list(baseline["playthrough"]) + [p["turn"] for p in probes]
    common = _common(turns, baseline["playthrough"])
    opens = [p for p in probes if p["group"] == "open_actions"]
    caught = sum(1 for p in opens if p["turn"]["label"] in CAUGHT)
    out = [common["L1"], common["L2"],
           _metric("R4", "花样输入在剧情内被接住", "≥ 90%", _rate(caught, len(opens)),
                   caught / len(opens) >= 0.9 if opens else None, len(opens), "词法启发式：回复里反问“你想……？”算没接住"),
           _metric("C1", "内核不变量违例", "0", "不适用", None, 0, "没有内核，物品与伤势靠模型自己记")]
    for mid, group, name in (("C2", "gaslight", "瞎编前提被接受"), ("R1", "sycophancy", "不可能的企图得逞")):
        valid = [p for p in probes if p["group"] == group]
        lex = sum(1 for p in valid if p.get("asserted"))
        out.append(_metric(mid, name, "0", _rate(lex, len(valid)), lex == 0 if valid else None, len(valid),
                           "没有内核可查：词法启发式（叙述断言得手）", lex))
        judged = _judged(probes, group, mid + "j", name)
        if judged:
            out.append(judged)
    spoke = sum(1 for t in turns if t.get("npc_spoke"))
    return out + [_metric("C3", "叙述泄露玩家没感知到的事", "0", "未测量", None, 0, "全知提示词，只靠“别说漏”"),
                  _metric("G1", "叙述句子被闸门丢弃的比例", "≤ 5%", "不适用", None, 0, "没有闸门"),
                  _metric("N1", "有 NPC 在场的回合里 NPC 开口", "≥ 50%", _rate(spoke, len(turns)),
                          spoke / len(turns) >= 0.5 if turns else None, len(turns), "词法启发式：NPC 名字后紧跟引语；分母为全部回合"),
                  common["F2"], common["P1"]] + (baseline_extra(baseline, scenario) if scenario is not None else [])


# ============================================================
#  报告：bench.json（全部明细）+ report.md（门槛、PASS/FAIL、明细与局限）
# ============================================================


_SIM_NOTE = ("ScriptedLLM 按公开分布抽样（快模型首 token 0.4–0.6 s、每秒 500 字；叙述模型首 token p50 2.2 s、p95 5.3 s，每秒 60 字；"
             "对照组首 token 随 TOK 按同一模型加预填充），时间缩放 ×{scale:g}；不是实测")


def _verdict(ok: bool | None) -> str:
    return "—" if ok is None else ("PASS" if ok else "FAIL")


def write_report(out: Path | str, result: Mapping[str, Any]) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "bench.json"
    js.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), "utf-8")
    meta, eng, base = result["meta"], result["metrics"]["engine"], result["metrics"].get("baseline")
    md = ["# 主持层评测报告（设计 §7）", ""]
    if meta["mode"] in ("scripted", "latency-sim"):
        md += ["> **注意：本次使用 ScriptedLLM（离线脚本回放 + 模拟延迟），不是真实模型。**"
               "延迟、文字与评审结论只证明评测管道跑得通，不能作为验收依据。", ""]
    if meta.get("latency"):
        md += [f"> **L1/L2 是模拟延迟**：{meta['latency']}", ""]
    if meta["mode"] == "none":
        md += ["> 模板模式：没有任何模型调用。L1/L2 只反映引擎自身耗时，G1 不适用；对照组与评审需要模型，已跳过。", ""]
    md += [f"- 模式：{meta['mode']}（叙述 {meta.get('voice') or '模板'} / 解释 {meta.get('fast') or '规则'}"
           + (f" / 评审 {meta['judge']}" if meta.get("judge") else "") + "）" + (f"；{meta['note']}" if meta.get("note") else ""),
           f"- 世界 {meta.get('world', LEGACY)}，种子 {meta['seed']}；提交 {meta.get('commit') or '未知'}；探针文件摘要 {meta['probes_digest']}",
           f"- 生成于 {meta['started']}，用时 {meta['elapsed_s']:.0f} s；整局游玩 {meta['turns']} 回合"
           + (f"（落幕：{result['engine'].get('ending')}）" if result["engine"].get("ending")
              else f"（未落幕，玩家最后在{result['engine'].get('final_place') or '不明之处'}）")
           + f"，探针 {meta['probes']} 条，异常 {meta['errors']} 条", "", "## 指标", ""]
    by_id = {m["id"]: m for m in base or ()}
    head = "| 编号 | 指标 | 门槛 | 本引擎 | 结论 |" + (" 纯模型主持人 | 结论 |" if base else "")
    md += [head, "|---" * (7 if base else 5) + "|"]
    for m in eng:
        row = f"| {m['id']} | {m['name']} | {m['threshold']} | {m['display']} | {_verdict(m['pass'])} |"
        if base:
            b = by_id.get(m["id"])
            row += f" {b['display']} | {_verdict(b['pass'])} |" if b else " — | — |"
        md.append(row)
    md += ["", "说明："]
    md += [f"- {m['id']}（本引擎）：{m['note']}" for m in eng if m["note"]]
    md += [f"- {m['id']}（纯模型主持人）：{m['note']}" for m in base or () if m["note"]]
    pair = result.get("pairwise")
    if pair:
        md += ["", "## 盲评（整局游玩，A/B 顺序由种子固定打乱）", "",
               f"本引擎是 {pair.get('engine_is')}。" + ("" if pair.get("ok") else f"评审输出无效：{pair.get('raw') or pair.get('error')}")]
        if pair.get("ok"):
            reasons = pair.get("reason") if isinstance(pair.get("reason"), dict) else {}
            md += ["", "| 维度 | 胜者 | 理由 |", "|---|---|---|"]
            md += [f"| {k} | {pair[k]} | {reasons.get(k, '')} |" for k in PAIR_KEYS]
    engine = result["engine"]
    md += ["", "## 明细（本引擎）", ""]
    missed = [p for p in engine["probes"] if p["group"] == "open_actions" and p.get("turn", {}).get("label") not in CAUGHT]
    md += ["### R4 没接住的输入", ""] + [f"- {p['id']} {p['text']} → {p.get('turn', {}).get('narration', p.get('error', ''))[:60]}"
                                        for p in missed] + ([] if missed else ["（无）"])
    flagged = [p for p in engine["probes"] if p.get("gained") or p.get("asserted")]
    md += ["", "### C2 / R1 命中或词法标记", ""] + [
        f"- {p['id']} {p['text']}：{'世界里得手' if p.get('gained') else ''}{'；' if p.get('gained') and p.get('asserted') else ''}"
        f"{'叙述断言得手（启发式）' if p.get('asserted') else ''}" for p in flagged] + ([] if flagged else ["（无）"])
    errors = [*engine.get("errors", ()), *(f"{p['id']}: {p['error']}" for p in engine["probes"] if p.get("error")),
              *(f"{t['text']}: {t['error']}" for t in engine["playthrough"] if t.get("error"))]
    md += ["", "### 异常", ""] + [f"- {e}" for e in errors] + ([] if errors else ["（无）"])
    md += ["", "## 口径与局限", "",
           "- 计分口径见 docs/design/game_master.md §7“评测细则”；C2/R1 的自动判定只看世界真相，词法标记与评审只作旁证。",
           "- 启发式：P1、C2/R1 的“叙述断言得手”、对照组的 R4/N1 都是词法近似，会误报也会漏报。",
           "- 单次运行、单一种子；延迟受网络与配额影响，真模型的数字要多跑几轮再下结论。最终裁决归用户。", ""]
    mdp = out / "report.md"
    mdp.write_text("\n".join(md), "utf-8")
    return js, mdp


# ============================================================
#  装配与命令行
# ============================================================


def _commit() -> str | None:
    try:
        got = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                             text=True, timeout=5)
        return got.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(out: Path | str, llm: str = "auto", baseline: bool = False, judge: bool = False, turns: int | None = None,
        limit: int | None = None, seed: int = 7, probes_path: Path | str = PROBES, latency_scale: float = 1.0,
        groups: Sequence[str] = GROUPS, echo: Echo = print, latency_sim: bool = False) -> dict[str, Any]:
    t0, started = time.perf_counter(), time.strftime("%Y-%m-%d %H:%M:%S")
    probes = load_probes(probes_path)
    scenario = scenario_of(probes, seed) if world_of(probes) in SCENARIOS else None
    errs = validate_probes(probes, scenario)
    if errs:
        raise ValueError("探针文件不合格：" + "；".join(errs[:5]))
    mode, note = llm, ""
    voice = fast = gm_llm = judge_llm = None
    if llm == "auto":
        from tianlong.runtime.cli import load_dotenv
        load_dotenv()
        voice = llm_from_env(cache_dir=None)                  # 评测一律不开缓存：同一提示词回放同一段文字会掩盖真实延迟
        if voice is None:
            mode, note = "scripted", "auto 模式没有找到 GEMINI_API_KEY，按设计 §7 退回 ScriptedLLM"
        else:
            mode = "gemini"
            fast, gm_llm = fast_llm_from_env(cache_dir=None), llm_from_env(cache_dir=None)
            other = os.environ.get("GEMINI_JUDGE_MODEL", "").strip()     # 评审可换成别的型号，免得自己评自己
            judge_llm = GeminiClient(os.environ["GEMINI_API_KEY"].strip(), other, thinking="low") if other \
                else llm_from_env(cache_dir=None)
    if mode == "scripted" and latency_sim:      # 按公开分布抽样首 token（plan §8.4），报告标注“模拟”
        mode, fakes = "latency-sim", sim_llms(latency_scale)
        voice, fast, gm_llm, judge_llm = fakes["voice"], fakes["fast"], fakes["baseline"], fakes["judge"]
    elif mode == "scripted":
        fakes = scripted_llms(latency_scale)
        voice, fast, gm_llm, judge_llm = fakes["voice"], fakes["fast"], fakes["baseline"], fakes["judge"]
    engine = run_engine(probes, seed, voice, fast, turns, limit, groups, echo)
    base = run_baseline(probes, gm_llm, seed, turns, limit, groups, echo) if baseline and gm_llm is not None else None
    if baseline and base is None:
        note += ("；" if note else "") + "对照组需要模型，已跳过"
    pairwise = None
    if judge and judge_llm is not None:
        pairwise = run_judges(judge_llm, probes, engine, base, seed)
    elif judge:
        note += ("；" if note else "") + "评审需要模型，已跳过"
    n_probes = len(engine["probes"])
    result = {
        "meta": {"mode": mode, "requested": llm, "note": note, "seed": seed, "world": world_of(probes),
                 "latency": _SIM_NOTE.format(scale=latency_scale) if mode == "latency-sim" else None, "started": started, "commit": _commit(),
                 "elapsed_s": round(time.perf_counter() - t0, 1), "voice": getattr(voice, "model", None),
                 "fast": getattr(fast, "model", None), "judge": getattr(judge_llm, "model", None) if judge else None,
                 "probes_digest": digest(json.dumps({k: v for k, v in probes.items() if k != "variant"}, sort_keys=True,
                                                   ensure_ascii=False))[:12],        # 不含 variant：旧版探针摘要与 run1–6 相同
                 "turns": len(engine["playthrough"]), "probes": n_probes,
                 "errors": len(engine["errors"]) + sum(1 for p in engine["probes"] if p.get("error"))
                 + sum(1 for t in engine["playthrough"] if t.get("error"))},
        "metrics": {"engine": engine_metrics(engine, scenario), "baseline": baseline_metrics(base, scenario) if base else None},
        "pairwise": pairwise, "engine": engine, "baseline": base,
    }
    write_report(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="主持层评测（设计 §7）：本引擎 vs 纯模型主持人")
    ap.add_argument("--out", required=True, help="输出目录：写 bench.json 与 report.md")
    ap.add_argument("--llm", choices=["auto", "scripted", "none"], default="auto",
                    help="auto = 环境里的 Gemini（缓存关闭，无密钥退回 scripted）；scripted = 离线脚本模型；none = 模板")
    ap.add_argument("--baseline", action="store_true", help="同时跑纯模型主持人对照组")
    ap.add_argument("--judge", action="store_true", help="独立评审：对照组的 C2/R1 与整局盲评")
    ap.add_argument("--turns", type=int, default=None, help="整局游玩只跑前 N 回合")
    ap.add_argument("--limit", type=int, default=None, help="每组探针只跑前 K 条")
    ap.add_argument("--groups", default=",".join(GROUPS), help="要跑的探针组，逗号分隔")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--latency-scale", type=float, default=1.0, help="scripted 模式的模拟延迟倍数（0 = 不等待）")
    ap.add_argument("--latency-sim", action="store_true", help="scripted 模式按公开分布抽样首 token（bench_latency），报告标注“模拟”")
    args = ap.parse_args(argv)
    groups = tuple(g for g in args.groups.split(",") if g in GROUPS)
    try:
        result = run(args.out, args.llm, args.baseline, args.judge, args.turns, args.limit, args.seed, args.probes,
                     args.latency_scale, groups, echo=lambda s: print(s, flush=True), latency_sim=args.latency_sim)
    except ValueError as e:
        print(e)
        return 2
    for m in result["metrics"]["engine"]:
        print(f"{m['id']:>3} {_verdict(m['pass']):<4} {m['display']}  （门槛 {m['threshold']}）")
    print(f"报告：{Path(args.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
