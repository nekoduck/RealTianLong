"""
[INPUT]: 依赖 tianlong.core 的 Op / Rel，tianlong.language.gate 的 violations（FPD：与叙述闸门同一条路），
         tests/gate_fixtures.py（按文件加载：误杀语料的解码器）与 tests/data/gate_corpus.json
[OUTPUT]: metric() / rate()（指标行与比例的写法，bench_gm 同用），TOKEN_CHARS / tokens()，Metered（包一层模型：记下每次调用的提示词字数与原始回复），
          snapshot() / observe()（本引擎一回合的新字段），ledger()，name_forms() / speaker_forms() / quotes() / repeats()，seam_of()，
          b1() / e1() / names() / seam() / fpd() / r5() / f3() / tok() / end()，engine_extra() / baseline_extra()
[POS]: scripts/bench_gm 的新指标（plan §8.3），单独成文件只为 bench_gm 不超 800 行。
       B1 看点（TurnReport.beats）、E1 空转推进回合、NAME 越过相识账本的名字（会话暴露了玩家的账本才量，否则判“不可得”）、
       SEAM 模型最后一句之后追加的模板行（Metered 记下的原始回复里没有交付的最后一句）、FPD 标注语料上的误杀、
       R5 当面搭话同回合交付回应、F3 同一 NPC 引语与他更早引语的 3-gram 重合、TOK 每回合提示词 token（字数 ÷1.6）、END 结局与回合数。
       逐回合的原始字段由 observe() 在 bench_gm.engine_turn 里记下（只读会话与真相，不改变会话），指标函数只读回合记录——手造记录即可测。
       对照组没有内核、没有账本：R5 / F3 / END 用词法（“某某道：“……””），其余判不适用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from tianlong.core import Op, Rel

ROOT = Path(__file__).resolve().parents[1]
TOKEN_CHARS = 1.6                    # 中文提示词按字数 ÷1.6 估 token
B1_NEED, E1_MAX, R5_MIN, F3_MAX = 8, 0.10, 0.85, 0.05
_BARE = re.compile(r"[\W_]+")
_SENT = re.compile(r"[。！？!?\n]")
_TO = r"(?:对|问|向|求|跟|给|告诉|劝|请|朝|冲)"


def metric(mid: str, name: str, threshold: str, display: str, ok: bool | None, n: int, note: str = "",
           value: Any = None) -> dict[str, Any]:
    return {"id": mid, "name": name, "threshold": threshold, "display": display, "pass": ok, "n": n, "note": note,
            "value": value}


def rate(hit: int, n: int) -> str:
    return f"{hit}/{n} = {hit / n:.0%}" if n else "无数据"


def tokens(chars: int) -> int:
    return round(chars / TOKEN_CHARS)


def _bare(text: str) -> str:
    return _BARE.sub("", text or "")


# ============================================================
#  逐回合记录：Metered 记下叙述调用，snapshot / observe 在回合前后读会话
# ============================================================


class Metered:
    """包一层模型：记下每次调用的 system + prompt 字数、是否结构化（解释器带 schema）与原始回复；其余属性原样转给被包的模型。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        if not callable(getattr(inner, "stream", None)):
            self.stream = None                     # 被包的模型没有流式接口：叙述者照旧一次拿全文

    def __getattr__(self, key: str) -> Any:
        return getattr(self.inner, key)

    def _open(self, prompt: str, kw: Mapping[str, Any], structured: bool) -> dict[str, Any]:
        call = {"chars": len(kw.get("system") or "") + len(prompt), "raw": "", "structured": structured}
        self.calls.append(call)
        return call

    def generate(self, prompt: str, **kw: Any) -> str:
        call = self._open(prompt, kw, kw.get("schema") is not None)
        call["raw"] = self.inner.generate(prompt, **kw)
        return call["raw"]

    def stream(self, prompt: str, **kw: Any):  # type: ignore[no-redef]
        call = self._open(prompt, kw, False)
        for piece in self.inner.stream(prompt, **kw):
            call["raw"] += piece
            yield piece


def ledger(session: Any) -> set[str] | None:
    """玩家此刻的相识账本（认得名字的人）：会话运行态里的 names / acquaintance / acq（{角色: [实体]} 或直接是玩家的列表），
    或会话的 _acq 属性；都没有即不可得（None）。"""
    state = session.session_state() if callable(getattr(session, "session_state", None)) else {}
    for got in (*(state.get(k) for k in ("names", "acquaintance", "acq")), getattr(session, "_acq", None)):
        if isinstance(got, Mapping):
            got = got.get(session.player)
        if isinstance(got, (list, tuple, set, frozenset)):
            return {str(x) for x in got}
    return None


def snapshot(session: Any) -> dict[str, Any]:
    """回合开始前：世界真相、玩家已认识的实体、叙述调用的水位。"""
    calls = getattr(session.llm, "calls", None)
    return {"head": session.authority.head(), "known": frozenset(session.beliefs(session.player).entities),
            "mark": len(calls) if isinstance(calls, list) else None}


def name_forms(scenario: Any) -> dict[str, str]:
    """人名与别称 → 实体（不含玩家、不含普通名词、不含外貌称呼里的字眼）：叙述里出现即算“点了名”。"""
    st, out = scenario.state, {}
    for a in scenario.npcs:
        epithet = scenario.epithets.get(a, "")
        for f in (st.entity(a).name, *scenario.aliases.get(a, ())):
            if len(f) >= 2 and f not in scenario.common_words and f not in epithet:
                out[f] = a
    return out


def speaker_forms(scenario: Any) -> dict[str, str]:
    """说话者的各种叫法（名字、别称、外貌称呼）→ 实体：抽引语用。"""
    return {**{e: a for a, e in scenario.epithets.items()}, **name_forms(scenario)}


def seam_of(narration: str, raw: str) -> bool | None:
    """模型交付过正文、但交付的最后一句不在模型的原始回复里：那是追加在模型最后一句之后的模板行。"""
    said = [s for s in _SENT.split(narration or "") if len(_bare(s)) >= 2]
    got = _bare(raw)
    if not said or not got or not any(_bare(s) in got for s in said):
        return None                                  # 模型一句也没交付（整段是模板）：不是缝，归 G1
    return _bare(said[-1]) not in got


def _answered(report: Any, who: str) -> bool:
    narration = report.narration or ""
    lines = getattr(getattr(report, "brief", None), "lines", None)
    if lines is None:
        return any(e.actor == who and e.op in (Op.TELL, Op.ASK) for e in report.events)
    return any(vl.speaker == who and (vl.answering or getattr(vl, "said", False) or vl.listener_name == "你")
               and ((vl.speaker_name or "") in narration or _bare(vl.template or "")[:6] in _bare(narration))
               for vl in lines)


def observe(session: Any, report: Any, snap: Mapping[str, Any], text: str) -> dict[str, Any]:
    """一回合的新字段：beats、idle（E1）、names 与 name_hits（NAME）、seam / patched（SEAM）、addressed / answered（R5）、tok（TOK）。"""
    player, sc = session.player, session.scenario
    before, me = snap["head"], session.beliefs(session.player)
    beats = list(getattr(report, "beats", ()) or ())
    acted = any(ep.tick >= before.clock and ep.event.actor not in (None, player) for ep in me.episodes)
    known = ledger(session)
    narration = report.narration or ""
    calls = getattr(session.llm, "calls", None)
    mine = [c for c in calls[snap["mark"]:] if not c["structured"]] if snap.get("mark") is not None and calls else []
    here = before.target(player, Rel.AT)
    addressed = next((e.intent.target for e in report.events if e.actor == player and e.op in (Op.TELL, Op.ASK)
                      and e.intent.target in sc.npcs and before.target(e.intent.target, Rel.AT) == here), None)
    violations = getattr(getattr(report, "render", None), "violations", None) or ()
    return {"beats": beats,
            "idle": bool(report.advanced) and not (acted or beats or set(me.entities) - snap["known"]),
            "names": None if known is None else sorted(known),
            "name_hits": None if known is None else sorted({f for f, a in name_forms(sc).items()
                                                            if a not in known and f in narration and f not in text}),
            "seam": seam_of(narration, "".join(c["raw"] for c in mine)) if mine else None,
            "patched": any(getattr(v, "kind", "") == "omitted" for v in violations),
            "addressed": addressed, "answered": None if addressed is None else _answered(report, addressed),
            "tok": tokens(sum(c["chars"] for c in mine)) if mine else None}


# ============================================================
#  词法：引语（某某道：“……”）与 3-gram 重合
# ============================================================


def quotes(text: str, forms: Mapping[str, str]) -> list[tuple[str, str]]:
    """（说话者, 引语）：叫法后面 12 字之内接着引号。"""
    if not forms:
        return []
    names = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
    pat = re.compile(f"({names})" + r"[^“”「」。！？\n]{0,12}?[“「]([^”」]+)[”」]")
    return [(forms[m.group(1)], m.group(2)) for m in pat.finditer(text or "")]


def _grams(text: str, n: int = 3) -> set[str]:
    t = _bare(text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def repeats(narrations: Iterable[str], forms: Mapping[str, str], bar: float = 0.6) -> tuple[int, int]:
    """（与同一人更早的某句引语 3-gram 重合 ≥ bar 的引语数, 引语总数）。不足 3 字的引语不计。"""
    said: dict[str, list[set[str]]] = {}
    hit = total = 0
    for text in narrations:
        for who, line in quotes(text, forms):
            g = _grams(line)
            if not g:
                continue
            total += 1
            hit += any(len(g & p) / len(g) >= bar for p in said.get(who, ()))
            said.setdefault(who, []).append(g)
    return hit, total


# ============================================================
#  指标：只读回合记录（手造记录即可测）
# ============================================================


def b1(turns: Sequence[Mapping], keys: Sequence[str]) -> dict[str, Any]:
    seen = [k for k in keys if any(k in (t.get("beats") or ()) for t in turns)]
    if not keys:
        return metric("B1", "目击的看点", f"≥ {B1_NEED}", "不适用（场景没有看点）", None, 0)
    return metric("B1", "目击的看点", f"≥ {B1_NEED}/{len(keys)}", f"{len(seen)}/{len(keys)}", len(seen) >= B1_NEED,
                  len(turns), "只算整局游玩里玩家感知到的（TurnReport.beats）：" + ("、".join(seen) or "无"), seen)


def _e1(turns: Sequence[Mapping]) -> tuple[int, int]:
    moved = [t for t in turns if t.get("advanced") and t.get("idle") is not None]
    return sum(1 for t in moved if t["idle"]), len(moved)


def e1(turns: Sequence[Mapping]) -> dict[str, Any]:
    hit, n = _e1(turns)
    return metric("E1", "空转的推进回合", f"≤ {E1_MAX:.0%}", rate(hit, n), hit / n <= E1_MAX if n else None, n,
                  "玩家没感知到任何 NPC 的动作或言语、没有新认识的东西、没有看点与景观", hit / n if n else None)


def names(turns: Sequence[Mapping]) -> dict[str, Any]:
    known = [t for t in turns if t.get("name_hits") is not None]
    if not known:
        return metric("NAME", "越过相识账本的名字", "0", "不可得", None, 0, "会话没有暴露玩家的相识账本（session_state 里没有 names）")
    hits = [f for t in known for f in t["name_hits"]]
    return metric("NAME", "越过相识账本的名字", "0", str(len(hits)), not hits, len(known),
                  "交付的句子里出现了玩家账本当时还没有的人名或别称（玩家自己打出的名字不算）"
                  + (f"；例：{'、'.join(dict.fromkeys(hits))}" if hits else ""), len(hits))


def seam(turns: Sequence[Mapping]) -> dict[str, Any]:
    judged = [t for t in turns if t.get("seam") is not None]
    hit = sum(1 for t in judged if t["seam"])
    patched = sum(1 for t in turns if t.get("patched"))
    if not judged:
        return metric("SEAM", "模型最后一句之后追加的模板行", "0", "不适用（没有模型叙述）", None, 0)
    return metric("SEAM", "模型最后一句之后追加的模板行", "0", str(hit), hit == 0, len(judged),
                  f"交付的最后一句不在模型的原始回复里；补句 {patched} 回合（插在钩子之前的不算缝）", hit)


@cache
def _corpus() -> tuple[Any, ...]:
    path = ROOT / "tests" / "gate_fixtures.py"
    spec = importlib.util.spec_from_file_location("bench_gate_fixtures", path)
    if spec is None or spec.loader is None or not path.exists():
        return ()
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                     # 数据类要在 sys.modules 里找得到自己的模块
    spec.loader.exec_module(mod)
    return tuple(mod.corpus())


def fpd() -> dict[str, Any]:
    from tianlong.language.gate import violations
    fps = [c for c in _corpus() if c.verdict == "FP"]
    if not fps:
        return metric("FPD", "标注语料上的误杀", "0", "不可得", None, 0, "找不到 tests/data/gate_corpus.json")
    bad = [c.id for c in fps if violations(c.piece, c.text, c.before, c.plan, c.brief, c.known, c.command)]
    return metric("FPD", "标注语料上的误杀", "0", f"{len(bad)}/{len(fps)}", not bad, len(fps),
                  "tests/data/gate_corpus.json 的 FP 条目经 language.gate.violations" + (f"；误杀：{'、'.join(bad)}" if bad else ""),
                  len(bad))


def r5(turns: Sequence[Mapping], how: str = "") -> dict[str, Any]:
    asked = [t for t in turns if t.get("addressed")]
    hit = sum(1 for t in asked if t.get("answered"))
    return metric("R5", "当面搭话，同回合交付此人的回应", f"≥ {R5_MIN:.0%}", rate(hit, len(asked)),
                  hit / len(asked) >= R5_MIN if asked else None, len(asked),
                  how or "玩家的言语事件冲着回合开始时在场的 NPC；回应 = 交给叙述者的他的台词（answering 或照录的原话）且出现在正文里",
                  hit / len(asked) if asked else None)


def f3(narrations: Sequence[str], forms: Mapping[str, str]) -> dict[str, Any]:
    hit, n = repeats(narrations, forms)
    return metric("F3", "同一 NPC 的引语重复（3-gram ≥ 0.6）", f"≤ {F3_MAX:.0%}", rate(hit, n),
                  hit / n <= F3_MAX if n else None, n, "只算整局游玩；引语按“某某道：“……””从正文里抽（两边同一口径）",
                  hit / n if n else None)


def tok(turns: Sequence[Mapping], what: str) -> dict[str, Any]:
    xs = [t["tok"] for t in turns if t.get("tok")]
    if not xs:
        return metric("TOK", "每回合提示词 token", "—", "不适用（没有模型调用）", None, 0, what)
    mid = sorted(xs)[len(xs) // 2]
    return metric("TOK", "每回合提示词 token", "—", f"中位 {mid} / 首 {xs[0]} → 末 {xs[-1]}（最大 {max(xs)}）", None, len(xs),
                  what + f"；按字数 ÷{TOKEN_CHARS} 估", {"p50": mid, "first": xs[0], "last": xs[-1], "max": max(xs)})


def _ended(rec: Mapping[str, Any]) -> str:
    turns = len(rec.get("playthrough") or ())
    return f"{rec['ending']}（第 {turns} 回合）" if rec.get("ending") else f"未落幕（{turns} 回合）"


def end(rec: Mapping[str, Any], extras: Mapping[str, Mapping[str, Any]] | None = None, how: str = "") -> dict[str, Any]:
    more = "；".join(f"{k}：{_ended(v)}" for k, v in (extras or {}).items())
    return metric("END", "到达的结局与回合数", "—", _ended(rec), None, len(rec.get("playthrough") or ()),
                  "；".join(x for x in (how, more) if x), {"ending": rec.get("ending"), "turns": len(rec.get("playthrough") or ())})


# ============================================================
#  两边的新指标一览
# ============================================================


def engine_extra(engine: Mapping[str, Any], scenario: Any) -> list[dict[str, Any]]:
    play = list(engine.get("playthrough") or ())
    turns = play + [p["turn"] for p in engine.get("probes") or () if "turn" in p]
    extras = engine.get("extras") or {}
    keys = tuple(dict.fromkeys(b.key for b in getattr(scenario, "beats", ()) or ()))
    out = [b1(play, keys), e1(play), names(turns), seam(turns), fpd(), r5(turns),
           f3([t.get("narration", "") for t in play], speaker_forms(scenario)),
           tok(play, "本引擎：叙述调用的 system + prompt"), end(engine, extras)]
    for k, v in extras.items():                       # 留守型、夜遁型：只量分布，不设门槛
        seen = b1(v.get("playthrough") or (), keys)["value"] or []
        hit, n = _e1(v.get("playthrough") or ())
        out[0]["note"] += f"；{k} {len(seen)}/{len(keys)}"
        out[1]["note"] += f"；{k} {rate(hit, n)}"
    return out


def _lexical_r5(turns: Sequence[Mapping], forms: Mapping[str, str]) -> list[dict[str, Any]]:
    names_ = "|".join(sorted(map(re.escape, forms), key=len, reverse=True))
    out = []
    for t in turns:
        m = re.search(_TO + f"({names_})", t.get("text", "")) if names_ else None
        who = forms[m.group(1)] if m else None
        out.append({"addressed": who, "answered": who is not None and any(w == who for w, _ in quotes(t.get("narration", ""), forms))})
    return out


def baseline_extra(baseline: Mapping[str, Any], scenario: Any) -> list[dict[str, Any]]:
    play = list(baseline.get("playthrough") or ())
    turns = play + [p["turn"] for p in baseline.get("probes") or ()]
    forms = speaker_forms(scenario)
    titles = {e.title: e.key for e in getattr(scenario, "endings", ())}
    hit = next((i for i, t in enumerate(play) if any(x in t.get("narration", "") for x in titles)), None)
    rec = {"playthrough": play[:hit + 1] if hit is not None else play,
           "ending": next((k for x, k in titles.items() if x in play[hit]["narration"]), None) if hit is not None else None}
    na = "没有内核与看点识别"
    return [metric("B1", "目击的看点", "—", "不适用", None, 0, na), metric("E1", "空转的推进回合", "—", "不适用", None, 0, na),
            metric("NAME", "越过相识账本的名字", "0", "未测量", None, 0, "没有相识账本，全凭模型自觉"),
            metric("SEAM", "模型最后一句之后追加的模板行", "0", "不适用", None, 0, "没有模板"),
            metric("FPD", "标注语料上的误杀", "0", "不适用", None, 0, "没有闸门"),
            r5(_lexical_r5(turns, forms), "词法启发式：玩家输入“对/问/向……某人”，回复里此人接着引语"),
            f3([t.get("narration", "") for t in play], forms),
            tok(play, "对照组：圣经 + 完整对话 + 新输入"), end(rec, how="词法：回复里出现结局标题")]
