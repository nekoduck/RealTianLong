"""
[INPUT]: 依赖同目录 bench_gm（会话装配、逐回合计分、对照组回放、评审、报告）与 bench_rival 的 PureLLMGM，
         tianlong.language.llm 的 ScriptedLLM，tianlong.core 的 Rel；DIR 下的代理答案（online/*.answers.json、interp_answers.json、
         rival_answers.json、judge_answers.json）
[OUTPUT]: 命令行 python scripts/bench_online.py {keys|step|answer|assemble|dump} DIR [KEY] [--seed N] [--open-actions K]，
          也是可导入的库：load_probes() / session_keys() / load_answers() / load_interp() / run_key()（给定答案重放一个会话）
[POS]: scripts 的在线代理评测（设计 §8.4：没有模型密钥时，由只看这一次调用的 system 与 prompt 的隔离代理替模型作答）。
       与事后按录制正文作答不同，代理看着真实的前文逐次作答，和真接模型时一样：step 从头重放一个会话（同样的答案 → 同样的轨迹），
       停在第一个没答的叙述调用上、打印它的提示词并立刻退出进程（os._exit：模型调用在叙述的读流线程里，评测的 except 也不能吞掉它），
       answer 把 stdin 追加为它的答案；全部 DONE 之后 assemble 拼出本引擎的记录，按对照组已有的代理回复回放纯模型主持人，
       写出盲评面板、评审提示词与报告；dump 把一个会话的每次叙述调用连同答案写下来。
       解释器（快模型）按玩家原文取 interp_answers.json 的答案（同一句话同一份 JSON），没有就走规则解析。
       代理作答是瞬时的：迟到的先声本就不会上场，重放时关掉它，免得机器一卡轨迹就分叉（与先声没上场时逐字相同）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tianlong.core import Rel
from tianlong.language.llm import ScriptedLLM

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 bench_gm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gm  # noqa: E402
from bench_rival import PureLLMGM  # noqa: E402

GROUPS = ("gaslight", "sycophancy", "open_actions")     # 会话键与对照组回放的顺序（与录制时一致，不可改）
Missing = Callable[[dict[str, Any]], str]


# ============================================================
#  会话键与代理答案
# ============================================================


def load_probes(open_actions: int | None = 20) -> dict[str, Any]:
    """bench_gm 的探针；花样输入只取前 open_actions 条（代理作答按条计费）。"""
    probes = bench_gm.load_probes()
    return dict(probes, open_actions=probes["open_actions"][:open_actions])


def session_keys(probes: Mapping[str, Any]) -> list[str]:
    """全部会话键：整局游玩 playthrough，其余每条探针一个 “组:探针 ID”。"""
    return ["playthrough"] + [f"{g}:{p['id']}" for g in GROUPS for p in probes[g]]


def _file(d: Path, key: str, suffix: str) -> Path:
    return d / "online" / f"{key.replace(':', '__')}.{suffix}"


def load_answers(path: Path) -> list[str]:
    """叙述调用的代理答案：第 i 个答第 i 次调用。"""
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def load_interp(path: Path) -> dict[str, str]:
    """解释器的代理答案：玩家原文 → JSON 字符串（同一句话同一份）。"""
    return {a["input"]: a["json"] for a in json.loads(path.read_text("utf-8"))} if path.exists() else {}


# ============================================================
#  重放一个会话
# ============================================================


def run_key(key: str, given: Sequence[str], interp: Mapping[str, str], probes: Mapping[str, Any], seed: int = 7,
            on_missing: Missing | None = None, dump: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """重放一个会话：第 i 次叙述调用取 given[i]，解释器按玩家原文取 interp；答案用完时交给 on_missing（缺省答空串）。
    dump 给出时记下每次叙述调用的回合、system、prompt 与答案。返回 bench_gm 同样的会话记录，另加 calls（叙述调用次数）。"""
    calls: list[str] = []
    tag = ["intro"]

    def voice(prompt: str, system: str | None, schema: Any) -> str:
        i = len(calls)
        calls.append(tag[0])
        if i < len(given):
            if dump is not None:
                dump.append({"call": i, "turn": tag[0], "system": system, "prompt": prompt, "answer": given[i]})
            return given[i]
        pending = {"key": key, "call": i, "turn": tag[0], "system": system, "prompt": prompt}
        return on_missing(pending) if on_missing is not None else ""

    def fast(prompt: str, system: str | None, schema: Any) -> str:
        return interp.get(prompt.rsplit("玩家输入：", 1)[-1].strip(), "")

    turn, session = bench_gm.engine_turn, bench_gm._session

    def tagged(s: Any, text: str, group: str, probe: str | None = None) -> tuple:
        tag[0] = f"{group}|{probe or ''}|{text}"          # 每次叙述调用标上它属于哪一回合（探针的回合经 bench_gm 内部调用）
        return turn(s, text, group, probe)

    def instant(scenario: Any, v: Any, f: Any) -> Any:
        s = session(scenario, v, f)
        s.narrator.lead_after = None                       # 代理作答是瞬时的：迟到的先声本就不上场
        return s

    bench_gm.engine_turn, bench_gm._session = tagged, instant
    try:
        v, f = ScriptedLLM(voice), ScriptedLLM(fast)
        if key == "playthrough":
            out: dict[str, Any] = {"intro": "", "playthrough": [], "errors": [], "ending": None, "final_place": None}
            s = bench_gm._session(bench_gm.build_wuliang(seed), v, f)
            out["intro"] = s.intro()
            for text in probes["playthrough"]:
                rec, *_ = bench_gm.engine_turn(s, text, "playthrough")
                out["playthrough"].append(rec)
                if rec.get("ending"):
                    out["ending"] = rec["ending"]
                    tag[0] = "epilogue"
                    out["epilogue"] = s.epilogue()
                    break
            head = s.authority.head()
            out["final_place"] = head.entity(head.target(s.player, Rel.AT)).name
        else:
            g, pid = key.split(":", 1)
            out = bench_gm._engine_probe(next(x for x in probes[g] if x["id"] == pid), g, seed, v, f)
    finally:
        bench_gm.engine_turn, bench_gm._session = turn, session
    out["calls"] = len(calls)
    return out


# ============================================================
#  命令行的五种模式
# ============================================================


def _halt(d: Path) -> Missing:
    """step：停在第一个没答的叙述调用上——写下它、打印它，立刻退出进程。"""
    def halt(pending: dict[str, Any]) -> str:
        key, system = pending["key"], pending["system"] or ""
        _file(d, key, "pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=1), "utf-8")
        last = _file(d, key, "system.txt")
        same = last.exists() and last.read_text("utf-8") == system
        last.write_text(system, "utf-8")
        shown = "（与上一次调用相同，见前文）" if same else system
        print(f"PENDING {key} call {pending['call']} (turn: {pending['turn']})\n===== SYSTEM =====\n{shown}\n"
              f"===== PROMPT =====\n{pending['prompt']}", flush=True)
        os._exit(0)
    return halt


def _answer(d: Path, key: str, text: str) -> str:
    got = load_answers(_file(d, key, "answers.json"))
    pend = _file(d, key, "pending.json")
    if not pend.exists() or json.loads(pend.read_text("utf-8"))["call"] != len(got):
        raise SystemExit("没有待答的调用，或序号对不上")
    _file(d, key, "answers.json").write_text(json.dumps([*got, text], ensure_ascii=False, indent=1), "utf-8")
    return f"saved answer {len(got)} for {key}"


def assemble(d: Path, probes: Mapping[str, Any], seed: int = 7) -> dict[str, Any]:
    """所有会话都 DONE 之后：拼出本引擎的记录，按对照组已有的代理回复回放纯模型主持人，写出盲评面板、评审提示词与报告。"""
    missing = [k for k in session_keys(probes) if not _file(d, k, "done.json").exists()]
    if missing:
        raise SystemExit(f"还有会话没跑完: {missing}")
    play = json.loads(_file(d, "playthrough", "done.json").read_text("utf-8"))
    engine = {k: play.get(k) for k in ("intro", "playthrough", "errors", "ending", "final_place")}
    if play.get("epilogue"):
        engine["epilogue"] = play["epilogue"]
    engine["probes"] = [json.loads(_file(d, k, "done.json").read_text("utf-8")) for k in session_keys(probes)[1:]]
    # ---- 对照组：按 run_baseline 的调用顺序回放已有的代理回复（对照组的对话本就是在线作答的） ----
    rival = json.loads((d / "rival_answers.json").read_text("utf-8"))
    order = [("playthrough", i) for i in range(len(probes["playthrough"]))]
    for g in GROUPS:
        for p in probes[g]:
            order += [(f"{g}:{p['id']}", i) for i in range(len(p.get("setup", ())) + 1)]
    served = iter(order)

    def rival_respond(prompt: str, system: str | None, schema: Any) -> str:
        key, i = next(served)
        replies = rival.get(key, [])
        return replies[i] if i < len(replies) else ""

    base = bench_gm.run_baseline(probes, ScriptedLLM(rival_respond), seed, None, None, GROUPS, echo=lambda s: None)
    eng_rows = [["（开场）", engine["intro"]]] + [[t["text"], t["narration"]] for t in engine["playthrough"]]
    base_rows = [["（开场）", PureLLMGM(None, bench_gm.build_wuliang(seed)).opening]] + \
        [[t["text"], t["narration"]] for t in base["playthrough"]]
    n = min(len(eng_rows), len(base_rows))
    # 终章的故事部分是主持人的最后一段；“真相揭晓”是落幕后另起的一张卡片（网页上单独展示），不算主持人的叙述
    story = (engine.get("epilogue") or "").split("—— 真相揭晓")[0].strip()
    eng_rows = eng_rows[:n] + ([["（落幕）", story]] if story else [])
    base_rows = base_rows[:n]
    engine_is_a = bench_gm.pairwise_order(seed)
    a, b = (eng_rows, base_rows) if engine_is_a else (base_rows, eng_rows)
    (d / "panel.json").write_text(json.dumps({"bible": base["system"], "A": a, "B": b}, ensure_ascii=False, indent=1), "utf-8")
    (d / "panel_key.json").write_text(json.dumps({"A": "engine" if engine_is_a else "baseline",
                                                  "B": "baseline" if engine_is_a else "engine"}), "utf-8")
    judged = d / "judge_answers.json"
    judge_ans = {x["id"]: x["text"] for x in json.loads(judged.read_text("utf-8"))} if judged.exists() else {}
    jcalls: list[dict[str, Any]] = []

    def judge_respond(prompt: str, system: str | None, schema: Any) -> str:
        i = len(jcalls)
        jcalls.append({"id": f"j{i}", "system": system, "prompt": prompt})
        return judge_ans.get(f"j{i}", "")

    pairwise = bench_gm.run_judges(ScriptedLLM(judge_respond), probes, engine, base, seed)
    (d / "judge_prompts.json").write_text(json.dumps(jcalls, ensure_ascii=False, indent=1), "utf-8")
    result = {"meta": {"mode": "proxy-online", "requested": "proxy", "note": "代理模型在线逐次作答（非真实模型）",
                       "seed": seed, "started": "", "commit": bench_gm._commit(), "elapsed_s": 0, "voice": "proxy",
                       "fast": "proxy", "judge": "proxy" if judge_ans else None, "probes_digest": "",
                       "turns": len(engine["playthrough"]), "probes": len(engine["probes"]), "errors": 0},
              "metrics": {"engine": bench_gm.engine_metrics(engine), "baseline": bench_gm.baseline_metrics(base)},
              "pairwise": pairwise, "engine": engine, "baseline": base}
    bench_gm.write_report(d / "report", result)
    for side in ("engine", "baseline"):
        for m in result["metrics"][side] or ():
            print(side, m["id"], bench_gm._verdict(m["pass"]), m["display"])
    print("judge prompts:", len(jcalls), "answered:", len(judge_ans))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="在线代理评测（设计 §8.4）：代理看着真实的前文逐次替模型作答")
    ap.add_argument("mode", choices=("keys", "step", "answer", "assemble", "dump"),
                    help="keys 列出会话键；step 重放到第一个没答的调用；answer 以 stdin 作答；assemble 拼报告；dump 写下全部调用")
    ap.add_argument("dir", type=Path, help="一轮评测的目录（代理答案与产物都在这里）")
    ap.add_argument("key", nargs="?", help="会话键（playthrough 或 组:探针 ID）；step / answer / dump 必填")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--open-actions", type=int, default=20, help="花样输入探针只取前 K 条")
    args = ap.parse_args(argv)
    d, key = args.dir, args.key
    probes = load_probes(args.open_actions)
    if args.mode == "keys":
        print("\n".join(session_keys(probes)))
        return 0
    if args.mode == "assemble":
        assemble(d, probes, args.seed)
        return 0
    if key is None:
        ap.error(f"{args.mode} 需要会话键")
    (d / "online").mkdir(parents=True, exist_ok=True)
    if args.mode == "answer":
        print(_answer(d, key, sys.stdin.read().strip()))
        return 0
    given, interp = load_answers(_file(d, key, "answers.json")), load_interp(d / "interp_answers.json")
    if args.mode == "dump":
        calls: list[dict[str, Any]] = []
        run_key(key, given, interp, probes, args.seed, dump=calls)
        _file(d, key, "dump.json").write_text(json.dumps(calls, ensure_ascii=False, indent=1), "utf-8")
        print(f"dumped {len(calls)} calls")
        return 0
    rec = run_key(key, given, interp, probes, args.seed, on_missing=_halt(d))
    _file(d, key, "done.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), "utf-8")
    _file(d, key, "pending.json").unlink(missing_ok=True)
    print(f"DONE {key} ({rec['calls']} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
