"""
[INPUT]: 依赖同目录 bench_gm（会话装配、整局与探针的逐回合计分、对照组回放、评审、报告、探针文件与 variant）与 bench_rival 的
         PureLLMGM / epilogue_story / panel_name，
         tianlong.language.llm 的 ScriptedLLM；DIR 下的代理答案（<运行目录>/online/*.answers.json、interp_answers.json、rival_answers.json、
         judge_answers.json）
[OUTPUT]: 命令行 python scripts/bench_online.py {keys|step|answer|assemble|dump} DIR [KEY] [--world W] [--seed N] [--probes FILE]
          [--open-actions K]，也是可导入的库：PROBE_FILES / RIVAL，load_probes() / session_keys() / run_dir() / online_dir() /
          load_answers() / load_interp() / run_key()（给定答案重放本引擎的一个会话）/ rival_key()（给定答案重放对照组的一个会话）/ assemble()
[POS]: scripts 的在线代理评测（设计 §8.4：没有模型密钥时，由只看这一次调用的 system 与 prompt 的隔离代理替模型作答）。
       与事后按录制正文作答不同，代理看着真实的前文逐次作答，和真接模型时一样：step 从头重放一个会话（同样的答案 → 同样的轨迹），
       停在第一个没答的调用上、打印它的提示词并立刻退出进程（os._exit：模型调用在叙述的读流线程里，评测的 except 也不能吞掉它），
       answer 把 stdin 追加为它的答案。本引擎的键是 playthrough / playthrough_hall / playthrough_flee / “组:探针 ID”，
       对照组（纯模型主持人，圣经 + 完整对话 + 新输入）的键加前缀 “B:”，同一套 step / answer；每个种子一个运行目录，
       两个种子（7 与 11）即对照组独立跑两次。全部 DONE 之后 assemble 拼出本引擎的记录（不进盲评的整局记在 extras），
       按对照组的代理回复（B:* 全部 DONE；一个都没跑且是旧版种子 7 才取录制的 DIR/rival_answers.json，否则报出没跑完的键）
       回放纯模型主持人，写出盲评面板（A/B 先后按面板 bench_rival.panel_name 轮流、终章只取收束）、评审提示词与报告；
       dump 把一个会话的每次叙述调用连同答案写下来。
       --world 选场景（wuliang 普通人版 / wuliang-duanyu 旧版，缺省看探针文件的 variant），--probes 选探针文件（缺省按世界取）；
       运行目录：旧版种子 7 就是 DIR 本身（run1–6 的布局原样可读），其余是 DIR/<world>-s<seed>。
       解释器（快模型）按玩家原文取 interp_answers.json 的答案（运行目录里有就用它，否则取 DIR 的；同一句话同一份 JSON），没有就走规则解析。
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

from tianlong.language.llm import ScriptedLLM

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 bench_gm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gm  # noqa: E402
from bench_rival import PureLLMGM, epilogue_story, panel_name  # noqa: E402

GROUPS = ("gaslight", "sycophancy", "open_actions")     # 会话键与对照组回放的顺序（与录制时一致，不可改）
PROBE_FILES = {"wuliang": bench_gm.PROBES, bench_gm.LEGACY: bench_gm.PROBES.with_name("bench_probes_duanyu.json")}
RIVAL = "B:"                                            # 对照组的会话键前缀
Missing = Callable[[dict[str, Any]], str]


# ============================================================
#  会话键、运行目录与代理答案
# ============================================================


def load_probes(open_actions: int | None = 20, path: Path | str = bench_gm.PROBES) -> dict[str, Any]:
    """bench_gm 的探针；花样输入只取前 open_actions 条（代理作答按条计费）。"""
    probes = bench_gm.load_probes(path)
    return dict(probes, open_actions=probes["open_actions"][:open_actions])


def session_keys(probes: Mapping[str, Any], rival: bool = False) -> list[str]:
    """本引擎：playthrough、不进盲评的整局、每条探针一个 “组:探针 ID”；对照组（rival）：键加 B: 前缀，没有不进盲评的整局。"""
    extras = [] if rival else [k for k in bench_gm.EXTRAS if k in probes]
    keys = ["playthrough", *extras] + [f"{g}:{p['id']}" for g in GROUPS for p in probes[g]]
    return [RIVAL + k for k in keys] if rival else keys


def run_dir(d: Path, world: str, seed: int) -> Path:
    """一个（世界, 种子）的运行目录：旧版种子 7 是 DIR 本身（run1–6 的布局），其余 DIR/<world>-s<seed>。"""
    return d if (world, seed) == (bench_gm.LEGACY, 7) else d / f"{world}-s{seed}"


def online_dir(d: Path, world: str, seed: int) -> Path:
    return run_dir(d, world, seed) / "online"


def _file(od: Path, key: str, suffix: str) -> Path:
    return od / f"{key.replace(':', '__')}.{suffix}"


def load_answers(path: Path) -> list[str]:
    """一个会话的代理答案：第 i 个答第 i 次调用。"""
    return json.loads(path.read_text("utf-8")) if path.exists() else []


def load_interp(path: Path) -> dict[str, str]:
    """解释器的代理答案：玩家原文 → JSON 字符串（同一句话同一份）。"""
    return {a["input"]: a["json"] for a in json.loads(path.read_text("utf-8"))} if path.exists() else {}


# ============================================================
#  重放一个会话：本引擎（run_key）与对照组（rival_key）
# ============================================================


def run_key(key: str, given: Sequence[str], interp: Mapping[str, str], probes: Mapping[str, Any], seed: int = 7,
            on_missing: Missing | None = None, dump: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """重放本引擎的一个会话：第 i 次叙述调用取 given[i]，解释器按玩家原文取 interp；答案用完时交给 on_missing（缺省答空串）。
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

    def epilogue() -> None:
        tag[0] = "epilogue"

    bench_gm.engine_turn, bench_gm._session = tagged, instant
    world = bench_gm.world_of(probes)
    try:
        v, f = ScriptedLLM(voice), ScriptedLLM(fast)
        if key == "playthrough" or key in bench_gm.EXTRAS:
            out = bench_gm.play(bench_gm.SCENARIOS[world](seed), probes[key], v, f, key, lambda s: None, epilogue)
        else:
            g, pid = key.split(":", 1)
            out = bench_gm._engine_probe(next(x for x in probes[g] if x["id"] == pid), g, seed, v, f, world)
    finally:
        bench_gm.engine_turn, bench_gm._session = turn, session
    out["calls"] = len(calls)
    return out


def rival_key(key: str, given: Sequence[str], probes: Mapping[str, Any], seed: int = 7,
              on_missing: Missing | None = None) -> dict[str, Any]:
    """重放对照组的一个会话（键带 B: 前缀）：第 i 次调用取 given[i]，用完交给 on_missing。返回 {replies, calls}。"""
    bare, calls = key.removeprefix(RIVAL), []
    tag = [""]

    def respond(prompt: str, system: str | None, schema: Any) -> str:
        i = len(calls)
        calls.append(tag[0])
        if i < len(given):
            return given[i]
        pending = {"key": key, "call": i, "turn": tag[0], "system": system, "prompt": prompt}
        return on_missing(pending) if on_missing is not None else ""

    gm = PureLLMGM(ScriptedLLM(respond), bench_gm.scenario_of(probes, seed))
    if bare == "playthrough":
        texts = list(probes["playthrough"])
    else:
        g, pid = bare.split(":", 1)
        p = next(x for x in probes[g] if x["id"] == pid)
        texts = [*p.get("setup", ()), p["text"]]
    for text in texts:
        tag[0] = text
        gm.turn(text)
    return {"replies": [reply for _, reply in gm.transcript], "calls": len(calls)}


# ============================================================
#  命令行：step 停在第一个没答的调用上，answer 以 stdin 作答
# ============================================================


def _halt(od: Path) -> Missing:
    """step：停在第一个没答的调用上——写下它、打印它，立刻退出进程。"""
    def halt(pending: dict[str, Any]) -> str:
        key, system = pending["key"], pending["system"] or ""
        _file(od, key, "pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=1), "utf-8")
        last = _file(od, key, "system.txt")
        same = last.exists() and last.read_text("utf-8") == system
        last.write_text(system, "utf-8")
        shown = "（与上一次调用相同，见前文）" if same else system
        print(f"PENDING {key} call {pending['call']} (turn: {pending['turn']})\n===== SYSTEM =====\n{shown}\n"
              f"===== PROMPT =====\n{pending['prompt']}", flush=True)
        os._exit(0)
    return halt


def _answer(od: Path, key: str, text: str) -> str:
    got = load_answers(_file(od, key, "answers.json"))
    pend = _file(od, key, "pending.json")
    if not pend.exists() or json.loads(pend.read_text("utf-8"))["call"] != len(got):
        raise SystemExit("没有待答的调用，或序号对不上")
    _file(od, key, "answers.json").write_text(json.dumps([*got, text], ensure_ascii=False, indent=1), "utf-8")
    return f"saved answer {len(got)} for {key}"


def _interp(d: Path, rd: Path) -> dict[str, str]:
    return load_interp(rd / "interp_answers.json" if (rd / "interp_answers.json").exists() else d / "interp_answers.json")


def _rival_answers(rd: Path, od: Path, probes: Mapping[str, Any], legacy: bool) -> dict[str, list[str]]:
    """对照组的代理回复 {会话键: [回复]}：B:* 全部 DONE 就用它们（在线逐次作答）；一个都没跑、且是旧版种子 7 的运行目录
    （run1–6 的布局）才取录制的 rival_answers.json；其余一律报出没跑完的 B:* 键——绝不拿别的世界、别的种子的录制顶替。"""
    keys = session_keys(probes, rival=True)
    done = {k: _file(od, k, "done.json") for k in keys}
    left = [k for k, p in done.items() if not p.exists()]
    if not left:
        return {k.removeprefix(RIVAL): json.loads(p.read_text("utf-8"))["replies"] for k, p in done.items()}
    if legacy and len(left) == len(keys) and (rd / "rival_answers.json").exists():
        return json.loads((rd / "rival_answers.json").read_text("utf-8"))
    raise SystemExit(f"对照组还有会话没跑完: {left}")


def assemble(d: Path, probes: Mapping[str, Any], seed: int = 7) -> dict[str, Any]:
    """所有会话都 DONE 之后：拼出本引擎的记录，按对照组的代理回复回放纯模型主持人，写出盲评面板、评审提示词与报告。"""
    world = bench_gm.world_of(probes)
    rd = run_dir(d, world, seed)
    od = rd / "online"
    missing = [k for k in session_keys(probes) if not _file(od, k, "done.json").exists()]
    if missing:
        raise SystemExit(f"还有会话没跑完: {missing}")

    def done(k: str) -> dict[str, Any]:
        return json.loads(_file(od, k, "done.json").read_text("utf-8"))
    play = done("playthrough")
    engine = {k: play.get(k) for k in ("intro", "playthrough", "errors", "ending", "final_place")}
    if play.get("epilogue"):
        engine["epilogue"] = play["epilogue"]
    extras = [k for k in bench_gm.EXTRAS if k in probes]
    engine["probes"] = [done(k) for k in session_keys(probes) if k != "playthrough" and k not in extras]
    if extras:
        engine["extras"] = {k: done(k) for k in extras}
    # ---- 对照组：按 run_baseline 的调用顺序回放已有的代理回复（对照组的对话本就是在线作答的） ----
    rival = _rival_answers(rd, od, probes, legacy=rd == d)
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
    scenario = bench_gm.scenario_of(probes, seed)
    eng_rows = [["（开场）", engine["intro"]]] + [[t["text"], t["narration"]] for t in engine["playthrough"]]
    base_rows = [["（开场）", PureLLMGM(None, scenario).opening]] + [[t["text"], t["narration"]] for t in base["playthrough"]]
    n = min(len(eng_rows), len(base_rows))
    # 终章的故事部分是主持人的最后一段；真相揭晓 / 纪事是落幕后另起的一张卡片（网页上单独展示），不算主持人的叙述
    story = epilogue_story(engine.get("epilogue"))
    eng_rows = eng_rows[:n] + ([["（落幕）", story]] if story else [])
    base_rows = base_rows[:n]
    engine_is_a = bench_gm.pairwise_order(seed, panel_name(world, seed))
    a, b = (eng_rows, base_rows) if engine_is_a else (base_rows, eng_rows)
    (rd / "panel.json").write_text(json.dumps({"bible": base["system"], "A": a, "B": b}, ensure_ascii=False, indent=1), "utf-8")
    (rd / "panel_key.json").write_text(json.dumps({"A": "engine" if engine_is_a else "baseline",
                                                   "B": "baseline" if engine_is_a else "engine"}), "utf-8")
    judged = rd / "judge_answers.json"
    judge_ans = {x["id"]: x["text"] for x in json.loads(judged.read_text("utf-8"))} if judged.exists() else {}
    jcalls: list[dict[str, Any]] = []

    def judge_respond(prompt: str, system: str | None, schema: Any) -> str:
        i = len(jcalls)
        jcalls.append({"id": f"j{i}", "system": system, "prompt": prompt})
        return judge_ans.get(f"j{i}", "")

    pairwise = bench_gm.run_judges(ScriptedLLM(judge_respond), probes, engine, base, seed)
    (rd / "judge_prompts.json").write_text(json.dumps(jcalls, ensure_ascii=False, indent=1), "utf-8")
    result = {"meta": {"mode": "proxy-online", "requested": "proxy", "note": "代理模型在线逐次作答（非真实模型）",
                       "seed": seed, "world": world, "started": "", "commit": bench_gm._commit(), "elapsed_s": 0,
                       "voice": "proxy", "fast": "proxy", "judge": "proxy" if judge_ans else None, "probes_digest": "",
                       "turns": len(engine["playthrough"]), "probes": len(engine["probes"]), "errors": 0},
              "metrics": {"engine": bench_gm.engine_metrics(engine, scenario),
                          "baseline": bench_gm.baseline_metrics(base, scenario)},
              "pairwise": pairwise, "engine": engine, "baseline": base}
    bench_gm.write_report(rd / "report", result)
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
    ap.add_argument("key", nargs="?", help="会话键（playthrough、组:探针 ID，对照组加 B: 前缀）；step / answer / dump 必填")
    ap.add_argument("--world", choices=sorted(PROBE_FILES), default=None, help="场景（缺省看探针文件的 variant）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probes", type=Path, default=None, help="探针文件（缺省按 --world 取，都没给就是普通人版）")
    ap.add_argument("--open-actions", type=int, default=20, help="花样输入探针只取前 K 条")
    args = ap.parse_args(argv)
    d, key = args.dir, args.key
    probes = load_probes(args.open_actions, args.probes or PROBE_FILES[args.world or "wuliang"])
    world = bench_gm.world_of(probes)
    if args.world and args.world != world:
        ap.error(f"--world {args.world} 与探针文件的 variant {world} 不符")
    rd = run_dir(d, world, args.seed)
    od = rd / "online"
    if args.mode == "keys":
        print("\n".join(session_keys(probes) + session_keys(probes, rival=True)))
        return 0
    if args.mode == "assemble":
        assemble(d, probes, args.seed)
        return 0
    if key is None:
        ap.error(f"{args.mode} 需要会话键")
    od.mkdir(parents=True, exist_ok=True)
    if args.mode == "answer":
        print(_answer(od, key, sys.stdin.read().strip()))
        return 0
    given = load_answers(_file(od, key, "answers.json"))
    if key.startswith(RIVAL):
        if args.mode == "dump":
            ap.error("dump 只用于本引擎的会话")
        rec = rival_key(key, given, probes, args.seed, on_missing=_halt(od))
    elif args.mode == "dump":
        calls: list[dict[str, Any]] = []
        run_key(key, given, _interp(d, rd), probes, args.seed, dump=calls)
        _file(od, key, "dump.json").write_text(json.dumps(calls, ensure_ascii=False, indent=1), "utf-8")
        print(f"dumped {len(calls)} calls")
        return 0
    else:
        rec = run_key(key, given, _interp(d, rd), probes, args.seed, on_missing=_halt(od))
    _file(od, key, "done.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), "utf-8")
    _file(od, key, "pending.json").unlink(missing_ok=True)
    print(f"DONE {key} ({rec['calls']} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
