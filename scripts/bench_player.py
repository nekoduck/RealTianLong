"""
[INPUT]: 依赖同目录 bench_gm（会话装配 _session、逐回合计分 engine_turn、world_of / SCENARIOS、盲评顺序 pairwise_order）与
         bench_rival 的 PureLLMGM / epilogue_story / panel_name，bench_metrics 的 b1 / e1 / ending_of，tianlong.language.llm 的 ScriptedLLM；DIR/player/<A|B>/ 下各角色的代理答案
[OUTPUT]: 命令行 python scripts/bench_player.py {step|answer|panel} DIR [SIDE] [--world W] [--seed N] [--turns K]，
          也是可导入的库：PLAYER_SYSTEM / MAX_TURNS / MAX_CHARS / ROLES，player_prompt()，game()（一整局，每次模型调用经 ask 作答），
          recorded()（按已录答案作答、缺了交给 on_missing），record()（追加一条答案），panel()
[POS]: scripts 的玩家代理模式（plan §8.4）：固定脚本不适合涌现的世界，于是让一个隔离的“玩家”代理（第一次玩、谨慎而好奇的普通玩家）
       只看本方看得见的正文（开场 + 历回合交付的文字），每回合写一句 ≤30 字的输入，最多 30 回合，到结局即停。
       A 是本引擎（叙述 narrator、解释 interp 各由隔离代理作答），B 是纯模型主持人（rival：圣经 + 完整对话 + 新输入），各跑一局。
       step 从头重放一局（同样的答案 → 同样的轨迹），停在第一个没答的调用上：打印待答的角色（player / narrator / interp / rival）
       与它的完整提示词，写下 pending.json 并立刻退出进程（叙述调用在读流线程里，只能 os._exit）；answer 以 stdin 追加这个角色的答案
       （玩家的一句超过 30 字即拒收）；两局都走完后 panel 写出盲评面板（A/B 先后按面板 wuliang-player-s7 与两个种子的面板轮流换边、终章只取收束）
       与各自的结局、B1、E1；B 的结局按标题词法判（去掉标点空白再比）。
       它是辅证：面板只做单局打分加总体偏好
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tianlong.language.llm import ScriptedLLM

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 bench_gm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gm  # noqa: E402
from bench_metrics import b1, e1, ending_of  # noqa: E402
from bench_rival import PureLLMGM, epilogue_story, panel_name  # noqa: E402

PLAYER_SYSTEM = (
    "你是第一次玩这款中文武侠文字游戏的玩家：谨慎而好奇的普通玩家。你只知道主持人讲给你的正文，不知道后面的剧情，也不去猜攻略；"
    "想活下去，也想看看会发生什么。每回合只写一句你此刻想做的事或想说的话（第一人称，说话可以写成“对某某说：……”），"
    "不超过 30 个字；不写旁白，不替别人做事，只输出这一句。"
)
MAX_TURNS, MAX_CHARS = 30, 30
ROLES = ("player", "narrator", "interp", "rival")
Ask = Callable[[str, int, str, str], str]          # (角色, 该角色的第几次调用, system, prompt) → 答案


def player_prompt(shown: Sequence[tuple[str, str]]) -> str:
    """玩家代理的提示词：只有本方看得见的正文（开场 + 历回合交付的文字）。"""
    rows = [x for said, text in shown for x in ((f"你：{said}",) if said else ()) + (f"主持人：{text}",)]
    return "以下是你到目前为止看到的全部正文：\n" + "\n".join(rows) + f"\n\n写下你的下一句输入（不超过 {MAX_CHARS} 字）："


def game(side: str, world: str, seed: int, ask: Ask, turns: int = MAX_TURNS) -> dict[str, Any]:
    """一整局：side="A" 本引擎、"B" 纯模型主持人。每次模型调用（玩家、叙述、解释、对照组）都经 ask 作答。"""
    scenario = bench_gm.SCENARIOS[world](seed)
    count: Counter[str] = Counter()

    def call(role: str, system: str | None, prompt: str) -> str:
        count[role] += 1
        return ask(role, count[role] - 1, system or "", prompt)

    out: dict[str, Any] = {"side": side, "world": world, "seed": seed, "turns": [], "ending": None, "epilogue": ""}
    shown: list[tuple[str, str]] = []
    if side == "A":
        s = bench_gm._session(scenario, ScriptedLLM(lambda p, sy, sc: call("narrator", sy, p)),
                              ScriptedLLM(lambda p, sy, sc: call("interp", sy, p)))
        s.narrator.lead_after = None                   # 代理作答是瞬时的：迟到的先声本就不上场（同 bench_online）
        shown.append(("", s.intro()))
    else:
        gm = PureLLMGM(ScriptedLLM(lambda p, sy, sc: call("rival", sy, p)), scenario)
        shown.append(("", gm.opening))
    out["opening"] = shown[0][1]
    for _ in range(turns):
        text = call("player", PLAYER_SYSTEM, player_prompt(shown)).strip()[:MAX_CHARS]
        if side == "A":
            rec, *_ = bench_gm.engine_turn(s, text, "player")
        else:
            rec = dict(gm.turn(text), ending=ending_of(gm.transcript[-1][1], scenario))   # 词法：标题去掉标点空白再比
        out["turns"].append(rec)
        shown.append((text, rec["narration"]))
        if rec.get("ending"):
            out["ending"] = rec["ending"]
            out["epilogue"] = s.epilogue() if side == "A" else ""
            break
    out["calls"] = dict(count)
    return out


# ============================================================
#  代理答案：DIR/player/<side>/<role>.json（第 i 个答该角色的第 i 次调用）
# ============================================================


def _dir(d: Path, side: str) -> Path:
    return d / "player" / side


def _answers(d: Path, side: str, role: str) -> list[str]:
    p = _dir(d, side) / f"{role}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else []


def record(d: Path, side: str, role: str, text: str) -> int:
    """追加一条答案，返回它的序号；玩家的一句超过 MAX_CHARS 字即拒收。"""
    if role == "player" and (not text.strip() or len(text.strip()) > MAX_CHARS):
        raise ValueError(f"玩家的输入须是 1–{MAX_CHARS} 字的一句话：{text!r}")
    got = _answers(d, side, role)
    _dir(d, side).mkdir(parents=True, exist_ok=True)
    (_dir(d, side) / f"{role}.json").write_text(json.dumps([*got, text], ensure_ascii=False, indent=1), "utf-8")
    return len(got)


def recorded(d: Path, side: str, on_missing: Callable[[dict[str, Any]], str]) -> Ask:
    """按已录的答案作答；某角色的答案用完时，把待答的调用交给 on_missing。"""
    cache = {r: _answers(d, side, r) for r in ROLES}

    def ask(role: str, n: int, system: str, prompt: str) -> str:
        if n < len(cache[role]):
            return cache[role][n]
        return on_missing({"side": side, "role": role, "n": n, "system": system, "prompt": prompt})
    return ask


def _halt(d: Path) -> Callable[[dict[str, Any]], str]:
    def halt(pending: dict[str, Any]) -> str:
        (_dir(d, pending["side"]) / "pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=1), "utf-8")
        print(f"PENDING {pending['side']} {pending['role']} #{pending['n']}\n===== SYSTEM =====\n{pending['system']}\n"
              f"===== PROMPT =====\n{pending['prompt']}", flush=True)
        os._exit(0)
    return halt


def panel(d: Path, seed: int) -> dict[str, Any]:
    """两局都走完后：盲评面板（A/B 先后由种子打乱）与各自的结局、B1、E1（B 只有结局）。"""
    games = {side: json.loads((_dir(d, side) / "game.json").read_text("utf-8")) for side in ("A", "B")}
    rows = {side: [["（开场）", g["opening"]]] + [[t["text"], t["narration"]] for t in g["turns"]]
            + ([["（落幕）", epilogue_story(g.get("epilogue"))]] if epilogue_story(g.get("epilogue")) else [])
            for side, g in games.items()}
    engine_is_a = bench_gm.pairwise_order(seed, panel_name(games["A"]["world"], seed, player=True))
    first, second = ("A", "B") if engine_is_a else ("B", "A")
    (d / "player" / "panel.json").write_text(json.dumps({"A": rows[first], "B": rows[second]}, ensure_ascii=False, indent=1),
                                             "utf-8")
    keys = tuple(dict.fromkeys(b.key for b in bench_gm.SCENARIOS[games["A"]["world"]](seed).beats))
    summary = {"panel_key": {"A": "engine" if engine_is_a else "baseline", "B": "baseline" if engine_is_a else "engine"},
               "engine": {"ending": games["A"]["ending"], "turns": len(games["A"]["turns"]),
                          "B1": b1(games["A"]["turns"], keys)["display"], "E1": e1(games["A"]["turns"])["display"]},
               "baseline": {"ending": games["B"]["ending"], "turns": len(games["B"]["turns"])}}
    (d / "player" / "panel_key.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="玩家代理模式（plan §8.4）：隔离的玩家代理与隔离的主持代理各自逐次作答，A 与 B 各一局")
    ap.add_argument("mode", choices=("step", "answer", "panel"))
    ap.add_argument("dir", type=Path)
    ap.add_argument("side", nargs="?", choices=("A", "B"), help="A 本引擎 / B 纯模型主持人；step 与 answer 必填")
    ap.add_argument("--world", choices=sorted(bench_gm.SCENARIOS), default="wuliang")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--turns", type=int, default=MAX_TURNS, help=f"至多几回合（缺省 {MAX_TURNS}）")
    args = ap.parse_args(argv)
    d = args.dir
    if args.mode == "panel":
        print(json.dumps(panel(d, args.seed), ensure_ascii=False, indent=1))
        return 0
    if args.side is None:
        ap.error(f"{args.mode} 需要 SIDE（A 或 B）")
    _dir(d, args.side).mkdir(parents=True, exist_ok=True)
    if args.mode == "answer":
        pend = _dir(d, args.side) / "pending.json"
        pending: Mapping[str, Any] = json.loads(pend.read_text("utf-8")) if pend.exists() else {}
        if not pending or pending["n"] != len(_answers(d, args.side, pending["role"])):
            raise SystemExit("没有待答的调用，或序号对不上")
        try:
            n = record(d, args.side, pending["role"], sys.stdin.read().strip())
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(f"saved {pending['role']} #{n} for {args.side}")
        return 0
    result = game(args.side, args.world, args.seed, recorded(d, args.side, _halt(d)), args.turns)
    (_dir(d, args.side) / "game.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), "utf-8")
    (_dir(d, args.side) / "pending.json").unlink(missing_ok=True)
    print(f"DONE {args.side}：{len(result['turns'])} 回合，结局 {result['ending'] or '（未落幕）'}，调用 {result['calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
