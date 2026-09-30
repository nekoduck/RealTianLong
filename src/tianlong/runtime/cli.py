"""
[INPUT]: 依赖 runtime/session 的 GameSession / TurnReport，runtime/versions 的 IncompatibleSave，scenarios 的 SCENARIOS 注册表，
         language/llm 的 llm_from_env / fast_llm_from_env，language/interpret 的 Interpreter（主持层解释器）；按需加载
         persistence/neo4j_store、learning/bundle（部署包）
[OUTPUT]: 对外提供 main()（命令行入口 `tianlong` / `python -m tianlong`）、load_dotenv()、interpreter_for()
[POS]: runtime 的终端前端：叙述经 on_text 流式逐句打印；开场只讲玩家所见，附上不剧透的输入示例（Scenario.hints）；
       /hint /recap /beliefs 与“GM：”场外提问都交给会话（不推进时间），/debug 显示真相、NPC 理由与分阶段耗时（含首字耗时）——
       开发者视角与玩家自己的认知刻意分开；落幕即打印终章与真相揭晓并退出。
       --llm auto 有密钥即启用：叙述用 llm_from_env()，解释用 fast_llm_from_env()；
       --world 选择世界（默认天龙八部·无量山普通人版，wuliang-duanyu 复现段誉作主角的旧版），--store/--save 选择持久化与存档（存档版本不符时一句话说明并退出，
       --allow-migration 显式接续旧档），--predictor/--policy 让部署包里的 GNN 与 RL 策略驱动 NPC（部署包逐项核对语义版本与文件哈希，
       策略配套的预测器随包决定；缺包、被改动或与当前代码不兼容时一句话说明并退出）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from tianlong.language.interpret import Interpreter
from tianlong.language.llm import LLMClient, fast_llm_from_env, llm_from_env
from tianlong.runtime.session import GameSession, TurnReport
from tianlong.runtime.versions import IncompatibleSave
from tianlong.scenarios import SCENARIOS, Scenario

_META = "元指令：/hint 提示  /recap 前情回顾  /beliefs 你所知道的  /debug 开发者视角  /quit 退出；以“GM：”开头向主持人提问"


def _debug_lines(r: TurnReport) -> list[str]:
    """推进了时间的回合列出真相与 NPC 理由；每个回合都列分阶段耗时与首字耗时。"""
    out = ["  ── 真相 ──"] if r.advanced else []
    for e in r.events:
        if e.op.value != "wait":
            out.append(f"  {e.actor} {e.op.value} {e.intent.target or ''} {e.intent.obj or ''} → {e.outcome.value}"
                       f"{' (' + e.reason + ')' if e.reason else ''}")
    for d in r.deliberations:
        if d.intent.op.value != "wait":
            out.append(f"  [{d.agent}] {d.intent.op.value} {d.intent.target or ''} ← {d.rationale}")
    first = f" first_text={r.first_text_ms}ms" if r.first_text_ms is not None else ""
    out.append(f"  ⏱ [{r.kind.value}]" + first + "".join(f" {k}={v}ms" for k, v in r.timings.items()))
    return out


def load_dotenv(path: Path = Path(".env")) -> None:
    """读取 .env（KEY=VALUE，# 为注释）；已存在的环境变量优先，文件只补缺。"""
    if not path.is_file():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def interpreter_for(llm: LLMClient | None, scenario: Scenario) -> Interpreter:
    """主持层解释器（快模型；没有模型时它自己退回规则解析）；场景的名字全集只用于拒绝回显玩家不认识的名字。"""
    return Interpreter(llm, aliases=scenario.aliases, universe=(e.name for e in scenario.state.entities.values()),
                       kowtow_ticks=scenario.kowtow_ticks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong", description="图世界文字游戏")
    ap.add_argument("--llm", choices=["auto", "none"], default="auto", help="auto：有 GEMINI_API_KEY 则启用")
    ap.add_argument("--world", choices=sorted(SCENARIOS), default="wuliang",
                    help="wuliang：天龙八部·无量山（普通人版：你是挑茶的伙计阿顺）；wuliang-duanyu：旧版（你是段誉）；warehouse：仓库钥匙")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="neo4j：读取 NEO4J_URI 等环境变量")
    ap.add_argument("--save", default=None, help="存档名（作为 world_id，Neo4j 下可跨进程保留）")
    ap.add_argument("--predictor", choices=["heuristic", "gnn"], default="heuristic",
                    help="gnn：用部署包里的角色视角动态模型作为 NPC 的后果预测器")
    ap.add_argument("--policy", choices=["scripted", "learned"], default="scripted",
                    help="learned：用部署包里的策略驱动 NPC（配套的预测器随包决定）")
    ap.add_argument("--artifacts", default="artifacts", help="部署包目录（含 bundle.json，由 python -m tianlong.learning.bundle 生成）")
    ap.add_argument("--allow-migration", action="store_true", help="存档版本与当前代码不符时仍显式接续（不补写旧档信息）")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    load_dotenv()

    voice = llm_from_env() if args.llm == "auto" else None
    fast = fast_llm_from_env() if args.llm == "auto" else None
    scenario = SCENARIOS[args.world](args.seed)
    if args.save:
        scenario = replace(scenario, world_id=args.save)
    store = None
    if args.store == "neo4j":
        from tianlong.persistence.neo4j_store import Neo4jWorldStore
        store = Neo4jWorldStore.from_env()
    predictor, policies, max_cands = None, None, 64
    try:
        if args.predictor == "gnn" or args.policy == "learned":
            # 部署包逐项核对语义版本与文件哈希；策略训练时用哪个预测器，上线就用哪个
            from tianlong.learning.bundle import load_bundle
            loaded = load_bundle(args.artifacts, want_predictor=args.predictor == "gnn",
                                 want_policy=args.policy == "learned")
            predictor = loaded.predictor
            if loaded.policy is not None:
                policies = dict.fromkeys(scenario.npcs, loaded.policy)
                max_cands = loaded.policy.spec.max_cands  # type: ignore[attr-defined]
    except (FileNotFoundError, ValueError) as e:     # 缺部署包或模型过期（StaleModel）：说清楚，不甩一屏张量报错
        print(f"无法加载训练好的模型：{e}")
        return 2
    try:
        session = GameSession(scenario, store=store, llm=voice, policies=policies, predictor=predictor,
                              max_candidates=max_cands, allow_migration=args.allow_migration,
                              interpreter=interpreter_for(fast, scenario))
    except IncompatibleSave as e:                   # 旧规则下建的档：说清楚，由玩家决定是否显式迁移
        print(f"无法读档：{e}")
        return 2
    debug = args.debug
    print(f"【{session.clock()}】{'（Gemini 叙述）' if voice else '（模板叙述）'}")
    if session.ending is not None:                  # 读到的是已落幕的存档
        print(session.epilogue())
        return 0
    if scenario.setting and not session.resumed:
        print(scenario.setting + "\n")
    print(session.intro())
    if scenario.hints:
        print("\n" + scenario.hints)
    print(_META)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/debug":
            debug = not debug
            print(f"开发者视角：{'开' if debug else '关'}")
            continue
        print(f"【{session.clock()}】", end="", flush=True)
        r = session.turn(text, on_text=lambda piece: print(piece, end="", flush=True))
        print()
        if debug:
            print("\n".join(_debug_lines(r)))
        if r.ending is not None:
            print("\n" + session.epilogue())
            return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
