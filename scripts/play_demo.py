"""
[INPUT]: 依赖 tianlong.runtime.session 的 GameSession，tianlong.scenarios 的 SCENARIOS，tianlong.language.llm 的 llm_from_env，
         tianlong.kernel.perception 的 sketches_for
[OUTPUT]: 命令行脚本：按固定指令序列游玩一局，写出 Markdown 记录（玩家所见 / 世界真相 / NPC 理由 / 结局时玩家认知与真相对照）
[POS]: scripts 的演示工具；不属于引擎本体。用它展示“同一时刻，玩家被告知了什么，而世界里实际发生了什么”；
       指令序列是段誉的原著路线，默认录旧版（wuliang-duanyu → docs/demo/wuliang.md），普通人版写到 wuliang_c.md、不覆盖旧录像；
       叙述经 CachedLLM 落盘，重跑同一局不再花费模型调用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tianlong.core import Op, Outcome, Proposition, Rel
from tianlong.kernel.perception import sketches_for
from tianlong.language.llm import llm_from_env
from tianlong.language.templates import REASONS
from tianlong.runtime.cli import load_dotenv
from tianlong.runtime.session import GameSession
from tianlong.scenarios import SCENARIOS

WULIANG = [
    "环顾四周", "等待", "等待", "等待", "等待", "去后院", "往后山走", "去崖顶", "跳下断崖", "查看玉璧",
    "等到天黑", "查看玉璧", "钻进石缝", "进石门", "磕头", "拿凌波微步", "拿北冥神功",
    "研读凌波微步", "研读凌波微步", "研读凌波微步", "钻进隧道", "等半个时辰",
]
_VERB = {Op.MOVE: "走向", Op.TAKE: "拿起", Op.PUT: "放下", Op.GIVE: "交出", Op.INSPECT: "查看", Op.TELL: "告诉",
         Op.ASK: "询问", Op.ATTACK: "出手攻击", Op.STUDY: "研读", Op.USE: "施用于", Op.UNLOCK: "开锁", Op.LOCK: "上锁"}


def _truth(session: GameSession, events) -> list[str]:
    st = session.authority.head()
    names = {sk.id: sk.name for sk in sketches_for(st, st.entities)}
    out = []
    for e in events:
        if e.op == Op.WAIT:
            continue
        it = e.intent
        obj = f"（{names.get(it.obj, it.obj)}）" if it.obj else ""
        tail = "" if e.outcome == Outcome.SUCCESS else f" ✗ {REASONS.get(e.reason or '', e.reason)}"
        out.append(f"{names.get(it.actor, it.actor)} {_VERB.get(e.op, e.op.value)} {names.get(it.target, it.target or '')}{obj}{tail}")
    return out


def _belief_snapshot(session: GameSession) -> dict[str, tuple[str, str]]:
    """玩家对几件关键事的认知 vs 真相：认知差异正是这台引擎要保留的东西。"""
    me = session.beliefs(session.player)
    st = session.authority.head()
    names = {sk.id: sk.name for sk in sketches_for(st, st.entities)}
    where = lambda eid: names.get(eid, "不知道") if eid else "不知道"      # noqa: E731
    out = {}
    for person in ("gongguangjie", "zhongling", "ganguanghao", "geguangpei"):
        if person in st.entities:
            believed = "被制住" if me.holds(Proposition.attr(person, "subdued", True)) else (
                "中毒" if me.holds(Proposition.attr(person, "poisoned", True)) else "安然")
            truth = "被制住" if st.clock < int(st.attr(person, "subdued_until", 0) or 0) else (
                "中毒" if st.attr(person, "poisoned") else "安然")
            out[f"{names[person]}的状况"] = (f"{believed}，在{where(me.location_of(person))}",
                                           f"{truth}，在{where(st.target(person, Rel.AT))}")
    if "antidote" in st.entities:
        out["解药在谁手里"] = (where(me.location_of("antidote")), where(st.target("antidote", Rel.AT)))
    return out


# 指令序列是段誉的原著路线：默认录旧版，沿用入库的 docs/demo/wuliang.md；普通人版另写一个文件，不覆盖旧录像
_FILES = {"wuliang-duanyu": "wuliang", "wuliang": "wuliang_c"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="录制一局游戏：玩家所见 vs 世界真相")
    ap.add_argument("--world", default="wuliang-duanyu", choices=sorted(SCENARIOS))
    ap.add_argument("--llm", choices=["auto", "none"], default="auto")
    ap.add_argument("--out", default="docs/demo")
    args = ap.parse_args(argv)
    load_dotenv()
    llm = llm_from_env() if args.llm == "auto" else None
    scenario = SCENARIOS[args.world](7)
    session = GameSession(scenario, llm=llm)
    md = [f"# {args.world} 演示（叙述：{llm.model if llm else '模板'}）", "",
          "> 每回合三栏：玩家读到的叙述（只来自玩家自己的感知）、世界真相（内核裁定的全部事件，含玩家看不见的）、"
          "NPC 各自的决策理由（只来自各自的认知）。", "", scenario.setting, "", session.intro(), ""]
    for cmd in WULIANG:
        before = session.authority.head().clock
        r = session.turn(cmd)
        when = r.clock if session.authority.head().clock - before <= 1 else f"{r.clock} → {session.clock()}"
        md += [f"## > {cmd}  【{when}】", "", r.narration, ""]
        truth = _truth(session, r.events)
        npc = [f"{session.authority.head().entity(d.agent).name}：{d.rationale}" for d in r.deliberations
               if d.intent.op != Op.WAIT]
        if truth:
            md += ["**世界真相：** " + "；".join(truth), ""]
        if npc:
            md += ["**NPC 理由：** " + "；".join(dict.fromkeys(npc)), ""]
        print(f"> {cmd}\n【{when}】{r.narration}\n", flush=True)
    md += ["## 结局：段誉以为 vs 真相", "", "| 事 | 段誉以为 | 真相 |", "|---|---|---|"]
    md += [f"| {k} | {v[0]} | {v[1]} |" for k, v in _belief_snapshot(session).items()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{_FILES.get(args.world, args.world)}.md").write_text("\n".join(md) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
