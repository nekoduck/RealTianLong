"""
[INPUT]: 依赖 tianlong.runtime.session 的 GameSession（模板模式，不接模型），tianlong.scenarios 的 build_wuliang_commoner，
         tianlong.scenarios.tianlong.stagecraft 的 BEAT_KEYS，tianlong.scenarios.tianlong.drives_c 的时刻，cognition/navigation 的 believed_place，
         scenarios/tianlong/wuliang 的 NIGHTFALL，core 的 Op / Outcome / Rel / SetAttr / WorldState / clock_label
[OUTPUT]: 命令行 python scripts/sim_beats.py [--seeds 1-20|7,11] [--players passive,follower,...] [--json FILE] [--check] [--jobs N]
          [--walk PROBES]，打印 Markdown 结果表（并行与否逐项相同）；
          也是可导入的库：PLAYERS（五种脚本化玩家）、play(seed, player) -> Run、summarize(runs) -> dict、pooled_e1(runs)、table(runs) -> str、GATES，
          走查核对 CANON / WHEN / WALKED / walk(seed, inputs, key) -> Walk
[POS]: scripts 的普通人版调参台（plan §7 M2）：种子 × 脚本化玩家逐回合跑真实会话，报告
       B1（玩家目击的看点数，按场景的识别器、只算玩家自己的感知）、段誉到达琅嬛福地 / 澜沧江畔的种子数、到达的结局、钟灵是否被制、
       讨价还价是否成立、各驱力的兑现次数、E1（空转的推进回合所占比例：这一回合玩家没感知到任何 NPC 的动作或言语、没有新认识的东西、
       没有看点）。脚本化玩家只凭玩家自己的认知与时钟行事（和真人一样看不到真相）；真相只用于统计。
       --check 按 GATES 判定出口条件（跟随型 ≥16/20 个种子目击 ≥8/11 个看点、跟随型与全体合计的 E1 各 ≤ 10%（逐脚本不设门槛）、
       被动玩家下段誉 ≥12/20 到琅嬛且 ≥8/20 到澜沧江、讨价还价 ≥16/20），不达标以非零码退出。
       --walk 核对评测探针里的整局（plan §8.1，进 run7 之前）：模板会话逐句走（CANON 是代理解释器会怎样理解那一句的规则说法），
       每句在回合开始时的真相上按 WHEN 求值（“跟上段公子”时他不在眼前、“查看蒲团”时身在琅嬛……），落空或没接住的句子逐条列出、以非零码退出
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from tianlong.cognition.navigation import believed_place
from tianlong.core import Op, Outcome, Proposition, Rel, SetAttr, WorldState, clock_label
from tianlong.runtime.session import GameSession
from tianlong.scenarios import build_wuliang_commoner
from tianlong.scenarios.tianlong.drives_c import HUNT, MOONRISE, SUPPER, SUPPER_END, TRYST
from tianlong.scenarios.tianlong.stagecraft import BEAT_KEYS
from tianlong.scenarios.tianlong.wuliang import NIGHTFALL

PLAYER, DUANYU = "ashun", "duanyu"
MAX_TURNS = 400
# E1 的口径：跟随型与全体推进回合合计各须 ≤ 10%（plan 的出口条件 E1 不限脚本）；逐脚本不设门槛——
# liar 在后院、slipper 在大殿掐着换班的时辰干等，是脚本故意在没事的地方耗时间，报告里照列
GATES = {"follower_b1": (16, 8), "follower_e1": 0.10, "e1": 0.10, "langhuan": 12, "lancang": 8, "bargain": 16}

# ============================================================
#  脚本化玩家：只读玩家自己的认知（以为段公子在哪、有没有人问他话）与时钟
# ============================================================

Script = Callable[[GameSession, dict], str]


def _where(s: GameSession) -> tuple[str | None, str | None]:
    me = s.beliefs(PLAYER)
    return me.location_of(PLAYER), believed_place(me, DUANYU)


def _passive(s: GameSession, memo: dict) -> str:
    return "等下去"


def _follower(s: GameSession, memo: dict) -> str:
    """跟着段公子：他走开就跟上，崖顶上不见了他就攀下去；在后山陪着他时回后院取一趟水（撞见私语）。"""
    here, dy = _where(s)
    clock = s.authority.head().clock
    if here == "yading" and dy != "yading":
        return "跳下断崖"
    if here == "houshan" and dy == "houshan" and not memo.get("water"):
        if clock < TRYST + 5:
            return "等一会儿"
        if clock < HUNT - 8:
            memo["water"] = True
            return "去后院"
    if dy is not None and dy != here:
        return "跟上段公子"
    if dy is None and not memo.get(("lost", here)):
        memo[("lost", here)] = True            # 跟丢了：往里走（石洞里只有那道石门），别处先四下看看
        return {"shidong": "进石门"}.get(here or "", "环顾四周")
    return "等下去"


def _liar(s: GameSession, memo: dict) -> str:
    """留在剑湖宫；天黑前溜去后院歇着，龚光杰问起段公子就说他往山道去了。"""
    me = s.beliefs(PLAYER)
    here = me.location_of(PLAYER)
    asked = any(o.kind == "answer" and o.counterpart == "gongguangjie" for o in me.obligations)
    if asked and not memo.get("lied"):
        memo["lied"] = True
        return "告诉龚光杰段公子在山道"
    if here == "hall" and s.authority.head().clock >= HUNT - 15 and not memo.get("yard"):
        memo["yard"] = True
        return "去后院"
    return "等下去"


def _grabber(s: GameSession, memo: dict) -> str:
    """跟着段公子到琅嬛福地，抢先翻蒲团、拿走凌波微步，自己研读，学成就钻隧道。"""
    here, dy = _where(s)
    me = s.beliefs(PLAYER)
    if here == "langhuan":
        if me.location_of("scroll_lb") == PLAYER:
            return "钻进隧道" if me.holds(Proposition.attr(PLAYER, "evasion", True)) else "研读凌波微步"
        if me.knows("scroll_lb") and me.location_of("scroll_lb") is not None:
            return "拿凌波微步"
        if not memo.get("searched"):
            memo["searched"] = True
            return "查看蒲团"
    return _follower(s, memo)


def _slipper(s: GameSession, memo: dict) -> str:
    """躲在大殿；夜饭换班的时辰一到，溜去山道、下山。"""
    here = s.beliefs(PLAYER).location_of(PLAYER)
    left = SUPPER + 5 - s.authority.head().clock
    if left > 0:
        return "等半个时辰" if left > 60 else "等一会儿"      # 掐着换班的时辰：别一觉等过了头
    return {"hall": "悄悄溜去山道", "shandao": "去山脚"}.get(here or "", "等下去")


PLAYERS: dict[str, Script] = {"passive": _passive, "follower": _follower, "liar": _liar, "grabber": _grabber,
                              "slipper": _slipper}


# ============================================================
#  一局
# ============================================================


@dataclass
class Run:
    seed: int
    player: str
    beats: list[str] = field(default_factory=list)
    langhuan: bool = False
    lancang: bool = False
    ending: str | None = None
    zl_subdued: bool = False
    bargain: bool = False
    marks: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    idle: int = 0
    clock: str = ""


def _idle(s: GameSession, start: int, known: frozenset[str], beats: tuple[str, ...]) -> bool:
    """空转：这一回合玩家没感知到任何 NPC 的动作或言语、没有新认识的东西、没有看点。"""
    me = s.beliefs(PLAYER)
    acted = any(ep.tick >= start and ep.event.actor not in (None, PLAYER) for ep in me.episodes)
    return not (acted or beats or set(me.entities) - known)


def play(seed: int, player: str, max_turns: int = MAX_TURNS) -> Run:
    s = GameSession(build_wuliang_commoner(seed), pipeline=False)
    s.intro()
    run, memo = Run(seed, player), {}
    seen: set[str] = set()
    for _ in range(max_turns):
        start, known = s.authority.head().clock, frozenset(s.beliefs(PLAYER).entities)
        r = s.turn(PLAYERS[player](s, memo))
        if r.advanced:
            run.turns += 1
            run.idle += _idle(s, start, known, r.beats)
        seen |= set(r.beats)
        if r.ending is not None:
            run.ending = r.ending.key
            break
    events = s.store.events(s.ref)
    run.beats = [k for k in BEAT_KEYS if k in seen]
    run.langhuan = any(e.actor == DUANYU and e.op == Op.MOVE and e.intent.target == "langhuan"
                       and e.outcome == Outcome.SUCCESS for e in events)
    run.lancang = any(e.actor == DUANYU and e.op == Op.MOVE and e.intent.target == "lancang"
                      and e.outcome == Outcome.SUCCESS for e in events)
    run.zl_subdued, run.bargain = _opening(events)
    run.marks = {f"{a}.{k}": len(v) for a, m in sorted(s._marks.items()) for k, v in sorted(m.items())}
    run.clock = clock_label(s.authority.head().clock)
    return run


def _opening(events: Iterable) -> tuple[bool, bool]:
    """(钟灵被制过没有, 讨价还价成立：她离开大殿前没被制、龚光杰的毒由她的 USE 解开、左子穆说了 AGREE)。"""
    subdued_at, left_at, use_ok, agree, taken = None, None, False, False, False
    for e in events:
        for c in e.changes:
            if isinstance(c, SetAttr) and c.entity == "zhongling" and c.key == "subdued_until" and c.new \
                    and int(c.new) > e.tick and subdued_at is None:
                subdued_at = e.tick
        ok = e.outcome == Outcome.SUCCESS
        if e.actor == "zhongling" and e.op == Op.MOVE and ok and left_at is None:
            left_at = e.tick
        use_ok |= ok and e.actor == "zhongling" and e.op == Op.USE and e.intent.target == "gongguangjie"
        taken |= ok and e.op == Op.TAKE and e.intent.target == "antidote"
        agree |= ok and e.actor == "zuozimu" and e.op == Op.TELL and e.intent.social is not None \
            and e.intent.social.value == "agree"
    free = subdued_at is None or (left_at is not None and left_at < subdued_at)
    return subdued_at is not None, free and use_ok and agree and not taken


# ============================================================
#  汇总与报告
# ============================================================


def summarize(runs: list[Run]) -> dict:
    out: dict = {}
    for player in dict.fromkeys(r.player for r in runs):
        mine = [r for r in runs if r.player == player]
        turns = sum(r.turns for r in mine)
        out[player] = {
            "seeds": len(mine),
            "b1_mean": round(sum(len(r.beats) for r in mine) / len(mine), 2),
            "b1_ge8": sum(len(r.beats) >= 8 for r in mine),
            "beats": dict(Counter(k for r in mine for k in r.beats)),
            "langhuan": sum(r.langhuan for r in mine), "lancang": sum(r.lancang for r in mine),
            "endings": dict(Counter(r.ending or "（未落幕）" for r in mine)),
            "zl_subdued": sum(r.zl_subdued for r in mine), "bargain": sum(r.bargain for r in mine),
            "e1": round(sum(r.idle for r in mine) / turns, 3) if turns else 0.0,
            "turns": turns,
            "marks": dict(sorted(sum((Counter(r.marks) for r in mine), Counter()).items())),
        }
    return out


def pooled_e1(runs: list[Run]) -> float:
    """全体推进回合合计的 E1（不分脚本）。"""
    turns = sum(r.turns for r in runs)
    return round(sum(r.idle for r in runs) / turns, 3) if turns else 0.0


def table(runs: list[Run]) -> str:
    summary = summarize(runs)
    rows = ["| 玩家 | 种子 | B1 均值 | B1≥8 | 到琅嬛 | 到澜沧江 | 讨价还价 | 钟灵被制 | E1 | 结局 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for player, x in summary.items():
        ends = "、".join(f"{k} {v}" for k, v in sorted(x["endings"].items()))
        rows.append(f"| {player} | {x['seeds']} | {x['b1_mean']} | {x['b1_ge8']} | {x['langhuan']} | {x['lancang']} | "
                    f"{x['bargain']} | {x['zl_subdued']} | {x['e1']:.1%} | {ends} |")
    rows.append(f"| 全体合计 | {len(runs)} | | | | | | | {pooled_e1(runs):.1%} | |")
    rows += ["", "| 看点 | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for k in BEAT_KEYS:
        rows.append(f"| {k} | " + " | ".join(str(x["beats"].get(k, 0)) for x in summary.values()) + " |")
    marks = sorted({k for x in summary.values() for k in x["marks"]})
    rows += ["", "| 驱力兑现（次） | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for k in marks:
        rows.append(f"| {k} | " + " | ".join(str(x["marks"].get(k, 0)) for x in summary.values()) + " |")
    return "\n".join(rows)


def check(runs: list[Run]) -> list[str]:
    """出口条件（GATES）：返回没达标的条目。"""
    x = summarize(runs)
    bad = []
    need, at_least = GATES["follower_b1"]
    if "follower" in x and sum(len(r.beats) >= at_least for r in runs if r.player == "follower") < need:
        bad.append(f"跟随型 B1≥{at_least} 的种子数 {x['follower']['b1_ge8']} < {need}")
    if "follower" in x and x["follower"]["e1"] > GATES["follower_e1"]:
        bad.append(f"跟随型 E1 {x['follower']['e1']:.1%} > {GATES['follower_e1']:.0%}")
    if pooled_e1(runs) > GATES["e1"]:
        bad.append(f"全体合计 E1 {pooled_e1(runs):.1%} > {GATES['e1']:.0%}")
    if "passive" in x:
        for key in ("langhuan", "lancang", "bargain"):
            if x["passive"][key] < GATES[key]:
                bad.append(f"被动玩家 {key} {x['passive'][key]} < {GATES[key]}")
    return bad


# ============================================================
#  走查核对（plan §8.1）：评测探针里的整局，每一句都要落在有意义的状态上
#  （“跟上段公子”时他确实不在眼前、“查看蒲团”时确实身在琅嬛）。模板会话只认规则解析的说法：
#  CANON 是代理解释器会怎样理解这一句的规则说法；WHEN 按关键词（先中先用）在回合开始时的世界真相上求值
# ============================================================

CANON: dict[str, str] = {
    "扯了扯段公子的袖子，低声道：公子，快别笑了": "悄悄对段公子说：公子，快别笑了",
    "向龚光杰赔笑道：这位爷息怒，我家公子是读书人": "向龚光杰赔不是",
    "对那姑娘说：多谢姑娘替我家公子出头": "对钟灵说：多谢姑娘替我家公子出头",
    "问段公子：伤得重不重？": "问段公子伤得重不重",
    "陪段公子在井边歇一会儿": "等一会儿",
    "在林边找块石头坐下，歇一会儿": "等一会儿",
    "我回后院取些水来": "去后院",
    "回后山": "去后山",
    "守着段公子，等下去": "等下去",
    "警觉地四下张望": "环顾四周",
    "探头往崖下望了望": "环顾四周",
    "攀着藤萝往下爬": "跳下断崖",
    "晃亮火折子，生堆火取暖": "叹了口气",
    "对那姑娘说：你怎么也跟下来了？": "对钟灵说：你怎么也跟下来了",
    "跟着段公子钻进石缝": "钻进石缝",
    "看段公子在做什么": "看看段公子",
    "拿起那卷凌波微步": "拿凌波微步",
    "问段公子：那帛卷上写的什么？": "问段公子那帛卷上写的什么",
    "长长舒了一口气": "叹了口气",
    "问马五爷：这可怎么是好？": "问马五爷这可怎么是好",
    "问马五爷：段公子跑去哪儿了？": "问马五爷段公子在哪",
    "给马五爷续茶": "叹了口气",
    "问马五爷：这山上可有什么稀奇的传闻？": "问马五爷这山上有什么传闻",
    "把茶饼献给左掌门": "把茶饼给左子穆",
    "向辛掌门作了个揖": "向辛双清作揖",
    "找个角落坐下，歇半个时辰": "等半个时辰",
    "再坐一盏茶的工夫": "等一盏茶",
    "把茶担挑去后院厨下": "去后院",
    "在厨下坐着，一直等到天黑": "等到天黑",
    "埋头收拾茶担，等到天黑": "等到天黑",
    "对龚光杰说：段公子往山道去了": "告诉龚光杰段公子在山道",
    "回大殿去": "去大殿",
    "靠着柱子打个盹，等下去": "等下去",
    "往后缩了缩，躲到东家身后": "叹了口气",
    "问马五爷：神农帮为什么围着山？": "问马五爷神农帮为什么围山",
    "问马五爷：咱们怎么下山？": "问马五爷怎么下山",
    "找个不起眼的角落坐下": "叹了口气",
    "向辛掌门打听：山道上的关卡几时换班？": "问辛双清关卡几时换班",
    "在角落里等到天黑": "等到天黑",
    "在角落里一直等到天黑": "等到天黑",
    "再等一个时辰": "等一个时辰",
    "接着再等半个时辰": "等半个时辰",
    "悄悄溜出大殿，去山道": "悄悄溜去山道",
    "看看关卡上的栅门": "查看山道关卡",
    "悄悄溜去山脚": "去山脚",
    "回头望了望无量山": "叹了口气",
}
Check = Callable[[WorldState, GameSession], bool]


def _at(st: WorldState, e: str) -> str | None:
    return st.target(e, Rel.AT)


def _here(who: str) -> Check:
    return lambda st, s: _at(st, who) == _at(st, PLAYER)


def _in(*places: str) -> Check:
    return lambda st, s: _at(st, PLAYER) in places


def _both(*checks: Check) -> Check:
    return lambda st, s: all(c(st, s) for c in checks)


WHEN: tuple[tuple[tuple[str, ...], Check], ...] = (
    (("跟上段公子",), lambda st, s: not _here(DUANYU)(st, s)),
    (("跟着段公子钻进石缝",), _both(_in("jianhu"), lambda st, s: s.beliefs(PLAYER).knows("d_cave"),
                                    lambda st, s: _at(st, DUANYU) in ("jianhu", "shidong"))),
    (("研读凌波微步",), lambda st, s: _at(st, "scroll_lb") == PLAYER),
    (("拿起那卷凌波微步",), _both(_in("langhuan"), lambda st, s: s.beliefs(PLAYER).knows("scroll_lb"))),
    (("隧道",), _both(_in("langhuan"), lambda st, s: s.beliefs(PLAYER).knows("d_tunnel"))),
    (("玉像", "蒲团"), _in("langhuan")),
    (("问段公子", "扯了扯段公子", "看段公子", "陪段公子", "守着段公子"), _here(DUANYU)),
    (("对那姑娘", "梁上那少女"), _here("zhongling")),
    (("对龚光杰", "向龚光杰", "警觉"), _here("gongguangjie")),
    (("左掌门",), _here("zuozimu")),
    (("辛掌门",), _here("xinshuangqing")),
    (("马五爷", "东家"), _here("mawude")),
    (("后院取些水",), _both(_in("houshan"), lambda st, s: _at(st, "ganguanghao") == _at(st, "geguangpei") == "houyuan")),
    (("后院厨下",), _in("hall")),
    (("回后山",), _in("houyuan")),
    (("回大殿",), lambda st, s: _at(st, PLAYER) != "hall"),
    (("禁地", "林边"), _in("houshan")),
    (("崖下", "藤萝"), _in("yading")),
    (("玉璧", "火折子"), _in("jianhu")),
    (("石门",), _in("shidong")),
    (("月亮出来",), lambda st, s: st.clock < MOONRISE),
    (("等到天黑",), lambda st, s: st.clock < NIGHTFALL),
    (("去山道",), _both(_in("hall"), lambda st, s: SUPPER <= st.clock < SUPPER_END)),
    (("关卡", "山脚"), _both(_in("shandao"), lambda st, s: not st.attr("d_downhill", "locked", False))),
)


WALKED = ("playthrough", "playthrough_hall", "playthrough_flee")


@dataclass
class Walk:
    seed: int
    key: str
    problems: list[str] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)
    ending: str | None = None
    turns: int = 0


def walk(seed: int, inputs: Iterable[str], key: str = "playthrough") -> Walk:
    """模板会话逐句走一局（每句按 CANON 换成规则解析认得的说法）：回合开始时 WHEN 不成立、或该推进却没推进的句子记为问题。"""
    s = GameSession(build_wuliang_commoner(seed), pipeline=False)
    s.intro()
    out, seen = Walk(seed, key), set()
    for i, text in enumerate(inputs, 1):
        st = s.authority.head()
        check = next((c for words, c in WHEN if any(w in text for w in words)), None)
        if check is not None and not check(st, s):
            out.problems.append(f"{i}「{text}」落空（{clock_label(st.clock)}，你在{_at(st, PLAYER)}）")
        r = s.turn(CANON.get(text, text))
        out.turns = i
        if not r.advanced and r.kind.value != "ask_gm":
            out.problems.append(f"{i}「{text}」没接住：{r.narration[:40]}")
        seen |= set(r.beats)
        if r.ending is not None:
            out.ending = r.ending.key
            break
    out.beats = [k for k in BEAT_KEYS if k in seen]
    return out


def _seeds(text: str) -> list[int]:
    if "," in text:
        return [int(x) for x in text.split(",")]
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="普通人版调参台：种子 × 脚本化玩家，报告看点、结局与驱力")
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--players", default=",".join(PLAYERS))
    ap.add_argument("--json", help="逐局结果写到这个文件")
    ap.add_argument("--check", action="store_true", help="按出口条件判定，不达标以非零码退出")
    ap.add_argument("--jobs", type=int, default=1, help="并行的进程数（每局各自确定，并行与否结果逐项相同）")
    ap.add_argument("--walk", metavar="PROBES", help="走查核对：探针文件里的每条整局在各种子上逐句落在有意义的状态上"
                                                    "（配 --seeds 7,11），有落空的句子以非零码退出")
    args = ap.parse_args(argv)
    if args.walk:
        with open(args.walk, encoding="utf-8") as f:
            probes = json.load(f)
        walks = [walk(seed, probes[k], k) for k in WALKED if k in probes for seed in _seeds(args.seeds)]
        for w in walks:
            print(f"{w.key} 种子 {w.seed}：{w.turns} 句，结局 {w.ending or '（未落幕）'}，看点 {len(w.beats)}/{len(BEAT_KEYS)}"
                  f"（{'、'.join(w.beats)}）" + "".join(f"\n  {p}" for p in w.problems))
        return 1 if any(w.problems for w in walks) else 0
    jobs = [(seed, p) for p in args.players.split(",") for seed in _seeds(args.seeds)]
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            runs = list(pool.map(play, *zip(*jobs, strict=True)))
    else:
        runs = [play(seed, p) for seed, p in jobs]
    print(table(runs))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in runs], f, ensure_ascii=False, indent=1)
    bad = check(runs) if args.check else []
    for b in bad:
        print("未达标：" + b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
