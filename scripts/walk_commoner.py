"""
[INPUT]: 依赖 tianlong.runtime.session 的 GameSession，tianlong.runtime.names 的 forms（本名与带名的别称），
         tianlong.language.llm 的 ScriptedLLM / LLMUnavailable，tianlong.language.render 的 sentence_ends，
         tianlong.scenarios 的 build_wuliang_commoner，同目录 sim_beats 的 PLAYERS（跟随型脚本化玩家）
[OUTPUT]: 命令行 python scripts/walk_commoner.py [--seeds 7,11] [--mode playthrough|follower] [--show] [--check]，
          打印每回合的丢句、SEAM、NAME 与三项合计（G1 丢句比例 ≤ 5%、SEAM = 0、NAME = 0）；--show 另打印每回合玩家所见；
          也是可导入的库：PLAYTHROUGH（plan §8.1 跟随型 38 句）、echo()（脚本化叙述者的答案）、walk() -> list[Turn]、metrics()
[POS]: scripts 的 M3 出口检查：scripted 管线走一遍普通人版。叙述者是 ScriptedLLM，它的“答案”照提示词写：天色、事实清单、
       本回合的看点（照写成一句）、照录的原话与要说出口的话（写成“某某道：“……””）、眼前的景象、初见外观、这回多看出的细节，
       再加上本回合点到的写法卡卡文——也就是“模板正文 + 看点 + 看点卡写法”，全部交给真实的闸门逐句验收。不解释玩家输入（带 schema 的
       调用一律 LLMUnavailable，解释退回规则），迟到先声关掉（逐字可复现）。
       G1 = 丢句 / (丢句 + 交付的句子)；SEAM = 模型最后一句之后还追加了模板行的回合数（补句插在钩子之前不算）；
       NAME = 交付的句子里出现玩家账本当时（叙述开始那一刻：听见的原话已学、交付的正文还没学）叫不出名字的人的本名或带名别称的句数
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 sim_beats
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_beats  # noqa: E402

from tianlong.language.llm import LLMUnavailable, ScriptedLLM  # noqa: E402
from tianlong.language.render import sentence_ends  # noqa: E402
from tianlong.runtime.names import forms  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import build_wuliang_commoner  # noqa: E402

PLAYER = "ashun"
GATES = {"G1": 0.05, "SEAM": 0, "NAME": 0}
# plan §8.1 跟随型整局游玩（38 句）：规则解析认不得的句子照样送进去——追问与场外回答不推进，也不算叙述
PLAYTHROUGH: tuple[str, ...] = (
    "环顾四周", "扯了扯段公子的袖子，低声道：公子，快别笑了", "向龚光杰赔笑道：这位爷息怒，我家公子是读书人",
    "问马五爷这位龚爷是什么来头", "我该怎么办？", "抬头看看梁上那少女", "等待", "对那姑娘说：多谢姑娘替我家公子出头",
    "跟上段公子", "四下打量一番", "问段公子：伤得重不重？", "跟上段公子", "叹了口气，自言自语道：这禁地可不能久留",
    "我回后院取些水来", "四下打量一番", "回后山", "等待", "跟上段公子", "探头往崖下望了望", "攀着藤萝往下爬", "查看玉璧",
    "我身上还有什么？", "晃亮火折子，生堆火取暖", "对那姑娘说：你怎么也跟下来了？", "一直等到月亮出来", "查看玉璧",
    "跟着段公子钻进石缝", "进石门", "看看那尊玉像", "看段公子在做什么", "查看蒲团", "问段公子：那帛卷上写的什么？",
    "求段公子把凌波微步借我看看", "研读凌波微步", "研读凌波微步", "接下来该往哪儿走？", "钻进隧道", "长长舒了一口气",
)
FOLLOWER_TURNS = 120          # --mode follower：跟随型脚本化玩家至多走这么多回合（落幕即停）

# ============================================================
#  脚本化叙述者：照提示词写“模板正文 + 看点卡写法里的一句”
# ============================================================

_ITEM = re.compile(r"^\d+\. (?P<who>.+?)(?:（[^）]*）)?对(?P<to>[^（：]+?)(?:（[^）]*）)?：(?P<body>.*)$")
_WORDS = re.compile(r"(?:原话|说法)“(?P<w>[^”]*)”")
_CATALOG = re.compile(r"^- (?P<key>\w+)：(?P<text>.+)$", re.M)


def _prose(line: str) -> str:
    line = line.strip()
    return line if not line or line[-1] in "。！？”…）" else line + "。"


def echo(prompt: str, system: str | None, schema: object = None) -> str:
    """脚本化叙述者的答案。带 schema 的调用（解释玩家输入）一律不接：解释退回规则。"""
    if schema is not None:
        raise LLMUnavailable("scripted walk: 不解释玩家输入")
    cards = dict(_CATALOG.findall(system or "")) if "写法卡目录" in (system or "") else {}
    head: list[str] = []
    body: list[str] = []
    for part in prompt.split("\n\n"):
        title, _, rest = part.partition("\n")
        if title.startswith("天色"):
            head.append(title.split("：", 1)[1])
        elif title.startswith("本回合的看点"):                # 看点也照写成一句：它说的事须真、须过得了闸门
            body.append(title.split("：", 1)[1])
        elif title.startswith("本回合玩家感知到的事实"):
            body += [x for x in rest.splitlines() if not x.startswith("（除下列言语外") and not x.startswith("（无事发生")]
        elif title.startswith(("照录的原话", "要说出口的话")):
            for row in rest.splitlines():
                m = _ITEM.match(row)
                if m is None:
                    continue
                w = _WORDS.search(m["body"]) or re.search(r"“(?P<w>[^”]*)”$", m["body"])
                body.append(f"{m['who']}道：“{w['w']}”" if w else f"{m['who']}看了{m['to']}一眼。")
        elif title.startswith(("眼前的景象", "玩家初次看清的人与物")):
            body += rest.splitlines()
        elif title.startswith(("这回多看出的一处细节", "写法")):
            text = title.split("：", 1)[1]
            body += [cards[k] for k in text.split("、") if k in cards] if title.startswith("写法") else [text]
    return "\n".join(_prose(x) for x in [*head, *body] if x.strip())


# ============================================================
#  走一遍
# ============================================================


@dataclass
class Turn:
    clock: str
    text: str
    narration: str
    advanced: bool
    answer: str = ""
    dropped: int = 0
    sentences: int = 0
    seam: bool = False
    names: list[str] = field(default_factory=list)
    beats: tuple[str, ...] = ()
    focus: str | None = None
    violations: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    cut, out = 0, []
    for end in sentence_ends(text, True):
        if text[cut:end].strip():
            out.append(text[cut:end].strip())
        cut = end
    return out + ([text[cut:].strip()] if text[cut:].strip() else [])


def walk(seed: int = 7, mode: str = "playthrough") -> list[Turn]:
    sc = build_wuliang_commoner(seed)
    answers: list[str] = []

    def respond(prompt: str, system: str | None, schema: object) -> str:
        answers.append(echo(prompt, system, schema))
        return answers[-1]

    s = GameSession(sc, llm=ScriptedLLM(respond), pipeline=False)
    s.narrator.lead_after = None                                 # 迟到先声由网速决定出不出场：关掉，逐字可复现
    known: list[frozenset[str]] = []
    render = s._render

    def spy(*args, **kwargs):                                    # 叙述开始那一刻玩家叫得出谁（听见的已学，交付的还没学）
        known.append(s._acq.of(PLAYER))
        return render(*args, **kwargs)
    s._render = spy
    s.intro()
    out: list[Turn] = []
    memo: dict = {}
    inputs = iter(PLAYTHROUGH) if mode == "playthrough" else None
    for _ in range(len(PLAYTHROUGH) if inputs is not None else FOLLOWER_TURNS):
        text = next(inputs) if inputs is not None else sim_beats.PLAYERS["follower"](s, memo)
        before = len(answers)
        r = s.turn(text)
        t = Turn(r.clock, text, r.narration, r.advanced, beats=r.beats, focus=r.brief.focus if r.brief else None)
        if r.advanced and r.render is not None and len(answers) > before:
            t.answer = answers[-1]
            t.dropped = r.render.dropped
            mine = set(_sentences(t.answer))
            said = _sentences(r.narration)
            t.sentences = sum(1 for x in said if x in mine)
            t.seam = bool(said) and said[-1] not in mine and any(x in mine for x in said)
            hidden = {f for eid in sc.epithets if eid not in known[-1] for f in forms(sc, eid)}
            t.names = [x for x in said if any(f in x for f in hidden)]
            t.violations = [f"{v.kind}:{v.detail}" for v in r.render.violations]
        out.append(t)
        if r.ending is not None:
            break
    return out


def metrics(turns: Sequence[Turn]) -> dict[str, float]:
    dropped = sum(t.dropped for t in turns)
    delivered = sum(t.sentences for t in turns)
    return {"G1": dropped / (dropped + delivered) if dropped + delivered else 0.0, "SEAM": sum(t.seam for t in turns),
            "NAME": sum(len(t.names) for t in turns), "dropped": dropped, "sentences": delivered,
            "narrated": sum(1 for t in turns if t.answer)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="普通人版的 scripted 走查：G1 / SEAM / NAME")
    ap.add_argument("--seeds", default="7,11")
    ap.add_argument("--mode", choices=("playthrough", "follower"), default="playthrough")
    ap.add_argument("--show", action="store_true", help="打印每回合玩家所见")
    ap.add_argument("--check", action="store_true", help="不达标以非零码退出")
    args = ap.parse_args(argv)
    logging.getLogger("tianlong.language.interpret").setLevel(logging.ERROR)   # 解释一律退回规则：不必每回合说一遍
    total: list[Turn] = []
    for seed in (int(x) for x in args.seeds.split(",")):
        turns = walk(seed, args.mode)
        total += turns
        print(f"## 种子 {seed}（{args.mode}，{len(turns)} 回合）")
        for t in turns:
            flag = " ".join(x for x in (f"丢{t.dropped}" if t.dropped else "", "SEAM" if t.seam else "",
                                        f"NAME{t.names}" if t.names else "", f"看点={t.focus}" if t.focus else "") if x)
            print(f"[{t.clock}] > {t.text}  {flag}" + (f"  {t.violations}" if t.violations else ""))
            if args.show:
                print(t.narration + "\n")
    m = metrics(total)
    print(f"\nG1 = {m['G1']:.1%}（丢 {m['dropped']} / 交付 {m['sentences']} 句，叙述 {m['narrated']} 回合）"
          f"  SEAM = {m['SEAM']}  NAME = {m['NAME']}")
    bad = m["G1"] > GATES["G1"] or m["SEAM"] > GATES["SEAM"] or m["NAME"] > GATES["NAME"]
    return 1 if args.check and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
