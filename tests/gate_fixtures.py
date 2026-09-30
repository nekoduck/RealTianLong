"""
[INPUT]: 依赖 core 的 Social，language/render 的 RenderPlan，language/scene 的 SceneBrief / VoiceLine，
         tests/data 的 gate_corpus.json（误杀语料）与 gate_seeded.json（植入错误与阴性对照）
[OUTPUT]: 对外提供 decode()（按 dataclasses.fields 与类型注解把 JSON 还原成 RenderPlan / SceneBrief / VoiceLine / Sky：
          tuple 与 frozenset 分得清，Social 还原成枚举）、Case（一条闸门输入 + 期望）、corpus() / seeded()（读出全部条目）
[POS]: tests 的闸门语料解码器，只被 test_gate_precision 使用（不进 src：只有测试要把冻结的闸门输入读回来）。
       语料里同一次叙述调用的 plan / brief 去重存放在 "plans" / "briefs" 表里，条目按下标引用；
       植入条目以语料里的某条为底（base），换掉本句与此前正文；可另给 "plan" / "brief" 两个对象，按字段覆盖底里的计划与
       叙述上下文（M3 审计的植入：天色 sky、被制的人 afflicted、谁身上有什么 holdings、本回合做成了什么 done），其余照底
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tianlong.core import Social
from tianlong.language.render import RenderPlan
from tianlong.language.scene import SceneBrief

DATA = Path(__file__).with_name("data")


# ============================================================
#  解码：JSON（列表、对象）→ 冻结的数据类
# ============================================================


def decode(tp: typing.Any, v: typing.Any) -> typing.Any:
    """按类型注解还原：frozenset 与 tuple 都存成 JSON 数组，靠注解分开；可空类型取第一个解得通的分支。"""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if v is None:
        return None
    if tp is frozenset or origin is frozenset:
        return frozenset(decode(args[0], x) if args else x for x in v)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], x) for x in v)
        return tuple(decode(a, x) for a, x in zip(args, v, strict=True))
    if origin is typing.Union or origin is types.UnionType:
        return next(decode(a, v) for a in args if a is not type(None))
    if tp is Social:
        return Social(v)
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        return tp(**{f.name: decode(hints[f.name], v[f.name]) for f in dataclasses.fields(tp) if f.name in v})
    return v


# ============================================================
#  条目：闸门的全部输入 + 期望
# ============================================================


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    piece: str
    before: str
    known: frozenset[str]
    command: str
    plan: RenderPlan
    brief: SceneBrief
    verdict: str            # 语料：FP（误杀，应放行）/ TP（真拦）；植入：TP（应拦）/ OK（阴性对照，应放行）
    category: str
    reason: str

    @property
    def text(self) -> str:
        return self.before + self.piece


def _patched(obj: typing.Any, over: dict | None) -> typing.Any:
    """按字段覆盖一个冻结的数据类：值照字段的类型注解解码（"sky": {...} 还原成 Sky）。"""
    if not over:
        return obj
    hints = typing.get_type_hints(type(obj))
    return dataclasses.replace(obj, **{k: decode(hints[k], v) for k, v in over.items()})


def _cases(path: Path) -> list[Case]:
    raw = json.loads(path.read_text("utf-8"))
    plans = [decode(RenderPlan, p) for p in raw["plans"]]
    briefs = [decode(SceneBrief, b) for b in raw["briefs"]]
    return [Case(r["id"], r["piece"], r["before"], frozenset(r["known"]), r["command"], plans[r["plan"]],
                 briefs[r["brief"]], r["verdict"], r["category"], r["reason"]) for r in raw["cases"]]


@cache
def corpus() -> tuple[Case, ...]:
    return tuple(_cases(DATA / "gate_corpus.json"))


@cache
def seeded() -> tuple[Case, ...]:
    """植入条目：底（base 指向语料里的一条）+ 换掉的本句与此前正文 + 按字段覆盖的计划与叙述上下文。"""
    base = {c.id: c for c in corpus()}
    raw = json.loads((DATA / "gate_seeded.json").read_text("utf-8"))
    return tuple(dataclasses.replace(base[r["base"]], id=r["id"], piece=r["piece"], before=r.get("before", ""),
                                     command=r.get("command", base[r["base"]].command), verdict=r["verdict"],
                                     category=r["category"], reason=r["reason"],
                                     plan=_patched(base[r["base"]].plan, r.get("plan")),
                                     brief=_patched(base[r["base"]].brief, r.get("brief")))
                 for r in raw["cases"])
