"""
[INPUT]: 依赖 agents/drives 的 Driven / Marks，agents/policies 的 Policy / ScriptedPolicy，agents/orchestrator 的 Deliberation（只作类型标注），
         runtime/authority 的 Settlement，core 的 Outcome，core/drives 的 Drive / Between / Not / AnyOf，scenarios 的 Scenario
[OUTPUT]: 对外提供 policy_for()（给 NPC 套上驱力）、wakes()（驱力的时间窗打开即唤醒）、advance_marks()（只记兑现成功的驱力）、
          MARKS_KEEP、Cast（{角色: {驱力: [兑现时刻]}}）
[POS]: runtime 的角色装配：会话经它把场景的驱力表接到策略、调度与会话运行态上。
       驱力标记是会话运行态的一部分：在 annotate 的副本上推进、与世界同一事务落库、读档原样恢复；
       只认意图在事件里结算成功（Outcome.SUCCESS）的驱力——once 与 cooldown 据此判断，没做成的不算兑现。
       不改 Scheduler：时间窗的起点落在上次决策与此刻之间就把该 NPC 列为需决策，其余照调度器
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from tianlong.agents.drives import Driven, Marks
from tianlong.agents.policies import Policy, ScriptedPolicy
from tianlong.core import Outcome
from tianlong.core.drives import AnyOf, Between, Cond, Drive, Not
from tianlong.runtime.authority import Settlement
from tianlong.scenarios import Scenario

if TYPE_CHECKING:                     # 只为类型标注：cast 本身不依赖 LangGraph（编排层是可选依赖）
    from tianlong.agents.orchestrator import Deliberation

MARKS_KEEP = 32          # 每条驱力只留最近这么多次兑现（once / cooldown / Fired 的次数与时间窗都够用；次数到此封顶）
Cast = Mapping[str, Marks]


def policy_for(agent: str, scenario: Scenario, override: Policy | None, marks: Marks,
               proposal: Drive | None = None) -> Policy:
    """有驱力（或一次性提议）就把它套在策略外面——学得的策略（override）也一样；否则原路径。"""
    inner = override or ScriptedPolicy()
    drives = scenario.drives.get(agent, ())
    return Driven(inner, drives, marks, proposal) if drives or proposal is not None else inner


def _edges(c: Cond) -> Iterable[int]:
    if isinstance(c, Between):
        yield c.start
        if c.end is not None:
            yield c.end
    elif isinstance(c, Not):
        yield from _edges(c.cond)
    elif isinstance(c, AnyOf):
        for x in c.conds:
            yield from _edges(x)


def wakes(drives: Sequence[Drive], last_tick: int | None, now: int) -> bool:
    """驱力的时间窗在上次决策之后、此刻或之前打开（或关上）：该醒了。从没决策过的交给调度器（它本就会叫醒）。"""
    if last_tick is None:
        return False
    return any(last_tick < t <= now for d in drives for c in d.when for t in _edges(c))


def advance_marks(marks: Cast, deliberations: Iterable[Deliberation], settlement: Settlement) -> dict[str, dict[str, list[int]]]:
    """新的一份驱力标记（不改入参）：本 tick 出自驱力、且意图结算成功的，记下结算时刻。"""
    done = {e.intent.id: e.tick for e in settlement.events if e.outcome == Outcome.SUCCESS}
    out = {a: {k: list(v) for k, v in m.items()} for a, m in marks.items()}
    for d in deliberations:
        if d.drive and d.intent.id in done:
            mine = out.setdefault(d.agent, {})
            mine[d.drive] = [*mine.get(d.drive, ()), done[d.intent.id]][-MARKS_KEEP:]
    return out
