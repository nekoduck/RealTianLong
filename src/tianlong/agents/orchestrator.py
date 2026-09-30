"""
[INPUT]: 依赖 langgraph 的 StateGraph / Send / Runtime，agents/npc_graph 的 build_npc_graph / NpcContext / checkpoint_serde，
         agents/policies 的 hush_chatter，cognition 的 Candidate，core 的 Intent / Op
[OUTPUT]: 对外提供 Deliberation（一次决策的可解释轨迹，带策略标签与驱力出处 drive）、Orchestrator（一个 tick 内多个 NPC 的并行决策；
          每处每 tick 至多一句闲谈；决策轨迹检查点须显式开启，开启后按条数修剪）
[POS]: agents 的多智能体编排：基于同一版本观察，把需要决策的角色扇出（Send）并行运行各自的决策流程，汇总结构化意图。
       汇总时裁决话头（hush_chatter）：同处几个人同时想闲谈只留一句、已有人正经开口则闲谈让出——地点取各人自己认为的所在，
       按捺住的改为原地等待（同一意图 ID），与扇出顺序无关、确定性可重算。驱力的选择从不让出话头（按正经开口 SPEAK 算），
       改写成等待时也不丢驱力的原话。
       它只产出意图，从不写世界——提交与冲突结算归 WorldAuthority。
       检查点默认关闭：剖析显示它占 NPC 决策耗时的 65~75%，而会话从不读它（重试靠意图 ID 去重，不靠恢复轨迹）；
       要看决策轨迹（调试、测试）就显式传 checkpoint=True
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import operator
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from tianlong.agents.npc_graph import NpcContext, build_npc_graph, checkpoint_serde
from tianlong.agents.policies import hush_chatter
from tianlong.agents.policy_kit import CHATTER, SPEAK
from tianlong.cognition import Candidate
from tianlong.core import Intent, Op


@dataclass(frozen=True, slots=True)
class Deliberation:
    agent: str
    intent: Intent
    rationale: str
    recalled: tuple[str, ...]
    considered: int
    tag: str = ""        # 策略的结构化标签（等待原因、explore、speak、chatter）
    drive: str = ""      # 这一步出自哪条驱力（空 = 不是驱力）：会话据此只给兑现成功的驱力记标记


class _TickState(TypedDict, total=False):
    agents: list[str]
    deliberations: Annotated[list[Deliberation], operator.add]


@dataclass
class _TickContext:
    npcs: Mapping[str, NpcContext]


class Orchestrator:
    """checkpoint：是否把每次决策的轨迹写进 LangGraph 检查点（默认否——没人读，却是决策耗时的大头）。
    keep_threads：开启检查点时保留最近多少条决策轨迹。每个角色每个 tick 一条线程，不修剪就会在长局中无限增长。"""

    def __init__(self, keep_threads: int = 64, checkpoint: bool = False) -> None:
        self.checkpointer: InMemorySaver | None = InMemorySaver(serde=checkpoint_serde()) if checkpoint else None
        self.npc_graph = build_npc_graph(self.checkpointer)
        self.graph = self._build()
        self.keep_threads = keep_threads
        self._threads: deque[str] = deque()

    @staticmethod
    def thread_id(port) -> str:
        return f"{port.world_id}:{port.branch_id}:{port.agent}:{port.version}"

    def _build(self):
        npc_graph = self.npc_graph
        traced = self.checkpointer is not None

        def fan_out(state: _TickState) -> list[Send]:
            return [Send("deliberate", {"agent": a}) for a in state["agents"]]

        def deliberate(state: dict, runtime: Runtime[_TickContext]) -> _TickState:
            ctx = runtime.context.npcs[state["agent"]]
            port = ctx.port
            config = {"configurable": {"thread_id": Orchestrator.thread_id(port)}} if traced else None
            out = npc_graph.invoke({"agent": port.agent}, config=config, context=ctx)
            d = Deliberation(port.agent, out["intent"], out["rationale"],
                             tuple(out.get("recent", [])) + tuple(out.get("related", [])),
                             len(out["candidates"]), out.get("tag", ""), out.get("drive") or "")
            return {"deliberations": [d]}

        g = StateGraph(_TickState, context_schema=_TickContext)
        g.add_node("deliberate", deliberate)
        g.add_conditional_edges(START, fan_out, ["deliberate"])
        g.add_edge("deliberate", END)
        return g.compile()

    def decide(self, npcs: Mapping[str, NpcContext]) -> list[Deliberation]:
        if not npcs:
            return []
        out = self.graph.invoke({"agents": sorted(npcs), "deliberations": []}, context=_TickContext(npcs))
        if self.checkpointer is not None:
            for ctx in npcs.values():
                tid = self.thread_id(ctx.port)
                if tid not in self._threads:
                    self._threads.append(tid)
            while len(self._threads) > self.keep_threads:
                self.checkpointer.delete_thread(self._threads.popleft())
        return self._share_the_floor(sorted(out["deliberations"], key=lambda d: d.agent), npcs)

    @staticmethod
    def _share_the_floor(delibs: list[Deliberation], npcs: Mapping[str, NpcContext]) -> list[Deliberation]:
        """每处每 tick 至多一句闲谈：只有开口的人参与裁决，地点取各人自己认为的所在；按捺住的原地等待。
        驱力的选择从不让出话头（按正经开口算），万一被改写成等待也带着它的原话。"""
        talking = [d for d in delibs if d.intent.op in (Op.TELL, Op.ASK)]
        if not any(d.tag == CHATTER and not d.drive for d in talking):
            return delibs
        port = npcs[talking[0].agent].port
        hushed = hush_chatter(port.now, {d.agent: (npcs[d.agent].port.beliefs().location_of(d.agent),
                                                   SPEAK if d.drive else d.tag, Candidate.of(d.intent))
                                         for d in talking})
        return [replace(d, intent=Intent(d.intent.id, d.agent, Op.WAIT, based_on=d.intent.based_on,
                                         utterance=d.intent.utterance if d.drive else None),
                        rationale=f"{d.rationale}（旁人正说着，没插上嘴）", tag="idle") if d.agent in hushed else d
                for d in delibs]
