"""
[INPUT]: 依赖 langgraph 的 StateGraph / Runtime / InMemorySaver / JsonPlusSerializer，agents 的 port（含长期记忆摘要）/ policies / predictors，
         cognition 的 candidates，language/speaker 的 Speaker，language/templates 的 render_experience
[OUTPUT]: 对外提供 NpcState、NpcContext（可带主角 player，转交 Situation；recall 开关：只有读 Situation.memories 的策略才需要向量回忆）、
          build_npc_graph()、checkpoint_serde()
[POS]: agents 的单角色决策流程：观察 → 回忆 → 形成候选 → 预测后果 → 选择 → 表达 → 提交意图。
       选择带出驱力的原话（Choice.line，在 chosen() 之后挂上）与出处（Choice.drive）：表达时有原话就用它作任何行动的 utterance，
       没有才走措辞器（只给带命题的 TELL/ASK 措辞）。
       依赖通过 LangGraph runtime context 注入（不进检查点）；检查点只保存本次决策的轨迹，不是世界状态。
       观察与回忆只为 Situation.memories 服务：现有策略（脚本、学得的）都不读它，默认跳过（省下每人每 tick 一次向量检索）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from tianlong.agents.policies import Policy, ScriptedPolicy, Situation
from tianlong.agents.port import AgentPort
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor, Prediction
from tianlong.cognition import Candidate, candidates
from tianlong.core import Intent, Op, make_id
from tianlong.language.speaker import Speaker, TemplateSpeaker
from tianlong.language.templates import Names, render_experience


class NpcState(TypedDict, total=False):
    agent: str
    query: str
    recent: list[str]
    related: list[str]
    candidates: list[Candidate]
    predictions: list[Prediction]
    choice: int
    chosen: Candidate          # 交给内核的行动（候选或候选之外的言语，附言语行为）
    rationale: str
    tag: str                   # 策略的结构化标签（Choice.tag）：编排据 CHATTER 给闲谈裁决话头
    line: str | None           # 驱力给的原话（Choice.line）：有就是这一步的 utterance，不论什么行动
    drive: str | None          # 这一步出自哪条驱力（Choice.drive）
    utterance: str | None
    intent: Intent


@dataclass
class NpcContext:
    port: AgentPort
    policy: Policy = field(default_factory=ScriptedPolicy)
    predictor: OutcomePredictor = field(default_factory=HeuristicPredictor)
    speaker: Speaker = field(default_factory=TemplateSpeaker)
    max_candidates: int = 64
    player: str | None = None      # 主角是谁（公开身份）：交给 Situation.player，搭话与见义出声以他为准
    recall: bool = False           # 是否跑回忆节点（向量检索）：只有读 Situation.memories 的策略需要，现有策略都不读；
                                   # 会话按策略的 reads_memories 属性设它

    def interests(self) -> list[str]:
        return list(self.port.profile.interests())


def checkpoint_serde() -> JsonPlusSerializer:
    """检查点反序列化白名单：只允许本流程状态里出现的类型，杜绝任意类构造。"""
    return JsonPlusSerializer(allowed_msgpack_modules=[
        ("tianlong.core.schema", "Op"),
        ("tianlong.core.schema", "Manner"),
        ("tianlong.core.schema", "Social"),
        ("tianlong.core.propositions", "Proposition"),
        ("tianlong.core.propositions", "Fact"),
        ("tianlong.core.events", "Intent"),
        ("tianlong.cognition.candidates", "Candidate"),
        ("tianlong.agents.predictors", "Prediction"),
    ])


# ============================================================
#  节点：每个节点只经由 port 读取本角色的认知与回忆
# ============================================================


def _names(ctx: NpcContext) -> Names:
    return ctx.port.beliefs().entities


def observe(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    """把最近的经历凝成一句“当下处境”，作为回忆的线索（不回忆就不必凝）。"""
    ctx = runtime.context
    if not ctx.recall:
        return {"query": ""}
    store = ctx.port.beliefs()
    names = _names(ctx)
    lines = [
        render_experience(ep.modality, ep.event, names, ctx.port.agent)
        for ep in store.episodes[-3:]
        if ctx.port.now - ep.tick <= 10
    ]
    return {"query": "；".join(lines) or "一切如常"}


def recall(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    port = runtime.context.port
    if not runtime.context.recall or port.recall is None:
        return {"recent": [], "related": []}
    r = port.recall(state.get("query", ""))
    return {"recent": [m.text for m in r.recent], "related": [x.record.text for x in r.related]}


def propose(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    ctx = runtime.context
    return {"candidates": list(candidates(ctx.port.beliefs(), ctx.interests(), ctx.max_candidates))}


def predict(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    ctx = runtime.context
    preds = ctx.predictor.predict(ctx.port.beliefs(), ctx.port.now, state["candidates"], ctx.interests(),
                                  profile=ctx.port.profile)
    return {"predictions": list(preds)}


def decide(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    ctx = runtime.context
    sit = Situation(
        ctx.port.agent, ctx.port.profile, ctx.port.beliefs(), ctx.port.now,
        tuple(state["candidates"]), tuple(state["predictions"]),
        tuple(state.get("recent", [])) + tuple(state.get("related", [])),
        ctx.port.memory() if ctx.port.memory is not None else None,
        player=ctx.player,
    )
    choice = ctx.policy.choose(sit)
    chosen = choice.chosen(state["candidates"], sit.beliefs)    # 越界或不合规的 free（含说的不是自己相信的事）在这里抛错
    return {"choice": choice.index, "chosen": chosen, "rationale": choice.rationale, "tag": choice.tag,
            "line": choice.line, "drive": choice.drive}


def express(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    ctx = runtime.context
    cand = state["chosen"]
    if state.get("line"):
        return {"utterance": state["line"]}   # 驱力的原话：任何行动都带得上（走、打、用、等、说），在场的人照原样听见看见
    if cand.op not in (Op.TELL, Op.ASK) or cand.topic is None:
        return {"utterance": None}       # 闲话没有命题可说：措辞留给主持人之声（叙述时按说话者的认知与腔调写出）；带命题的答话照常措辞
    return {"utterance": ctx.speaker.utter(ctx.port.profile, cand, _names(ctx))}


def submit(state: NpcState, runtime: Runtime[NpcContext]) -> NpcState:
    """意图 ID 由 (世界, 分支, 角色, 版本) 派生：流程重试产出同一 ID，权威写入器据此去重。"""
    port = runtime.context.port
    cand = state["chosen"]
    iid = make_id("int", port.world_id, port.branch_id, port.agent, port.version)
    return {"intent": cand.to_intent(iid, port.agent, port.version, state.get("utterance"))}


def build_npc_graph(checkpointer: InMemorySaver | None = None):
    g = StateGraph(NpcState, context_schema=NpcContext)
    steps = [("observe", observe), ("recall", recall), ("propose", propose), ("predict", predict),
             ("decide", decide), ("express", express), ("submit", submit)]
    for name, fn in steps:
        g.add_node(name, fn)
    g.add_edge(START, steps[0][0])
    for (a, _), (b, _) in zip(steps, steps[1:], strict=False):
        g.add_edge(a, b)
    g.add_edge(steps[-1][0], END)
    return g.compile(checkpointer=checkpointer)
