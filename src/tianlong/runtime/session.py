"""
[INPUT]: 依赖 runtime/authority 的 WorldAuthority / Settlement，runtime/versions 的 current_versions / check_save，runtime/talk 的谈资账本与台词账本，
         runtime/gm 的主持层纯函数（gm_command / companions / salient / build_brief），runtime/aside 的 AsideMixin / ENDED（不推进的回合），
         runtime/endings 的 EndingMixin（落幕与终章），runtime/cast 的 policy_for / wakes / advance_marks（驱力），
         runtime/staging 的 recognize / stops_wait / lore_at / staging / dress（看点、写法卡、景观、细节、天色：只读玩家自己的感知），runtime/suggest 的 suggestions，
         runtime/names 的相识账本（Acquaintance / masked / voiced / gate_aliases / may_name / learn_heard / learn_delivered / early_line），
         agents 的 Orchestrator / NpcContext / AgentPort / Scheduler / Policy / OutcomePredictor，
         memory 的 QdrantMemoryIndex / Recall / MemoryIndexer / MemoryScope，language 的 IntentParser / MoveKind / Parsed / Narrator /
         TemplateSpeaker / LLMClient，language/scene 的 SceneBrief / TextSink，
         language/render 的 Rendered，persistence 的 WorldStore / InMemoryWorldStore / WorldRef / TurnEnvelope /
         RequestConflict / VersionConflict，scenarios 的 Scenario / Ending，cognition 的 Candidate / believed_place，
         language/templates 的 render_fact，memory/view 的 MemoryView（NPC 的长期记忆摘要，增量汇总——水位含边界、按记录 ID 去重，与读档后重建逐项相同）
[OUTPUT]: 对外提供 GameSession（可玩会话：turn() 一回合、intro()/epilogue() 开场与终章、belief_lines() 玩家自己的认知；
          读档接续并恢复调度标记、已描写实体、最近几段正文、提示进度、谈资账本、台词账本、驱力标记、看点账本与细节账本 facets；请求幂等、存档版本闸门）、
          TurnReport（一回合的全部产物：世界侧与文字侧分开记录，含这句话的类别、结局、首字耗时、分阶段耗时、叙述上下文与玩家感知到的看点 beats）、
          ENDED（再导出自 runtime/aside）
[POS]: runtime 的装配中心（主持层的回合循环）：一回合 = 解释玩家输入（后台同时算好本 tick 的 NPC 决策；元指令与“GM：”一眼认得，不算）→ 按类别推进：
       ACT 走 1~3 步计划（失败即止；多 tick 叩首的每一步都带着原话）、SAY/GESTURE 与冲着在场之人的行动再加一个反应 tick、
       普通等待按时长（等到入夜/月出/天亮按场景的时刻换算，上限 MAX_WAIT）且只被要紧的事打断
       （当面动手、对同伴——自己人与 DEFEND 目标——动手都算）或在玩家感知到看点的那个 tick 停下（一次性的看点只停一回），
       时钟到了结局的时刻（天亮）也停、
       ASK_GM/META/追问不推进时间也不落库（AsideMixin：场外回答边生成边逐句过名字闸门；模型写的追问同样过闸门，玩家自己说出的名字不算）→
       权威结算（同一事务附上请求进度与会话运行态）→ 同步记忆索引 →
       主持人之声据玩家感知与 SceneBrief 流式叙述（过语义闸门；迟到先声的时限从回车算起：deadline = 回车时刻 + lead_after；SceneBrief 带前后照应——拿本回合开始前玩家的认知比对，
       此前已知下落的东西再翻出来不算发现，抵达结局的那一回合收幕）→ 幂等记下叙述（先写者为准，返回与记住的都是落库的那一段）→
       抵达结局地点即落幕（EndingMixin）。
       后台预算的决策只依赖同一版本；首 tick 时版本未变才用，否则或本回合不推进就丢弃——结果与顺序执行逐项相同。
       带 request_id 的请求：同 ID 同内容返回既有结果、不再结算；同 ID 异内容抛 RequestConflict；提交后崩溃的重试只重写文字，
       多 tick 请求（等待、多步计划、反应 tick）中途崩溃的重试由已提交的 tick 数推出剩下的步骤，只走剩下的 tick。
       不推进的回合不落库，带 request_id 的只记在本进程里（最近 ASIDE_KEEP 个）：重试原样返回，提示不多翻、模型不再问。
       请求绑定由存储在提交内检查：并发的重复投递只有一次能提交某个 tick，被越过的一方即停、以落库的那一份为准返回，
       对方尚未走完时只给出目前为止的文字、不落库。建档后尚无提交就读档，开场已描写的实体按开场规则补回。
       NPC 决策图里从不调模型（台词由主持人之声一并写出，驱力的原话除外——它随意图落库），向量回忆只为声明 reads_memories 的策略而跑。
       场景给了驱力的 NPC 经 cast.policy_for 套上 Driven，时间窗打开即唤醒（cast.wakes，不改调度器）；
       驱力标记与看点账本（staged：已让等待停过的一次性看点）在 annotate 的副本上推进（驱力只记兑现成功的），与调度标记同一事务落库；
       相识账本同样：听见的原话在 annotate 里推进（names.learn_heard），交付的正文在叙述之后推进、随下一次提交落库。
       给玩家看的一切（解释、叙述、行动建议、场外问答、终章）都用 _view()——还叫不出名字的人换成外貌称呼的展示用副本，
       叙述与台词读的感知同样经 names.voiced（旁人原话里他叫不出名字的人换成外貌称呼）；
       叙述闸门的别称按账本给（没引介的人的名字只用于拒绝），NPC 台词可点名的经 names.may_name；NPC 的决策从不经过它。
       SceneBrief 另经 staging.dress 带上本回合的看点、写法卡、景观、细节与天色（旧版全空、逐字不变）；给过的细节在叙述之后记账。
       CLI、测试、未来的 Web 前端都只和它打交道
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace

from tianlong.agents.npc_graph import NpcContext
from tianlong.agents.orchestrator import Deliberation, Orchestrator
from tianlong.agents.policies import Policy
from tianlong.agents.port import AgentPort
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor
from tianlong.agents.scheduler import Scheduler
from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import believed_place
from tianlong.core import (
    Event,
    Fact,
    Intent,
    Kind,
    Modality,
    Op,
    Outcome,
    Percept,
    Rel,
    WorldState,
    clock_label,
    digest,
    make_id,
    minutes_until_night,
)
from tianlong.language.interpret import Interpreter
from tianlong.language.llm import CachedLLM, LLMClient
from tianlong.language.narrator import LEAD_AFTER, Narrator, lore_keys
from tianlong.language.parser import IntentParser, MoveKind, Parsed
from tianlong.language.render import Rendered
from tianlong.language.scene import SceneBrief, TextSink
from tianlong.language.speaker import Speaker, TemplateSpeaker
from tianlong.language.templates import render_fact
from tianlong.memory.index import MemoryIndex, MemoryScope, QdrantMemoryIndex
from tianlong.memory.indexer import MemoryIndexer
from tianlong.memory.recall import Recall
from tianlong.memory.view import MemoryView
from tianlong.persistence import (
    InMemoryWorldStore,
    RequestConflict,
    TurnEnvelope,
    VersionConflict,
    WorldRef,
    WorldStore,
)
from tianlong.runtime import cast, gm, names, staging, talk
from tianlong.runtime.aside import ENDED, AsideMixin
from tianlong.runtime.authority import Settlement, WorldAuthority
from tianlong.runtime.endings import EndingMixin
from tianlong.runtime.suggest import suggestions
from tianlong.runtime.versions import check_save, current_versions
from tianlong.scenarios import Ending, Scenario

__all__ = ["ENDED", "GameSession", "TurnReport"]     # ENDED 再导出自 runtime/aside


@dataclass(frozen=True)
class TurnReport:
    clock: str
    parsed: Parsed
    narration: str
    advanced: bool                                    # False：时间未推进（没解析成行动、场外问答、元指令、已落幕）
    events: tuple[Event, ...] = ()                    # 真相（调试用，不给玩家看）
    deliberations: tuple[Deliberation, ...] = ()      # NPC 的决策理由（调试用）
    settlement: Settlement | None = field(default=None, repr=False)
    timings: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）：interpret / npc_decide / settle / index / narrate
    render: Rendered | None = None                    # 文字侧结果（来源 + 闸门结论）；None = 叙述取自已落库的结果
    request_id: str | None = None
    replayed: bool = False                            # True：本次调用没有推进世界，只返回（或补写文字）既有请求的结果
    kind: MoveKind = MoveKind.ACT                     # 这句话的类别（行动/说话/姿态/问主持人/元指令/听不懂）
    ending: Ending | None = None                      # 本幕已落幕（玩家据世界真相身处结局地点）
    first_text_ms: float | None = None                # 回车到第一段文字交付的毫秒数；None = 没有交付任何文字
    brief: SceneBrief | None = field(default=None, repr=False)   # 交给叙述者的上下文（要说出口的话、最近正文、玩家原话）
    beats: tuple[str, ...] = ()                       # 本回合玩家感知到的看点编号（B1；只读玩家自己的感知）


MAX_WAIT = 240       # 一次最多等四个时辰（240 分钟）
RECENT_KEEP = 3      # 最近几段正文：随会话运行态落库，交给叙述者与解释器接续上下文
_ENGAGE = frozenset({Op.ATTACK, Op.GIVE, Op.USE, Op.TELL, Op.ASK})


def _last(env: TurnEnvelope) -> int:
    """请求最后一次提交的世界版本（尚未提交任何 tick 时是开始时的版本）。"""
    return env.versions[-1] if env.versions else env.start_version


def _bound(env: TurnEnvelope | None, payload: str) -> TurnEnvelope:
    """已落库的请求必须绑定同一份内容：同 ID 异内容抛 RequestConflict。"""
    if env is None:
        raise RuntimeError("请求进度不在存储里：会话与存储不一致")
    if env.payload_hash != payload:
        raise RequestConflict(f"请求 {env.request_id} 已绑定另一份内容：拒绝执行")
    return env


def _compact(percepts: tuple[Percept, ...]) -> tuple[Percept, ...]:
    """请求进度里的感知只留事件感知与最后一次环顾：多 tick 等待从不讲中途的所见（只有移动/查看才附所见，而它们只占一个 tick），
    等上四个时辰也不让每次提交整份重写的进度随 tick 平方增长。正常叙述与崩溃后的重写用的是同一份。"""
    last = max((i for i, p in enumerate(percepts) if p.modality == Modality.SCENE), default=-1)
    return tuple(p for i, p in enumerate(percepts) if p.modality != Modality.SCENE or i == last)


def _next_dawn(now: int) -> int:
    """场景没给天亮的时刻：下一个卯时（05:00）。"""
    dawn = now - now % 1440 + 5 * 60
    return dawn if dawn > now else dawn + 1440


def _dawned(scenario: Scenario, st: WorldState) -> bool:
    """时钟到了某个结局的时刻（天亮）：再长的等待也在这一 tick 停下。只看时钟结局，地点结局由走到那里的那一步定。"""
    return any(e.at_clock is not None and st.clock >= e.at_clock for e in scenario.endings)


def _interruptible(env: TurnEnvelope) -> bool:
    """只有普通等待（不带计划、不带反应 tick）会被身边的事打断；计划的步骤与反应 tick 从不被截短。"""
    return env.intent.op == Op.WAIT and not env.followups and not env.reaction


class _Stopwatch:
    def __init__(self) -> None:
        self.start = self._t = time.perf_counter()
        self.entered = time.monotonic()       # 回车的时刻（先声时限的起点，与 narrate_scene 的 deadline 同一口径）
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(self.laps.get(name, 0.0) + (now - self._t) * 1000, 1)   # 多 tick 时累加
        self._t = now


class _Sink:
    """流式交付：记下第一段文字到达的时刻（回车到首字），再转交前端。"""

    def __init__(self, on_text: TextSink | None, start: float) -> None:
        self.on_text = on_text
        self.start = start
        self.first_ms: float | None = None

    def __call__(self, text: str) -> None:
        if not text:
            return
        if self.first_ms is None:
            self.first_ms = round((time.perf_counter() - self.start) * 1000, 1)
        if self.on_text is not None:
            self.on_text(text)


@dataclass
class _Ahead:
    """解释玩家输入期间在后台算好的首 tick NPC 决策：(例行等待者, 决策)，只依赖 version 这一个版本。"""

    version: int
    future: Future

    def take(self, version: int) -> tuple[list[str], list[Deliberation]] | None:
        """等后台算完（本回合的任何写入之前后台必已收手）；版本变了就丢弃，由调用方按新版本重算。"""
        try:
            got = self.future.result()
        except Exception:
            if version == self.version:
                raise                              # 与顺序执行一样抛出
            return None
        return got if version == self.version else None


class GameSession(AsideMixin, EndingMixin):
    def __init__(
        self,
        scenario: Scenario,
        store: WorldStore | None = None,
        index: MemoryIndex | None = None,
        llm: LLMClient | None = None,
        branch_id: str = "main",
        policies: Mapping[str, Policy] | None = None,
        predictor: OutcomePredictor | None = None,
        max_candidates: int = 64,
        allow_migration: bool = False,
        interpreter: Interpreter | None = None,
        pipeline: bool = True,
        fast_llm: LLMClient | None = None,
    ) -> None:
        """llm 是主持人之声（叙述、场外问答、终章）；fast_llm 是解释玩家输入的快模型（缺省用 llm）；
        interpreter 缺省按快模型装配（没有模型时它退回规则解析）；
        pipeline=False 关掉“解释时后台预算 NPC 决策”（与之逐项相同，只是慢一些）。"""
        if scenario.player is None:
            raise ValueError("场景没有玩家角色")
        self.scenario = scenario
        self.player: str = scenario.player
        store = store or InMemoryWorldStore()
        self.store: WorldStore = store
        self.index = index or QdrantMemoryIndex()
        self.recall = Recall(store, self.index)
        self.indexer = MemoryIndexer(store, self.index)
        ref = WorldRef(scenario.world_id, branch_id)
        self.versions = current_versions()
        self.migrated_from: dict[str, str] | None = None   # 显式迁移时记下旧档的版本表（不改写它，也不补写旧档未记录的信息）
        self.resumed = store.exists(ref)
        session_state: Mapping | None = None
        if self.resumed:
            # 读档：先过版本闸门；世界与认知来自权威存储；向量索引是派生数据，从经历记录重建
            stored = store.save_versions(ref)
            if check_save(stored, self.versions, allow_migration):
                self.migrated_from = dict(stored)
            self.authority = WorldAuthority(store, ref)
            for agent in scenario.profiles:
                self.index.upsert(store.recent_memories(ref, agent, 0))
            session_state = store.session_state(ref)
            if session_state is None:
                # 建档之后还没有任何提交：开场已描写过的实体只记在上一个会话的内存里，按开场同样的规则补回，免得首回合重讲一遍
                session_state = {"described": self._opening_keys()}
        else:
            self.authority = WorldAuthority.found(store, scenario, branch_id, versions=self.versions)
        self.orchestrator = Orchestrator()
        # 会话运行态随每次提交落库，读档原样恢复：否则读档那一刻所有 NPC 都“该决策了”，初见描写也会重来一遍
        self.scheduler = Scheduler()
        self._described: set[str] = set()   # 已向玩家描写过外观的实体：只在初见时描写
        self._recent: list[str] = []        # 最近几段正文（派生数据：叙述之后更新，随下一次提交落库）
        self._hint = 0                      # 已给出的逐级提示条数
        self._told: dict[str, set[int]] = {}   # 谈资账本：每个 NPC 已经出现在正文里的谈资条目（派生数据，随下一次提交落库）
        self._said: dict[str, list[str]] = {}  # 台词账本：每个 NPC 最近说过的原话（同上）
        self._marks: dict[str, dict[str, list[int]]] = {}   # 驱力标记：角色 → 驱力 → 兑现成功的时刻（随世界同一事务落库）
        self._staged: set[str] = set()      # 已让等待停过的一次性看点（随世界同一事务落库）
        self._acq = names.initial(scenario)   # 相识账本：谁叫得出谁的名字（听见的原话随世界落库，交付的正文随下一次提交）
        self._facets: set[str] = set()      # 细节卡组里一幕之内已经给过的细节（派生数据，随下一次提交落库）
        self._asides: dict[str, tuple[str, TurnReport]] = {}   # 带 request_id 的不推进回合：ID → (原文摘要, 报告)
        self._restore(session_state)
        self.llm = llm
        self.parser = IntentParser(fast_llm or llm, aliases=names.lookup(scenario))     # 外貌称呼与真名都认得
        self.interpreter = interpreter or Interpreter(fast_llm or llm, aliases=names.lookup(scenario), fallback=self.parser,
                                                      universe=(e.name for e in scenario.state.entities.values()),
                                                      kowtow_ticks=scenario.kowtow_ticks)
        self.pipeline = pipeline
        # 开了磁盘缓存（评测、演示录像）就不用迟到的先声：它由网速决定出不出场，同一局重跑的文字就对不上了
        self.narrator = Narrator(llm, scenario.setting, scenario.lore, scenario.style, scenario.gate_aliases, scenario.secrets,
                                 lead_after=None if isinstance(llm, CachedLLM) else LEAD_AFTER,
                                 cards={k: c.text for k, c in scenario.cards.items()})
        self._universe = frozenset(e.name for e in scenario.state.entities.values())  # 闸门拒绝用的名字全集
        self._friends = gm.companions(scenario.profiles[self.player])   # 有人对他们动手即打断等待
        self.speaker: Speaker = TemplateSpeaker()     # 决策图里从不调模型：NPC 的台词由主持人之声一并写出
        # 长期记忆摘要的增量缓存（派生数据）：(水位 tick, 水位 tick 上已并入的记录 ID, 摘要)
        self._memory_views: dict[str, tuple[int, frozenset[str], MemoryView]] = {}
        self.policies = dict(policies or {})
        self.predictor = predictor or HeuristicPredictor()
        self.max_candidates = max_candidates   # 学得的策略按训练时的候选上限看世界
        self._pool: ThreadPoolExecutor | None = None
        self._ahead: _Ahead | None = None
        self.ending: Ending | None = self._ended()    # 读档时落幕与否同样由世界真相推出

    # ------------------------------------------------------------
    #  读
    # ------------------------------------------------------------

    @property
    def ref(self):
        return self.authority.ref

    def beliefs(self, agent: str) -> BeliefStore:
        return self.authority.store.beliefs(self.ref, agent)

    def clock(self) -> str:
        return clock_label(self.authority.head().clock)

    def session_state(self) -> dict:
        """会话运行态：随每次世界提交一起落库的那一份（调度标记 + 已描写实体 + 最近几段正文 + 提示进度 + 驱力标记）。"""
        return self._state(self.scheduler, self._described)

    def _state(self, sched: Scheduler, described: set[str], marks: cast.Cast | None = None,
               staged: set[str] | None = None, acq: names.Acquaintance | None = None) -> dict:
        marks = self._marks if marks is None else marks
        staged = self._staged if staged is None else staged
        acq = self._acq if acq is None else acq
        return {"scheduler": sched.to_state(), "described": sorted(described), "recent": list(self._recent),
                "hint": self._hint, "told": {k: sorted(v) for k, v in sorted(self._told.items()) if v},
                "drives": {a: {k: list(v) for k, v in sorted(m.items())} for a, m in sorted(marks.items()) if m},
                **({"said": dict(sorted(self._said.items()))} if self._said else {}),
                **({"staged": sorted(staged)} if staged else {}), **({"names": acq.to_state()} if acq.known else {}),
                **({"facets": sorted(self._facets)} if self._facets else {})}

    def _restore(self, state: Mapping | None) -> None:
        """会话运行态以落库的那一份为准：读档时，以及一次请求被同一请求的另一次投递越过之后。"""
        state = state or {}
        self.scheduler = Scheduler.from_state(state.get("scheduler", {}), self.scheduler.idle_interval)
        self._described = set(state.get("described", ()))
        self._recent = [str(x) for x in state.get("recent", ())][-RECENT_KEEP:]
        self._hint = int(state.get("hint", 0))
        self._told = {str(k): {int(i) for i in v} for k, v in (state.get("told") or {}).items()}
        self._said = talk.said(state.get("said") or {}, "", None)
        self._marks = {str(a): {str(k): [int(t) for t in v] for k, v in m.items()}
                       for a, m in (state.get("drives") or {}).items()}
        self._staged = {str(k) for k in state.get("staged", ())}
        self._acq = names.Acquaintance.from_state(state["names"]) if "names" in state else names.initial(self.scenario)
        self._facets = {str(x) for x in state.get("facets", ())}

    def _opening_keys(self) -> list[str]:
        """开场讲的初始认知里应当描写外观的实体：新游戏的 intro() 描写它们，建档后尚无提交就读档时据此补回“已描写”。"""
        sc = self.scenario
        return staging.lore_at(lore_keys(self.player, sc.priors.get(self.player, ()), sc.lore), sc.state.clock, sc.lore,
                               sc.moments)

    def _known(self, me: BeliefStore) -> frozenset[str]:
        return self._universe | {sk.name for sk in me.entities.values()}

    def _view(self) -> BeliefStore:
        """玩家的认知给他看的样子：还叫不出名字的人换成外貌称呼（names.masked；展示用副本，永不落库）。"""
        return names.masked(self.beliefs(self.player), self._acq, self.scenario)

    def _veiled(self) -> dict[str, tuple[str, ...]]:
        return names.veiled(self._acq, self.player, self.scenario)

    def intro(self) -> str:
        """开场：新游戏讲初始认知；读档讲玩家此刻以为的周遭（不是世界真相）。"""
        me = self._view()
        if not self.resumed:
            prior = self.scenario.priors.get(self.player, ())
            fresh = self._opening_keys()
            self._described.update(fresh)
            return self.narrator.narrate(self.player, prior, me.entities, show_scene=True, fresh=fresh,
                                         known=self._known(me), aliases=names.gate_aliases(self._acq, self.player,
                                                                                           self.scenario))
        here = believed_place(me, self.player)
        around = [
            render_fact(Fact(b.prop, True), me.entities, self.player, me="你")
            for b in me.sorted_beliefs()
            if b.holds and b.prop.predicate == Rel.AT.value and b.prop.subject != self.player
            and believed_place(me, b.prop.subject) == here
        ]
        where = me.sketch(here).name if here and me.sketch(here) else "某处"
        return f"（读档）你在{where}。" + ("；".join(around) if around else "")

    def belief_lines(self) -> list[str]:
        """玩家自己的认知（/beliefs）：亲见与传闻分开标注——是他以为的，不是世界真相。"""
        me = self._view()
        return [f"  [{'传闻' if b.hearsay else '亲见'} {b.confidence:.1f}] "
                f"{render_fact(Fact(b.prop, b.holds), me.entities, self.player)}"
                for b in me.sorted_beliefs() if not b.prop.is_attr or b.holds]

    # ------------------------------------------------------------
    #  一回合
    # ------------------------------------------------------------

    def turn(self, text: str, request_id: str | None = None, on_text: TextSink | None = None) -> TurnReport:
        """request_id 为 None 时每次都推进；给出时，请求以 request_id + 原文摘要绑定，重试不会二次结算。
        on_text 收到每一段交付给玩家的文字（流式）；TurnReport.first_text_ms 记下回车到首字的时间。"""
        clock = _Stopwatch()
        sink = _Sink(on_text, clock.start)
        self._settle_background()
        payload = digest("turn", text)
        if request_id is not None:
            prior = self.store.request(self.ref, request_id)
            if prior is not None:
                return self._resume_request(_bound(prior, payload), text, clock, sink)
            if request_id in self._asides:
                return self._replay_aside(request_id, payload, clock, sink)
        head = self.authority.head()
        me = self._view()
        quick = gm.gm_command(text)          # 元指令与“GM：”前缀一眼认得、从不推进：不必预算 NPC 决策
        ahead = self._look_ahead(head) if self.ending is None and quick is None else None
        parsed = quick or self._parse(text, me)
        clock.lap("interpret")
        if (parsed.kind in (MoveKind.ASK_GM, MoveKind.META) or parsed.candidate is None
                or self.ending is not None):
            return self._aside(parsed, text, head, me, clock, sink, request_id)   # 后台的决策随之作废：它不写任何东西
        steps, planned, reaction = self._plan(parsed, me, head.clock)
        slot = next((i for i, c in enumerate(steps) if c.op in (Op.TELL, Op.ASK, Op.WAIT)), 0)   # 原话归属的那一步
        # 重复的姿态步（一连几个叩首）每一步都带着原话：不带字的等待谁也看不见
        plan = [c.to_intent(self._intent_id(self.player, head.version), self.player, head.version,
                            parsed.utterance if i == slot or (c.op == Op.WAIT and c == steps[slot]) else None)
                for i, c in enumerate(steps)]
        env = TurnEnvelope(request_id or "", payload, plan[0], planned, head.version, head.clock,
                           source=parsed.source, followups=tuple(plan[1:]), reaction=reaction)
        env, events, deliberations, settlement = self._advance(env, clock, request_id is not None, ahead)
        if not env.done:
            # 被越过：同一请求的另一次投递抢先推进了世界——以落库的那一份为准，本次不再多走
            stored = self.store.request(self.ref, env.request_id)
            return self._resume_request(_bound(stored, payload), text, clock, sink, execute=False,
                                        committed=settlement is not None)
        render, brief = self._render(env, text, clock, sink, before=me,
                                     closing=self.ending is None and self._ended() is not None)
        narration = render.text if request_id is None else self._record(request_id, render.text)
        self._remember(narration, brief, text)
        return TurnReport(clock_label(head.clock), parsed, narration, True, tuple(events), tuple(deliberations),
                          settlement, clock.laps, render, request_id, kind=parsed.kind, ending=self._reach_ending(),
                          first_text_ms=sink.first_ms, brief=brief, beats=self._beats(env))

    def _parse(self, text: str, me: BeliefStore) -> Parsed:
        """解释玩家输入：元指令先认（会话自己的命令表），其余交主持层解释器（带上最近几段正文消解“她/那人”）。"""
        return gm.gm_command(text) or self.interpreter.interpret(text, me, tuple(self._recent))

    def _plan(self, parsed: Parsed, me: BeliefStore, now: int) -> tuple[tuple[Candidate, ...], int, bool]:
        """(计划步骤, 计划 tick 数, 是否追加反应 tick)。普通等待照旧按时长；其余按计划的步数，
        对人说了话、做了姿态、或冲着（玩家以为）在场的人动手/递物/施用/说话，再加一个反应 tick 让在场的人当场回应；
        同伴在身边时走开也加一个，让跟着你的人当场跟上。"""
        assert parsed.candidate is not None
        steps = (parsed.candidate, *parsed.followups)
        if parsed.kind != MoveKind.GESTURE and len(steps) == 1 and steps[0].op == Op.WAIT:
            return steps, self._ticks_for(parsed, now), False
        here = believed_place(me, self.player)

        def engages(c: Candidate) -> bool:
            sk = me.sketch(c.target) if c.target and c.target != self.player else None
            return (c.op in _ENGAGE and sk is not None and sk.kind == Kind.PERSON and here is not None
                    and believed_place(me, sk.id) == here)

        # 身边有同伴（自己人、要护着的人）时走开：多给一个 tick，让跟着你的人当场跟上，而不是晚一回合才冒出来
        friends = gm.companions(self.scenario.profiles[self.player]) if self.player in self.scenario.profiles else ()
        follow = any(c.op == Op.MOVE for c in steps) and here is not None and any(
            believed_place(me, f) == here for f in friends)
        reaction = parsed.kind in (MoveKind.SAY, MoveKind.GESTURE) or any(engages(c) for c in steps) or follow
        return steps, len(steps) + reaction, reaction

    def _ticks_for(self, parsed: Parsed, now: int) -> int:
        """等待的 tick 数：等到某个时刻（场景的 moments；没有月出按入夜，没有天亮按卯时）之前，上限 MAX_WAIT。"""
        until, moments = parsed.until, self.scenario.moments
        if until == "moon" and "moon" not in moments:
            until = "night"
        if until == "night":
            wanted = minutes_until_night(now)
        elif until in ("moon", "dawn"):
            wanted = moments.get(until, _next_dawn(now)) - now
        else:
            wanted = parsed.repeat
        return max(1, min(wanted, MAX_WAIT))

    def _beats(self, env: TurnEnvelope) -> tuple[str, ...]:
        """本回合玩家感知到的看点编号（只读落库的请求进度里玩家自己的感知）。"""
        found = staging.recognize(self.scenario.beats, env.percepts, self.player, moments=self.scenario.moments)
        return tuple(b.key for b in found)

    def _resume_request(self, env: TurnEnvelope, text: str, clock: _Stopwatch, sink: _Sink, execute: bool = True,
                        committed: bool = False) -> TurnReport:
        """既有请求：世界侧没走完且没人越过它就只走剩下的 tick；文字没落库就按已持久化的感知重写；否则原样返回。
        execute=False：本次执行已被同一请求的另一次投递越过，只以落库的那一份为准、不再推进世界；
        那一份若还没走完（对方仍在进行），先给出目前为止的文字但不落库——终稿由走完它的那一方写。
        committed：本次调用在被越过之前是否已提交过 tick（决定 replayed）。"""
        parsed = self._parsed_of(env)
        deliberations: list[Deliberation] = []
        settlement = None
        stuck = False
        if execute and not env.done:
            env, version = self._progress(env.request_id)
            if not env.done and version == _last(env):
                env, _, deliberations, settlement = self._advance(env, clock, True)
                if not env.done:                    # 续跑途中又被另一次投递越过
                    env = _bound(self.store.request(self.ref, env.request_id), env.payload_hash)
            else:
                stuck = not env.done                # 别的写入者越过了它：这个请求再也走不完，按已走的 tick 定稿
        render = brief = None
        narration = env.narration
        if narration is None or settlement is not None:
            render, brief = self._render(env, text, clock, sink)
            narration = render.text
            if env.done or stuck:
                narration = self._record(env.request_id, render.text)
                self._remember(narration, brief, text)
        else:
            sink(narration)
        wanted = {v - 1 for v in env.versions}       # 本请求各 tick 的意图都基于提交前的那个版本
        events = tuple(e for e in self.store.events(self.ref) if e.intent.based_on in wanted)
        return TurnReport(clock_label(env.start_clock), parsed, narration, True, events, tuple(deliberations),
                          settlement, clock.laps, render, env.request_id,
                          replayed=not (committed or settlement is not None), kind=parsed.kind,
                          ending=self._reach_ending(), first_text_ms=sink.first_ms, brief=brief, beats=self._beats(env))

    @staticmethod
    def _parsed_of(env: TurnEnvelope) -> Parsed:
        """已落库的请求还原成解析结果（续跑与重放的报告用）：类别由计划推出。"""
        it = env.intent
        kind = (MoveKind.GESTURE if it.op == Op.WAIT and it.utterance and env.reaction
                else MoveKind.SAY if it.op in (Op.TELL, Op.ASK) and not env.followups else MoveKind.ACT)
        return Parsed(Candidate.of(it), it.utterance, source=env.source, repeat=env.planned_ticks, kind=kind,
                      followups=tuple(Candidate.of(f) for f in env.followups))

    def _progress(self, request_id: str) -> tuple[TurnEnvelope, int]:
        """请求进度与世界版本的一致快照：前后两次读到同一份进度，夹在中间读到的版本才与它相符
        （每个 tick 的进度与世界同一事务落库、只增不减）。免得把另一次投递刚提交的 tick 误当成“别人越过了它”。"""
        env = self.store.request(self.ref, request_id)
        while True:
            version = self.authority.head().version
            again = self.store.request(self.ref, request_id)
            if again.versions == env.versions:
                return again, version
            env = again

    def _advance(self, env: TurnEnvelope, clock: _Stopwatch, persist: bool, ahead: _Ahead | None = None,
                 ) -> tuple[TurnEnvelope, list[Event], list[Deliberation], Settlement | None]:
        """一次输入可能跨越多个 tick（多步计划、反应 tick、“等到天黑”）；每个 tick 的进度与世界同一事务落库。
        第 i 个 tick 走计划的第 i 步——由已提交的 tick 数推出，崩溃后的续跑因此接在正确的一步上。
        带请求时被越过即停、不再多走：世界已不在本请求最后提交的版本上，或这一 tick 的提交被拒（同一请求的另一次投递抢先）。
        此时返回的进度未完结，内存里的会话运行态改回落库的那一份，调用方以落库的请求进度为准。"""
        events: list[Event] = []
        deliberations: list[Deliberation] = []
        settlement = None
        while not env.done:
            now = self.authority.head()
            if persist and env.versions and now.version != env.versions[-1]:
                break
            try:
                settlement, delibs, env = self._tick(self._step(env, now.version), env, clock, persist, ahead)
            except (VersionConflict, RequestConflict):
                if not persist or self.store.request(self.ref, env.request_id) is None:
                    raise                            # 与本请求无关的写入者抢先：原样抛出
                break
            ahead = None                             # 后台只预算首 tick
            events += settlement.events
            deliberations += delibs
        if not env.done:
            self._restore(self.store.session_state(self.ref))
        return env, events, deliberations, settlement

    def _step(self, env: TurnEnvelope, version: int) -> Intent:
        """第 len(versions) 个 tick 的玩家意图：计划之内取对应的一步，计划走完之后原地等待（反应 tick、多 tick 等待）。"""
        plan = (env.intent, *env.followups)
        i = len(env.versions)
        base = plan[i] if i < len(plan) else (replace(env.intent, utterance=None, social=None)
                                              if env.intent.op == Op.WAIT else Intent("", self.player, Op.WAIT))
        return replace(base, id=self._intent_id(self.player, version), based_on=version)

    def _render(self, env: TurnEnvelope, text: str, clock: _Stopwatch, sink: _Sink, before: BeliefStore | None = None,
                closing: bool = False) -> tuple[Rendered, SceneBrief]:
        """只依据已持久化的请求进度（加此刻各人的认知）渲染：提交后崩溃的重试写出的是同一回合的文字。
        叙述者有 narrate_scene（主持人之声）就交给它流式写，否则一次写完再交付。
        before 是本回合之前玩家的认知（前后照应的比对基准；重试补写时没有）；closing：这一回合抵达了结局，叙述收在余韵上。"""
        me, acq, sc = self._view(), self._acq, self.scenario      # 名字经相识账本：玩家叫不出的人用外貌称呼
        env = replace(env, percepts=names.voiced(env.percepts, acq, self.player, sc))   # 旁人原话里同样（展示用）
        lapse = clock_label(env.ticks[-1]) if _interruptible(env) and env.planned_ticks > 1 and env.ticks else ""
        brief = gm.build_brief(env, me, sc, self.beliefs, self._recent, before, closing, self._told,
                               lambda a: self.store.recent_memories(self.ref, a, 0), suggestions(me, friends=self._friends),
                               self._said, lambda a, mind: names.may_name(a, mind, acq, sc), self._veiled())
        brief = staging.dress(brief, staging.staging(sc, env, self._facets), sc.lore)   # 看点、写法、景观、细节与天色
        aliases = names.gate_aliases(acq, self.player, sc, text)
        # 此前已知下落的东西：再翻出来不算“发现”（重试补写时没有 before，照旧）
        familiar = frozenset(e for e in before.entities if before.location_of(e) is not None) if before else frozenset()
        scene = getattr(self.narrator, "narrate_scene", None)
        after = getattr(self.narrator, "lead_after", None)      # 迟到先声的时限从回车算起：解释与结算花掉的时间不再另等
        if scene is not None:
            render = scene(self.player, env.percepts, me.entities, brief=brief, fresh=env.fresh, command=text,
                           familiar=familiar, since=clock_label(env.start_clock) if lapse else "",
                           lapse=lapse, known=self._known(me), on_text=sink,
                           deadline=clock.entered + after if after else None, aliases=aliases)
        else:
            render = self.narrator.narrate_rendered(self.player, env.percepts, me.entities, fresh=env.fresh,
                                                    command=text, lapse=lapse, known=self._known(me), aliases=aliases)
            sink(render.text)
        early = names.early_line(names.named_early(acq, self.player, text, brief, sc), text, sc)   # 抢先叫出真名：只一次
        if early:
            sink(early)
            render = replace(render, text=render.text + early)
        clock.lap("narrate")
        return render, brief

    def _remember(self, narration: str, brief: SceneBrief | None = None, command: str = "") -> None:
        """最近几段正文、谈资账本、台词账本与玩家的相识账本：叙述写成之后更新（派生数据），随下一次提交落库。
        交给叙述者的谈资（brief 里的台词）换了说法也认得出来：说话者开了口、这条谈资的字眼大半出现了，就算说过。"""
        self._recent = [*self._recent, narration][-RECENT_KEEP:]
        self._acq, _ = names.learn_delivered(self._acq, self.player, narration, brief, command, self.scenario)
        self._said = talk.said(self._said, narration, brief)
        self._facets |= set(brief.details if brief else ())
        offered = {vl.speaker: vl.knows for vl in (brief.lines if brief else ()) if vl.speaker_name in narration}
        for npc in self.scenario.npcs:
            said = talk.told(self.scenario.profiles[npc].knows, narration, offered.get(npc, ""))
            if said:
                self._told[npc] = self._told.get(npc, set()) | said

    def _record(self, request_id: str, text: str) -> str:
        """记下请求的文字，返回已落库的那一份：先写者为准（并发的重复投递可能抢先写下自己的），同一请求永远只有一段正文。"""
        self.store.record_render(self.ref, request_id, text)
        stored = self.store.request(self.ref, request_id)
        return stored.narration if stored is not None and stored.narration else text

    def _intent_id(self, agent: str, version: int) -> str:
        return make_id("int", self.ref.world_id, self.ref.branch_id, agent, version)

    def _tick(self, player_intent: Intent, env: TurnEnvelope, clock: _Stopwatch, persist: bool,
              ahead: _Ahead | None = None) -> tuple[Settlement, list[Deliberation], TurnEnvelope]:
        head = self.authority.head()
        got = ahead.take(head.version) if ahead is not None else None
        routine, deliberations = got if got is not None else self._decide(head)
        clock.lap("npc_decide")
        intents = [player_intent, *(d.intent for d in deliberations)]
        intents += [Intent(self._intent_id(a, head.version), a, Op.WAIT, based_on=head.version) for a in routine]
        # 调度标记与已描写实体先在副本上推进，随世界同一事务落库；提交成功后才替换内存中的那份
        sched = Scheduler.from_state(self.scheduler.to_state(), self.scheduler.idle_interval)
        for d in deliberations:
            sched.record(d.agent, head.clock, d.intent)
        after: dict = {}

        def annotate(s: Settlement) -> tuple[TurnEnvelope | None, dict]:
            mine = tuple(o.percept for o in s.observations_of(self.player))
            keys = staging.lore_at(lore_keys(self.player, mine, self.narrator.lore), s.state.clock, self.narrator.lore,
                                   self.scenario.moments)
            fresh = tuple(k for k in keys if k not in self._described)
            n = len(env.versions) + 1
            plan = env
            failed = any(e.intent.id == player_intent.id and e.outcome != Outcome.SUCCESS for e in s.events)
            if failed and n <= len(env.followups):
                # 计划中的一步落空：后面的步骤不再走，反应 tick 照旧；截短的计划随这一 tick 落库，续跑据此只走剩下的
                plan = replace(env, followups=env.followups[:n - 1], planned_ticks=n + env.reaction)
            sc = self.scenario
            staged = self._staged | {b.key for b in staging.recognize(sc.beats, mine, self.player, moments=sc.moments)
                                     if b.once}
            # 普通等待被要紧的事、或玩家感知到的看点（一次性的只停一回）打断；时钟到了结局的时刻也停
            done = n >= plan.planned_ticks or (_interruptible(env) and (gm.salient(
                mine, self.player, self._friends, s.state.target(self.player, Rel.AT)) or staging.stops_wait(
                sc.beats, mine, self.player, sc.moments, self._staged))) or _dawned(sc, s.state)
            progressed = replace(plan, versions=(*env.versions, s.state.version), ticks=(*env.ticks, s.state.clock),
                                 percepts=_compact(env.percepts + mine), fresh=env.fresh + fresh, done=done)
            described = self._described | set(fresh)
            marks = cast.advance_marks(self._marks, deliberations, s)     # 驱力标记：只记兑现成功的
            acq = names.learn_heard(self._acq, s, sc)                     # 相识账本：听见（看见）的原话里点了谁的名
            after.update(env=progressed, described=described, marks=marks, staged=staged, acq=acq)
            return (progressed if persist else None), self._state(sched, described, marks, staged, acq)

        settlement = self.authority.settle(intents, annotate)
        if "env" not in after:
            if persist:     # 带请求：多半是同一请求的另一次投递抢先结算了这一 tick，交由 _advance 按“被越过”处理
                raise VersionConflict(f"版本 {head.version} 的意图早已由别的写入者结算")
            raise RuntimeError(f"版本 {head.version} 的意图早已结算过：会话与存储不一致")
        self.scheduler, self._described, self._marks = sched, after["described"], after["marks"]
        self._staged, self._acq = after["staged"], after["acq"]
        clock.lap("settle")
        self.indexer.drain()
        clock.lap("index")
        return settlement, deliberations, after["env"]

    # ------------------------------------------------------------
    #  后台预算：解释玩家输入的同时算好首 tick 的 NPC 决策（纯计算，不写存储）
    # ------------------------------------------------------------

    def _decide(self, head: WorldState) -> tuple[list[str], list[Deliberation]]:
        due, routine = self._npc_split(head.clock)
        return routine, self.orchestrator.decide(self._contexts(due, head.version, head.clock))

    def _look_ahead(self, head: WorldState) -> _Ahead | None:
        if not self.pipeline:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="npc-ahead")
        self._ahead = _Ahead(head.version, self._pool.submit(self._decide, head))
        return self._ahead

    def _settle_background(self) -> None:
        """上一回合作废的预算若还在跑，先等它收手：后台与本回合的写入从不重叠。"""
        if self._ahead is not None:
            wait((self._ahead.future,))              # 作废的预算：结果与异常一并丢弃
            self._ahead = None

    # ------------------------------------------------------------
    #  NPC 装配：每个角色只拿到自己的 port
    # ------------------------------------------------------------

    def _npc_split(self, now: int) -> tuple[list[str], list[str]]:
        due, routine = [], []
        last = self.scheduler.to_state()
        for a in self.scenario.npcs:
            woke = cast.wakes(self.scenario.drives.get(a, ()), last[a][0] if a in last else None, now)  # 驱力的时间窗到了
            (due if woke or self.scheduler.due(a, self.beliefs(a), now, self.scenario.profiles[a]) else routine).append(a)
        return due, routine

    def _memory_view(self, agent: str, now: int) -> MemoryView:
        """长期记忆摘要：从权威经历记录汇总（可重建的派生数据），按角色增量缓存。
        known_at 不是逐次提交唯一的（本 tick 末的环顾与下一次结算写下的记录同一个 tick）：水位含边界、按记录 ID 去重，
        与从全部记录重建（读档后）逐项相同。"""
        seen, ids, view = self._memory_views.get(agent, (0, frozenset(), MemoryView()))
        fresh = [m for m in self.authority.store.recent_memories(self.ref, agent, seen)
                 if m.known_at <= now and not (m.known_at == seen and m.id in ids)]
        if fresh:
            view = view.add(fresh, now)
            top = max(seen, max(m.known_at for m in fresh))
            ids = (ids if top == seen else frozenset()) | {m.id for m in fresh if m.known_at == top}
            self._memory_views[agent] = (top, ids, view)
        return view

    def _contexts(self, agents: list[str], version: int, now: int) -> dict[str, NpcContext]:
        out = {}
        for a in agents:
            scope = MemoryScope(self.ref.world_id, self.ref.branch_id, a, now)
            port = AgentPort(
                a, self.scenario.profiles[a], self.ref.world_id, self.ref.branch_id, version, now,
                beliefs=lambda a=a: self.beliefs(a),
                recall=lambda q, scope=scope: self.recall.recall(scope, q),
                memory=lambda a=a, now=now: self._memory_view(a, now),
            )
            policy = cast.policy_for(a, self.scenario, self.policies.get(a), self._marks.get(a, {}))   # 有驱力就套上
            # 向量回忆只为读 Situation.memories 的策略而跑（策略以 reads_memories = True 声明）；现有策略都不读
            out[a] = NpcContext(port, policy, self.predictor, self.speaker, self.max_candidates,
                                player=self.player, recall=bool(getattr(policy, "reads_memories", False)))
        return out
