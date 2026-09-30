"""
[INPUT]: 依赖 core 的 WorldState / Event / Observation / Intent / Percept / MemoryRecord，cognition 的 BeliefStore
[OUTPUT]: 对外提供 WorldRef / TurnEnvelope / ChoiceUse / CommitBatch / VersionConflict / RequestConflict / ChoiceConflict / UnknownWorld / WorldStore 协议、
          check_request_progress()（请求绑定检查：两个后端在提交的同一临界区 / 事务里调用）
[POS]: persistence 的契约；内存实现与 Neo4j 实现都遵守它。commit 是唯一写路径：版本检查 + 请求绑定检查 + 世界变化 + 事件 + 观察 + 认知
       + 待索引经历 + 请求进度（TurnEnvelope）+ 会话运行态，一次原子提交——“世界推进了”与“这个请求推进到哪了”不可能只落一半，
       同一请求的重复投递也不可能各结算一次（进度必须恰好接在已落库的那一份之后）。
       冻结菜单经 decision / publish_decision 旁路发布；consume_decision 与首 tick 同事务，菜单与世界不会只落一半。
       叙述文字另走幂等的 record_render()：文字失败不回滚世界，世界也不因文字重试而再结算一次
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Intent, Observation, Percept, WorldState
from tianlong.core.memories import MemoryRecord


@dataclass(frozen=True, slots=True)
class WorldRef:
    world_id: str
    branch_id: str = "main"

    def __str__(self) -> str:
        return f"{self.world_id}@{self.branch_id}"


@dataclass(frozen=True, slots=True)
class ChoiceUse:
    decision_id: str
    choice_id: str


# ============================================================
#  TurnEnvelope：一个玩家请求的持久化进度
#  - 以 request_id + payload_hash 绑定：同 ID 同内容 → 返回既有结果；同 ID 异内容 → RequestConflict
#  - 与每个 tick 的世界提交同事务写入：崩溃后要么看到“这一 tick 连同进度都落了”，要么都没落
#  - percepts / fresh 是重新渲染的全部依据：提交后、叙述前崩溃，重试只重写文字，不重走世界
# ============================================================


@dataclass(frozen=True, slots=True)
class TurnEnvelope:
    request_id: str
    payload_hash: str
    intent: Intent                        # 玩家行动模板（首 tick 的意图）；续跑的 tick 只换 id 与 based_on
    planned_ticks: int                    # 计划推进的 tick 数（“等到天黑”可跨越多个 tick）
    start_version: int                    # 请求开始时的世界版本
    start_clock: int
    versions: tuple[int, ...] = ()        # 已提交的世界版本，每 tick 一个
    ticks: tuple[int, ...] = ()           # 已提交各 tick 结束时的时钟
    percepts: tuple[Percept, ...] = ()    # 玩家在本请求中获得的感知（事件感知 + 最后一次环顾，足以重新渲染）
    fresh: tuple[str, ...] = ()           # 本请求里初见、应当描写外观的实体键
    done: bool = False                    # 世界侧已完结（计划 tick 走完，或被身边动静打断）
    source: str = "rules"                 # 解析来源（rules / llm），供重放时还原报告
    narration: str | None = None          # 已记录的叙述；None = 世界已结算但文字尚未落库
    followups: tuple[Intent, ...] = ()    # 多步计划的后续步骤（“拿起长剑向龚光杰刺去”= 拿 → 刺）：第 i+1 个 tick 用第 i 个；
                                          # 计划走完之后的 tick 玩家原地等待（反应 tick、多 tick 等待）
    reaction: bool = False                # 计划之后是否追加一个反应 tick：对人说了话、做了姿态、动了手，在场的人当场回应
    choice: ChoiceUse | None = None       # 首 tick 原子消费的冻结菜单选项；后续 tick 与重放不再消费
    command: str = ""                     # 冻结的展示文字供崩溃后叙述；不重新解释它


@dataclass(frozen=True, slots=True)
class CommitBatch:
    ref: WorldRef
    expected_version: int              # 乐观并发：只有 head 仍是这个版本才允许提交
    state: WorldState                  # 新版本
    events: tuple[Event, ...]
    observations: tuple[Observation, ...]
    beliefs: Mapping[str, BeliefStore]  # 本次发生变化的角色认知（整份替换）
    memories: tuple[MemoryRecord, ...]  # outbox：待索引的经历
    request: TurnEnvelope | None = None           # 请求进度（整份替换；须接续已落库的那一份；narration 不随提交改写）
    session_state: Mapping[str, Any] | None = None  # 会话运行态（调度标记、已描写实体），JSON 兼容，整份替换


class VersionConflict(Exception):
    """head 已不是 expected_version：有别的写入者抢先提交了。"""


class RequestConflict(Exception):
    """同一个 request_id 带来了不同的请求内容，或这份进度接不上已落库的那一份：拒绝执行，而不是猜哪一份才算数。"""


class ChoiceConflict(RequestConflict):
    """菜单已过期、被消费或选项不存在。"""


def consume_decision(decision: Mapping[str, Any] | None, progress: TurnEnvelope,
                     expected_version: int) -> Mapping[str, Any] | None:
    """三个后端在提交临界区调用。只有首 tick 消费，与世界和请求进度一起提交或回滚。"""
    use = progress.choice
    if use is None or len(progress.versions) != 1:
        return decision
    if (decision is None or decision["id"] != use.decision_id or decision["version"] != expected_version
            or decision.get("consumed_request") or not any(c["id"] == use.choice_id for c in decision["choices"])):
        raise ChoiceConflict("局势已经变化，请刷新后重新选择")
    return {**decision, "consumed_request": progress.request_id, "consumed_choice": use.choice_id}


def check_request_progress(prior: TurnEnvelope | None, progress: TurnEnvelope, version: int) -> None:
    """请求进度只能逐 tick 向前接续：首 tick 建立绑定，此后每次提交恰好在已落库的进度上多出本次的版本。
    两个后端都在 commit 的同一临界区 / 事务里调用它——检查与写入之间没有空隙：
    重复投递抢在另一次之后提交，整次提交随之回滚，不会二次结算，也不会改写别人的绑定。"""
    if progress.versions[-1:] != (version,):
        raise RequestConflict(f"请求 {progress.request_id} 的进度没有记下本次提交的版本 {version}")
    if prior is None:
        if len(progress.versions) != 1:
            raise RequestConflict(f"请求 {progress.request_id} 尚未落库，进度却不是从首 tick 开始")
        return
    if prior.payload_hash != progress.payload_hash:
        raise RequestConflict(f"请求 {progress.request_id} 已绑定另一份内容：拒绝执行")
    if prior.done or prior.versions != progress.versions[:-1]:
        raise RequestConflict(f"请求 {progress.request_id} 已由另一次投递推进到 {prior.versions}：拒绝重复结算")


class UnknownWorld(KeyError):
    pass


class WorldStore(Protocol):
    # ---- 生命周期 ----
    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore],
               versions: Mapping[str, str] | None = None) -> None: ...
    def exists(self, ref: WorldRef) -> bool: ...

    # ---- 读 ----
    def head(self, ref: WorldRef) -> WorldState: ...
    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore: ...
    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None: ...
    def events(self, ref: WorldRef) -> tuple[Event, ...]: ...
    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]: ...
    def request(self, ref: WorldRef, request_id: str) -> TurnEnvelope | None: ...
    def session_state(self, ref: WorldRef) -> Mapping[str, Any] | None: ...
    def save_versions(self, ref: WorldRef) -> Mapping[str, str]: ...
    def decision(self, ref: WorldRef) -> Mapping[str, Any] | None: ...
    def publish_decision(self, ref: WorldRef, decision: Mapping[str, Any],
                         expected_version: int) -> Mapping[str, Any]: ...

    # ---- 写（唯一路径）----
    def commit(self, batch: CommitBatch) -> None: ...

    # ---- 文字（幂等旁路：只补写一次，不触碰世界）----
    def record_render(self, ref: WorldRef, request_id: str, narration: str) -> None: ...

    # ---- outbox ----
    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]: ...
    def mark_indexed(self, ids: Sequence[str]) -> None: ...
