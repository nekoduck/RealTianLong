"""
[INPUT]: 依赖 persistence/store 的协议、值对象与 check_request_progress，core / cognition 的不可变类型，
         标准库 json（会话运行态按 JSON 往返存取）
[OUTPUT]: 对外提供 InMemoryWorldStore
[POS]: persistence 的内存实现；测试、训练、离线游玩的默认后端。与 Neo4j 实现遵守同一协议，用同一组契约测试验证：
       请求绑定检查、冻结菜单消费、请求进度与会话运行态随世界提交同一临界区完成；菜单发布不推进世界，叙述文字只补写一次
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Observation, WorldState
from tianlong.core.memories import MemoryRecord
from tianlong.persistence.store import (
    CommitBatch,
    TurnEnvelope,
    UnknownWorld,
    VersionConflict,
    WorldRef,
    check_request_progress,
    consume_decision,
)


def _json_copy(state: Mapping[str, Any]) -> dict[str, Any]:
    """会话运行态按 JSON 往返保存：与 Neo4j 后端取回的形状逐字节一致，也拒绝混进不可序列化的对象。"""
    return json.loads(json.dumps(state, ensure_ascii=False, sort_keys=True))


@dataclass
class _World:
    head: WorldState
    beliefs: dict[str, BeliefStore]
    events: list[Event] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    intent_index: dict[str, Event] = field(default_factory=dict)
    memories: list[MemoryRecord] = field(default_factory=list)
    requests: dict[str, TurnEnvelope] = field(default_factory=dict)
    session: dict[str, Any] | None = None
    versions: dict[str, str] = field(default_factory=dict)
    decision: dict[str, Any] | None = None


class InMemoryWorldStore:
    """所有值都是不可变对象，直接保存引用即可；锁只保护“检查版本 + 写入”这一临界区。"""

    def __init__(self) -> None:
        self._worlds: dict[WorldRef, _World] = {}
        self._pending: dict[str, MemoryRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------

    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore],
               versions: Mapping[str, str] | None = None) -> None:
        with self._lock:
            if ref in self._worlds:
                raise ValueError(f"世界已存在: {ref}")
            self._worlds[ref] = _World(state, dict(beliefs), versions=dict(versions or {}))

    def exists(self, ref: WorldRef) -> bool:
        return ref in self._worlds

    # ------------------------------------------------------------
    #  读
    # ------------------------------------------------------------

    def _world(self, ref: WorldRef) -> _World:
        try:
            return self._worlds[ref]
        except KeyError:
            raise UnknownWorld(str(ref)) from None

    def head(self, ref: WorldRef) -> WorldState:
        return self._world(ref).head

    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore:
        return self._world(ref).beliefs.get(agent) or BeliefStore(agent)

    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None:
        return self._world(ref).intent_index.get(intent_id)

    def events(self, ref: WorldRef) -> tuple[Event, ...]:
        return tuple(self._world(ref).events)

    def observations(self, ref: WorldRef) -> tuple[Observation, ...]:
        return tuple(self._world(ref).observations)

    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]:
        return tuple(m for m in self._world(ref).memories if m.owner == owner and m.known_at >= since)

    def request(self, ref: WorldRef, request_id: str) -> TurnEnvelope | None:
        return self._world(ref).requests.get(request_id)

    def session_state(self, ref: WorldRef) -> Mapping[str, Any] | None:
        s = self._world(ref).session
        return None if s is None else _json_copy(s)

    def save_versions(self, ref: WorldRef) -> Mapping[str, str]:
        return dict(self._world(ref).versions)

    def decision(self, ref: WorldRef) -> Mapping[str, Any] | None:
        with self._lock:
            data = self._world(ref).decision
            return None if data is None else _json_copy(data)

    def publish_decision(self, ref: WorldRef, decision: Mapping[str, Any],
                         expected_version: int) -> Mapping[str, Any]:
        data = _json_copy(decision)
        with self._lock:
            w = self._world(ref)
            if w.head.version != expected_version or data["version"] != expected_version:
                raise VersionConflict("发布选项时局势已变化")
            if w.decision is None or w.decision["version"] != expected_version:
                w.decision = data
            return _json_copy(w.decision)

    # ------------------------------------------------------------
    #  写
    # ------------------------------------------------------------

    def commit(self, batch: CommitBatch) -> None:
        session = None if batch.session_state is None else _json_copy(batch.session_state)
        with self._lock:
            w = self._world(batch.ref)
            if w.head.version != batch.expected_version:
                raise VersionConflict(f"{batch.ref}: head={w.head.version} expected={batch.expected_version}")
            if batch.state.version != batch.expected_version + 1:
                raise ValueError("新状态版本必须恰好 +1")
            if batch.request is not None:
                # 请求绑定与写入同一临界区：重复投递抢在后面提交即被拒，世界一处不改
                check_request_progress(w.requests.get(batch.request.request_id), batch.request, batch.state.version)
            decision = w.decision if batch.request is None else consume_decision(
                w.decision, batch.request, batch.expected_version)
            w.head = batch.state
            w.beliefs.update(batch.beliefs)
            w.events.extend(batch.events)
            w.observations.extend(batch.observations)
            for e in batch.events:
                w.intent_index[e.intent.id] = e
            w.memories.extend(batch.memories)
            for m in batch.memories:
                self._pending[m.id] = m
            if batch.request is not None:
                # 叙述只由 record_render 写：提交整份替换进度，但保留已落库的文字
                prior = w.requests.get(batch.request.request_id)
                w.requests[batch.request.request_id] = replace(
                    batch.request, narration=prior.narration if prior else None)
            if session is not None:
                w.session = session
            w.decision = None if decision is None else dict(decision)

    def record_render(self, ref: WorldRef, request_id: str, narration: str) -> None:
        """幂等：第一次写入的叙述为准，重试不改写；请求不存在是调用方的错。"""
        with self._lock:
            w = self._world(ref)
            env = w.requests.get(request_id)
            if env is None:
                raise KeyError(f"未知请求: {request_id}")
            if env.narration is None:
                w.requests[request_id] = replace(env, narration=narration)

    # ------------------------------------------------------------
    #  outbox
    # ------------------------------------------------------------

    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(list(self._pending.values())[:limit])

    def mark_indexed(self, ids: Sequence[str]) -> None:
        with self._lock:
            for i in ids:
                self._pending.pop(i, None)
