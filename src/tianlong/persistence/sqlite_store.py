"""
[INPUT]: 标准库 sqlite3 / json；persistence 的显式 codec 与 WorldStore 契约
[OUTPUT]: SQLiteWorldStore：无需外部数据库的磁盘存档；web_state/save_web_state 保存网页记录
[POS]: persistence 的 SQLite 后端。BEGIN IMMEDIATE 内检查版本与请求绑定，再原子写世界、认知、事件、outbox 和会话进度。
       决策菜单单独冻结，首 tick 同事务消费；每次操作独立连接，可跨线程与进程；叙述只补写一次；没有 pickle。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Observation, WorldState
from tianlong.core.memories import MemoryRecord
from tianlong.persistence import codec
from tianlong.persistence.store import (
    CommitBatch,
    TurnEnvelope,
    UnknownWorld,
    VersionConflict,
    WorldRef,
    check_request_progress,
    consume_decision,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class SQLiteWorldStore:
    """包括冻结菜单的原子发布与首 tick 消费，旧数据库新增 decisions 表即可兼容。"""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS worlds (
                    w TEXT, b TEXT, state TEXT NOT NULL, minds TEXT NOT NULL,
                    session TEXT, versions TEXT NOT NULL, PRIMARY KEY (w,b));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, w TEXT, b TEXT, intent TEXT, data TEXT NOT NULL,
                    UNIQUE (w,b,intent));
                CREATE TABLE IF NOT EXISTS observations (
                    seq INTEGER PRIMARY KEY, w TEXT, b TEXT, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memories (
                    seq INTEGER PRIMARY KEY, id TEXT UNIQUE, w TEXT, b TEXT, owner TEXT,
                    known_at INTEGER, data TEXT NOT NULL, pending INTEGER NOT NULL DEFAULT 1);
                CREATE INDEX IF NOT EXISTS memory_scope ON memories(w,b,owner,known_at);
                CREATE TABLE IF NOT EXISTS requests (
                    w TEXT, b TEXT, id TEXT, data TEXT NOT NULL, narration TEXT, PRIMARY KEY(w,b,id));
                CREATE TABLE IF NOT EXISTS web (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (w TEXT, b TEXT, data TEXT NOT NULL, PRIMARY KEY(w,b));
            """)

    @contextmanager
    def _connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _world(db: sqlite3.Connection, ref: WorldRef) -> sqlite3.Row:
        row = db.execute("SELECT * FROM worlds WHERE w=? AND b=?", (ref.world_id, ref.branch_id)).fetchone()
        if row is None:
            raise UnknownWorld(str(ref))
        return row

    @staticmethod
    def _request(db: sqlite3.Connection, ref: WorldRef, request_id: str) -> TurnEnvelope | None:
        row = db.execute("SELECT data,narration FROM requests WHERE w=? AND b=? AND id=?",
                         (ref.world_id, ref.branch_id, request_id)).fetchone()
        return codec.envelope_from(json.loads(row["data"]), row["narration"]) if row else None

    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore],
               versions: Mapping[str, str] | None = None) -> None:
        with self._connection(True) as db:
            if db.execute("SELECT 1 FROM worlds WHERE w=? AND b=?", (ref.world_id, ref.branch_id)).fetchone():
                raise ValueError(f"世界已存在: {ref}")
            db.execute("INSERT INTO worlds(w,b,state,minds,versions) VALUES (?,?,?,?,?)",
                       (ref.world_id, ref.branch_id, _json(codec.world_to(state)),
                        _json({a: codec.mind_to(s) for a, s in beliefs.items()}), _json(dict(versions or {}))))

    def exists(self, ref: WorldRef) -> bool:
        with self._connection() as db:
            return db.execute("SELECT 1 FROM worlds WHERE w=? AND b=?", (ref.world_id, ref.branch_id)).fetchone() is not None

    def head(self, ref: WorldRef) -> WorldState:
        with self._connection() as db:
            return codec.world_from(json.loads(self._world(db, ref)["state"]))

    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore:
        with self._connection() as db:
            mind = json.loads(self._world(db, ref)["minds"]).get(agent)
            return codec.mind_from(mind) if mind else BeliefStore(agent)

    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None:
        with self._connection() as db:
            self._world(db, ref)
            row = db.execute("SELECT data FROM events WHERE w=? AND b=? AND intent=?",
                             (ref.world_id, ref.branch_id, intent_id)).fetchone()
            return codec.event_from(json.loads(row["data"])) if row else None

    def events(self, ref: WorldRef) -> tuple[Event, ...]:
        with self._connection() as db:
            self._world(db, ref)
            return tuple(codec.event_from(json.loads(r[0])) for r in db.execute(
                "SELECT data FROM events WHERE w=? AND b=? ORDER BY seq", (ref.world_id, ref.branch_id)))

    def observations(self, ref: WorldRef) -> tuple[Observation, ...]:
        with self._connection() as db:
            self._world(db, ref)
            return tuple(Observation(d["id"], d["observer"], d["source_event"], codec.percept_from(d["percept"]))
                         for r in db.execute("SELECT data FROM observations WHERE w=? AND b=? ORDER BY seq",
                                             (ref.world_id, ref.branch_id)) for d in [json.loads(r[0])])

    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]:
        with self._connection() as db:
            self._world(db, ref)
            return tuple(codec.memory_from(json.loads(r[0])) for r in db.execute(
                "SELECT data FROM memories WHERE w=? AND b=? AND owner=? AND known_at>=? ORDER BY seq",
                (ref.world_id, ref.branch_id, owner, since)))

    def request(self, ref: WorldRef, request_id: str) -> TurnEnvelope | None:
        with self._connection() as db:
            self._world(db, ref)
            return self._request(db, ref, request_id)

    def session_state(self, ref: WorldRef) -> Mapping[str, Any] | None:
        with self._connection() as db:
            raw = self._world(db, ref)["session"]
            return json.loads(raw) if raw is not None else None

    def save_versions(self, ref: WorldRef) -> Mapping[str, str]:
        with self._connection() as db:
            return json.loads(self._world(db, ref)["versions"])

    @staticmethod
    def _decision(db: sqlite3.Connection, ref: WorldRef) -> dict[str, Any] | None:
        row = db.execute("SELECT data FROM decisions WHERE w=? AND b=?", (ref.world_id, ref.branch_id)).fetchone()
        return json.loads(row[0]) if row else None

    def decision(self, ref: WorldRef) -> Mapping[str, Any] | None:
        with self._connection() as db:
            self._world(db, ref)
            return self._decision(db, ref)

    def publish_decision(self, ref: WorldRef, decision: Mapping[str, Any],
                         expected_version: int) -> Mapping[str, Any]:
        data = _json(dict(decision))
        with self._connection(True) as db:
            head = json.loads(self._world(db, ref)["state"])
            if head["version"] != expected_version or decision["version"] != expected_version:
                raise VersionConflict("发布选项时局势已变化")
            prior = self._decision(db, ref)
            if prior is not None and prior["version"] == expected_version:
                return prior
            db.execute("INSERT INTO decisions(w,b,data) VALUES (?,?,?) ON CONFLICT(w,b) DO UPDATE SET data=excluded.data",
                       (ref.world_id, ref.branch_id, data))
            return json.loads(data)

    def commit(self, batch: CommitBatch) -> None:
        ref = batch.ref
        with self._connection(True) as db:
            row = self._world(db, ref)
            head = codec.world_from(json.loads(row["state"]))
            if head.version != batch.expected_version:
                raise VersionConflict(f"{ref}: head={head.version} expected={batch.expected_version}")
            if batch.state.version != batch.expected_version + 1:
                raise ValueError("新状态版本必须恰好 +1")
            if batch.request is not None:
                prior = self._request(db, ref, batch.request.request_id)
                check_request_progress(prior, batch.request, batch.state.version)
                decision = consume_decision(self._decision(db, ref), batch.request, batch.expected_version)
                if batch.request.choice is not None and len(batch.request.versions) == 1:
                    db.execute("UPDATE decisions SET data=? WHERE w=? AND b=?",
                               (_json(decision), ref.world_id, ref.branch_id))
                db.execute("INSERT INTO requests(w,b,id,data) VALUES (?,?,?,?) "
                           "ON CONFLICT(w,b,id) DO UPDATE SET data=excluded.data",
                           (ref.world_id, ref.branch_id, batch.request.request_id, _json(codec.envelope_to(batch.request))))
            minds = json.loads(row["minds"])
            minds.update({a: codec.mind_to(s) for a, s in batch.beliefs.items()})
            session = row["session"] if batch.session_state is None else _json(dict(batch.session_state))
            db.execute("UPDATE worlds SET state=?,minds=?,session=? WHERE w=? AND b=?",
                       (_json(codec.world_to(batch.state)), _json(minds), session, ref.world_id, ref.branch_id))
            db.executemany("INSERT INTO events(w,b,intent,data) VALUES (?,?,?,?)",
                           [(ref.world_id, ref.branch_id, e.intent.id, _json(codec.event_to(e))) for e in batch.events])
            db.executemany("INSERT INTO observations(w,b,data) VALUES (?,?,?)", [
                (ref.world_id, ref.branch_id, _json({"id": o.id, "observer": o.observer,
                                                   "source_event": o.source_event, "percept": codec.percept_to(o.percept)}))
                for o in batch.observations])
            db.executemany("INSERT INTO memories(id,w,b,owner,known_at,data) VALUES (?,?,?,?,?,?)", [
                (m.id, ref.world_id, ref.branch_id, m.owner, m.known_at, _json(codec.memory_to(m))) for m in batch.memories])

    def record_render(self, ref: WorldRef, request_id: str, narration: str) -> None:
        with self._connection(True) as db:
            self._world(db, ref)
            if self._request(db, ref, request_id) is None:
                raise KeyError(f"未知请求: {request_id}")
            db.execute("UPDATE requests SET narration=? WHERE w=? AND b=? AND id=? AND narration IS NULL",
                       (narration, ref.world_id, ref.branch_id, request_id))

    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]:
        with self._connection() as db:
            return tuple(codec.memory_from(json.loads(r[0])) for r in db.execute(
                "SELECT data FROM memories WHERE pending=1 ORDER BY seq LIMIT ?", (limit,)))

    def mark_indexed(self, ids: Sequence[str]) -> None:
        with self._connection(True) as db:
            db.executemany("UPDATE memories SET pending=0 WHERE id=?", [(i,) for i in ids])

    def web_state(self) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT data FROM web WHERE id=1").fetchone()
            return json.loads(row[0]) if row else {}

    def save_web_state(self, state: dict[str, Any]) -> None:
        with self._connection(True) as db:
            db.execute("INSERT INTO web(id,data) VALUES (1,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data", (_json(state),))
