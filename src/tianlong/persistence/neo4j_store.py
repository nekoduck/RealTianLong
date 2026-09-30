"""
[INPUT]: 依赖 neo4j 驱动的 Driver / GraphDatabase，persistence/store 的协议、值对象与 check_request_progress，
         persistence/codec 的编解码，core / cognition 的不可变类型
[OUTPUT]: 对外提供 Neo4jWorldStore（WorldStore 协议的图数据库实现）、net_relation_diff()
[POS]: persistence 的 Neo4j 后端。图模型：
       (:World) 版本锚点；(:Entity:{Person|Place|Item|Surface|Door}) 以 AT/OWNS/MATCHES/CONNECTS 相连；
       (:Event)-[:BY|TARGET|OBJ|OCCURRED_AT]->(:Entity)；(:Entity)-[:OBSERVED]->(:Observation)-[:OF]->(:Event)；
       命题与相信分离：(:Entity)-[:BELIEVES {holds, confidence, ...}]->(:Proposition)-[:ABOUT]->(:Entity)；
       (:Entity)-[:HAS_MIND]->(:Mind)-[:KNOWS]->(:Entity)，Mind 节点另以 JSON 属性存勘察记录、承诺状态与社交状态
       （cues/attitudes/company/allies/yielded，旧存档缺省为空）；(:Memory {indexed}) 即 outbox；
       (:Request {data, narration}) 是玩家请求的进度与叙述；World 节点另存存档版本、会话运行态与冻结菜单；首 tick 原子消费菜单。
       commit 先对 World 节点加写锁再比对版本——read-committed 隔离下由锁保证串行，而不是指望 ACID 自动解决并发；
       持锁后再做请求绑定检查（进度必须接在已落库的那一份之后），请求进度与会话运行态与世界变化同一事务写入，
       叙述文字由 record_render() 只补写一次
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from neo4j import Driver, GraphDatabase, ManagedTransaction

from tianlong.cognition import BeliefStore
from tianlong.core import (
    AddRelation,
    Change,
    Entity,
    Event,
    Kind,
    Modality,
    Rel,
    Relation,
    RemoveRelation,
    SetAttr,
    WorldState,
)
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

_LABELS = {k: k.value.capitalize() for k in Kind}   # 标签与关系类型只来自枚举白名单，绝不拼接外部输入
_ROLES = (("BY", "actor"), ("TARGET", "target"), ("OBJ", "obj"), ("OCCURRED_AT", "place"))
_NODES = ("World", "Entity", "Event", "Observation", "Proposition", "Mind", "Memory", "Request")


def _uid(ref: WorldRef, *parts: str) -> str:
    return "|".join((ref.world_id, ref.branch_id, *parts))


def net_relation_diff(changes: Iterable[Change]) -> tuple[set[Relation], set[Relation]]:
    """把一个 tick 内按顺序发生的关系变化折叠成净差异 (removed, added)。

    钥匙 桌→玩家 再 玩家→守卫：朴素地“先删后加”会让钥匙同时出现在两处；折叠后只剩 删(桌) 加(守卫)。
    """
    added: set[Relation] = set()
    removed: set[Relation] = set()
    for c in changes:
        if isinstance(c, AddRelation):
            if c.rel in removed:
                removed.discard(c.rel)
            else:
                added.add(c.rel)
        elif isinstance(c, RemoveRelation):
            if c.rel in added:
                added.discard(c.rel)
            else:
                removed.add(c.rel)
    return removed, added


class Neo4jWorldStore:
    def __init__(self, driver: Driver, database: str | None = None) -> None:
        self._driver = driver
        self._db = database
        self._ensure_schema()

    @classmethod
    def from_env(cls) -> Neo4jWorldStore:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
        return cls(GraphDatabase.driver(uri, auth=auth), os.environ.get("NEO4J_DATABASE") or None)

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------
    #  基础设施
    # ------------------------------------------------------------

    def _write(self, fn, **kw):
        with self._driver.session(database=self._db) as s:
            return s.execute_write(fn, **kw)

    def _read(self, fn, **kw):
        with self._driver.session(database=self._db) as s:
            return s.execute_read(fn, **kw)

    def _ensure_schema(self) -> None:
        stmts = [f"CREATE CONSTRAINT {n.lower()}_uid IF NOT EXISTS FOR (x:{n}) REQUIRE x.uid IS UNIQUE" for n in _NODES]
        stmts += [
            "CREATE CONSTRAINT event_intent IF NOT EXISTS FOR (x:Event) REQUIRE x.intent_uid IS UNIQUE",
            "CREATE INDEX memory_pending IF NOT EXISTS FOR (x:Memory) ON (x.indexed)",
        ]
        with self._driver.session(database=self._db) as s:
            for q in stmts:
                s.run(q).consume()

    def drop_world(self, ref: WorldRef) -> None:
        """删除一个世界分支的全部节点（测试清理与存档删除）。"""
        self._write(lambda t: t.run(
            "MATCH (x) WHERE x.w = $w AND x.b = $b DETACH DELETE x", w=ref.world_id, b=ref.branch_id).consume())

    # ------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------

    def exists(self, ref: WorldRef) -> bool:
        return self._read(lambda t: t.run("MATCH (w:World {uid:$u}) RETURN count(w) AS n",
                                          u=_uid(ref)).single()["n"] > 0)

    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore],
               versions: Mapping[str, str] | None = None) -> None:
        def tx(t: ManagedTransaction) -> None:
            if t.run("MATCH (w:World {uid:$u}) RETURN count(w) AS n", u=_uid(ref)).single()["n"]:
                raise ValueError(f"世界已存在: {ref}")
            t.run("CREATE (:World {uid:$u, w:$w, b:$b, version:$v, clock:$c, seed:$s, versions:$vs})",
                  u=_uid(ref), w=ref.world_id, b=ref.branch_id, v=state.version, c=state.clock,
                  s=str(state.seed), vs=json.dumps(dict(versions or {}), sort_keys=True)).consume()
            by_kind: dict[Kind, list[dict]] = defaultdict(list)
            for e in state.entities.values():
                by_kind[e.kind].append(self._entity_row(ref, e))
            for kind, rows in by_kind.items():
                t.run(f"UNWIND $rows AS r CREATE (e:Entity:{_LABELS[kind]}) SET e = r", rows=rows).consume()
            self._write_relations(t, ref, removed=set(), added=set(state.relations))
            for store in beliefs.values():
                self._write_mind(t, ref, store)

        self._write(tx)

    @staticmethod
    def _entity_row(ref: WorldRef, e: Entity) -> dict[str, Any]:
        return {"uid": _uid(ref, e.id), "w": ref.world_id, "b": ref.branch_id, "id": e.id,
                "kind": e.kind.value, "name": e.name, "attrs": json.dumps(e.attrs, ensure_ascii=False)}

    # ------------------------------------------------------------
    #  读
    # ------------------------------------------------------------

    def head(self, ref: WorldRef) -> WorldState:
        def tx(t: ManagedTransaction) -> WorldState:
            w = t.run("MATCH (w:World {uid:$u}) RETURN w", u=_uid(ref)).single()
            if w is None:
                raise UnknownWorld(str(ref))
            w = w["w"]
            ents = [
                Entity(r["id"], Kind(r["kind"]), r["name"], tuple((k, v) for k, v in json.loads(r["attrs"])))
                for r in t.run("MATCH (e:Entity {w:$w, b:$b}) RETURN e.id AS id, e.kind AS kind, e.name AS name, "
                               "e.attrs AS attrs", w=ref.world_id, b=ref.branch_id)
            ]
            rels = [
                Relation(r["s"], Rel(r["t"]), r["d"])
                for r in t.run("MATCH (a:Entity {w:$w, b:$b})-[x]->(c:Entity {w:$w, b:$b}) "
                               "WHERE type(x) IN $types RETURN a.id AS s, type(x) AS t, c.id AS d",
                               w=ref.world_id, b=ref.branch_id, types=[r.value for r in Rel])
            ]
            return WorldState.build(int(w["seed"]), w["clock"], ents, rels, version=w["version"])

        return self._read(tx)

    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore:
        def tx(t: ManagedTransaction) -> BeliefStore:
            m = t.run("MATCH (m:Mind {uid:$u}) OPTIONAL MATCH (m)-[k:KNOWS]->() "
                      "RETURN m, collect(k.sketch) AS sketches", u=_uid(ref, "mind", agent)).single()
            if m is None or m["m"] is None:
                return BeliefStore(agent)
            mind = m["m"]
            sketches = [codec.sketch_from(json.loads(s)) for s in m["sketches"]]
            beliefs = {}
            for r in t.run("MATCH (a:Entity {uid:$a})-[bel:BELIEVES]->(p:Proposition) "
                           "RETURN p.subject AS s, p.predicate AS p, p.value AS v, properties(bel) AS b",
                           a=_uid(ref, agent)):
                prop = codec.prop_from({"s": r["s"], "p": r["p"], "v": json.loads(r["v"])})
                beliefs[prop] = codec.belief_from(prop, r["b"])
            return BeliefStore(
                agent, {s.id: s for s in sketches}, beliefs,
                tuple(codec.episode_from(e) for e in json.loads(mind["episodes"])),
                json.loads(mind["trust"]), mind["last_tick"],
                json.loads(mind.get("surveyed") or "{}"), json.loads(mind.get("searched") or "{}"),
                tuple(codec.obligation_from(o) for o in json.loads(mind.get("obligations") or "[]")),
                tuple(codec.said_from(x) for x in json.loads(mind.get("said") or "[]")),
                cues=tuple(codec.cue_from(c) for c in json.loads(mind.get("cues") or "[]")),
                attitudes=codec.attitudes_from(json.loads(mind.get("attitudes") or "{}")),
                company=json.loads(mind.get("company") or "{}"),
                allies=tuple(json.loads(mind.get("allies") or "[]")),
                yielded=json.loads(mind.get("yielded") or "{}"),
            )

        return self._read(tx)

    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None:
        rec = self._read(lambda t: t.run("MATCH (e:Event {intent_uid:$i}) RETURN e.data AS d",
                                         i=_uid(ref, "intent", intent_id)).single())
        return codec.event_from(json.loads(rec["d"])) if rec else None

    def events(self, ref: WorldRef) -> tuple[Event, ...]:
        rows = self._read(lambda t: list(t.run(
            "MATCH (e:Event {w:$w, b:$b}) RETURN e.data AS d ORDER BY e.version, e.seq",
            w=ref.world_id, b=ref.branch_id)))
        return tuple(codec.event_from(json.loads(r["d"])) for r in rows)

    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]:
        rows = self._read(lambda t: list(t.run(
            "MATCH (m:Memory {w:$w, b:$b, owner:$o}) WHERE m.known_at >= $since RETURN m ORDER BY m.known_at, m.id",
            w=ref.world_id, b=ref.branch_id, o=owner, since=since)))
        return tuple(self._memory_from(r["m"]) for r in rows)

    def request(self, ref: WorldRef, request_id: str) -> TurnEnvelope | None:
        rec = self._read(lambda t: t.run("MATCH (r:Request {uid:$u}) RETURN r.data AS d, r.narration AS n",
                                         u=_uid(ref, "request", request_id)).single())
        return codec.envelope_from(json.loads(rec["d"]), rec["n"]) if rec else None

    def session_state(self, ref: WorldRef) -> Mapping[str, Any] | None:
        rec = self._read(lambda t: t.run("MATCH (w:World {uid:$u}) RETURN w.session AS s", u=_uid(ref)).single())
        if rec is None:
            raise UnknownWorld(str(ref))
        return json.loads(rec["s"]) if rec["s"] is not None else None

    def save_versions(self, ref: WorldRef) -> Mapping[str, str]:
        rec = self._read(lambda t: t.run("MATCH (w:World {uid:$u}) RETURN w.versions AS v", u=_uid(ref)).single())
        if rec is None:
            raise UnknownWorld(str(ref))
        return json.loads(rec["v"]) if rec["v"] is not None else {}

    def decision(self, ref: WorldRef) -> Mapping[str, Any] | None:
        rec = self._read(lambda t: t.run("MATCH (w:World {uid:$u}) RETURN w.decision AS d", u=_uid(ref)).single())
        if rec is None:
            raise UnknownWorld(str(ref))
        return json.loads(rec["d"]) if rec["d"] else None

    def publish_decision(self, ref: WorldRef, decision: Mapping[str, Any],
                         expected_version: int) -> Mapping[str, Any]:
        data = json.dumps(decision, ensure_ascii=False, sort_keys=True)

        def tx(t: ManagedTransaction) -> Mapping[str, Any]:
            rec = t.run("MATCH (w:World {uid:$u}) SET w._lock = true RETURN w.version AS v, w.decision AS d",
                        u=_uid(ref)).single()
            if rec is None:
                raise UnknownWorld(str(ref))
            if rec["v"] != expected_version or decision["version"] != expected_version:
                raise VersionConflict("发布选项时局势已变化")
            prior = json.loads(rec["d"]) if rec["d"] else None
            if prior is not None and prior["version"] == expected_version:
                t.run("MATCH (w:World {uid:$u}) REMOVE w._lock", u=_uid(ref)).consume()
                return prior
            t.run("MATCH (w:World {uid:$u}) SET w.decision=$d REMOVE w._lock", u=_uid(ref), d=data).consume()
            return json.loads(data)

        return self._write(tx)

    # ------------------------------------------------------------
    #  写：唯一路径
    # ------------------------------------------------------------

    def commit(self, batch: CommitBatch) -> None:
        ref = batch.ref

        def tx(t: ManagedTransaction) -> None:
            # ---- 1. 先加锁，再比对版本 ----
            rec = t.run("MATCH (w:World {uid:$u}) SET w._lock = true RETURN w.version AS v", u=_uid(ref)).single()
            if rec is None:
                raise UnknownWorld(str(ref))
            if rec["v"] != batch.expected_version:
                raise VersionConflict(f"{ref}: head={rec['v']} expected={batch.expected_version}")
            if batch.state.version != batch.expected_version + 1:
                raise ValueError("新状态版本必须恰好 +1")
            if batch.request is not None:
                # 持有 World 锁时读已落库的进度：绑定检查与写入同一事务，重复投递抢在后面提交即整体回滚
                rec = t.run("MATCH (r:Request {uid:$u}) RETURN r.data AS d",
                            u=_uid(ref, "request", batch.request.request_id)).single()
                prior = codec.envelope_from(json.loads(rec["d"])) if rec else None
                check_request_progress(prior, batch.request, batch.state.version)
                if batch.request.choice is not None and len(batch.request.versions) == 1:
                    rec = t.run("MATCH (w:World {uid:$u}) RETURN w.decision AS d", u=_uid(ref)).single()
                    decision = consume_decision(json.loads(rec["d"]) if rec["d"] else None,
                                                batch.request, batch.expected_version)
                    t.run("MATCH (w:World {uid:$u}) SET w.decision=$d", u=_uid(ref),
                          d=json.dumps(decision, ensure_ascii=False, sort_keys=True)).consume()

            # ---- 2. 世界变化：关系按净差异，属性按新状态整体覆盖 ----
            changes = [c for e in batch.events for c in e.changes]
            removed, added = net_relation_diff(changes)
            self._write_relations(t, ref, removed, added)
            touched = sorted({c.entity for c in changes if isinstance(c, SetAttr)})
            rows = [{"uid": _uid(ref, eid), "attrs": json.dumps(batch.state.entity(eid).attrs, ensure_ascii=False)}
                    for eid in touched]
            t.run("UNWIND $rows AS r MATCH (e:Entity {uid:r.uid}) SET e.attrs = r.attrs", rows=rows).consume()
            t.run("MATCH (w:World {uid:$u}) SET w.version = $v, w.clock = $c REMOVE w._lock",
                  u=_uid(ref), v=batch.state.version, c=batch.state.clock).consume()
            if batch.session_state is not None:
                t.run("MATCH (w:World {uid:$u}) SET w.session = $s", u=_uid(ref),
                      s=json.dumps(batch.session_state, ensure_ascii=False, sort_keys=True)).consume()

            # ---- 3. 事件与观察（环顾类观察量大且可由认知反映，不落库）----
            self._write_events(t, ref, batch.state.version, batch.events)
            obs = [
                {"uid": _uid(ref, o.id), "w": ref.world_id, "b": ref.branch_id, "id": o.id, "observer": o.observer,
                 "tick": o.percept.tick, "modality": o.percept.modality.value,
                 "data": json.dumps(codec.percept_to(o.percept), ensure_ascii=False),
                 "ev": _uid(ref, o.source_event) if o.source_event else None, "who": _uid(ref, o.observer)}
                for o in batch.observations if o.percept.modality != Modality.SCENE
            ]
            t.run("UNWIND $rows AS r MATCH (who:Entity {uid:r.who}) "
                  "CREATE (o:Observation {uid:r.uid, w:r.w, b:r.b, id:r.id, observer:r.observer, tick:r.tick, "
                  "modality:r.modality, data:r.data}) CREATE (who)-[:OBSERVED]->(o) "
                  "WITH o, r MATCH (ev:Event {uid:r.ev}) CREATE (o)-[:OF]->(ev)", rows=obs).consume()

            # ---- 4. 认知与 outbox ----
            for store in batch.beliefs.values():
                self._write_mind(t, ref, store)
            mems = [
                {"uid": _uid(ref, m.id), "w": ref.world_id, "b": ref.branch_id, "id": m.id, "owner": m.owner,
                 "kind": m.kind, "text": m.text, "occurred_at": m.occurred_at, "known_at": m.known_at,
                 "source": m.source, "subjects": list(m.subjects), "indexed": False, "who": _uid(ref, m.owner),
                 "informant": m.informant, "verdict": m.verdict}
                for m in batch.memories
            ]
            t.run("UNWIND $rows AS r MATCH (who:Entity {uid:r.who}) "
                  "CREATE (m:Memory {uid:r.uid, w:r.w, b:r.b, id:r.id, owner:r.owner, kind:r.kind, text:r.text, "
                  "occurred_at:r.occurred_at, known_at:r.known_at, source:r.source, subjects:r.subjects, "
                  "informant:r.informant, verdict:r.verdict, indexed:false}) CREATE (m)-[:OF]->(who)",
                  rows=mems).consume()

            # ---- 5. 请求进度：整份替换 data，narration 只由 record_render 写 ----
            if batch.request is not None:
                env = batch.request
                t.run("MERGE (r:Request {uid:$u}) ON CREATE SET r.w = $w, r.b = $b, r.request_id = $rid "
                      "SET r.payload_hash = $ph, r.data = $d",
                      u=_uid(ref, "request", env.request_id), w=ref.world_id, b=ref.branch_id, rid=env.request_id,
                      ph=env.payload_hash, d=json.dumps(codec.envelope_to(env), ensure_ascii=False)).consume()

        self._write(tx)

    def record_render(self, ref: WorldRef, request_id: str, narration: str) -> None:
        """幂等：第一次写入的叙述为准，重试不改写；请求不存在是调用方的错。"""
        def tx(t: ManagedTransaction) -> None:
            rec = t.run("MATCH (r:Request {uid:$u}) SET r.narration = coalesce(r.narration, $n) RETURN count(r) AS n",
                        u=_uid(ref, "request", request_id), n=narration).single()
            if not rec or not rec["n"]:
                raise KeyError(f"未知请求: {request_id}")

        self._write(tx)

    # ------------------------------------------------------------
    #  outbox
    # ------------------------------------------------------------

    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]:
        rows = self._read(lambda t: list(t.run(
            "MATCH (m:Memory {indexed:false}) RETURN m ORDER BY m.known_at, m.id LIMIT $n", n=limit)))
        return tuple(self._memory_from(r["m"]) for r in rows)

    def mark_indexed(self, ids: Sequence[str]) -> None:
        # 经历 ID 由 (世界, 分支, 观察) 派生，全局唯一
        self._write(lambda t: t.run("UNWIND $ids AS i MATCH (m:Memory {id:i}) SET m.indexed = true",
                                    ids=list(ids)).consume())

    # ------------------------------------------------------------
    #  写入积木
    # ------------------------------------------------------------

    @staticmethod
    def _write_relations(t: ManagedTransaction, ref: WorldRef, removed: set[Relation], added: set[Relation]) -> None:
        for rel_type in Rel:
            rm = [{"s": _uid(ref, r.src), "d": _uid(ref, r.dst)} for r in removed if r.type == rel_type]
            ad = [{"s": _uid(ref, r.src), "d": _uid(ref, r.dst)} for r in added if r.type == rel_type]
            if rm:
                t.run(f"UNWIND $rows AS r MATCH (:Entity {{uid:r.s}})-[x:{rel_type.value}]->(:Entity {{uid:r.d}}) "
                      "DELETE x", rows=rm).consume()
            if ad:
                t.run(f"UNWIND $rows AS r MATCH (a:Entity {{uid:r.s}}), (c:Entity {{uid:r.d}}) "
                      f"CREATE (a)-[:{rel_type.value}]->(c)", rows=ad).consume()

    @staticmethod
    def _write_events(t: ManagedTransaction, ref: WorldRef, version: int, events: Sequence[Event]) -> None:
        rows = [
            {"uid": _uid(ref, e.id), "w": ref.world_id, "b": ref.branch_id, "id": e.id, "version": version,
             "seq": i, "tick": e.tick, "op": e.op.value, "outcome": e.outcome.value, "reason": e.reason,
             "intent_uid": _uid(ref, "intent", e.intent.id),
             "data": json.dumps(codec.event_to(e), ensure_ascii=False)}
            for i, e in enumerate(events)
        ]
        t.run("UNWIND $rows AS r CREATE (e:Event) SET e = r", rows=rows).consume()
        for rel, field in _ROLES:
            links = []
            for e in events:
                val = e.place if field == "place" else getattr(e.intent, field)
                if val:
                    links.append({"e": _uid(ref, e.id), "x": _uid(ref, val)})
            if links:
                t.run(f"UNWIND $rows AS r MATCH (e:Event {{uid:r.e}}), (x:Entity {{uid:r.x}}) CREATE (e)-[:{rel}]->(x)",
                      rows=links).consume()

    @staticmethod
    def _write_mind(t: ManagedTransaction, ref: WorldRef, store: BeliefStore) -> None:
        """整份替换一个角色的认知：命题节点共享（MERGE），相信关系属于个人（先删后建）。"""
        agent_uid, mind_uid = _uid(ref, store.owner), _uid(ref, "mind", store.owner)
        t.run("MATCH (a:Entity {uid:$a}) MERGE (m:Mind {uid:$m}) "
              "SET m.w = $w, m.b = $b, m.owner = $o, m.trust = $trust, m.episodes = $eps, m.last_tick = $lt, "
              "m.surveyed = $sv, m.searched = $sr, m.obligations = $ob, m.said = $sd, "
              "m.cues = $cu, m.attitudes = $at, m.company = $cp, m.allies = $al, m.yielded = $yd "
              "MERGE (a)-[:HAS_MIND]->(m) "
              "WITH a, m OPTIONAL MATCH (m)-[k:KNOWS]->() DELETE k "
              "WITH a OPTIONAL MATCH (a)-[bel:BELIEVES]->() DELETE bel",
              a=agent_uid, m=mind_uid, w=ref.world_id, b=ref.branch_id, o=store.owner,
              trust=json.dumps(dict(store.trust)), lt=store.last_tick,
              sv=json.dumps(dict(store.surveyed), sort_keys=True), sr=json.dumps(dict(store.searched), sort_keys=True),
              ob=json.dumps([codec.obligation_to(o) for o in store.obligations], ensure_ascii=False),
              sd=json.dumps([codec.said_to(x) for x in store.said], ensure_ascii=False),
              cu=json.dumps([codec.cue_to(c) for c in store.cues], ensure_ascii=False),
              at=json.dumps(codec.attitudes_to(store.attitudes)), cp=json.dumps(dict(store.company), sort_keys=True),
              al=json.dumps(list(store.allies)), yd=json.dumps(dict(store.yielded), sort_keys=True),
              eps=json.dumps([codec.episode_to(e) for e in store.episodes], ensure_ascii=False)).consume()
        known = [{"uid": _uid(ref, sk.id), "sketch": json.dumps(codec.sketch_to(sk), ensure_ascii=False)}
                 for sk in store.entities.values()]
        t.run("UNWIND $rows AS r MATCH (m:Mind {uid:$m}), (e:Entity {uid:r.uid}) CREATE (m)-[:KNOWS {sketch:r.sketch}]->(e)",
              rows=known, m=mind_uid).consume()
        beliefs = []
        for b in store.sorted_beliefs():
            v = json.dumps(b.prop.value, ensure_ascii=False)
            beliefs.append({"puid": _uid(ref, "prop", b.prop.subject, b.prop.predicate, v), "s": b.prop.subject,
                            "p": b.prop.predicate, "v": v, "subj": _uid(ref, b.prop.subject), **codec.belief_to(b)})
        t.run("UNWIND $rows AS r MERGE (p:Proposition {uid:r.puid}) "
              "ON CREATE SET p.w = $w, p.b = $b, p.subject = r.s, p.predicate = r.p, p.value = r.v "
              "WITH p, r MATCH (a:Entity {uid:$a}) "
              "CREATE (a)-[:BELIEVES {holds:r.holds, confidence:r.confidence, modality:r.modality, "
              "learned_at:r.learned_at, informant:r.informant}]->(p) "
              "WITH p, r MATCH (s:Entity {uid:r.subj}) MERGE (p)-[:ABOUT]->(s)",
              rows=beliefs, a=agent_uid, w=ref.world_id, b=ref.branch_id).consume()

    @staticmethod
    def _memory_from(m: Any) -> MemoryRecord:
        return MemoryRecord(m["id"], m["w"], m["b"], m["owner"], m["kind"], m["text"], m["occurred_at"],
                            m["known_at"], m["source"], tuple(m["subjects"]), m.get("informant"), m.get("verdict"))
