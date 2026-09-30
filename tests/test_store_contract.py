"""
[INPUT]: 依赖 tianlong.persistence 的 InMemoryWorldStore / Neo4jWorldStore / net_relation_diff / TurnEnvelope / RequestConflict，
         tianlong.runtime.authority
[OUTPUT]: WorldStore 契约测试：两种后端跑同一组断言——往返一致、幂等、版本冲突、outbox、跨后端确定性、
          请求进度与会话运行态随提交同事务落库、叙述只补写一次、存档版本往返、
          请求绑定在提交内检查（重复投递 / 同 ID 异内容 / 接不上的进度 / 已完结请求的提交整体回滚）
[POS]: tests 的持久化层；Neo4j 用例在 NEO4J_URI 不可达时自动跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import pytest

from tianlong.cognition import BeliefStore
from tianlong.core import Intent, Op, Rel, Relation
from tianlong.persistence import (
    CommitBatch,
    InMemoryWorldStore,
    RequestConflict,
    SQLiteWorldStore,
    TurnEnvelope,
    VersionConflict,
)
from tianlong.persistence.store import ChoiceConflict, ChoiceUse
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_warehouse

SCRIPT = [
    [("player", Op.TAKE, "key"), ("guard", Op.MOVE, "warehouse", "door_main")],
    [("player", Op.UNLOCK, "door_store", "key"), ("guard", Op.INSPECT, "player")],
    [("player", Op.MOVE, "storeroom", "door_store"), ("guard", Op.TELL, "player")],   # 语法非法 → 被拒也要可回放
    [("player", Op.PUT, "storeroom", "key")],
]


def _neo4j_store():
    pytest.importorskip("neo4j")
    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI 未配置")
    from neo4j.exceptions import ServiceUnavailable

    from tianlong.persistence.neo4j_store import Neo4jWorldStore
    try:
        return Neo4jWorldStore.from_env()
    except (ServiceUnavailable, OSError) as e:  # pragma: no cover
        pytest.skip(f"Neo4j 不可达: {e}")


@pytest.fixture(params=["memory", "sqlite", pytest.param("neo4j", marks=pytest.mark.neo4j)])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryWorldStore()
        return
    if request.param == "sqlite":
        yield SQLiteWorldStore(tmp_path / "world.sqlite3")
        return
    s = _neo4j_store()
    s.created = []  # type: ignore[attr-defined]
    yield s
    for ref in s.created:  # type: ignore[attr-defined]
        s.drop_world(ref)
    s.close()


def found(store, versions=None):
    sc = replace(build_warehouse(), world_id=f"t-{uuid.uuid4().hex[:8]}")
    auth = WorldAuthority.found(store, sc, versions=versions)
    if hasattr(store, "created"):
        store.created.append(auth.ref)
    return sc, auth


def play(auth):
    for tick, specs in enumerate(SCRIPT):
        v = auth.head().version
        auth.settle([Intent(f"s{tick}-{s[0]}", s[0], s[1], *s[2:], based_on=v) for s in specs])


def test_create_round_trip(store):
    sc, auth = found(store)
    assert auth.head().fingerprint() == sc.state.fingerprint()
    for agent in sc.profiles:
        expected = BeliefStore(agent, trust=dict(sc.profiles[agent].trust)).revise_all(sc.priors[agent])[0]
        assert store.beliefs(auth.ref, agent) == expected
    with pytest.raises(ValueError):
        store.create(auth.ref, sc.state, {})


def test_backends_agree_after_play(store):
    _, auth = found(store)
    play(auth)
    _, ref_auth = found(InMemoryWorldStore())
    play(ref_auth)
    assert auth.head().fingerprint() == ref_auth.head().fingerprint()
    assert [e.id for e in store.events(auth.ref)] == [e.id for e in ref_auth.store.events(ref_auth.ref)]
    for agent in ("player", "guard", "captain"):
        assert store.beliefs(auth.ref, agent) == ref_auth.store.beliefs(ref_auth.ref, agent)
    assert auth.head().target("key", Rel.AT) == "storeroom"


def test_idempotency_and_conflict(store):
    _, auth = found(store)
    it = Intent("once", "player", Op.TAKE, "key", based_on=0)
    auth.settle([it])
    assert auth.settle([it]).replayed
    assert store.event_for_intent(auth.ref, "once").outcome.value == "success"
    head = store.head(auth.ref)
    stale = auth.kernel.step(head, [Intent("x", "guard", Op.WAIT, based_on=head.version)])
    auth.settle([Intent("y", "guard", Op.WAIT, based_on=head.version)])
    with pytest.raises(VersionConflict):
        store.commit(CommitBatch(auth.ref, head.version, stale.state, stale.events, (), {}, ()))


def test_outbox_cycle(store):
    _, auth = found(store)
    auth.settle([Intent("t", "player", Op.TAKE, "key", based_on=0)])
    mine = [m for m in store.pending_memories(1000) if m.world_id == auth.ref.world_id]
    assert any(m.owner == "guard" and "响动" in m.text for m in mine)
    store.mark_indexed([m.id for m in mine])
    assert not [m for m in store.pending_memories(1000) if m.world_id == auth.ref.world_id]
    assert store.recent_memories(auth.ref, "guard", 0)


def test_net_relation_diff_folds_chains():
    pytest.importorskip("neo4j")
    from tianlong.core import relocate
    from tianlong.persistence.neo4j_store import net_relation_diff
    changes = [*relocate("key", "table", "player"), *relocate("key", "player", "guard")]
    removed, added = net_relation_diff(changes)
    assert removed == {Relation("key", Rel.AT, "table")}
    assert added == {Relation("key", Rel.AT, "guard")}


def test_request_progress_and_session_state_ride_the_commit(store):
    versions = {"save": "save-v2", "kernel": "kernel-v1"}
    _, auth = found(store, versions)
    assert store.save_versions(auth.ref) == versions
    assert store.request(auth.ref, "r1") is None and store.session_state(auth.ref) is None
    it = Intent("r1-t0", "player", Op.TAKE, "key", based_on=0)
    seen = {}

    def annotate(s):
        mine = tuple(o.percept for o in s.observations_of("player"))
        seen["env"] = TurnEnvelope("r1", "h1", it, 2, 0, 0, (s.state.version,), (s.state.clock,), mine, ("key",))
        return seen["env"], {"scheduler": {"guard": [480, True]}, "described": ["key"]}

    auth.settle([it], annotate)
    assert store.request(auth.ref, "r1") == seen["env"], "感知、意图与进度逐字段往返"
    assert store.session_state(auth.ref) == {"scheduler": {"guard": [480, True]}, "described": ["key"]}
    store.record_render(auth.ref, "r1", "你拿起钥匙")
    store.record_render(auth.ref, "r1", "另一种说法")
    assert store.request(auth.ref, "r1").narration == "你拿起钥匙", "叙述只补写一次"
    with pytest.raises(KeyError):
        store.record_render(auth.ref, "nope", "……")
    # 下一次提交整份替换进度（接在已落库的那一份之后），但不抹掉已落库的叙述；不带附注的提交不动会话运行态
    def finish(s):
        env = seen["env"]
        seen["done"] = replace(env, versions=(*env.versions, s.state.version), ticks=(*env.ticks, s.state.clock),
                               done=True)
        return seen["done"], None

    head = auth.head()
    auth.settle([Intent("r1-t1", "player", Op.WAIT, based_on=head.version)], finish)
    assert store.request(auth.ref, "r1") == replace(seen["done"], narration="你拿起钥匙")
    assert store.session_state(auth.ref) == {"scheduler": {"guard": [480, True]}, "described": ["key"]}


def test_request_binding_is_checked_inside_the_commit(store):
    """同一 request_id 的两次提交交错：进度必须恰好接在已落库的那一份之后，否则整次提交回滚——
    不会二次结算，不会改写别人的绑定，同 ID 异内容也逃不过（检查与写入在同一临界区 / 事务里）。"""
    _, auth = found(store)
    base = Intent("rq-t0", "player", Op.TAKE, "key", based_on=0)
    envs = {}

    def progress(key, payload, prior_versions=(), done=False):
        def annotate(s):
            envs[key] = TurnEnvelope("rq", payload, base, 2, 0, 0, (*prior_versions, s.state.version), done=done)
            return envs[key], {"described": [key]}
        return annotate

    def refused(intent_id, annotate):
        head, n_events = store.head(auth.ref), len(store.events(auth.ref))
        with pytest.raises(RequestConflict):
            auth.settle([Intent(intent_id, "guard", Op.WAIT, based_on=head.version)], annotate)
        assert store.head(auth.ref).fingerprint() == head.fingerprint(), "被拒的提交一处不改：世界原样"
        assert len(store.events(auth.ref)) == n_events and store.event_for_intent(auth.ref, intent_id) is None

    auth.settle([base], progress("a", "h-take"))                                  # A：首 tick 建立绑定
    assert store.request(auth.ref, "rq") == envs["a"]
    refused("dup-1", progress("b", "h-take"))                                     # B：同内容重复投递，也想“首 tick”
    refused("dup-2", progress("c", "h-wait"))                                     # 同 ID 异内容
    refused("dup-3", progress("d", "h-take", prior_versions=(7,)))                # 接不上已落库的进度
    assert store.request(auth.ref, "rq") == envs["a"], "绑定没被改写"
    assert store.session_state(auth.ref) == {"described": ["a"]}, "会话运行态随被拒的提交一起回滚"
    auth.settle([Intent("rq-t1", "player", Op.WAIT, based_on=1)], progress("e", "h-take", envs["a"].versions, True))
    assert store.request(auth.ref, "rq").versions == (1, 2) and store.request(auth.ref, "rq").done
    refused("dup-4", progress("f", "h-take", envs["e"].versions))                 # 已完结：不再接受任何进度
    assert store.head(auth.ref).version == 2


def test_decision_publication_and_first_tick_consumption_are_atomic(store):
    _, auth = found(store)
    snapshot = {"schema": 1, "id": "decision-0", "version": 0,
                "choices": [{"id": "take-key", "label": "拿起钥匙"}], "consumed_request": None}
    assert store.decision(auth.ref) is None
    assert store.publish_decision(auth.ref, snapshot, 0) == snapshot
    assert store.publish_decision(auth.ref, {**snapshot, "choices": []}, 0) == snapshot
    it = Intent("choice-t0", "player", Op.TAKE, "key", based_on=0)

    def annotate(choice_id):
        def progress(s):
            return TurnEnvelope("choice-r", "payload", it, 1, 0, 0, versions=(s.state.version,), done=True,
                                choice=ChoiceUse("decision-0", choice_id), command="拿起钥匙"), {"choice-test": True}
        return progress

    with pytest.raises(ChoiceConflict):
        auth.settle([it], annotate("fake-choice"))
    assert auth.head().version == 0 and auth.head().target("key", Rel.AT) == "table"
    assert store.decision(auth.ref) == snapshot and store.request(auth.ref, "choice-r") is None
    assert store.session_state(auth.ref) is None and not store.events(auth.ref)
    auth.settle([it], annotate("take-key"))
    assert auth.head().version == 1 and auth.head().target("key", Rel.AT) == "player"
    assert store.decision(auth.ref)["consumed_request"] == "choice-r"
    assert store.request(auth.ref, "choice-r").choice == ChoiceUse("decision-0", "take-key")
    assert store.request(auth.ref, "choice-r").command == "拿起钥匙"
    assert store.session_state(auth.ref) == {"choice-test": True}
    with pytest.raises(VersionConflict):
        store.publish_decision(auth.ref, snapshot, 0)
