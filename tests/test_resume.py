"""
[INPUT]: 依赖 tianlong.core 的 FrozenMap / WorldState，tianlong.persistence 的 InMemoryWorldStore / RequestConflict（Neo4j 按需），
         tianlong.runtime 的 GameSession / versions，tianlong.scenarios 的 build_warehouse / build_wuliang
[OUTPUT]: 持久化与恢复验收 R01–R04：快照映射不可就地修改且可 pickle；连续运行与中途关闭（含首回合前关闭）再读档的事件、调度、
          认知、叙述逐项一致（内存后端必跑，Neo4j 可达时同跑）；同请求返回既有结果、同 ID 异内容显式冲突；
          并发的重复投递在查询与结算之间插入也只结算一次，重试仍在进行的多 tick 等待不多走 tick；
          提交后索引/叙述崩溃的重试不二次结算，多 tick 等待中途崩溃只走剩下的 tick；存档版本不一致即拒绝，除非显式迁移；
          NPC 长期记忆摘要的增量缓存与从全部记录重建、与读档后重建逐项相同
[POS]: tests 的恢复层；证明“重试与读档”不会让世界多走一步、也不会让角色忘掉自己的节奏
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import uuid
from dataclasses import replace

import pytest

from tianlong.core import FrozenMap, Op, Rel
from tianlong.persistence import InMemoryWorldStore, RequestConflict, SQLiteWorldStore, WorldRef

# ============================================================
#  R01：快照不可变
# ============================================================


def test_snapshot_mappings_reject_mutation(authority):
    st = authority.head()
    mind = authority.store.beliefs(authority.ref, "player")
    key = st.entity("key")
    for mapping in (st.entities, st._out, mind.entities, mind.beliefs, mind.trust):
        assert isinstance(mapping, FrozenMap)
        some = next(iter(mapping), "x")
        for attempt in (
            lambda m=mapping: m.__setitem__("x", None),
            lambda m=mapping, k=some: m.__delitem__(k),
            lambda m=mapping: m.update(x=None),
            lambda m=mapping, k=some: m.pop(k),
            lambda m=mapping: m.popitem(),
            lambda m=mapping: m.clear(),
            lambda m=mapping: m.setdefault("x", None),
        ):
            with pytest.raises(TypeError):
                attempt()
    with pytest.raises(TypeError):
        mind.trust |= {"captain": 0.0}
    with pytest.raises(TypeError):
        st.entities["key"] = key.with_attr("small", False)
    # 拷贝一份自己的可以随便改，快照纹丝不动；改世界只能经 apply() 形成新版本
    mine = dict(st.entities)
    mine.pop("key")
    assert "key" in st.entities and authority.head().entity("key") == key
    assert st.apply([]).entities == st.entities


def test_snapshots_survive_pickle_deepcopy_and_json(authority, act):
    act(("player", Op.TAKE, "key"))
    st = authority.head()
    mind = authority.store.beliefs(authority.ref, "guard")
    for obj in (st, mind):
        for clone in (pickle.loads(pickle.dumps(obj)), copy.deepcopy(obj)):
            assert clone == obj and isinstance(clone.entities, FrozenMap)
    assert pickle.loads(pickle.dumps(st)).fingerprint() == st.fingerprint()
    assert pickle.loads(pickle.dumps(st)).target("key", Rel.AT) == "player", "邻接索引随快照一起复原"
    assert json.loads(json.dumps(mind.trust)) == dict(mind.trust)
    assert FrozenMap({"a": 1}) == {"a": 1} and hash(FrozenMap({"a": 1})) == hash(FrozenMap({"a": 1}))


# ============================================================
#  会话层用例需要 LangGraph + Qdrant
# ============================================================

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.versions import IncompatibleSave  # noqa: E402
from tianlong.scenarios import build_warehouse, build_wuliang  # noqa: E402

# ============================================================
#  R02：读档等价——连续运行 vs 中途关闭再读档
# ============================================================

COMMANDS = ["等待", "等待", "去后院", "等待", "等待", "去大殿", "等待", "等待"]
SPLIT = 4


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


def _play(scenario, store, split=None, reopen=None):
    """逐条输入命令；split 处丢掉会话对象，另开一个读档接续。reopen 返回读档时使用的存储（默认同一个）。"""
    s = GameSession(scenario, store=store)
    s.intro()
    trace = []
    for i, cmd in enumerate(COMMANDS):
        if i == split:
            s = GameSession(scenario, store=reopen() if reopen else store)
            assert s.resumed
        r = s.turn(cmd)
        trace.append((r.narration, tuple(sorted(d.agent for d in r.deliberations))))
    events = [(e.id, e.op, e.outcome, e.changes) for e in s.store.events(s.ref)]
    minds = {a: s.store.beliefs(s.ref, a) for a in scenario.profiles}
    return events, minds, trace, s.session_state(), s.authority.head().fingerprint()


@pytest.mark.parametrize("split", [SPLIT, 0], ids=["mid-game", "right-after-opening"])
@pytest.mark.parametrize("backend", ["memory", "sqlite", pytest.param("neo4j", marks=pytest.mark.neo4j)])
def test_resume_is_equivalent_to_continuous_play(backend, split, tmp_path):
    """split=0：看完开场、第一回合提交之前就关闭——此时还没有任何提交带着会话运行态落库，开场的初见描写也不许重来。"""
    scenario = replace(build_wuliang(), world_id=f"r02-{uuid.uuid4().hex[:8]}")
    continuous = _play(scenario, InMemoryWorldStore())
    if backend == "memory":
        resumed = _play(scenario, InMemoryWorldStore(), split=split)
    elif backend == "sqlite":
        path = tmp_path / "save.sqlite3"
        resumed = _play(scenario, SQLiteWorldStore(path), split=split, reopen=lambda: SQLiteWorldStore(path))
    else:
        first, second = _neo4j_store(), _neo4j_store()        # 读档用新的连接：模拟另一个进程
        try:
            resumed = _play(scenario, first, split=split, reopen=lambda: second)
        finally:
            first.drop_world(WorldRef(scenario.world_id))
            first.close()
            second.close()
    events, minds, trace, state, fingerprint = resumed
    assert events == continuous[0], "结构化事件流（ID、操作、结果、变化）逐条一致"
    assert minds == continuous[1], "所有角色的认知一致"
    assert [t[1] for t in trace] == [t[1] for t in continuous[2]], "每一回合谁在做完整决策（调度）一致"
    assert [t[0] for t in trace] == [t[0] for t in continuous[2]], "叙述一致：读档后不会重复初见描写"
    assert state == continuous[3] and fingerprint == continuous[4]


def test_resume_without_session_state_would_diverge(monkeypatch):
    """反例对照：若不恢复会话运行态，读档那一刻所有 NPC 都“该决策了”，初见描写也重来一遍。"""
    scenario = build_wuliang()
    continuous = _play(scenario, InMemoryWorldStore())
    store = InMemoryWorldStore()

    def forgetful():                                                # 同一份数据，只是读不到会话运行态
        monkeypatch.setattr(store, "session_state", lambda ref: None)
        return store

    lost = _play(scenario, store, split=SPLIT, reopen=forgetful)
    assert [t[1] for t in lost[2]] != [t[1] for t in continuous[2]]
    assert [t[0] for t in lost[2]] != [t[0] for t in continuous[2]]


# ============================================================
#  R04：同请求返回既有结果；同 ID 异内容显式冲突
# ============================================================


def test_same_request_returns_persisted_result():
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    first = s.turn("拿走桌上的钥匙", request_id="req-take")
    version, n_events = s.authority.head().version, len(store.events(s.ref))
    for again in (s.turn("拿走桌上的钥匙", request_id="req-take"),
                  GameSession(build_warehouse(), store=store).turn("拿走桌上的钥匙", request_id="req-take")):
        assert again.replayed and again.narration == first.narration
        assert [e.id for e in again.events] == [e.id for e in first.events]
        assert s.authority.head().version == version and len(store.events(s.ref)) == n_events, "没有执行任何东西"
    env = store.request(s.ref, "req-take")
    assert env.done and env.versions == (version,) and env.narration == first.narration


def test_same_id_different_payload_conflicts():
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    s.turn("拿走桌上的钥匙", request_id="req-1")
    version = s.authority.head().version
    with pytest.raises(RequestConflict):
        s.turn("等待", request_id="req-1")
    assert s.authority.head().version == version
    s.turn("等待")
    s.turn("等待")
    assert s.authority.head().version == version + 2, "不带 request_id 的请求保持原有行为：每次都推进"


def _inject_before_settle(monkeypatch, store, duplicate):
    """在本次请求的 request() 检查之后、结算之前，让另一次投递整个跑完（模拟并发的重复投递）。"""
    real = store.request
    fired = []

    def racing(ref, rid):
        prior = real(ref, rid)
        if not fired:
            fired.append(True)                           # 先置位：另一次投递自己的 request() 检查不再触发
            fired.append(duplicate())
        return prior

    monkeypatch.setattr(store, "request", racing)
    return fired


@pytest.mark.parametrize("same_session", [True, False], ids=["one-session", "two-sessions"])
def test_duplicate_delivery_in_flight_is_settled_once(monkeypatch, same_session):
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    other = s if same_session else GameSession(build_warehouse(), store=store)
    fired = _inject_before_settle(monkeypatch, store, lambda: other.turn("拿走桌上的钥匙", request_id="req-x"))
    late = s.turn("拿走桌上的钥匙", request_id="req-x")
    first = fired[1]
    assert store.head(s.ref).version == 1 and len(store.events(s.ref)) == len(first.events), "世界只结算一次"
    assert late.replayed and late.narration == first.narration == "你拿起钥匙"
    assert [e.id for e in late.events] == [e.id for e in first.events]
    env = store.request(s.ref, "req-x")
    assert env.versions == (1,) and env.done and env.narration == first.narration


@pytest.mark.parametrize("same_session", [True, False], ids=["one-session", "two-sessions"])
def test_duplicate_id_with_other_payload_in_flight_conflicts(monkeypatch, same_session):
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    other = s if same_session else GameSession(build_warehouse(), store=store)
    fired = _inject_before_settle(monkeypatch, store, lambda: other.turn("拿走桌上的钥匙", request_id="req-x"))
    with pytest.raises(RequestConflict):
        s.turn("等待", request_id="req-x")
    assert store.head(s.ref).version == 1, "异内容的那一次一处不改"
    env = store.request(s.ref, "req-x")
    assert env.versions == (1,) and env.narration == fired[1].narration, "绑定与进度仍是先到的那一份"
    monkeypatch.undo()
    again = s.turn("拿走桌上的钥匙", request_id="req-x")
    assert again.replayed and again.narration == fired[1].narration
    with pytest.raises(RequestConflict):
        s.turn("等待", request_id="req-x")


@pytest.mark.parametrize("where", ["between-ticks", "same-version"])
def test_retry_during_multi_tick_wait_runs_no_extra_ticks(monkeypatch, where):
    """客户端超时重试一个仍在进行的“等一会”：两次执行合起来也只走计划的十个 tick，进度与叙述都与不被打扰时一致。"""
    ref_session = GameSession(build_warehouse())
    reference = ref_session.turn("等一会")
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    retried = []

    def retry():
        retried.append(GameSession(build_warehouse(), store=store).turn("等一会", request_id="req-wait"))

    if where == "between-ticks":                                     # 第 3 个 tick 提交之后，重试到达并走完剩下的 tick
        real, calls = s.indexer.drain, {"n": 0}

        def drain(*a, **kw):
            out = real(*a, **kw)
            calls["n"] += 1
            if calls["n"] == 3:
                retry()
            return out

        monkeypatch.setattr(s.indexer, "drain", drain)
    else:                                                            # 第 4 个 tick 已读过版本、尚未结算时，重试抢先走完
        real, calls = s.orchestrator.decide, {"n": 0}

        def decide(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 4:
                retry()
            return real(*a, **kw)

        monkeypatch.setattr(s.orchestrator, "decide", decide)
    first = s.turn("等一会", request_id="req-wait")
    assert store.head(s.ref).version == 10, "只走计划的十个 tick，不多不少"
    assert store.request(s.ref, "req-wait").versions == tuple(range(1, 11))
    assert [e.id for e in store.events(s.ref)] == [e.id for e in ref_session.store.events(ref_session.ref)]
    assert first.narration == retried[0].narration == reference.narration
    assert store.request(s.ref, "req-wait").narration == reference.narration
    assert not first.replayed and not retried[0].replayed, "两次调用都真的提交过 tick"


def test_unparsed_request_is_not_persisted():
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    r = s.turn("跳个舞", request_id="req-dance")
    assert not r.advanced and store.request(s.ref, "req-dance") is None and s.authority.head().version == 0


# ============================================================
#  R03：提交后、索引或叙述之前崩溃 → 重试返回既有结算，不二次执行
# ============================================================


def _fail_once(monkeypatch, obj, name, after=0):
    """第 after+1 次调用时抛出一次异常，其余照常。"""
    real = getattr(obj, name)
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == after + 1:
            raise RuntimeError("注入的故障")
        return real(*a, **kw)

    monkeypatch.setattr(obj, name, flaky)
    return calls


@pytest.mark.parametrize("fault", ["indexer", "narrator"])
def test_crash_after_commit_is_not_settled_twice(monkeypatch, fault):
    reference = GameSession(build_warehouse()).turn("拿走桌上的钥匙")
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    target = (s.indexer, "drain") if fault == "indexer" else (s.narrator, "narrate_scene")
    _fail_once(monkeypatch, *target)
    with pytest.raises(RuntimeError, match="注入的故障"):
        s.turn("拿走桌上的钥匙", request_id="req-crash")
    env = store.request(s.ref, "req-crash")
    assert env.done and env.narration is None, "世界已提交，文字尚未落库"
    assert s.authority.head().version == 1 and s.authority.head().target("key", Rel.AT) == "player"

    retry = GameSession(build_warehouse(), store=store).turn("拿走桌上的钥匙", request_id="req-crash")
    assert retry.replayed and retry.render is not None, "只按已持久化的感知重写文字"
    assert retry.narration == reference.narration
    assert store.head(s.ref).version == 1 and len(store.events(s.ref)) == len(reference.events), "没有二次结算"
    assert store.request(s.ref, "req-crash").narration == reference.narration
    again = GameSession(build_warehouse(), store=store).turn("拿走桌上的钥匙", request_id="req-crash")
    assert again.replayed and again.render is None and again.narration == reference.narration


def test_interrupted_multi_tick_wait_continues_only_remaining_ticks(monkeypatch):
    ref_session = GameSession(build_warehouse())
    reference = ref_session.turn("等一会")
    ticks = ref_session.authority.head().version
    assert ticks == 10, "仓库里无人打扰：等满十分钟"

    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    _fail_once(monkeypatch, s.indexer, "drain", after=2)            # 第 3 个 tick 提交之后崩溃
    with pytest.raises(RuntimeError, match="注入的故障"):
        s.turn("等一会", request_id="req-wait")
    env = store.request(s.ref, "req-wait")
    assert not env.done and env.versions == (1, 2, 3) and store.head(s.ref).version == 3

    retry = GameSession(build_warehouse(), store=store).turn("等一会", request_id="req-wait")
    assert not retry.replayed and store.head(s.ref).version == ticks, "只走剩下的 7 个 tick"
    assert [e.id for e in store.events(s.ref)] == [e.id for e in ref_session.store.events(ref_session.ref)]
    assert [e.id for e in retry.events] == [e.id for e in reference.events]
    assert retry.narration == reference.narration
    assert store.request(s.ref, "req-wait").versions == tuple(range(1, ticks + 1))


# ============================================================
#  存档版本：不一致即拒绝，迁移必须显式
# ============================================================


def test_incompatible_save_is_refused_unless_migration_is_explicit():
    store = InMemoryWorldStore()
    s = GameSession(build_warehouse(), store=store)
    s.turn("拿走桌上的钥匙")
    assert store.save_versions(s.ref) == s.versions
    store._worlds[s.ref].versions["kernel"] = "kernel-v0"           # 模拟旧规则下建的档
    with pytest.raises(IncompatibleSave, match="kernel"):
        GameSession(build_warehouse(), store=store)
    migrated = GameSession(build_warehouse(), store=store, allow_migration=True)
    assert migrated.resumed and migrated.migrated_from["kernel"] == "kernel-v0"
    assert store.save_versions(s.ref)["kernel"] == "kernel-v0", "迁移不改写旧档的版本记录"


def test_incremental_memory_view_equals_rebuild_from_all_records():
    """同一 tick 的经历分两次提交写下（本 tick 末的环顾 + 下一次结算）：增量缓存与读档后从全部记录重建必须逐项相同。"""
    from tianlong.memory.view import MemoryView
    scenario = replace(build_wuliang(), world_id=f"mv-{uuid.uuid4().hex[:8]}")
    store = InMemoryWorldStore()
    s = GameSession(scenario, store=store)
    s.intro()
    for cmd in ["问钟灵长剑在哪", "告诉钟灵长剑在兵器架", "问马五德易经在哪", "等待", "问左子穆长剑在哪", "等待",
                "告诉左子穆易经在兵器架", "去后院", "等待", "等待"]:
        s.turn(cmd)
    now = s.authority.head().clock
    reopened = GameSession(scenario, store=store)
    for a in scenario.npcs:
        full = MemoryView.from_records(store.recent_memories(s.ref, a, 0), now)
        assert s._memory_view(a, now) == full, a
        assert reopened._memory_view(a, now) == full, a
