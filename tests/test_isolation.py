"""
[INPUT]: 依赖 tianlong.kernel / cognition / scenarios，conftest 的 make_intent
[OUTPUT]: 认知隔离的性质测试：扰动角色未观察到的世界事实，其认知与认知视图必须逐字节不变；
          看点识别（runtime/staging）只收玩家的感知、引擎里只由会话调用，从不进入 NPC 的决策路径；
          相识账本（runtime/names）同样只由会话调用、Situation 里没有它
[POS]: tests 的隔离层；“没收到消息的人不能提前知道结果”被写成可证伪的断言
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from tianlong.cognition import BeliefStore, belief_view
from tianlong.core import Entity, Kind, Op, Percept, Rel, Relation, WorldState
from tianlong.kernel import Kernel
from tianlong.scenarios import build_warehouse

from .conftest import make_intent


def _perturbed(variant: str) -> WorldState:
    """在守卫看不到的地方改动真相。"""
    s = build_warehouse().state
    ents = dict(s.entities)
    rels = set(s.relations)
    if variant == "ledger_moved":
        rels.remove(Relation("ledger", Rel.AT, "storeroom"))
        rels.add(Relation("ledger", Rel.AT, "harbor"))
    elif variant == "door_unlocked":
        ents["door_store"] = ents["door_store"].with_attr("locked", False)
    elif variant == "extra_secret":
        ents["gold"] = Entity.make("gold", Kind.ITEM, "金条")
        rels.add(Relation("gold", Rel.AT, "storeroom"))
    return WorldState.build(s.seed, s.clock, ents.values(), rels)


def _guard_after(state: WorldState, ticks: int = 3) -> BeliefStore:
    sc = build_warehouse()
    store = BeliefStore("guard").revise_all(sc.priors["guard"])[0]
    k = Kernel()
    for _ in range(ticks):
        r = k.step(state, [make_intent("guard", Op.WAIT, based_on=state.version, intent_id=f"w{state.version}")])
        store = store.revise_all(o.percept for o in r.observations if o.observer == "guard")[0]
        state = r.state
    return store


@pytest.mark.parametrize("variant", ["ledger_moved", "door_unlocked", "extra_secret"])
def test_unobserved_truth_does_not_leak(variant):
    base = _guard_after(build_warehouse().state)
    other = _guard_after(_perturbed(variant))
    assert base == other
    assert belief_view(base, 999) == belief_view(other, 999)


def test_percept_has_no_provenance_field():
    names = {f.name for f in fields(Percept)}
    assert "source_event" not in names and "id" not in names, "角色拿到的感知不能顺着 ID 摸到真相"


def test_sound_percept_reveals_nothing_about_actor():
    s = build_warehouse().state
    r = Kernel().step(s, [make_intent("player", Op.TAKE, "key", based_on=0)])
    heard = [o.percept for o in r.observations if o.observer == "guard" and o.percept.event]
    assert heard and heard[0].facts == () and {sk.id for sk in heard[0].sketches} <= {"warehouse"}


def test_staging_reads_only_the_players_percepts_and_is_called_only_by_the_session():
    """看点识别读真相之外的东西（玩家自己的感知）也只为呈现：引擎里只有会话调用它，NPC 的决策路径（agents、cognition）从不碰它。"""
    src = Path(__file__).resolve().parents[1] / "src" / "tianlong"
    users = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                   if p.name != "staging.py" and re.search(
                       r"^from tianlong\.runtime import .*\bstaging\b|tianlong\.runtime\.staging", p.read_text("utf-8"), re.M))
    assert users == ["runtime/session.py"], users
    import inspect

    from tianlong.runtime import staging
    for fn in (staging.recognize, staging.stops_wait, staging.witnessed):
        assert "WorldState" not in str(inspect.signature(fn)), "只收感知，不收世界"


def test_names_are_called_only_by_the_session_and_never_reach_a_situation():
    """相识账本与展示用的外貌称呼只为呈现：引擎里只有会话调用 runtime/names，NPC 的决策路径（agents、cognition）从不碰它，
    Situation 里没有账本——NPC 叫得出谁的名字只影响主持人替他写的台词，不影响他做什么。"""
    src = Path(__file__).resolve().parents[1] / "src" / "tianlong"
    users = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                   if p.name != "names.py" and re.search(
                       r"^from tianlong\.runtime import .*\bnames\b|tianlong\.runtime\.names", p.read_text("utf-8"), re.M))
    assert users == ["runtime/session.py"], users
    from tianlong.agents.policy_kit import Situation
    assert not any("names" in f.name or "acq" in f.name or "Acquaintance" in str(f.type) for f in fields(Situation))
