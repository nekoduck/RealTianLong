"""
[INPUT]: 依赖 runtime/authority 的 WorldAuthority，persistence 的 InMemoryWorldStore，agents 的 ScriptedPolicy / HeuristicPredictor /
         Situation / hush_chatter，cognition 的 candidates，core 的 WorldState.fingerprint() / digest / make_id，
         scenarios 的 build_warehouse / build_wuliang，learning/task 的 TaskConfig（程序化世界），runtime/session 的 GameSession（模板模式），
         tests/data/playthrough_duanyu.json（旧版评测整局游玩的输入）与 tests/data/goldens.json（钉住的指纹）
[OUTPUT]: 金标准指纹验收：(a) 仓库——玩家拿钥匙、开内仓门、进内仓，NPC 照脚本策略反应，共 50 tick；(b) 程序化世界种子 0–9
          （jianghu=1）在脚本策略下各走 200 tick；(c) 旧版无量山 build_wuliang(7)、段誉作玩家、评测整局游玩的前 30 回合（模板模式，
          不接模型）——每一项的世界指纹（WorldState.fingerprint）与事件日志摘要逐字节等于 goldens.json；
          写入：从仓库根目录 PYTHONPATH=src python -m tests.test_goldens --write
[POS]: tests 的守护层（设计 §6）：之后的里程碑不许改动内核、ScriptedPolicy、候选规则与程序化世界的行为。
       NPC 一侧照会话的决策图走（候选 → 启发式预测 → 脚本策略 → 每处每 tick 至多一句闲谈），只是不经 LangGraph、不调措辞器——
       台词措辞不在钉住之列；(c) 走真实会话，缺 LangGraph / Qdrant 时跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from tianlong.agents.policies import CHATTER, ScriptedPolicy, Situation, hush_chatter
from tianlong.agents.predictors import HeuristicPredictor
from tianlong.cognition import Candidate, candidates
from tianlong.core import Event, Intent, Op, digest, make_id
from tianlong.learning.task import TaskConfig
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import Scenario, build_warehouse, build_wuliang

GOLDENS = Path(__file__).with_name("data") / "goldens.json"
PLAYTHROUGH = Path(__file__).with_name("data") / "playthrough_duanyu.json"     # 与评测探针文件脱钩：探针改版不动金标准
SEEDS = range(10)
TICKS = 200
WULIANG_TURNS = 30
# 仓库：玩家前三个 tick 的行动（与验收用例同一条路），此后原地等待；NPC 全程照脚本策略
WAREHOUSE_SCRIPT = (Candidate(Op.TAKE, "key"), Candidate(Op.UNLOCK, "door_store", "key"),
                    Candidate(Op.MOVE, "storeroom", "door_store"))
WAREHOUSE_TICKS = 50


# ============================================================
#  摘要：世界指纹 + 事件日志（台词措辞不算，言语的命题与言语行为算）
# ============================================================


def _events(events: Iterable[Event]) -> str:
    return digest(*((e.id, e.tick, e.place, e.outcome.value, e.reason, e.intent.actor, e.intent.op.value,
                     e.intent.target, e.intent.obj, e.intent.manner.value, e.intent.topic,
                     e.intent.social.value if e.intent.social else None) for e in events))


def _summary(auth: WorldAuthority, **extra: Any) -> dict[str, Any]:
    head = auth.head()
    return {"world": head.fingerprint(), "events": _events(auth.store.events(auth.ref)),
            "version": head.version, "clock": head.clock, **extra}


# ============================================================
#  不经 LangGraph 的 NPC 决策：候选 → 启发式预测 → 脚本策略 → 闲谈让出话头（与会话的决策图同一口径）
# ============================================================


def _tick(auth: WorldAuthority, sc: Scenario, fixed: Mapping[str, Candidate]) -> None:
    head, ref = auth.head(), auth.ref
    policy, predictor = ScriptedPolicy(), HeuristicPredictor()
    chosen: dict[str, tuple[str | None, str, Candidate]] = {}
    for a in sorted(sc.profiles):
        if a in fixed:
            chosen[a] = (None, "", fixed[a])
            continue
        me, prof = auth.store.beliefs(ref, a), sc.profiles[a]
        if prof.is_player:                                   # 玩家不在脚本里的 tick 原地等待
            chosen[a] = (None, "", Candidate(Op.WAIT))
            continue
        interests = list(prof.interests())
        cands = candidates(me, interests, 64)
        preds = tuple(predictor.predict(me, head.clock, cands, interests, profile=prof))
        choice = policy.choose(Situation(a, prof, me, head.clock, cands, preds, player=sc.player))
        chosen[a] = (me.location_of(a), choice.tag, choice.chosen(cands, me))
    talking = {a: v for a, v in chosen.items() if a not in fixed and v[2].op in (Op.TELL, Op.ASK)}
    hushed = hush_chatter(head.clock, talking) if any(t == CHATTER for _, t, _ in talking.values()) else frozenset()
    intents = [Intent(make_id("int", ref.world_id, ref.branch_id, a, head.version), a, Op.WAIT, based_on=head.version)
               if a in hushed else c.to_intent(make_id("int", ref.world_id, ref.branch_id, a, head.version), a, head.version)
               for a, (_, _, c) in chosen.items()]
    auth.settle(intents)


def _simulate(sc: Scenario, ticks: int, script: tuple[Candidate, ...] = ()) -> WorldAuthority:
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    for t in range(ticks):
        _tick(auth, sc, {sc.player: script[t]} if sc.player and t < len(script) else {})
    return auth


def warehouse() -> dict[str, Any]:
    return _summary(_simulate(build_warehouse(), WAREHOUSE_TICKS, WAREHOUSE_SCRIPT))


def procedural(seed: int) -> dict[str, Any]:
    return _summary(_simulate(TaskConfig(jianghu=1.0).scenario(seed), TICKS))


def wuliang() -> dict[str, Any]:
    """旧版无量山、段誉作玩家，评测整局游玩的前 30 回合，模板模式（不接模型：解释退回规则解析、叙述走模板）。"""
    from tianlong.runtime.session import GameSession
    s = GameSession(build_wuliang(7))
    s.intro()
    played = 0
    for text in json.loads(PLAYTHROUGH.read_text("utf-8"))[:WULIANG_TURNS]:
        played += 1
        if s.turn(text).ending is not None:
            break
    return _summary(s.authority, turns=played, ending=s.ending.key if s.ending else None)


def compute() -> dict[str, Any]:
    return {"warehouse": warehouse(), **{f"procedural/{sd}": procedural(sd) for sd in SEEDS}, "wuliang": wuliang()}


def write(path: Path = GOLDENS) -> dict[str, Any]:
    got = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return got


# ============================================================
#  验收
# ============================================================


def _golden(key: str) -> dict[str, Any]:
    return json.loads(GOLDENS.read_text("utf-8"))[key]


def test_warehouse_golden():
    assert warehouse() == _golden("warehouse")


@pytest.mark.parametrize("seed", SEEDS)
def test_procedural_golden(seed):
    assert procedural(seed) == _golden(f"procedural/{seed}")


def test_wuliang_golden():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    assert wuliang() == _golden("wuliang")


def test_goldens_cover_exactly_the_pinned_runs():
    assert set(json.loads(GOLDENS.read_text("utf-8"))) == {"warehouse", "wuliang", *(f"procedural/{s}" for s in SEEDS)}


def test_policy_run_is_not_trivially_idle():
    """钉住的是有内容的轨迹：程序化世界里真的有人走动、拿取或开口，仓库里 NPC 真的有反应。"""
    auth = _simulate(TaskConfig(jianghu=1.0).scenario(0), 30)
    assert any(e.intent.op != Op.WAIT for e in auth.store.events(auth.ref))
    house = _simulate(build_warehouse(), 10, WAREHOUSE_SCRIPT)
    assert any(e.intent.actor != "player" and e.intent.op != Op.WAIT for e in house.store.events(house.ref))


if __name__ == "__main__":
    if "--write" not in sys.argv[1:]:
        raise SystemExit("用法：PYTHONPATH=src python -m tests.test_goldens --write")
    out = write()
    print(f"写入 {GOLDENS}：{len(out)} 项")
