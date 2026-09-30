"""
[INPUT]: 依赖 runtime/authority 的 WorldAuthority，persistence 的 InMemoryWorldStore，agents 的 ScriptedPolicy / Policy / HeuristicPredictor / PRED_FIELDS /
         Situation / hush_chatter，cognition 的 candidates，memory/view 的 MemoryView，core 的 WorldState.fingerprint() / digest / make_id，
         scenarios 的 build_warehouse / build_wuliang，learning/task 的 TaskConfig（程序化世界），runtime/session 的 GameSession（模板模式），
         tests/data/playthrough_duanyu.json（旧版评测整局游玩的输入）与 tests/data/goldens.json（钉住的指纹）
[OUTPUT]: 金标准指纹验收：(a) 仓库——玩家拿钥匙、开内仓门、进内仓，NPC 照脚本策略反应，共 50 tick；(a') 仓库·报告——玩家拿了钥匙
          走到入口后原地等待，守卫查问、报告船长，船长问起玩家下落、守卫作答，共 30 tick（带命题的 TELL 落进听者的信念）；
          (b) 程序化世界种子 0–9（jianghu=1）在脚本策略下各走 200 tick；(c) 旧版无量山 build_wuliang(7)、段誉作玩家、
          评测整局游玩的前 30 回合（模板模式，不接模型）——每一项逐字节等于 goldens.json：世界指纹（WorldState.fingerprint）、
          事件日志、全部感知（Observation）、每个角色结束时的整份认知（BeliefStore）与经历记录；(a)(a')(b) 另钉住每一次 NPC 决策
          （候选集、预测、Choice.index 与 tag）。_tick/_simulate 可换策略（缺省脚本策略：test_drives::identity 借同一条路比对套了驱力的策略）。写入：从仓库根目录 PYTHONPATH=src python -m tests.test_goldens --write
[POS]: tests 的守护层（设计 §6）：之后的里程碑不许改动内核、认知、ScriptedPolicy、候选规则、启发式预测与程序化世界的行为。
       (a)(a')(b) 不经会话：NPC 一侧照会话决策图的口径（候选 → 启发式预测 → 脚本策略，Situation 带上由权威经历记录汇总的
       MemoryView，与会话 _memory_view 同一口径 → 每处每 tick 至多一句闲谈），但不经 LangGraph / Orchestrator、不调措辞器、
       不做向量回忆（现有策略都不读 Situation.memories）——经 Orchestrator 的那条路只由会话的 (c) 守护（M2 另有 test_drives::identity）；
       这几局从没走到的分支与没让任何一次掷骰翻面的参数微调，金标准守不住——MemoryView 虽已接上，按说话者可靠度取舍并存矛盾说法的
       那一支（policy_kit._whereabouts，只经 ACQUIRE / DELIVER）在这几局里一次也没走到（程序化种子 10–79 同样没有）。
       措辞不在钉住之列：台词（utterance）只记有无，经历记录不记文字（模板措辞归语言层），NPC 决策不记理由文字；
       (c) 走真实会话，缺 LangGraph / Qdrant 时跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import sys
from collections.abc import Collection, Iterable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from tianlong.agents.policies import CHATTER, Policy, ScriptedPolicy, Situation, hush_chatter
from tianlong.agents.predictors import PRED_FIELDS, HeuristicPredictor
from tianlong.cognition import Candidate, candidates
from tianlong.core import Event, Fact, Intent, Op, Outcome, Proposition, Rel, digest, make_id
from tianlong.learning.task import TaskConfig
from tianlong.memory.view import MemoryView
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
# 仓库·报告：玩家拿了钥匙走到入口，此后原地等待——NPC 之间一问一答（带命题的 TELL），“话即事实”这条路由此钉住
REPORT_SCRIPT = (Candidate(Op.TAKE, "key"), Candidate(Op.MOVE, "entrance", "door_main"))
REPORT_TICKS = 30
Decision = tuple[Any, ...]


# ============================================================
#  摘要：世界指纹 + 事件日志 + 感知 + 认知 + 经历 + NPC 决策（措辞不算：台词只记有无）
# ============================================================


def _canon(x: Any) -> Any:
    """可 repr 的规范形：数据类逐字段展开、映射按键排序（与插入顺序无关）、枚举取值；utterance 字段只记有无。"""
    if is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__, *(bool(getattr(x, f.name)) if f.name == "utterance" else _canon(getattr(x, f.name))
                                     for f in fields(x)))
    if isinstance(x, Mapping):
        return tuple(sorted(((_canon(k), _canon(v)) for k, v in x.items()), key=repr))
    if isinstance(x, (tuple, list)):
        return tuple(_canon(v) for v in x)
    if isinstance(x, (set, frozenset)):
        return tuple(sorted((_canon(v) for v in x), key=repr))
    return x.value if isinstance(x, Enum) else x


def _events(events: Iterable[Event]) -> str:
    return digest(*((e.id, e.tick, e.place, e.outcome.value, e.reason, e.intent.actor, e.intent.op.value,
                     e.intent.target, e.intent.obj, e.intent.manner.value, e.intent.topic,
                     e.intent.social.value if e.intent.social else None) for e in events))


def _summary(auth: WorldAuthority, agents: Collection[str], decisions: Iterable[Decision] | None = None,
             **extra: Any) -> dict[str, Any]:
    """agents：结束时逐个摘要其认知与经历的角色（玩家也算）；decisions 给出时一并摘要（会话那一局不给）。"""
    head, store, ref = auth.head(), auth.store, auth.ref
    out = {"world": head.fingerprint(), "events": _events(store.events(ref)),
           "percepts": digest(*(_canon(o) for o in store.observations(ref))),
           "minds": {a: digest(_canon(store.beliefs(ref, a))) for a in sorted(agents)},
           "memories": {a: digest(*((m.id, m.kind, m.occurred_at, m.known_at, m.source, m.subjects, m.informant, m.verdict)
                                    for m in store.recent_memories(ref, a, 0))) for a in sorted(agents)},
           "version": head.version, "clock": head.clock}
    if decisions is not None:
        out["decisions"] = digest(*decisions)
    return {**out, **extra}


# ============================================================
#  不经 LangGraph 的 NPC 决策：候选 → 启发式预测 → 脚本策略 → 闲谈让出话头（与会话的决策图同一口径）
# ============================================================


def _tick(auth: WorldAuthority, sc: Scenario, fixed: Mapping[str, Candidate],
          policy: Policy | None = None) -> list[Decision]:
    """结算一个 tick，交回本 tick 每个 NPC 的决策：(角色, 时刻, 候选集, 预测, Choice.index, Choice.tag)。
    policy 缺省是脚本策略（金标准）；test_drives 拿同一条路比对套了驱力的策略。"""
    head, ref = auth.head(), auth.ref
    policy, predictor = policy or ScriptedPolicy(), HeuristicPredictor()
    chosen: dict[str, tuple[str | None, str, Candidate]] = {}
    decisions: list[Decision] = []
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
        memory = MemoryView.from_records(auth.store.recent_memories(ref, a, 0), head.clock)   # 与会话 _memory_view 逐项相同
        choice = policy.choose(Situation(a, prof, me, head.clock, cands, preds, memory=memory, player=sc.player))
        chosen[a] = (me.location_of(a), choice.tag, choice.chosen(cands, me))
        decisions.append((a, head.clock, tuple(c.sort_key() for c in cands),
                          tuple(tuple(getattr(p, f) for f in PRED_FIELDS) for p in preds), choice.index, choice.tag))
    talking = {a: v for a, v in chosen.items() if a not in fixed and v[2].op in (Op.TELL, Op.ASK)}
    hushed = hush_chatter(head.clock, talking) if any(t == CHATTER for _, t, _ in talking.values()) else frozenset()
    intents = [Intent(make_id("int", ref.world_id, ref.branch_id, a, head.version), a, Op.WAIT, based_on=head.version)
               if a in hushed else c.to_intent(make_id("int", ref.world_id, ref.branch_id, a, head.version), a, head.version)
               for a, (_, _, c) in chosen.items()]
    auth.settle(intents)
    return decisions


def _simulate(sc: Scenario, ticks: int, script: tuple[Candidate, ...] = (),
              policy: Policy | None = None) -> tuple[WorldAuthority, list[Decision]]:
    auth, decisions = WorldAuthority.found(InMemoryWorldStore(), sc), []
    for t in range(ticks):
        decisions += _tick(auth, sc, {sc.player: script[t]} if sc.player and t < len(script) else {}, policy)
    return auth, decisions


def _pinned(sc: Scenario, ticks: int, script: tuple[Candidate, ...] = ()) -> dict[str, Any]:
    auth, decisions = _simulate(sc, ticks, script)
    return _summary(auth, sc.profiles, decisions)


def warehouse() -> dict[str, Any]:
    return _pinned(build_warehouse(), WAREHOUSE_TICKS, WAREHOUSE_SCRIPT)


def report() -> dict[str, Any]:
    return _pinned(build_warehouse(), REPORT_TICKS, REPORT_SCRIPT)


def procedural(seed: int) -> dict[str, Any]:
    return _pinned(TaskConfig(jianghu=1.0).scenario(seed), TICKS)


def wuliang() -> dict[str, Any]:
    """旧版无量山、段誉作玩家，评测整局游玩的前 30 回合，模板模式（不接模型：解释退回规则解析、叙述走模板）。"""
    from tianlong.runtime.session import GameSession
    sc = build_wuliang(7)
    s = GameSession(sc)
    s.intro()
    played = 0
    for text in json.loads(PLAYTHROUGH.read_text("utf-8"))[:WULIANG_TURNS]:
        played += 1
        if s.turn(text).ending is not None:
            break
    return _summary(s.authority, sc.profiles, turns=played, ending=s.ending.key if s.ending else None)


def compute() -> dict[str, Any]:
    return {"warehouse": warehouse(), "warehouse/report": report(),
            **{f"procedural/{sd}": procedural(sd) for sd in SEEDS}, "wuliang": wuliang()}


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


def test_report_golden():
    assert report() == _golden("warehouse/report")


@pytest.mark.parametrize("seed", SEEDS)
def test_procedural_golden(seed):
    assert procedural(seed) == _golden(f"procedural/{seed}")


def test_wuliang_golden():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    assert wuliang() == _golden("wuliang")


def test_goldens_cover_exactly_the_pinned_runs():
    assert set(json.loads(GOLDENS.read_text("utf-8"))) == {"warehouse", "warehouse/report", "wuliang",
                                                           *(f"procedural/{s}" for s in SEEDS)}


def test_policy_run_is_not_trivially_idle():
    """钉住的是有内容的轨迹：程序化世界里真的有人走动、拿取或开口，仓库里 NPC 真的有反应。"""
    auth, _ = _simulate(TaskConfig(jianghu=1.0).scenario(0), 30)
    assert any(e.intent.op != Op.WAIT for e in auth.store.events(auth.ref))
    house, _ = _simulate(build_warehouse(), 10, WAREHOUSE_SCRIPT)
    assert any(e.intent.actor != "player" and e.intent.op != Op.WAIT for e in house.store.events(house.ref))


def test_report_run_carries_words_as_facts():
    """仓库·报告确实走过“话即事实”：NPC 对 NPC 问起某人下落、被问的人带着命题作答，问的人亲耳听到这句说法（SPEECH 感知，
    说法归到答话的人名下）；守卫的报告同样带命题，船长没亲眼见却信了它。钉住的指纹因此覆盖 TELL → 感知 → 信念这条路。"""
    auth, _ = _simulate(build_warehouse(), REPORT_TICKS, REPORT_SCRIPT)
    store, ref = auth.store, auth.ref
    ok = [e for e in store.events(ref) if e.outcome == Outcome.SUCCESS and e.intent.topic is not None]
    asked = {(e.intent.actor, e.intent.target, e.intent.topic.prop.subject) for e in ok if e.intent.op == Op.ASK}
    answers = [e for e in ok if e.intent.op == Op.TELL and e.intent.actor != "player"
               and (e.intent.target, e.intent.actor, e.intent.topic.prop.subject) in asked]
    assert answers, "应有 NPC 带着命题回答了另一个 NPC 的提问"
    heard = {(o.observer, o.percept.informant, f.prop) for o in store.observations(ref) for f in o.percept.facts
             if o.percept.informant is not None}
    for e in answers:
        assert (e.intent.target, e.intent.actor, e.intent.topic.prop) in heard, e
    key_on_player = Proposition.rel("key", Rel.AT, "player")
    assert any(e.intent.op == Op.TELL and e.intent.topic == Fact(key_on_player) for e in ok), "守卫把“钥匙在玩家身上”报告出去"
    told = store.beliefs(ref, "captain").believed(key_on_player)
    assert told is not None and told.holds and told.informant == "guard", "船长没亲眼见，信的是守卫的说法"


if __name__ == "__main__":
    if "--write" not in sys.argv[1:]:
        raise SystemExit("用法：PYTHONPATH=src python -m tests.test_goldens --write")
    out = write()
    print(f"写入 {GOLDENS}：{len(out)} 项")
