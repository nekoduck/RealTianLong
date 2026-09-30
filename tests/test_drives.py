"""
[INPUT]: 依赖 agents/drives 的 Driven / holds / realize / oneway_doors，agents 的 ScriptedPolicy / Situation / Choice / HeuristicPredictor，
         core/drives 的驱力词表，runtime/cast 的 policy_for / wakes / advance_marks，runtime/session 的 GameSession（模板模式），
         scenarios 的 build_warehouse / build_wuliang，learning/task 的 TaskConfig（程序化世界），
         tests/test_goldens 的 _simulate（不经 LangGraph 的同一口径决策循环），tests/test_isolation 的 _perturbed（改动守卫看不到的真相）
[OUTPUT]: 驱力层验收（M2 第一段）：导入图不含 kernel / persistence / runtime；驱力为空（或一条也不成立）时 20 个程序化种子 × 60 tick
          的世界指纹与每次决策（候选、预测、Choice.index 与 tag）逐项相同；变形测试——只在角色不相信的事实上不同的两份世界，
          认知相同即驱力选择相同；每类条件的求值、每种行动的落地；优先级（URGENT 先于脚本、IDLE 只在脚本闲着时、VETO 把关）；
          once / cooldown 经 advance_marks 只认兑现成功的；时间窗打开即唤醒；驱力原话随任何行动落库、被在场的人感知；
          带驱力标记的会话读档接续与连续运行逐项相同；
          Flee 不撞认为锁着的门；多一条单向捷径不让绕开它的 Go 放弃开锁；台词在长时间窗里一直轮换；
          Saw(here=True) 只认此处发生的事、Near 只认刚才亲眼见过（传闻与早先的不算）、Unlock/Lock 只在会改变什么时才转钥匙；
          lint（普通人版）：台词与姿态不命中 SECRETS_C、每条有 gloss、姿态引号外不用 NPC 口吻里的昵称（书呆子、酸秀才……）、点的名字说话者在条件成立时都已认识（例外逐条写明理由），
          一夜里 once 的驱力至多兑现一次
[POS]: tests 的驱力层：“驱力只读自己的认知、只是意图、由内核裁定”被写成可证伪的断言
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tianlong
from tianlong.agents.drives import Driven, holds, oneway_doors, realize
from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.policy_kit import CHATTER, SPEAK, Choice
from tianlong.agents.predictors import HeuristicPredictor
from tianlong.cognition import BeliefStore, candidates
from tianlong.cognition.beliefs import Belief, Episode
from tianlong.core import (
    Alone,
    AnyOf,
    Arrived,
    Ask,
    At,
    Between,
    Cross,
    Drive,
    EntitySketch,
    Event,
    Fired,
    Flee,
    Follow,
    Fond,
    Give,
    Go,
    Heard,
    HeldBy,
    Here,
    Hold,
    Holds,
    Inspect,
    Intent,
    Kind,
    Knows,
    Level,
    Lock,
    Lost,
    Manner,
    Menaced,
    Modality,
    Near,
    Not,
    Op,
    Outcome,
    PerceivedEvent,
    Pose,
    Proposition,
    Pursue,
    Rel,
    Saw,
    Say,
    Searched,
    Social,
    Status,
    Study,
    Take,
    Unlock,
    Use,
    at,
    derive_seed,
)
from tianlong.learning.task import TaskConfig
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.runtime.cast import advance_marks, policy_for, wakes
from tianlong.scenarios import Scenario, build_warehouse, build_wuliang
from tianlong.scenarios.warehouse import START

from .test_goldens import WAREHOUSE_SCRIPT, _simulate
from .test_isolation import _perturbed

URGENT, IDLE, VETO = Level.URGENT, Level.IDLE, Level.VETO

# ============================================================
#  搭一个角色此刻的处境：真实场景建档后的认知，再按需添几笔
# ============================================================


def _mind(sc: Scenario, agent: str) -> BeliefStore:
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    return auth.store.beliefs(auth.ref, agent)


def _with(b: BeliefStore, *facts: tuple[str, str, object, bool], now: int = START, **fields) -> BeliefStore:
    """添上亲眼所见的信念 (主语, 谓词, 值, 真假)，没见过的主语按仓库真相补上样子；AT 是函数型谓词，先抹掉同一主语的旧位置。
    其余字段直接替换。"""
    beliefs, entities, truth_state = dict(b.beliefs), dict(b.entities), build_warehouse().state
    for subject, pred, value, truth in facts:
        if subject not in entities and truth_state.has_entity(subject):
            e = truth_state.entity(subject)
            entities[subject] = EntitySketch(subject, e.kind, e.name)
        if pred == Rel.AT.value:
            beliefs = {p: x for p, x in beliefs.items() if not (p.subject == subject and p.predicate == pred)}
        prop = Proposition(subject, pred, value)
        beliefs[prop] = Belief(prop, truth, 1.0, Modality.SIGHT, now)
    return replace(b, beliefs=beliefs, entities=entities, **fields)


def _at(who: str, place: str) -> tuple[str, str, object, bool]:
    return who, Rel.AT.value, place, True


def _attr(who: str, name: str, value: object = True) -> tuple[str, str, object, bool]:
    return who, f"attr.{name}", value, True


def _sit(sc: Scenario, b: BeliefStore, now: int = START) -> Situation:
    prof = sc.profiles[b.owner]
    cands = candidates(b, list(prof.interests()), 64)
    preds = tuple(HeuristicPredictor().predict(b, now, cands, list(prof.interests()), profile=prof))
    return Situation(b.owner, prof, b, now, cands, preds, player=sc.player)


def _guard(*facts, **fields) -> Situation:
    sc = build_warehouse()
    return _sit(sc, _with(_mind(sc, "guard"), *facts, **fields))


def _cand(sit: Situation, choice: Choice):
    return choice.chosen(sit.candidates, sit.beliefs)


def _ep(tick: int, kind: Op, actor: str, place: str = "entrance", modality: Modality = Modality.SIGHT, **kw) -> Episode:
    return Episode(tick, modality, PerceivedEvent(kind.value, place, actor, **kw))


# ============================================================
#  隔离：驱力层的导入图不碰内核、存储与装配
# ============================================================


def test_isolation_imports():
    src = str(Path(tianlong.__file__).resolve().parents[1])
    probe = ("import sys, tianlong.agents.drives, tianlong.core.drives; "
             "print(' '.join(m for m in sys.modules if m.startswith('tianlong')))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": src}).stdout.split()
    assert "tianlong.agents.drives" in out
    banned = [m for m in out if m.split(".")[1:2] in (["kernel"], ["persistence"], ["runtime"])]
    assert not banned, f"驱力层不许依赖 {banned}"
    core = subprocess.run([sys.executable, "-c", "import sys, tianlong.core.drives; print(' '.join(sys.modules))"],
                          capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": src}).stdout.split()
    assert all(m.startswith("tianlong.core") or not m.startswith("tianlong.") for m in core), "core/drives 只依赖标准库与 core"


# ============================================================
#  恒等：驱力为空、或一条也不成立时，与不套 Driven 逐项相同
# ============================================================

# 各级都有、条件永不成立或成立却实现不了的驱力：只要不触发，就不许改变任何一次决策
INERT = (
    Drive("never", URGENT, At("nowhere"), Hold()),
    Drive("unreal", URGENT, (), Take("no_such_item")),
    Drive("mute", IDLE, (), Say("no_such_person", Social.REMARK)),
    Drive("idle", IDLE, Knows("no_such_thing"), Pose("发呆")),
    Drive("veto", VETO, Not(Between(0)), veto=Op.ATTACK),
)


@pytest.mark.parametrize("seed", range(20))
def test_identity_without_drives(seed):
    sc = TaskConfig(jianghu=1.0).scenario(seed)
    base, plain = _simulate(sc, 60)
    for drives in ((), INERT):
        auth, wrapped = _simulate(sc, 60, policy=Driven(ScriptedPolicy(), drives))
        assert auth.head().fingerprint() == base.head().fingerprint()
        assert wrapped == plain, "每个 NPC 每次决策的候选、预测、Choice.index 与 tag 逐项相同"
    assert not sc.drives and not isinstance(policy_for(sc.npcs[0], sc, None, {}), Driven), "没有驱力就不构造 Driven"


# ============================================================
#  变形：两份世界只在角色不相信的事实上不同——认知相同，驱力的选择就相同
# ============================================================

BATTERY = (
    Drive("flee", URGENT, (Status("self", "wounded"), Menaced()), Flee(prefer=("harbor",)), line="走为上！", once=True),
    Drive("hail", URGENT, (Here("player"), Fond("player", at_least=0), Not(Fired("hail"))), Say("player", Social.COMMAND),
          line="站住！"),
    Drive("chase", URGENT, AnyOf((Saw(Op.TAKE, actor="player", within=5), HeldBy("key", "player"))),
          (Ask("player"), Pursue("player"))),
    Drive("look", URGENT, (Arrived(), Knows("player")), Pose("抬眼打量来人")),
    Drive("nod", IDLE, Heard("player", within=5), Pose("点了点头", Social.AGREE), cooldown=3),
    Drive("check", IDLE, (Knows("key"), Not(Searched("warehouse")), Between(START, START + 40)),
          (Go("warehouse"), Inspect("warehouse"))),
    Drive("stamp", IDLE, (At("entrance"), Alone()), Pose("跺了跺脚"), lines=("风真冷。", "几时换班？"), cooldown=2),
    Drive("hand", IDLE, Holds("key"), Give("key", "captain")),
    Drive("seek", IDLE, Lost("captain"), Follow("captain")),
    Drive("calm", VETO, Fired("hail", within=10), Say("target", Social.PLEAD), veto=Op.ATTACK),
)


def _trail(sc: Scenario, ticks: int) -> list[tuple[int, str, Situation]]:
    """玩家照仓库验收的三步走，NPC 照脚本策略；记下每个 tick 每个 NPC 决策时的处境。"""
    from .test_goldens import _tick
    auth, out = WorldAuthority.found(InMemoryWorldStore(), sc), []
    for t in range(ticks):
        for a in sc.npcs:
            out.append((t, a, _sit(sc, auth.store.beliefs(auth.ref, a), auth.head().clock)))
        _tick(auth, sc, {sc.player: WAREHOUSE_SCRIPT[t]} if t < len(WAREHOUSE_SCRIPT) else {})
    return out


def test_metamorphic_unbelieved_truth_does_not_change_drives():
    base = build_warehouse()
    same = 0
    for variant in ("ledger_moved", "door_unlocked", "extra_secret"):
        other = replace(base, state=_perturbed(variant))
        assert other.state.fingerprint() != base.state.fingerprint()
        for (t, a, s1), (_, _, s2) in zip(_trail(base, 30), _trail(other, 30), strict=True):
            if s1.beliefs != s2.beliefs:
                continue                                 # 前提不成立（他已察觉差别）：不是本测试的样本
            rng = random.Random(derive_seed("metamorphic", variant, a, t))
            marks = {d.key: sorted(rng.sample(range(s1.now - 12, s1.now + 1), rng.randint(1, 3)))
                     for d in BATTERY if rng.random() < 0.3}
            drives = rng.sample(BATTERY, len(BATTERY))
            c1 = Driven(ScriptedPolicy(), drives, marks).choose(s1)
            c2 = Driven(ScriptedPolicy(), drives, marks).choose(s2)
            assert c1 == c2 and _cand(s1, c1) == _cand(s2, c2)
            same += 1
    assert same >= 50, f"样本太少：{same}"


# ============================================================
#  条件：每类一例（成立与不成立）
# ============================================================


def test_holds_place_time_presence_status():
    s = _guard(_at("player", "entrance"), _attr("guard", "wounded"), _attr("guard", "evasion"))
    assert holds(At("entrance"), s, {}) and not holds(At("harbor"), s, {})
    assert holds(Between(START), s, {}) and holds(Between(START - 5, START + 1), s, {})
    assert not holds(Between(START + 1), s, {}) and not holds(Between(START - 5, START), s, {})
    assert holds(Here("player"), s, {}) and not holds(Here("captain"), s, {})
    assert holds(Here("target"), s, {}, target="player") and not holds(Here("target"), s, {})
    assert holds(Status(), s, {}) and holds(Status("self", "evasion"), s, {}), "技能也是状态"
    assert not holds(Status("player"), s, {}) and not holds(Status("self", "poisoned"), s, {})


def test_holds_saw_and_heard_read_episodes():
    took = _ep(START - 1, Op.TAKE, "player", "warehouse", target="key", outcome=Outcome.SUCCESS)
    cmd = _ep(START - 2, Op.TELL, "player", modality=Modality.SPEECH, target="guard", social=Social.COMMAND,
              utterance="让开")
    walk = _ep(START - 1, Op.MOVE, "captain", target="entrance", obj="path")
    mine = _ep(START, Op.TAKE, "guard")
    s = _guard(episodes=(cmd, took, walk, replace(mine, modality=Modality.SELF)))
    assert holds(Saw(Op.TAKE, actor="player"), s, {}) and holds(Saw(Op.TAKE, target="key", outcome=Outcome.SUCCESS), s, {})
    assert not holds(Saw(Op.TAKE, actor="player", within=0), s, {}), "窗口外不算"
    assert not holds(Saw(Op.TAKE, outcome=Outcome.FAILURE), s, {}) and not holds(Saw(Op.TAKE, actor="guard"), s, {})
    assert holds(Saw(Op.MOVE, obj="path"), s, {}) and not holds(Saw(Op.MOVE, obj="door_main"), s, {})
    assert holds(Heard("player", frozenset({Social.COMMAND}), to="self"), s, {})
    assert not holds(Heard("player", frozenset({Social.PLEAD})), s, {}) and not holds(Heard("captain"), s, {})
    assert not holds(Heard("player", within=1), s, {}), "拿钥匙那一下不是开口"


def test_holds_items_knowledge_and_search():
    s = _guard(_at("player", "entrance"), _at("key", "player"), searched={"entrance": START - 3})
    assert holds(HeldBy("key"), s, {}) and holds(HeldBy("key", "player"), s, {}) and not holds(HeldBy("key", "captain"), s, {})
    assert not holds(Holds("key"), s, {})
    assert holds(Holds("key"), _guard(_at("key", "guard")), {})
    assert holds(Knows("key"), s, {}) and not holds(Knows("scroll"), s, {})
    assert holds(Searched("entrance"), s, {}) and not holds(Searched("warehouse"), s, {})


def test_holds_attitude_company_and_marks():
    now = START
    s = _guard(_at("player", "entrance"), attitudes={"player": -2}, company={"player": now - 1})
    assert holds(Menaced(), s, {}) and not holds(Fond("player", at_least=0), s, {}) and holds(Fond("player", -2), s, {})
    assert not holds(Menaced(), _guard(_at("player", "entrance"), _attr("player", "poisoned"), attitudes={"player": -2}), {})
    assert not holds(Menaced(), _guard(_at("player", "entrance"), attitudes={"player": -1}), {})
    assert not holds(Alone(), s, {}) and holds(Alone(("player",)), s, {}) and holds(Alone(), _guard(), {})
    assert holds(Arrived(), s, {}) and not holds(Arrived(("player",)), s, {})
    assert not holds(Arrived(), _guard(_at("player", "entrance"), company={"player": now}), {}), "同一 tick 进来的推迟一 tick"
    assert not holds(Arrived(), _guard(_at("player", "entrance"), company={"player": now - 3}), {})
    marks = {"x": [now - 5, now - 1]}
    assert holds(Fired("x"), s, marks) and holds(Fired("x", times=2), s, marks) and not holds(Fired("x", times=3), s, marks)
    assert holds(Fired("x", within=1), s, marks) and not holds(Fired("x", within=0), s, marks)
    assert not holds(Fired("y"), s, marks)
    assert holds(Not(Fired("y")), s, marks) and holds(AnyOf((Fired("y"), Fired("x"))), s, marks)
    assert not holds(AnyOf((Fired("y"), At("harbor"))), s, marks)


def test_holds_lost_when_only_a_one_way_door_leads_there():
    sc = build_wuliang(7)
    b = _mind(sc, "gongguangjie")
    assert "d_cliff" in oneway_doors(b)
    down = _sit(sc, _with(b, _at("duanyu", "jianhu"), now=sc.state.clock), sc.state.clock)
    assert holds(Lost("duanyu"), down, {}), "以为要跳断崖才到得了：追不到了"
    assert realize(Pursue("duanyu"), down) is None
    jump = realize(Pursue("duanyu", avoid_oneway=False), down)
    assert jump is not None and _cand(down, jump).op == Op.MOVE, "不避单向门就照走"
    assert realize(Go("jianhu"), down) is None and realize(Go("jianhu", avoid_oneway=False), down) is not None
    behind = _sit(sc, _with(b, _at("duanyu", "houyuan"), now=sc.state.clock), sc.state.clock)
    assert not holds(Lost("duanyu"), behind, {})
    step = _cand(behind, realize(Pursue("duanyu"), behind))
    assert (step.op, step.target, step.obj) == (Op.MOVE, "houyuan", "d_corridor")
    here = _sit(sc, b, sc.state.clock)
    assert not holds(Lost("duanyu"), here, {}), "人在眼前就不算追丢"


# ============================================================
#  行动：每种至少一例
# ============================================================


def test_realize_movement():
    s = _guard()
    go = _cand(s, realize(Go("warehouse"), s))
    assert (go.op, go.target, go.obj, go.manner) == (Op.MOVE, "warehouse", "door_main", Manner.NORMAL)
    assert _cand(s, realize(Go("warehouse", careful=True), s)).manner == Manner.CAREFUL
    assert realize(Go("entrance"), s) is None, "已在此处"
    cross = _cand(s, realize(Cross("path"), s))
    assert (cross.op, cross.target, cross.obj) == (Op.MOVE, "harbor", "path")
    follow = _cand(s, realize(Follow("captain"), s))
    assert (follow.op, follow.target) == (Op.MOVE, "harbor")
    near = _guard(_at("captain", "entrance"))
    assert realize(Follow("captain"), near) is None, "人在眼前不必跟"
    pursue = _guard(_at("player", "entrance"))
    word = realize(Pursue("player"), pursue)
    assert word is not None and _cand(pursue, word).social == Social.CHALLENGE, "先礼后兵照旧"


def test_realize_flee_prefers_then_unvisited_then_seed():
    s = _guard(_at("player", "entrance"), attitudes={"player": -3})
    to = _cand(s, realize(Flee(prefer=("harbor",)), s, "flee"))
    assert (to.op, to.target, to.manner) == (Op.MOVE, "harbor", Manner.NORMAL)
    assert _cand(s, realize(Flee(prefer=("warehouse",)), s, "flee")).target == "warehouse"
    fresh = _guard(_at("player", "entrance"), attitudes={"player": -3}, surveyed={"harbor": START - 1})
    assert _cand(fresh, realize(Flee(), fresh, "flee")).target == "warehouse", "没去过的在前"
    feared = _guard(_at("player", "entrance"), _at("captain", "warehouse"), attitudes={"player": -3, "captain": -2})
    assert _cand(feared, realize(Flee(prefer=("warehouse",)), feared, "flee")).target == "harbor", "不往仇人那里逃"
    picks = {realize(Flee(), s, "flee") for _ in range(3)}
    assert len(picks) == 1, "平手由 derive_seed 定，重算不变"


def test_realize_flee_never_runs_into_a_door_it_believes_locked():
    locked = ("door_main", "attr.locked", True, True)
    s = _guard(_at("player", "entrance"), locked, attitudes={"player": -3})
    for flee in (Flee(prefer=("warehouse",)), Flee()):
        to = _cand(s, realize(flee, s, "flee"))
        assert (to.target, to.obj) == ("harbor", "path"), "认为锁着的门不去撞，换一条认为走得通的"
    shut = _guard(_at("player", "entrance"), locked, ("path", "attr.locked", True, True), attitudes={"player": -3})
    assert realize(Flee(), shut, "flee") is None, "无路可逃：退回里层策略"


def test_wary_unlocks_rather_than_giving_up_when_a_one_way_shortcut_exists():
    sc = build_warehouse()
    s = _guard(_at("key", "guard"), ("door_main", "attr.locked", True, True),
               ("key", Rel.MATCHES.value, "door_main", True))
    b = s.beliefs
    beliefs = dict(b.beliefs)
    for p in (Proposition("a_drop", Rel.CONNECTS.value, "entrance"),
              Proposition("a_drop", Rel.CONNECTS.value, "warehouse"), Proposition("a_drop", "attr.oneway", "warehouse")):
        beliefs[p] = Belief(p, True, 1.0, Modality.SIGHT, START)
    drop = _sit(sc, replace(b, beliefs=beliefs, entities={**b.entities, "a_drop": EntitySketch("a_drop", Kind.DOOR, "陡坡")}))
    assert oneway_doors(drop.beliefs) == {"a_drop"}
    go = _cand(drop, realize(Go("warehouse"), drop))
    assert (go.op, go.target, go.obj) == (Op.UNLOCK, "door_main", "key"), "多一条单向捷径不该让“开锁过去”这条路消失"
    assert _cand(drop, realize(Go("warehouse", avoid_oneway=False), drop)).obj == "a_drop", "不避单向门就走捷径"
    assert not holds(Lost("player"), _sit(sc, _with(drop.beliefs, _at("player", "warehouse"))), {}), "有路可走不算追丢"


def test_realize_items_and_inspection():
    s = _guard(_at("player", "entrance"), _at("key", "guard"))
    take_sit = _guard(_at("guard", "warehouse"))
    take = _cand(take_sit, realize(Take("key"), take_sit))
    assert (take.op, take.target, take.manner) == (Op.TAKE, "key", Manner.NORMAL)
    assert _cand(s, realize(Inspect("entrance"), s)).op == Op.INSPECT
    assert _cand(s, realize(Study("key"), s)).op == Op.STUDY
    use = _cand(s, realize(Use("key"), s))
    assert (use.op, use.target, use.obj) == (Op.USE, "guard", "key")
    give = _cand(s, realize(Give("key", "player"), s))
    assert (give.op, give.target, give.obj) == (Op.GIVE, "player", "key")
    assert realize(Give("key", "captain"), s) is None, "不在眼前递不过去"
    assert realize(Take("ledger"), s) is None


def test_saw_here_counts_only_what_happened_where_i_stand():
    walk = _ep(START - 1, Op.MOVE, "captain", "entrance", target="harbor", obj="path")
    there = _guard(episodes=(walk,))
    away = _guard(_at("guard", "warehouse"), episodes=(walk,))
    assert holds(Saw(Op.MOVE, actor="captain", here=True), there, {})
    assert not holds(Saw(Op.MOVE, actor="captain", here=True), away, {}), "他走开的地方不是我此刻所在之处"
    assert holds(Saw(Op.MOVE, actor="captain"), away, {})


def test_near_means_seen_firsthand_a_moment_ago():
    """Near：他刚才还在身边（亲眼所见的下落不出 within 个 tick），站着不动也算；早先见过的、听人说的都不算。"""
    beside = _guard(_at("player", "entrance"), now=START - 1)
    assert holds(Near("player"), beside, {}) and holds(Near("player", within=1), beside, {})
    assert not holds(Near("player", within=0), beside, {})
    assert not holds(Near("player"), _guard(_at("player", "entrance"), now=START - 20), {}), "早先见过的不算刚才"
    b = _with(_mind(build_warehouse(), "guard"), _at("player", "harbor"))
    prop = Proposition("player", Rel.AT.value, "harbor")
    told = replace(b, beliefs={**b.beliefs, prop: Belief(prop, True, 0.6, Modality.SPEECH, START, "captain")})
    assert not holds(Near("player"), _sit(build_warehouse(), told), {}), "听人说他在哪不算见过"
    assert not holds(Near("captain"), _guard(), {})


def test_unlock_and_lock_turn_the_key_only_when_it_changes_something():
    shut = _guard(_at("guard", "warehouse"), _at("key", "guard"), _attr("door_store", "locked"))
    opened = _guard(_at("guard", "warehouse"), _at("key", "guard"), ("door_store", "attr.locked", True, False))
    got = _cand(shut, realize(Unlock("door_store"), shut))
    assert (got.op, got.target, got.obj) == (Op.UNLOCK, "door_store", "key")
    assert realize(Lock("door_store"), shut) is None, "以为锁着就不必再锁"
    assert realize(Unlock("door_store"), opened) is None, "以为没锁就不去开"
    assert _cand(opened, realize(Lock("door_store"), opened)).op == Op.LOCK
    assert realize(Unlock("door_store"), _guard(_attr("door_store", "locked"))) is None, "门不在身边、手里也没钥匙"


def test_realize_speech_pose_hold_and_ask():
    s = _guard(_at("player", "entrance"))
    say = realize(Say("player", Social.COMMAND), s)
    assert say.tag == SPEAK and (say.free.op, say.free.target, say.free.social) == (Op.TELL, "player", Social.COMMAND)
    assert realize(Say("player", Social.REMARK, careful=True), s).free.manner == Manner.CAREFUL
    assert realize(Say("captain", Social.GREET), s) is None, "不在眼前的人说不上话"
    pose = realize(Pose("抱起胳膊", Social.THREATEN), s)
    assert pose.line == "抱起胳膊" and _cand(s, pose).op == Op.WAIT and _cand(s, pose).social == Social.THREATEN
    with pytest.raises(ValueError):
        Pose("")
    assert _cand(s, realize(Hold(), s)).op == Op.WAIT and realize(Hold(), s).line is None
    lost = _with(s.beliefs, now=START)
    lost = replace(lost, beliefs={p: x for p, x in lost.beliefs.items()
                                  if not (p.subject == "key" and p.predicate == Rel.AT.value)})
    ask_sit = _sit(build_warehouse(), lost)
    ask = _cand(ask_sit, realize(Ask("key"), ask_sit))
    assert (ask.op, ask.target, ask.topic.prop.subject) == (Op.ASK, "player", "key")
    assert realize(Ask("key"), s) is None, "以为知道下落就不问"
    asked = replace(lost, episodes=(_ep(START - 1, Op.ASK, "guard", modality=Modality.SELF, target="player"),))
    assert realize(Ask("key"), _sit(build_warehouse(), asked)) is None, "刚问过的人不再问"


# ============================================================
#  Driven：优先级、VETO、台词、冷却
# ============================================================


class _Fixed:
    """测试用的里层策略：永远做同一件事。"""

    def __init__(self, op: Op, target: str | None = None, tag: str = "") -> None:
        self.op, self.target, self.tag = op, target, tag

    def choose(self, sit: Situation) -> Choice:
        i = next(i for i, c in enumerate(sit.candidates) if c.op == self.op and c.target == self.target)
        return Choice(i, "里层", self.tag)


def test_priority_urgent_inner_idle():
    s = _guard(_at("player", "entrance"))
    shout = Drive("shout", URGENT, Here("player"), Say("player", Social.COMMAND), line="站住！")
    muse = Drive("muse", IDLE, (), Pose("望天"))
    busy, idle = _Fixed(Op.INSPECT, "entrance"), _Fixed(Op.WAIT, tag="idle")
    got = Driven(busy, (muse, shout)).choose(s)
    assert (got.drive, got.line, got.tag) == ("shout", "站住！", SPEAK), "URGENT 先于里层策略"
    assert Driven(busy, (muse,)).choose(s) == busy.choose(s), "里层有事做：IDLE 不插手"
    assert Driven(idle, (muse,)).choose(s).drive == "muse", "里层闲着：试 IDLE"
    chatter = Driven(_Fixed(Op.WAIT, tag=CHATTER), (muse,)).choose(s)
    assert chatter.drive == "muse" and chatter.tag != CHATTER, "驱力的选择从不标 CHATTER"
    gone = _guard()
    assert Driven(busy, (shout,)).choose(gone) == busy.choose(gone), "条件不成立：退回里层"
    unreal = Drive("unreal", URGENT, (), Take("ledger"))
    assert Driven(busy, (unreal,)).choose(s) == busy.choose(s), "实现不了：退回里层"
    offer = Drive("offer", URGENT, (), Pose("拱了拱手", Social.GREET))
    assert Driven(busy, (shout,), proposal=offer).choose(s).drive == "offer", "一次性提议最先"


def test_veto_turns_attack_into_plea():
    s = _guard(_at("player", "entrance"))
    brute = _Fixed(Op.ATTACK, "player")
    pacifist = Drive("pacifist", VETO, (), Say("target", Social.PLEAD), line="君子动口不动手！", veto=Op.ATTACK)
    got = Driven(brute, (pacifist,)).choose(s)
    cand = _cand(s, got)
    assert (cand.op, cand.target, cand.social) == (Op.TELL, "player", Social.PLEAD)
    assert got.line == "君子动口不动手！" and got.drive == "pacifist"
    hold = Drive("truce", VETO, Fired("deal", within=30), veto=Op.ATTACK)
    assert _cand(s, Driven(brute, (hold,), {"deal": [START - 3]}).choose(s)).op == Op.WAIT, "时间窗内：改为按兵不动"
    assert Driven(brute, (hold,), {"deal": [START - 31]}).choose(s) == brute.choose(s), "时间窗以驱力标记计，过了就不拦"
    fond = Drive("spare", VETO, Fond("target", at_least=1), veto=Op.ATTACK)
    assert Driven(brute, (fond,)).choose(s) == brute.choose(s), "条件对被否决的那个人求值"
    assert Driven(_Fixed(Op.INSPECT, "entrance"), (pacifist,)).choose(s).drive is None, "只否决指明的行动"
    with pytest.raises(ValueError):
        Drive("bad", VETO, ())


def test_lines_rotate_by_marks():
    s = _guard()
    d = Drive("stamp", URGENT, (), Pose("跺脚"), lines=("一", "二", "三"))
    said = [Driven(_Fixed(Op.WAIT), (d,), {"stamp": list(range(n))}).choose(s).line for n in range(6)]
    assert said[:3] == said[3:] and sorted(said[:3]) == ["一", "三", "二"], "按兑现次数轮换，起点由种子定"
    plain = Drive("plain", URGENT, (), Pose("跺脚"))
    assert Driven(_Fixed(Op.WAIT), (plain,)).choose(s).line == "跺脚", "没有台词时姿态的字就是原话"


def test_lines_keep_rotating_through_a_long_window():
    s = _guard()
    d = Drive("murmur", URGENT, (), Pose("低语"), lines=("一", "二", "三", "四"))
    marks: dict = {}
    said = []
    for t in range(40):
        line = Driven(_Fixed(Op.WAIT), (d,), marks.get("guard", {})).choose(replace(s, now=START + t)).line
        said.append(line)
        ok = Intent(f"i{t}", "guard", Op.WAIT)
        settle = SimpleNamespace(events=(Event(f"e{t}", START + t, ok, "entrance", Outcome.SUCCESS),))
        marks = advance_marks(marks, [_delib("guard", ok.id, "murmur")], settle)
    assert len(marks["guard"]["murmur"]) == 40, "标记不截断：累计次数准确"
    assert said[32:36] == said[:4] and len(set(said[36:])) == 4, "兑现三十多次之后照样轮换"


def _delib(agent: str, intent_id: str, drive: str = "") -> SimpleNamespace:
    return SimpleNamespace(agent=agent, intent=Intent(intent_id, agent, Op.WAIT), drive=drive)


def test_once_and_cooldown_count_only_successes():
    ok = Intent("i1", "guard", Op.WAIT)
    bad = Intent("i2", "captain", Op.MOVE, "entrance", "path")
    settle = SimpleNamespace(events=(Event("e1", START, ok, "entrance", Outcome.SUCCESS),
                                     Event("e2", START, bad, "harbor", Outcome.FAILURE)))
    before = {"guard": {"stamp": [START - 9]}}
    marks = advance_marks(before, [_delib("guard", "i1", "stamp"), _delib("captain", "i2", "go"),
                                   _delib("x", "i1")], settle)
    assert marks == {"guard": {"stamp": [START - 9, START]}} and before == {"guard": {"stamp": [START - 9]}}
    s = _guard()
    once = Drive("stamp", URGENT, (), Pose("跺脚"), once=True)
    assert Driven(_Fixed(Op.WAIT), (once,), marks["guard"]).choose(s).drive is None, "once：兑现过就不再触发"
    cool = Drive("stamp", URGENT, (), Pose("跺脚"), cooldown=3)
    assert Driven(_Fixed(Op.WAIT), (cool,), {"stamp": [START - 2]}).choose(s).drive is None
    assert Driven(_Fixed(Op.WAIT), (cool,), {"stamp": [START - 3]}).choose(s).drive == "stamp"


def test_wakes_when_a_window_opens():
    d = (Drive("dusk", URGENT, AnyOf((Not(Between(100, 200)), At("x"))), Hold()),)
    assert wakes(d, 90, 100) and wakes(d, 99, 150) and wakes(d, 199, 200) and not wakes(d, 101, 199)
    assert not wakes(d, 100, 110) and not wakes(d, None, 100) and not wakes((), 0, 10 ** 6)


def test_share_the_floor_never_hushes_drives():
    pytest.importorskip("langgraph")
    from tianlong.agents.npc_graph import NpcContext
    from tianlong.agents.orchestrator import Deliberation, Orchestrator
    from tianlong.agents.port import AgentPort
    sc = build_warehouse()
    b = {a: _with(_mind(sc, a), _at(a, "entrance")) for a in ("guard", "captain")}
    npcs = {a: NpcContext(AgentPort(a, sc.profiles[a], "w", "main", 0, START, beliefs=lambda a=a: b[a])) for a in b}
    chat = Deliberation("captain", Intent("i1", "captain", Op.TELL, "player", social=Social.JOKE), "", (), 1, CHATTER)
    drive = Deliberation("guard", Intent("i2", "guard", Op.TELL, "player", utterance="站住！", social=Social.COMMAND),
                         "", (), 1, CHATTER, drive="hail")
    out = {d.agent: d for d in Orchestrator._share_the_floor([chat, drive], npcs)}
    assert out["guard"] == drive, "驱力的话不让出话头"
    assert out["captain"].intent.op == Op.WAIT, "闲谈让给驱力"


# ============================================================
#  会话：原话随任何行动落库、被在场的人感知；时间窗唤醒；读档接续
# ============================================================

HAIL = "站住！哪里来的？"
STROLL = "我去入口瞧瞧。"
DRIVES = {
    "guard": (Drive("hail", URGENT, Here("player"), Say("player", Social.COMMAND), line=HAIL, once=True),
              Drive("stamp", IDLE, At("entrance"), Pose("跺了跺脚"), lines=("风真冷。", "几时换班？"), cooldown=3)),
    "captain": (Drive("stroll", URGENT, Between(START + 6), Go("entrance"), line=STROLL, once=True),),
}
COMMANDS = ["去仓库入口", "等待", "等待", "等10分钟", "等待", "等待", "等待", "等待"]


def _driven_world() -> Scenario:
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    return replace(build_warehouse(), drives=DRIVES)


def test_words_ride_any_op_and_are_perceived():
    pytest.importorskip("langgraph")                # 会话要 LangGraph：核心零依赖环境跳过
    from tianlong.runtime.session import GameSession
    s = GameSession(_driven_world())
    s.intro()
    for cmd in COMMANDS:
        s.turn(cmd)
    events = s.store.events(s.ref)
    hails = [e for e in events if e.intent.utterance == HAIL]
    assert len(hails) == 1 and hails[0].intent.op == Op.TELL and hails[0].outcome == Outcome.SUCCESS, "once"
    stroll = [e for e in events if e.intent.utterance == STROLL]
    assert len(stroll) == 1 and stroll[0].intent.op == Op.MOVE and stroll[0].outcome == Outcome.SUCCESS
    assert stroll[0].tick == START + 6, "时间窗一开就唤醒（调度器本要等到闲置满 15 分钟）"
    heard = {o.percept.event.utterance for o in s.store.observations(s.ref)
             if o.observer == "player" and o.percept.event is not None}
    assert {HAIL, STROLL} <= heard, "在场的玩家听见看见原话"
    assert any(ep.event.utterance == STROLL for ep in s.beliefs("guard").episodes), "在场的 NPC 也记得"
    marks = s.session_state()["drives"]
    assert marks["guard"]["hail"] == [hails[0].tick] and marks["captain"]["stroll"] == [START + 6]
    stamps = [e.tick for e in events if e.intent.actor == "guard" and e.intent.utterance in ("风真冷。", "几时换班？")]
    assert stamps and all(b - a >= 3 for a, b in zip(stamps, stamps[1:], strict=False)), "cooldown"
    assert marks["guard"].get("stamp", []) == stamps


def _play(sc: Scenario, split: int | None = None):
    pytest.importorskip("langgraph")                # 会话要 LangGraph：核心零依赖环境跳过
    from tianlong.runtime.session import GameSession
    store = InMemoryWorldStore()
    s = GameSession(sc, store=store)
    s.intro()
    for i, cmd in enumerate(COMMANDS):
        if i == split:
            s = GameSession(sc, store=store)
            assert s.resumed
        s.turn(cmd)
    return s.authority.head().fingerprint(), [(e.id, e.outcome) for e in store.events(s.ref)], s.session_state()


def test_resume_with_marks_equals_continuous_play():
    sc = _driven_world()
    continuous = _play(sc)
    assert continuous[2]["drives"], "这一局确实记下了驱力标记"
    assert json.loads(json.dumps(continuous[2])) == continuous[2], "会话运行态 JSON 往返不变"
    for split in (3, 5):
        assert _play(sc, split) == continuous


# ============================================================
#  普通人版驱力表的体检（lint）：台词不说破秘密；once 的驱力兑现后不再触发
# ============================================================


def _texts(d: Drive) -> tuple[str, ...]:
    return (d.line, *d.lines, *(a.text for a in d.do if isinstance(a, Pose)))


def test_lint_lines_never_say_a_secret():
    from tianlong.scenarios.tianlong.commoner import SECRETS_C
    from tianlong.scenarios.tianlong.drives_c import DRIVES_C
    said = [t for ds in DRIVES_C.values() for d in ds for t in _texts(d) if t]
    assert len(said) >= 30
    leaks = [(t, pat) for t in said for pat in SECRETS_C if re.search(pat, t)]
    assert not leaks, leaks
    glossed = [d.key for ds in DRIVES_C.values() for d in ds if not d.gloss]
    assert not glossed, f"每条驱力都要有给对照组圣经的 gloss：{glossed}"


def test_lint_poses_use_no_npc_nicknames():
    """姿态是叙述看得见的那一行（“看见钟灵护在……”）：引号外不许用只在某个 NPC 口吻里的昵称（书呆子、酸秀才、挑茶的……）——
    那是他嘴里的话，不是旁人眼里的样子；引号里的原话照旧由他说。"""
    from tianlong.scenarios import build_wuliang_commoner
    from tianlong.scenarios.tianlong.drives_c import DRIVES_C
    sc = build_wuliang_commoner(7)
    nicknames = {w for a in sc.npcs for w in re.findall(r"“([^”]+)”", sc.profiles[a].voice)}
    assert {"书呆子", "酸秀才", "挑茶的", "小伙计"} <= nicknames
    poses = [t for ds in DRIVES_C.values() for d in ds if any(isinstance(a, Pose) for a in d.do)
             for t in (*d.lines, *(a.text for a in d.do if isinstance(a, Pose)))]
    assert len(poses) >= 15
    bad = [(t, w) for t in poses for w in nicknames if w in re.sub(r"“[^”]*”", "", t)]
    assert not bad, bad


# 条件最早成立那一刻说话者凭自己的认知推不出、却点得出的名字：逐条写明为什么说得出口
_NAMED_ANYWAY = {
    ("zuozimu", "demand", "antidote"): "“解药”是泛称：弟子中了貂毒，掌门料定放貂的人身上有解药，他要的正是这个",
    ("shennong", "bribed", "suiyin"): "对阿顺态度到 +2 只能来自赠物：碎银是阿顺刚塞到他手里的",
}


def _ids(x) -> set[str]:
    """驱力条件与行动里点到的一切字符串字段（实体 ID 混在其中，交给调用方与实体表求交）。"""
    import dataclasses
    if isinstance(x, (tuple, list, frozenset)):
        return set().union(*map(_ids, x)) if x else set()
    if dataclasses.is_dataclass(x):
        return set().union(*(_ids(getattr(x, f.name)) for f in dataclasses.fields(x)))
    return {x} if isinstance(x, str) else set()


def test_lint_lines_only_name_what_the_speaker_knows():
    """plan M2 test_drives::lint：台词与姿态里点的名字（名或别称），说话者在条件最早可能成立的那一刻都已认识——
    开场就认识的（先验）、驱力自己的条件与行动点到的（Knows/Holds/Go……成立即已认识）、条件里 At 的那处看得见的东西；
    其余逐条写进 _NAMED_ANYWAY 并说明理由。"""
    from tianlong.scenarios import build_wuliang_commoner
    from tianlong.scenarios.tianlong.drives_c import DRIVES_C
    sc = build_wuliang_commoner(7)
    st = sc.state
    forms = {eid: (e.name, *sc.aliases.get(eid, ())) for eid, e in st.entities.items()}
    visible = lambda place: {e for e in st.entities if st.target(e, Rel.AT) == place     # noqa: E731
                             and not st.attr(e, "hidden")}
    bad, used = [], set()
    for who, ds in DRIVES_C.items():
        prior = _mind(sc, who)
        for d in ds:
            ok = set(prior.entities) | _ids((d.when, d.do)) | {who}
            ok |= {e for c in d.when if isinstance(c, At) for e in visible(c.place)}
            for text in filter(None, _texts(d)):
                for eid, fs in forms.items():
                    if eid in ok or not any(f and f in text for f in fs):
                        continue
                    if (who, d.key, eid) in _NAMED_ANYWAY:
                        used.add((who, d.key, eid))
                    else:
                        bad.append((who, d.key, eid, text))
    assert not bad, bad
    assert used == set(_NAMED_ANYWAY), "例外表里不留用不上的条目"


def test_lint_once_drives_never_fire_again():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    from tianlong.scenarios import build_wuliang_commoner
    from tianlong.scenarios.tianlong.drives_c import DRIVES_C
    once = {(a, d.key) for a, ds in DRIVES_C.items() for d in ds if d.once}
    s = GameSession(build_wuliang_commoner(7), pipeline=False)
    while s.authority.head().clock < at(1, 22, 0):          # 从开场一直等到换班之后：一夜的名场面都已上演
        s.turn("等下去")
    marks = s.session_state()["drives"]
    fired = {(a, k): v for a, m in marks.items() for k, v in m.items() if (a, k) in once}
    assert len(fired) >= 10, "这一夜确有十多条一次性的驱力兑现"
    assert all(len(v) == 1 for v in fired.values()), {k: v for k, v in fired.items() if len(v) > 1}
