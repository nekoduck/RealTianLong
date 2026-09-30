"""
[INPUT]: 依赖 scenarios 的 SCENARIOS / build_wuliang / build_wuliang_commoner，scenarios/tianlong/commoner 的 ENDINGS_C / SECRETS_C / EPITHETS，
         scenarios/tianlong/drives_c 的 DRIVES_C 与时刻，scenarios/tianlong/stagecraft 的 BEAT_KEYS，core 的 at / relocate / Op / Outcome / Rel，
         kernel 的 violations、perception 的 scene_percept，runtime/authority 的 WorldAuthority，runtime/session 的 GameSession（模板模式），
         runtime/endings 的 chronicle / names_for，scripts/sim_beats.py（按文件加载：脚本化玩家与统计）
[OUTPUT]: 普通人版验收（plan §7 M2）：legacy_pin（旧版 build_wuliang 的初始世界与初始认知逐字节不变、注册为 wuliang-duanyu）、
          世界自洽与内容（阿顺是玩家、段誉是没有目标的 NPC、先验让段誉与马五德对阿顺 +1 而阿顺不对东家生分、关卡锁着）、
          opening（被动玩家 15 tick 内：龚光杰对段誉叫阵或出手、钟灵出手打龚光杰、段誉从不动手）、bargain（20 个种子里 ≥16 个：
          钟灵离开大殿前没被制、龚光杰的毒由她的 USE 解开、没人从她身上搜走解药、左子穆说了 AGREE）、story_runs_offscreen
          （被动玩家下段誉 ≥12/20 到琅嬛、≥8/20 到澜沧江；纪事只写真相里发生过的事）、follower_beats（跟随型 ≥16/20 目击 ≥8/11 看点）、
          divergence 1–5（撒谎引开龚光杰、抢先拿走帛卷则段誉只能恳求、带段誉去山道就不跳崖、落单撞见私奔被打而求饶有人求情、
          两人同在则不灭口）、one_way_restraint（龚光杰从不经断崖、并说出作罢的话）、guard_supper（20:30 下山成功落幕“第一幕终 · 下山”，
          18:00 挨一下过不去，塞了碎银放行）；统计版标 slow，各留一个单种子冒烟版；stage() 供别的测试把人挪到某处、把时钟拨到某刻重新建档；
          审查回归：带字姿态只写成看得见的那一行（不做成“对众人道”、不套引号），马五德的差遣是 EXPLAIN（阿顺对东家不降、
          没有“向马五德赔罪”），茶饼碎银不劝人研读
[POS]: tests 的普通人版内容层：剧情不是写死的脚本，这里断言的是“会发生”的分布，与改变它的分歧
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from tianlong.core import Op, Outcome, Rel, Social, at, digest, relocate
from tianlong.core.profiles import GoalKind
from tianlong.kernel import violations
from tianlong.kernel.perception import scene_percept
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import SCENARIOS, Scenario, build_wuliang, build_wuliang_commoner
from tianlong.scenarios.tianlong.commoner import ENDINGS_C, EPITHETS, SECRETS_C
from tianlong.scenarios.tianlong.drives_c import DRIVES_C, HUNT, MOONRISE, SUPPER
from tianlong.scenarios.tianlong.stagecraft import BEAT_KEYS

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(1, 21)
# 旧版无量山的初始世界与初始认知：取自 M2 动工之前（6792d8e）的 build_wuliang，重构只许参数化，不许改一个字节
LEGACY_STATE = {7: "54428a31de89ee8c13cdbd12f4b8933a", 11: "ba1a334a9178d201e2a77b4539e5e3ea"}
LEGACY_PRIORS = "b72fbb40ad89bb09751be7759f9522a2"


def stage(sc: Scenario, where: dict[str, str], clock: int) -> Scenario:
    """把人挪到某处、把时钟拨到某刻重新建档：每人最后那条“此刻所见”按新世界重算，更早的先验原样保留。"""
    st = sc.state
    changes = [c for who, place in where.items() for c in relocate(who, st.target(who, Rel.AT), place)]
    st = st.apply(changes).stamp(st.version, clock)
    priors = {a: (*ps[:-1], replace(scene_percept(st, a), tick=clock - 1)) for a, ps in sc.priors.items()}
    return replace(sc, state=st, priors=priors)


# ============================================================
#  旧版钉死与内容（不需要会话）
# ============================================================


def test_legacy_pin():
    for seed, fp in LEGACY_STATE.items():
        sc = build_wuliang(seed)
        assert sc.state.fingerprint() == fp, "旧版初始世界逐字节不变"
        assert digest(repr(sorted((a, repr(p)) for a, p in sc.priors.items()))) == LEGACY_PRIORS, "旧版初始认知不变"
        assert sc.world_id == "wuliang" and sc.player == "duanyu" and not sc.drives and sc.kowtow_ticks == 1
        assert not (sc.moments or sc.beats or sc.chronicle or sc.epithets or sc.introduced)
    assert SCENARIOS["wuliang-duanyu"] is build_wuliang and SCENARIOS["wuliang"] is build_wuliang_commoner


def test_commoner_world_and_content():
    sc = build_wuliang_commoner(7)
    assert not violations(sc.state) and sc.world_id == "wuliang_c" and sc.player == "ashun"
    assert set(sc.profiles) == {e.id for e in sc.state.of_kind(sc.state.entity("ashun").kind)}
    duanyu = sc.profiles["duanyu"]
    assert not duanyu.is_player and not duanyu.goals and duanyu.temper < 0, "段誉降为没有目标的 NPC：弧线由驱力承载"
    assert {g.kind for g in sc.profiles["ashun"].goals} == {GoalKind.DEFEND, GoalKind.ESCAPE}
    assert sc.state.attr("d_downhill", "locked") and sc.state.target("guanyao", Rel.AT) == "shennong"
    assert all(k.split("@")[0] in sc.state.entities for k in sc.lore), "外观描写只针对存在的实体"
    for eid in ("mawude", "duanyu", "zhongling", "ganguanghao", "gongguangjie"):
        assert sc.state.entity(eid).name not in sc.lore[eid], "人物外观不带名字：名字要有人道出才算知道"
    assert "月" not in sc.lore["d_cave"] and "人影" in sc.lore["yubi@moon"] and "人影" not in sc.lore["yubi@night"]
    assert sc.setting.endswith("龚光杰霍地转过身，怒目朝你们这一席瞪来。")
    assert len(sc.guide) == 8 and sc.kowtow_ticks == 6 and sc.moments["moon"] == MOONRISE
    assert {e.key for e in sc.endings} == {"river", "downhill", "dawn"} and sc.endings == ENDINGS_C
    assert set(EPITHETS) == set(sc.epithets) and "zhongling" not in sc.introduced["ashun"]
    assert {"duanyu", "mawude", "gongguangjie"} <= sc.introduced["ashun"]
    assert all(s in SECRETS_C for s in ("王子", "世子", "镇南王", "段正淳"))
    assert set(sc.drives) == set(DRIVES_C) and tuple(dict.fromkeys(b.key for b in sc.beats)) == BEAT_KEYS
    assert len(BEAT_KEYS) == 11


def test_priors_make_duanyu_and_mawude_fond_of_ashun():
    auth = WorldAuthority.found(InMemoryWorldStore(), build_wuliang_commoner(7))
    mind = lambda a: auth.store.beliefs(auth.ref, a)       # noqa: E731
    assert mind("duanyu").attitude("ashun") == 1 and mind("mawude").attitude("ashun") == 1
    assert mind("ashun").attitude("mawude") == 0, "东家的叮嘱记在他自己那头：阿顺不因此对东家生分"
    assert not [o for a in ("duanyu", "mawude") for o in mind(a).obligations], "先验早于开场：不留待回应的义务"
    assert mind("ashun").knows("d_downhill") and not mind("ashun").knows("houshan"), "只认得上山的路与后院，不知后山"


# ============================================================
#  会话（模板模式）
# ============================================================


def _sim():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    if "sim_beats" not in sys.modules:
        spec = importlib.util.spec_from_file_location("sim_beats", ROOT / "scripts" / "sim_beats.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sim_beats"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["sim_beats"]


def _session(sc: Scenario | None = None):
    _sim()                                          # 会话要 LangGraph：核心零依赖环境先跳过，再导入
    from tianlong.runtime.session import GameSession
    return GameSession(sc or build_wuliang_commoner(7), pipeline=False)


def _until(s, clock: int) -> None:
    while s.authority.head().clock < clock:
        s.turn("等一会儿" if clock - s.authority.head().clock >= 10 else "等待")


def _events(s, actor: str | None = None, op: Op | None = None):
    return [e for e in s.store.events(s.ref) if (actor is None or e.actor == actor) and (op is None or e.op == op)]


@cache
def _openings() -> tuple:
    """20 个种子，被动玩家原地等 20 个 tick：开场冲突链与讨价还价在这段时间里定局。"""
    out = []
    for seed in SEEDS:
        s = _session(build_wuliang_commoner(seed))
        for _ in range(20):
            s.turn("等待")
        out.append(tuple(s.store.events(s.ref)))
    return tuple(out)


def test_opening():
    for events in _openings():
        early = [e for e in events if e.tick <= at(1, 17, 40) + 15]
        assert any(e.actor == "gongguangjie" and e.intent.target == "duanyu" and (
            e.op == Op.ATTACK or (e.op == Op.TELL and e.intent.social == Social.CHALLENGE)) for e in early)
        assert any(e.actor == "zhongling" and e.op == Op.ATTACK and e.intent.target == "gongguangjie" for e in early)
        assert not [e for e in events if e.actor == "duanyu" and e.op == Op.ATTACK], "段誉从不动手"


def test_poses_are_seen_not_voiced():
    """驱力的带字姿态只写成看得见的那一行：不做成“X对众人道：“<姿态>””，也不给姿态里的引语再套一层引号。"""
    from tianlong.core import Pose
    poses = {a.text for ds in DRIVES_C.values() for d in ds for a in d.do if isinstance(a, Pose)}
    s = _session()
    body = []
    for _ in range(8):                                      # 17:41–17:48：中毒的龚光杰、护着书呆子的钟灵都摆了姿态
        r = s.turn("等待")
        assert not [vl for vl in r.brief.lines if vl.op == Op.WAIT.value], "姿态不是台词"
        body.append(r.narration)
    text = "\n".join(body)
    assert "看见龚光杰捂着伤处" in text, "姿态照样看得见"
    assert not [p for p in poses if f"“{p}" in text], "姿态原文不出现在引语里"
    assert not re.search(r"“[^”]*“", text), "没有套引号"


def test_the_errand_is_kind_and_the_opening_suggests_no_studying_tea():
    """马五德差阿顺去后院是叮嘱（EXPLAIN）不是喝令：阿顺不因此对东家生分、建议里没有“向马五德赔罪”；
    茶饼、碎银、火折子不劝人去研读。"""
    from tianlong.runtime.suggest import suggestions
    s = _session()
    assert not [t for t in suggestions(s.beliefs("ashun"), friends={"duanyu"}) if t.startswith("研读")]
    while not _events(s, "mawude", Op.TELL) or _events(s, "mawude", Op.TELL)[-1].intent.target != "ashun":
        assert s.authority.head().clock < at(1, 18, 0), "讲和之后马五德会差阿顺去后院"
        s.turn("等待")
    errand = _events(s, "mawude", Op.TELL)[-1]
    assert errand.intent.social == Social.EXPLAIN and errand.intent.utterance.startswith("阿顺")
    s.turn("等待")
    me = s.beliefs("ashun")
    assert me.attitude("mawude") >= 0
    assert "向马五德赔罪" not in suggestions(me, friends={"duanyu"})
    assert not [t for t in suggestions(me, friends={"duanyu"}) if t.startswith("研读")]


def test_bargain():
    ok = [_sim()._opening(events)[1] for events in _openings()]
    assert sum(ok) >= 16, f"讨价还价只在 {sum(ok)}/20 个种子里成立"


def test_story_runs_offscreen_smoke():
    sim = _sim()
    run = sim.play(7, "passive")
    assert run.langhuan and run.lancang and run.ending == "dawn"
    assert {"challenge", "beating", "marten", "bargain"} <= set(run.beats), "留在大殿的人只看得见开场"
    assert "cliff" not in run.beats and "kowtow" not in run.beats


@pytest.mark.slow
def test_story_runs_offscreen():
    sim = _sim()
    runs = [sim.play(seed, "passive") for seed in SEEDS]
    assert sum(r.langhuan for r in runs) >= 12 and sum(r.lancang for r in runs) >= 8


def test_follower_beats_smoke():
    run = _sim().play(1, "follower")
    assert len(run.beats) >= 8 and run.ending == "river", run


@pytest.mark.slow
def test_follower_beats():
    sim = _sim()
    runs = [sim.play(seed, "follower") for seed in SEEDS]
    assert sum(len(r.beats) >= 8 for r in runs) >= 16


# ============================================================
#  分歧：玩家做了原著之外的事，世界照样接得住
# ============================================================


def test_divergence_liar_sends_the_hunter_up_the_mountain_road():
    sim, s = _sim(), _session()
    memo: dict = {}
    while not memo.get("lied") and s.authority.head().clock < HUNT + 30:
        s.turn(sim.PLAYERS["liar"](s, memo))
    assert memo.get("lied"), "龚光杰向阿顺打听段公子的下落"
    lie = max(e.tick for e in _events(s, "ashun", Op.TELL))
    for _ in range(6):
        s.turn("等待")
    moves = [e for e in _events(s, "gongguangjie", Op.MOVE) if e.tick > lie and e.outcome == Outcome.SUCCESS]
    assert moves and moves[0].intent.target == "hall", "下一步朝山道的方向去（经大殿）"
    assert any(e.intent.target == "shandao" for e in moves), "撞上把守山道的帮众"


def test_divergence_grabber_takes_the_scroll_and_duanyu_can_only_plead():
    sim, s = _sim(), _session()
    memo: dict = {}
    beats: set[str] = set()
    while s.authority.head().target("scroll_lb", Rel.AT) != "ashun" and s.authority.head().clock < at(1, 21, 0):
        beats |= set(s.turn(sim.PLAYERS["grabber"](s, memo)).beats)
    assert s.authority.head().target("ashun", Rel.AT) == "langhuan", "跟着段公子进了琅嬛福地、抢先拿走帛卷"
    for _ in range(20):
        s.turn("等待")
    takes = [e for e in _events(s, "duanyu", Op.TAKE) if e.intent.target == "scroll_lb"]
    assert not [e for e in takes if e.outcome == Outcome.SUCCESS], "阿顺先拿走了凌波微步：段誉拿不到"
    assert any(e.op == Op.TELL and e.intent.social == Social.PLEAD and e.intent.target == "ashun"
               for e in _events(s, "duanyu")), "他只能恳求借阅"
    assert "scroll" in beats


def test_divergence_leading_duanyu_to_the_road_means_no_leap():
    s = _session()
    s.turn("去山道")
    for _ in range(60):
        s.turn("等待")
    assert s.authority.head().target("duanyu", Rel.AT) != "yading"
    assert any(e.op == Op.MOVE and e.intent.target == "shandao" for e in _events(s, "duanyu")), "段誉跟着阿顺走"
    assert not [e for e in _events(s, "duanyu", Op.MOVE) if e.intent.obj == "d_cliff"], "60 tick 内不经断崖"


def _lone_witness(extra: dict[str, str]):
    """19:15 阿顺在后山、私奔那对在后院（龚光杰落在崖底：上不来，搅不了局）。"""
    where = {"ashun": "houshan", "ganguanghao": "houyuan", "geguangpei": "houyuan", "gongguangjie": "jianhu", **extra}
    return _session(stage(build_wuliang_commoner(7), where, at(1, 19, 15)))


def _struck(s) -> list:
    return [e for e in _events(s, op=Op.ATTACK) if e.intent.target == "ashun" and e.actor in ("ganguanghao", "geguangpei")]


def test_divergence_lone_witness_is_attacked_and_mercy_holds_the_blade():
    s = _lone_witness({})
    while s.authority.head().target("ganguanghao", Rel.AT) != "houshan" and s.authority.head().clock < at(1, 19, 30):
        s.turn("等待")
    s.turn("向干光豪求饶：饶命啊")                       # 那一对一进后山就开口求饶
    assert _struck(s), "19:21 独自在后山撞见私奔那对：灭口"
    plea = max(e.tick for e in _events(s, "ashun", Op.TELL))
    for _ in range(4):
        s.turn("等待")
    assert any(e.op == Op.TELL and e.intent.social == Social.PLEAD and e.intent.target == "ganguanghao"
               for e in _events(s, "geguangpei")), "西宗女弟子替他求情"
    assert not [e for e in _events(s, "ganguanghao", Op.ATTACK) if e.tick > plea], "干光豪不再动手"


def test_divergence_two_witnesses_are_left_alone():
    s = _lone_witness({"duanyu": "houshan"})
    while s.authority.head().clock < at(1, 19, 30):
        s.turn("等待")
    assert s.authority.head().target("ganguanghao", Rel.AT) == "camp", "那一对照样动身，投到营里"
    assert not [e for e in _events(s, op=Op.ATTACK) if e.actor in ("ganguanghao", "geguangpei")], "与段誉同在后山：不动手"


# ============================================================
#  单向门的克制与关卡
# ============================================================

_QUIT = tuple(d.do[0].text for d in DRIVES_C["gongguangjie"] if d.key.startswith("give_up"))


def _restraint(seed: int) -> tuple[bool, bool]:
    s = _session(build_wuliang_commoner(seed))
    _until(s, MOONRISE)
    ggj = _events(s, "gongguangjie")
    return (not [e for e in ggj if e.op == Op.MOVE and e.intent.obj == "d_cliff"],
            any(e.intent.utterance in _QUIT for e in ggj))


def test_one_way_restraint_smoke():
    assert _restraint(7) == (True, True)


@pytest.mark.slow
def test_one_way_restraint():
    for seed in SEEDS:
        assert _restraint(seed) == (True, True), seed


def test_guard_supper():
    from tianlong.runtime.endings import title_for
    s = _session()
    _until(s, at(1, 18, 0))
    s.turn("悄悄溜去山道")
    s.turn("去山脚")
    head = s.authority.head()
    assert head.target("ashun", Rel.AT) == "shandao" and head.attr("ashun", "wounded"), "18:00：挨帮众一下，关卡过不去"
    s.turn("把碎银给帮众")
    r = s.turn("去山脚")
    assert r.ending is not None and r.ending.key == "downhill", "塞了碎银，帮众开锁放行"

    s = _session()
    _until(s, SUPPER + 30)
    s.turn("悄悄溜去山道")
    r = s.turn("去山脚")
    assert r.ending is not None and r.ending.key == "downhill"
    assert title_for(r.ending, s.authority.head(), "ashun") == "第一幕终 · 下山"
    assert re.search(r"第一幕终 · 下山", s.epilogue())



# ============================================================
#  解释与建议：跟上认识的人、拉着人走、多 tick 叩首；同伴一族的行动建议
# ============================================================


def _mind(sc: Scenario, agent: str = "ashun"):
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    return auth.store.beliefs(auth.ref, agent)


def test_follow_and_tow_fast_paths():
    from tianlong.language.interpret import Interpreter
    sc = build_wuliang_commoner(7)
    it = Interpreter(None, aliases=sc.aliases, kowtow_ticks=sc.kowtow_ticks)
    me = _mind(sc)
    assert it.interpret("跟上段公子", me).clarification == "段誉就在你身边。"
    tow = it.interpret("拉着段公子逃去后院", me).candidate
    assert (tow.op, tow.target, tow.obj) == (Op.MOVE, "houyuan", "d_corridor")


def test_kowtow_spans_the_scenes_ticks_before_the_bow_turns_to_a_search():
    from tianlong.language.interpret import Interpreter
    sc = build_wuliang_commoner(7)
    me = _mind(stage(sc, {"ashun": "langhuan"}, at(1, 20, 0)))
    p = Interpreter(None, aliases=sc.aliases, kowtow_ticks=6).interpret("磕头", me)
    steps = (p.candidate, *p.followups)
    assert len(steps) == 6 and [c.op for c in steps] == [Op.WAIT] * 5 + [Op.INSPECT]
    assert all(c.social == Social.SUBMIT for c in steps) and p.utterance
    legacy = Interpreter(None, aliases=sc.aliases).interpret("磕头", me)
    assert legacy.candidate.op == Op.INSPECT and not legacy.followups, "kowtow_ticks=1：一拜即伏地细看（旧版）"
    hall = Interpreter(None, aliases=sc.aliases, kowtow_ticks=6).interpret("磕头", _mind(sc))
    assert hall.candidate.op == Op.WAIT and not hall.followups, "大殿里没有可拜的陈设：当众服软，不展开"


def test_kowtow_plan_bows_each_tick_with_words():
    s = _session(stage(build_wuliang_commoner(7), {"ashun": "langhuan"}, at(1, 20, 0)))
    s.turn("磕头")
    bows = [e for e in _events(s, "ashun") if e.op == Op.WAIT and e.intent.social == Social.SUBMIT]
    assert len(bows) == 5 and all(e.intent.utterance for e in bows), "每一拜都看得见"
    assert _events(s, "ashun", Op.INSPECT)[-1].intent.target == "langhuan"


def test_companion_suggestions():
    from tianlong.runtime.suggest import suggestions
    s = _session()
    for _ in range(3):
        s.turn("等待")
    me = s.beliefs("ashun")
    assert "向龚光杰替段誉求情" in suggestions(me, friends={"duanyu"}), "同伴挨了打：先替他求情"
    assert "向龚光杰替段誉求情" not in suggestions(me), "没有同伴就不提"
    while s.authority.head().target("duanyu", Rel.AT) == "hall":
        s.turn("等待")
    assert "跟上段誉" in suggestions(s.beliefs("ashun"), friends={"duanyu"}), "同伴刚走开：跟上他"
    left = [e for e in _events(s, "duanyu", Op.MOVE) if e.outcome == Outcome.SUCCESS][-1]
    follow = s.interpreter.interpret("跟上段誉", s.beliefs("ashun"))
    assert follow.source == "rules" and follow.candidate.op == Op.MOVE, "快路径，不经模型"
    assert (follow.candidate.target, follow.candidate.obj) == (left.intent.target, left.intent.obj), "走他走的那道门"
