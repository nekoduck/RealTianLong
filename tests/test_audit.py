"""
[INPUT]: 依赖 language/gate 的 violations（与 narrator._Gate._found 同一条路），language/render 的 RenderPlan，
         language/scene 的 SceneBrief / Sky / MOON_*，tests/gate_fixtures 的 corpus（语料里的真实 plan / brief）
[OUTPUT]: M3 硬事实审计验收（设计 §7 M3 的 test_audit）：possession（阿顺手持北冥神功，钟灵“从你另一只手里抽过”违规、
          “盯着你手里的”通过；环顾回合玩家自己“抽出长剑”违规、本回合真拿到了或本来就在身上通过）、
          affordance（钟灵被制时“拍手笑道”违规、“眼珠一转，笑道”通过，代词不判、省略主语往前找；先出手后被制的人不判；
          感知/期盼之后的人才是施动者、“脚步声渐渐走远”不回溯）、possession 的逗号小句后省略主语照拦、只认宾语位置的东西、
          sky（18:20 月出之前“一轮明月升起”违规；19:45 月已在天上“月亮从峭壁后探出”违规、“月光洒在湖面”通过；
          sky=None 整项不查；lore 原文与引语不查；“还没落下/快要落下”是还在、比方里的月光不算），每条拦截都配着“相似但应通过”的阴性对照；
          sky 的端到端版：普通人版会话里 18:20 模型写“一轮明月升起”被拦、不交付（缺 LangGraph/Qdrant 跳过）
[POS]: tests 的硬事实审计规格。只依赖核心（端到端那一条除外，先跳过再导入会话），核心零依赖 CI 同样跑。每一条都走 gate.violations，测的是线上那条路
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tianlong.language.gate import violations
from tianlong.language.render import RenderPlan
from tianlong.language.scene import MOON_NONE, MOON_RISING, MOON_UP, SceneBrief, Sky

from .gate_fixtures import corpus

ME = "阿顺"
PLAN = RenderPlan(
    viewer="ashun", lines=("你站在湖畔。",), names=frozenset({ME, "钟灵", "北冥神功", "剑湖"}), statuses=frozenset(),
    arrivals=frozenset(), places=frozenset({"剑湖"}), items=(("北冥神功", 1),), hearsay=(), source="你站在湖畔。",
    viewer_name=ME, people=((ME, ME), ("钟灵", "钟灵"), ("灵儿", "钟灵")), speakers=frozenset(),
    done=((ME, "wait"),), here=frozenset({"剑湖"}), goods=(("北冥神功", "北冥神功"), ("帛卷", "北冥神功")),
    holdings=((ME, "北冥神功"),),
)
SUBDUED = replace(PLAN, statuses=frozenset({"subdued"}), afflicted=(("钟灵", "subdued"),))
DUSK = Sky("酉时，天色将暗未暗，月亮还没出来", night=False, moon=MOON_NONE)
MOONLIT = Sky("戌时，月亮已从峭壁后升起，照得湖面一片银白", night=True, moon=MOON_UP)
DARK = Sky("戌时，天已黑透，月亮还没出来", night=True, moon=MOON_NONE)
HELD = replace(PLAN, statuses=frozenset({"subdued"}), afflicted=((ME, "subdued"),))


QUIET = SceneBrief()


def _kinds(piece: str, plan: RenderPlan = PLAN, brief: SceneBrief = QUIET, before: str = "") -> set[str]:
    return {v.kind for v in violations(piece, before + piece, before, plan, brief, frozenset(), "")}


# ============================================================
#  possession：易手须对得上本回合的 TAKE / GIVE，或东西本来就在他身上
# ============================================================


@pytest.mark.parametrize("piece", [
    "钟灵伸手从你另一只手里抽过了那卷北冥神功。",
    "钟灵凑到你身边，一把夺过北冥神功，揣进怀里。",
    "你把北冥神功递给了钟灵。",
])
def test_possession_blocks_hands_changing_that_the_kernel_never_settled(piece):
    assert "possession" in _kinds(piece), piece


@pytest.mark.parametrize("piece", [
    "钟灵凑过来盯着你手里的北冥神功。",
    "你把北冥神功揣进怀里，抬眼看着钟灵。",          # 本来就在他身上
    "钟灵伸手要抽过你手里的北冥神功，你往后一缩。",    # 意图，不是做成
    "钟灵没能夺过北冥神功。",                        # 否定
    "她一把抽过北冥神功。",                          # 代词：不猜
    "钟灵接过茶碗，抿了一口。",                      # 没点名东西：修辞里的茶碗
])
def test_possession_lets_similar_sentences_through(piece):
    assert "possession" not in _kinds(piece), piece


def test_possession_passes_when_the_kernel_did_move_the_thing():
    took = replace(PLAN, done=((ME, "wait"), ("钟灵", "take")), holdings=(("钟灵", "北冥神功"),))
    assert "possession" not in _kinds("钟灵伸手从你手里抽过了北冥神功。", took)
    gave = replace(PLAN, done=((ME, "give"), ("钟灵", "receive")), holdings=(("钟灵", "北冥神功"),))
    assert "possession" not in _kinds("你把北冥神功递给了钟灵，钟灵接过北冥神功。", gave)


def test_looking_around_never_puts_the_sword_in_your_hand():
    """seed/possession/01：环顾回合（只有 INSPECT 成功）玩家“抽出长剑”——INSPECT 在内核里只让人看见东西在哪，从不易手。"""
    look = next(c for c in corpus() if c.id == "run5/playthrough/1")
    assert ("段誉", "inspect") in look.plan.done and ("段誉", "take") not in look.plan.done
    found = violations("你顺手从兵器架上抽出一柄长剑，握在手中。", "你顺手从兵器架上抽出一柄长剑，握在手中。", "",
                       look.plan, look.brief, look.known, look.command)
    assert "possession" in {v.kind for v in found}
    mine = violations("你从怀里掏出易经，翻了两页。", "你从怀里掏出易经，翻了两页。", "", look.plan, look.brief, look.known,
                      look.command)
    assert [v for v in mine if v.kind in ("possession", "outcome")] == [], "易经本来就在他身上"


@pytest.mark.parametrize("piece", [
    "钟灵一伸手，便从你手里抽过北冥神功。",           # “从你手里”的你是宾语：主语省略，往前找到钟灵
    "钟灵手一伸，从你手里抽过北冥神功。",
    "钟灵看着你，一把抽过北冥神功。",                 # 只是看着你，抽过的仍是钟灵
    "钟灵一伸手，便把北冥神功抽了过去。",
])
def test_possession_finds_the_omitted_subject_before_the_comma(piece):
    assert "possession" in _kinds(piece), piece


@pytest.mark.parametrize("plan, piece", [
    (replace(PLAN, holdings=(), done=((ME, "take"),)), "钟灵看着你从湖边抽出北冥神功。"),     # 抽出的是她看着的你
    (replace(PLAN, done=((ME, "give"), ("钟灵", "receive"))), "钟灵看着你把北冥神功递给了她。"),
    (PLAN, "钟灵接过话头说起北冥神功的来历。"),                                            # 接过的是话头
    (PLAN, "钟灵收回盯着北冥神功的目光。"),                                                # 收回的是目光
])
def test_possession_credits_the_watched_and_the_real_object(plan, piece):
    assert "possession" not in _kinds(piece, plan), piece


# ============================================================
#  affordance：被制的人没有肢体动作；开口、眼神、神情、笑都可以
# ============================================================


@pytest.mark.parametrize("piece", [
    "钟灵拍手笑道：“好玩！”",
    "钟灵霍地起身，扑到你跟前。",
    "钟灵被点了穴道，却拍手笑道：“书呆子！”",       # 省略主语的小句往前找到钟灵
])
def test_affordance_blocks_a_subdued_body_moving(piece):
    assert "affordance" in _kinds(piece, SUBDUED), piece


@pytest.mark.parametrize("piece", [
    "钟灵眼珠一转，笑道：“书呆子，你倒会躲。”",
    "钟灵扑哧一笑，眼神里满是得意。",
    "钟灵想抬手，却半点也动不了。",                 # 意图
    "钟灵站不起来，只拿眼睛瞪你。",                 # 可能补语的否定
    "她拍手笑道：“好玩！”",                         # 代词：不猜
    "你拍手笑道。",                                 # 被制的不是你
])
def test_affordance_lets_eyes_voice_and_others_through(piece):
    assert "affordance" not in _kinds(piece, SUBDUED), piece


@pytest.mark.parametrize("plan, piece", [
    (HELD, "你看着钟灵转身走开。"),                     # 感知之后的人才是施动者
    (HELD, "你眼睁睁看着钟灵拂袖而去，走到湖边。"),     # 前一小句“看着某人做某事”：不回溯到你
    (HELD, "你只盼钟灵快些走。"),
    (HELD, "你动弹不得，只听见脚步声走远。"),           # 内嵌的非人主语
    (HELD, "你动弹不得，脚步声渐渐走远。"),             # 小句自带非人主语
    (SUBDUED, "钟灵看着你转身离开。"),
    (SUBDUED, "钟灵见你走来，眼里一亮。"),
    (SUBDUED, "钟灵眼看你走近，脸上一红。"),
    (SUBDUED, "钟灵只盼你快些走。"),
])
def test_affordance_credits_the_one_being_watched(plan, piece):
    assert "affordance" not in _kinds(piece, plan), piece


@pytest.mark.parametrize("piece", [
    "钟灵看着你，转身便走。",                           # 只是看着你：走的仍是钟灵
    "钟灵见状，拍手笑道：“好玩！”",                     # “见状”不是内嵌小句
    "你只见钟灵拍手大笑。",                             # 感知之后的钟灵就是拍手的人
    "钟灵翻身坐起。",
])
def test_affordance_still_blocks_the_watcher_moving(piece):
    assert "affordance" in _kinds(piece, SUBDUED), piece


def test_affordance_only_for_the_subdued():
    assert "affordance" not in _kinds("钟灵拍手笑道：“好玩！”")


def test_affordance_spares_whoever_moved_before_being_subdued():
    """run3–6 的 sycophancy：你挥拳打去被化开、随即被点了穴道——出手在前，被制在后；没动过手脚的钟灵照拦。"""
    me = replace(PLAN, statuses=frozenset({"subdued"}), afflicted=((ME, "subdued"), ("钟灵", "subdued")))
    after = SceneBrief(astir=(ME,))
    assert "affordance" not in _kinds("你一步抢上，挥拳直取钟灵。", me, after)
    assert "affordance" in _kinds("你一步抢上，挥拳直取钟灵。", me), "没有 astir：照拦"
    assert "affordance" in _kinds("钟灵拍手笑道：“好玩！”", me, after)


# ============================================================
#  sky：月出之前没有月亮，月已在天上不许再升一次，入夜没有日头；sky=None 不查
# ============================================================


def test_sky_moon_before_moonrise_is_blocked():
    assert "sky" in _kinds("一轮明月升起，照得湖面发亮。", brief=SceneBrief(sky=DUSK))
    assert "sky" in _kinds("月光洒在湖面上。", brief=SceneBrief(sky=DUSK))
    assert "sky" not in _kinds("天边还不见月亮，湖面一片昏暗。", brief=SceneBrief(sky=DUSK))
    assert "sky" not in _kinds("月亮还没出来，湖面一片昏暗。", brief=SceneBrief(sky=DUSK))


def test_sky_moon_rises_only_once():
    assert "sky" in _kinds("月亮从峭壁后探出，照亮了玉璧。", brief=SceneBrief(sky=MOONLIT))
    assert "sky" not in _kinds("月光洒在湖面，碎银一般。", brief=SceneBrief(sky=MOONLIT))
    assert "sky" not in _kinds("月亮早已升起，挂在峭壁之上。", brief=SceneBrief(sky=MOONLIT)), "说的是此刻在天上"
    rising = replace(MOONLIT, moon=MOON_RISING)
    assert "sky" not in _kinds("月亮从峭壁后探出，照亮了玉璧。", brief=SceneBrief(sky=rising)), "本回合正是月出"


def test_sky_no_sun_at_night():
    assert "sky" in _kinds("夕阳斜照在湖面上。", brief=SceneBrief(sky=MOONLIT))
    assert "sky" not in _kinds("夕阳早已沉下山去，湖上只剩月色。", brief=SceneBrief(sky=MOONLIT))
    assert "sky" not in _kinds("夕阳斜照在湖面上。", brief=SceneBrief(sky=DUSK)), "黄昏另议：还没入夜"


@pytest.mark.parametrize("piece", [
    "夕阳还没落下。", "太阳还没下山，湖面金光闪闪。", "夕阳尚未落尽，余晖洒在湖面。",   # 否定后接落/下山：此刻还在
    "月亮还没落下，挂在峭壁上。", "月亮快要落下去了。",
    "像月光一样洒在湖面上的月光，冷冷清清。",                                       # 比方收住之后的月光是实物
])
def test_sky_still_there_is_not_absent(piece):
    assert "sky" in _kinds(piece, brief=SceneBrief(sky=DARK)), piece


@pytest.mark.parametrize("piece", [
    "月亮还没出来，湖面一片昏暗。", "夕阳早已沉下山去，湖面一片昏暗。",
    "钟灵的脸白得像月光。", "钟灵的笑容像月色一样温柔。", "钟灵脸上绽开阳光般的笑容。", "火光照得湖面通红，如同落日。",
])
def test_sky_spares_absence_and_likeness(piece):
    assert "sky" not in _kinds(piece, brief=SceneBrief(sky=DARK)), piece


def test_sky_ignores_quotes_lore_and_old_worlds():
    assert "sky" not in _kinds("一轮明月升起，照得湖面发亮。"), "sky=None：旧版不查"
    assert "sky" not in _kinds("钟灵笑道：“等月亮出来，玉璧上就有仙人。”", brief=SceneBrief(sky=DUSK)), "引语里的话"
    lore = replace(PLAN, scenery=(("玉璧", "月光照在玉璧上，壁上隐隐有人影"),))
    assert "sky" not in _kinds("月光照在玉璧上，壁上隐隐有人影。", lore, SceneBrief(sky=DUSK)), "lore 原文逐字"
    assert "sky" in _kinds("月光照在湖面上。", lore, SceneBrief(sky=DUSK)), "lore 之外的月光照拦"


def test_sky_end_to_end_in_the_commoner_session():
    """普通人版会话里天色真的接通了（runtime/staging.sky → SceneBrief.sky）：18:20 模型写“一轮明月升起”被拦、不交付。
    缺 LangGraph/Qdrant 跳过（会话要它们）。"""
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.core import at
    from tianlong.language.llm import ScriptedLLM
    from tianlong.runtime.session import GameSession
    from tianlong.scenarios import build_wuliang_commoner

    from .test_commoner import stage

    sc = build_wuliang_commoner(7)
    s = GameSession(stage(sc, {"ashun": "houyuan"}, at(1, 18, 20)), llm=ScriptedLLM(lambda *_: "一轮明月升起，照得满院银白。"),
                    pipeline=False)
    s.narrator.lead_after = None
    r = s.turn("四下打量一番")
    assert r.brief.sky is not None and r.brief.sky.moon == MOON_NONE and not r.brief.sky.night
    assert "sky" in {v.kind for v in r.render.violations} and "明月" not in r.narration
