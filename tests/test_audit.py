"""
[INPUT]: 依赖 language/gate 的 violations（与 narrator._Gate._found 同一条路），language/render 的 RenderPlan，
         language/scene 的 SceneBrief / Sky / MOON_*，tests/gate_fixtures 的 corpus（语料里的真实 plan / brief）
[OUTPUT]: M3 硬事实审计验收（设计 §7 M3 的 test_audit）：possession（阿顺手持北冥神功，钟灵“从你另一只手里抽过”违规、
          “盯着你手里的”通过；环顾回合玩家自己“抽出长剑”违规、本回合真拿到了或本来就在身上通过）、
          affordance（钟灵被制时“拍手笑道”违规、“眼珠一转，笑道”通过，代词不判、省略主语往前找；先出手后被制的人不判）、
          sky（18:20 月出之前“一轮明月升起”违规；19:45 月已在天上“月亮从峭壁后探出”违规、“月光洒在湖面”通过；
          sky=None 整项不查；lore 原文与引语不查），每条拦截都配着“相似但应通过”的阴性对照
[POS]: tests 的硬事实审计规格。只依赖核心，核心零依赖 CI 同样跑。每一条都走 gate.violations，测的是线上那条路
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


def test_sky_ignores_quotes_lore_and_old_worlds():
    assert "sky" not in _kinds("一轮明月升起，照得湖面发亮。"), "sky=None：旧版不查"
    assert "sky" not in _kinds("钟灵笑道：“等月亮出来，玉璧上就有仙人。”", brief=SceneBrief(sky=DUSK)), "引语里的话"
    lore = replace(PLAN, scenery=(("玉璧", "月光照在玉璧上，壁上隐隐有人影"),))
    assert "sky" not in _kinds("月光照在玉璧上，壁上隐隐有人影。", lore, SceneBrief(sky=DUSK)), "lore 原文逐字"
    assert "sky" in _kinds("月光照在湖面上。", lore, SceneBrief(sky=DUSK)), "lore 之外的月光照拦"
