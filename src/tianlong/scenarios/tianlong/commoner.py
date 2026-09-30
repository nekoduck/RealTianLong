"""
[INPUT]: 依赖 core 的实体/关系/时间/命题/感知/言语行为类型，core/profiles 的 Goal / GoalKind / Profile，kernel/perception 的 make_percept /
         scene_percept，scenarios/base 的 Scenario / Ending / Variant，scenarios/tianlong/wuliang 的 entities / relations / layout /
         profiles / DOORS / PLACEMENT / OWNS / START / NIGHTFALL / SECT / EAST / WEST / SHENNONG / LOVERS / SECRETS，
         scenarios/tianlong/lore_commoner 的文字素材，scenarios/tianlong/drives_c 的 DRIVES_C 与时刻，scenarios/tianlong/stagecraft 的 BEATS_C
[OUTPUT]: 对外提供 build_wuliang_commoner()：无量山·普通人版（世界 ID wuliang_c），ENDINGS_C / SECRETS_C / CHRONICLE / EPITHETS / MOMENTS
[POS]: scenarios/tianlong 第一幕的普通人版：玩家是马五德茶号的伙计阿顺（martial .05），段誉降为同行的 NPC（没有目标，
       整条弧线由驱力承载，不还手）。世界在旧版之上加了阿顺、无量山脚（经山道尽头神农帮扎的木栅关卡相连：栅门上锁，钥匙在帮众身上——
       他们守着时一个挑夫闯不过去，夜饭换班或收了碎银才开）、茶饼、火折子、碎银与关卡钥匙。
       先验照 game_master §8 第 1 条：段誉、马五德喜欢阿顺，放在**他们**的先验里（阿顺向他们见礼/道谢的言语感知），
       他们自己的见礼与叮嘱记成 SELF 先验；阿顺的先验只有路与眼前所见，不带马五德的“叮嘱”（免得他对东家 −1）。
       先验都比开场早 30 个 tick，不留待回应的义务
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

from tianlong.core import (
    Entity,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Proposition,
    Rel,
    Relation,
    Social,
    WorldState,
)
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.scenarios.base import Ending, Scenario, Variant
from tianlong.scenarios.tianlong.drives_c import DAWN, DRIVES_C, MOONRISE
from tianlong.scenarios.tianlong.lore import STYLE
from tianlong.scenarios.tianlong.lore_commoner import (
    ALIASES_C,
    COMMON_WORDS_C,
    EPILOGUE_DAWN_C,
    EPILOGUE_DOWNHILL_C,
    EPILOGUE_RIVER_C,
    GUIDE_AT_C,
    GUIDE_C,
    HINTS_C,
    LORE_C,
    SETTING_C,
)
from tianlong.scenarios.tianlong.stagecraft import BEATS_C
from tianlong.scenarios.tianlong.wuliang import (
    DOORS,
    EAST,
    LOVERS,
    NIGHTFALL,
    OWNS,
    PLACEMENT,
    SECRETS,
    SECT,
    SHENNONG,
    START,
    WEST,
    entities,
    layout,
    profiles,
    relations,
)

# ============================================================
#  地图：旧版之上，山道尽头多一道关卡通往山脚
#
#       无量山脚 ══关卡（木栅，上锁；钥匙在帮众身上）══ 无量山山道 ──宫门── 剑湖宫大殿 ── …（同旧版）
# ============================================================

MOMENTS = {"night": NIGHTFALL, "moon": MOONRISE, "dawn": DAWN}
DOORS_C = {**DOORS, "d_downhill": ("shandao", "shanjiao")}
PLACEMENT_C = {**PLACEMENT, "ashun": "hall", "chabing": "ashun", "huozhezi": "ashun", "suiyin": "ashun",
               "guanyao": "shennong"}
OWNS_C = (*OWNS, ("ashun", "chabing"), ("ashun", "huozhezi"), ("ashun", "suiyin"), ("shennong", "guanyao"))
# 熟读原著的模型最容易说破的是段誉的身世：与旧版的秘密一样只交给叙述闸门
SECRETS_C = (*SECRETS, "王子", "世子", "镇南王", "段正淳")
CHRONICLE = ("duanyu", "zhongling", "ganguanghao", "geguangpei", "gongguangjie")   # 终章纪事取材的角色
# 没人道出姓名之前怎样称呼（帮众本来就是泛称）
EPITHETS = {"zhongling": "梁上的青衫少女", "ganguanghao": "高个子的东宗弟子", "geguangpei": "清秀的西宗女弟子",
            "sikongxuan": "须发花白的药农头领"}
_MATCH = ("gongguangjie", "zuozimu", "xinshuangqing")      # 比剑时唱过名的人：满堂都知其名


def _extra() -> tuple[Entity, ...]:
    P, It, D, H = Kind.PLACE, Kind.ITEM, Kind.DOOR, Kind.PERSON
    return (
        Entity.make("shanjiao", P, "无量山脚"),
        Entity.make("d_downhill", D, "山道关卡", locked=True),
        Entity.make("chabing", It, "普洱茶饼", small=True),
        Entity.make("huozhezi", It, "火折子", small=True),       # 纯作风味：补上“来历不明的光”，没有内核属性
        Entity.make("suiyin", It, "碎银", small=True),
        Entity.make("guanyao", It, "关卡钥匙", small=True),
        Entity.make("ashun", H, "阿顺", martial=0.05, agility=0.5, alertness=0.55),
    )


def _introduced() -> dict[str, frozenset[str]]:
    """开场谁已认得谁：唱过名的比剑三人满堂皆知；同门、同帮彼此相识；阿顺、段誉、马五德同行；马五德与各派都有交情。"""
    everyone = ("ashun", "duanyu", "mawude", *SECT, *SHENNONG, "zhongling")
    known: dict[str, set[str]] = {a: set(_MATCH) for a in everyone}
    for group in (EAST + WEST, SHENNONG + LOVERS, ("ashun", "duanyu", "mawude")):
        for a in group:
            known[a] |= set(group)
    for a in SECT:
        known[a].add("mawude")
    return {a: frozenset(k - {a}) for a, k in known.items()}


# ============================================================
#  角色：旧版设定之上改三处——段誉降为 NPC、阿顺是玩家、各人对阿顺的称呼写进腔调
# ============================================================


def _profiles() -> dict[str, Profile]:
    base = profiles()
    duanyu = base["duanyu"]
    out = {
        **base,
        "duanyu": replace(
            duanyu, goals=(), is_player=False, chatty=0.6, temper=-0.8,
            persona="大理段氏子弟，饱读诗书，生性仁厚，最厌习武；离家出走，随马五德上山观剑，宁可挨打也不肯还手",
            knows=duanyu.knows + "；与马五德同路上山，马五德的伙计阿顺一路挑着茶担，是个老实人",
            intro="大理来的书生，姓段"),
        "ashun": Profile(
            "ashun", "伙计", "普洱马五德茶号的伙计，替东家挑茶上山；东家叮嘱他照看同行的段公子",
            goals=(Goal(GoalKind.DEFEND, person="duanyu", weight=0.5), Goal(GoalKind.ESCAPE, home="shanjiao")),
            is_player=True, voice="老实本分的挑夫，说话直来直去，称人“爷”“公子”",
            knows="普洱马五德茶号的伙计，跟着东家跑过几趟茶山", intro="马五德茶号的伙计，挑茶上山的"),
        "mawude": replace(
            base["mawude"],
            knows=base["mawude"].knows + "；听说剑湖宫后头有面玉璧，月夜里显过仙人影子；伙计阿顺跟了他三年，手脚勤快"),
        "gongguangjie": replace(base["gongguangjie"], voice=base["gongguangjie"].voice + "；管阿顺叫“挑茶的”"),
        "zhongling": replace(base["zhongling"], voice=base["zhongling"].voice + "；管阿顺叫“小伙计”"),
        "shennong": replace(base["shennong"], persona="神农帮帮众，奉命把守山道、锁着下山的关卡，不许任何人离开",
                            knows="奉司空玄帮主之命把守山道，关卡的钥匙就在身上；神农帮的人个个识得毒草"),
    }
    return out


ENDINGS_C = (
    Ending("river", "第一幕终 · 澜沧江畔", "lancang", EPILOGUE_RIVER_C, variants=(
        Variant("身负奇功", ("skill:evasion", "skill:absorb")), Variant("与段公子同行", ("with:duanyu",)),
        Variant("怀揣帛卷", ("holds:scroll_lb", "holds:scroll_bm")))),
    Ending("downhill", "第一幕终 · 下山", "shanjiao", EPILOGUE_DOWNHILL_C,
           variants=(Variant("与段公子同行", ("with:duanyu",)),)),
    Ending("dawn", "第一幕终 · 天亮了", None, EPILOGUE_DAWN_C, at_clock=DAWN, variants=(
        Variant("身在剑湖宫", ("at:hall", "at:houyuan")), Variant("困在崖底", ("at:jianhu", "at:shidong")),
        Variant("身在石室", ("at:langhuan",)))),
)


def build_wuliang_commoner(seed: int = 7) -> Scenario:
    rels = relations(DOORS_C, PLACEMENT_C, OWNS_C, extra=(Relation("guanyao", Rel.MATCHES, "d_downhill"),))
    state = WorldState.build(seed, START, entities(*_extra()), rels)

    def past(facts: tuple[Fact, ...]):
        return replace(make_percept(state, Modality.SCENE, facts=facts), tick=START - 30)

    def now_seen(agent: str):
        return replace(scene_percept(state, agent), tick=START - 1)

    def said(speaker: str, listener: str, social: Social, modality: Modality, tick: int):
        """一句过去的话：听者记成 SPEECH（态度据此回暖），说话者自己记成 SELF（记下“说过”，开场不再重说一遍）。"""
        view = PerceivedEvent(Op.TELL.value, "hall", speaker, listener, None, Outcome.SUCCESS, None, None, None, social)
        who = speaker if modality == Modality.SPEECH else None
        return replace(make_percept(state, modality, view, (), informant=who, vantage="hall"), tick=tick)

    def ways(*doors: str) -> tuple[Fact, ...]:
        return layout(*doors, table=DOORS_C)

    palace = ways("d_gate", "d_corridor", "d_backgate", "d_path", "d_trail", "d_cliff")
    swords = (Fact(Proposition.rel("swordrack", Rel.AT, "hall")), Fact(Proposition.rel("sword", Rel.AT, "swordrack")))
    legend = (Fact(Proposition.rel("yubi", Rel.AT, "jianhu")),)
    priors: dict[str, tuple] = {a: (past(palace + swords + legend), now_seen(a)) for a in SECT}
    # 段誉：随马五德从山道上山；阿顺一路向他见礼，他也回过礼（开场不再招呼第二遍）
    priors["duanyu"] = (past(ways("d_gate")), said("ashun", "duanyu", Social.GREET, Modality.SPEECH, START - 30),
                        said("duanyu", "ashun", Social.GREET, Modality.SELF, START - 29), now_seen("duanyu"))
    # 马五德：伙计道过谢；茶担先挑去后院厨下是他自己吩咐过的（记在他自己这头，阿顺不因此对东家生分）
    priors["mawude"] = (past(ways("d_gate", "d_corridor")),
                        said("ashun", "mawude", Social.THANK, Modality.SPEECH, START - 30),
                        said("mawude", "ashun", Social.COMMAND, Modality.SELF, START - 29), now_seen("mawude"))
    # 阿顺：从山脚经关卡、山道上山；东家说过茶担要挑去后院厨下（回廊通后院）
    priors["ashun"] = (past(ways("d_downhill", "d_gate", "d_corridor")), now_seen("ashun"))
    priors["zhongling"] = (past(ways("d_gate")), now_seen("zhongling"))
    camp = ways("d_camproad", "d_trail", "d_gate", "d_downhill")
    gate = (Fact(Proposition.rel("guanyao", Rel.MATCHES, "d_downhill")), Fact(Proposition.attr("d_downhill", "locked", True)))
    priors["sikongxuan"] = (past(camp + (Fact(Proposition.rel("shennong", Rel.AT, "shandao")),)), now_seen("sikongxuan"))
    priors["shennong"] = (past(camp + gate + (Fact(Proposition.rel("sikongxuan", Rel.AT, "camp")),)), now_seen("shennong"))
    return Scenario("wuliang_c", state, _profiles(), priors, setting=SETTING_C, lore=LORE_C, aliases=ALIASES_C,
                    hints=HINTS_C, style=STYLE, guide=GUIDE_C, endings=ENDINGS_C, secrets=SECRETS_C, guide_at=GUIDE_AT_C,
                    common_words=COMMON_WORDS_C, drives=DRIVES_C, epithets=EPITHETS, introduced=_introduced(),
                    moments=MOMENTS, beats=BEATS_C, chronicle=CHRONICLE, kowtow_ticks=6)

