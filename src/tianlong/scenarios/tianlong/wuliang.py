"""
[INPUT]: 依赖 core 的实体/关系/时间/命题/感知类型，core/profiles 的 Goal / GoalKind / Profile，kernel/perception 的 make_percept / scene_percept，
         scenarios/base 的 Scenario / Ending，scenarios/tianlong/lore 的 SETTING / STYLE / LORE / ALIASES / COMMON_WORDS / HINTS / GUIDE / GUIDE_AT / EPILOGUE_RIVER
[OUTPUT]: 对外提供 build_wuliang()：天龙八部·无量山小范围世界（含每个角色的腔调/谈资/话多/脾气、段誉的目标、逐级提示与结局），
          EAST / WEST / SHENNONG / LOVERS 名册（东宗、西宗、神农帮、私奔的一对）、ENDINGS 与 SECRETS（秘密的说法，只交给叙述闸门）
[POS]: scenarios/tianlong 的第一幕。以金庸《天龙八部》世纪新修版开篇为蓝本，但不写剧本——只摆好世界、角色目标与各自所知，
       剧情由规则内核与角色认知自然涌现：比剑之后龚光杰寻衅、钟灵放貂护人、左子穆护短（西宗掌门袖手旁观）、
       入夜后干葛私奔投奔神农帮、途中撞见外人便灭口、崖底玉璧月夜显影、琅嬛福地里的两卷帛书、山腹隧道通往澜沧江畔（第一幕终）。
       腔调 voice 与谈资 knows 只写公开的一面：谈资里点到的实体都是此人开场就认识的，不含秘密，也不与世界真相矛盾
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

from tianlong.core import Entity, Fact, Kind, Modality, Proposition, Rel, Relation, WorldState, at
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.scenarios.base import Ending, Scenario
from tianlong.scenarios.tianlong.lore import (
    ALIASES,
    COMMON_WORDS,
    EPILOGUE_RIVER,
    GUIDE,
    GUIDE_AT,
    HINTS,
    LORE,
    SETTING,
    STYLE,
)

# ============================================================
#  地图
#
#          神农帮营地 ──林间山径── 后山(禁地) ──后山小径── 后山崖顶
#           │                      │                        │ 断崖（只能下）
#       营地下山路             后院小门                     ▼
#           │                      │                  剑湖畔（无量玉璧）
#       无量山山道 ──宫门── 剑湖宫大殿 ──回廊── 剑湖宫后院    ┊ 石缝（暗门，月夜显形）
#                                                         石洞 ──石门── 琅嬛福地 ══隧道(只能出)══▶ 澜沧江畔
# ============================================================

START = at(1, 17, 40)         # 比剑方罢，酉时将近；戌时（19:00）入夜
NIGHTFALL = at(1, 19, 0)
ELOPE = at(1, 19, 20)          # 干光豪、葛光佩入夜后动身
EAST = ("zuozimu", "gongguangjie", "ganguanghao")   # 无量剑东宗：掌门与两名弟子
WEST = ("xinshuangqing", "geguangpei")              # 无量剑西宗：掌门与女弟子
SECT = EAST + WEST
SHENNONG = ("sikongxuan", "shennong")               # 神农帮：帮主与把守山道的帮众
LOVERS = ("ganguanghao", "geguangpei")              # 东西两宗私下相好的一对，入夜投奔神农帮
# 秘密的说法：叙述闸门据此拦住熟读原著的模型替玩家剧透（只在清单与台词里有出处时才许说）
SECRETS = ("私奔", "私订终身", "暗通款曲", "通了声气", r"投[奔靠]?.{0,2}神农帮")


def _entities() -> list[Entity]:
    P, S, It, D, H = Kind.PLACE, Kind.SURFACE, Kind.ITEM, Kind.DOOR, Kind.PERSON
    return [
        # ---- 地点 ----
        Entity.make("shandao", P, "无量山山道"),
        Entity.make("hall", P, "剑湖宫大殿"),
        Entity.make("houyuan", P, "剑湖宫后院"),
        Entity.make("houshan", P, "后山"),
        Entity.make("yading", P, "后山崖顶"),
        Entity.make("jianhu", P, "剑湖畔"),
        Entity.make("shidong", P, "石洞"),
        Entity.make("langhuan", P, "琅嬛福地"),
        Entity.make("lancang", P, "澜沧江畔"),
        Entity.make("camp", P, "神农帮营地"),
        # ---- 通道 ----
        Entity.make("d_gate", D, "剑湖宫宫门"),
        Entity.make("d_corridor", D, "回廊"),
        Entity.make("d_backgate", D, "后院小门"),
        Entity.make("d_path", D, "后山小径"),
        Entity.make("d_trail", D, "林间山径"),
        Entity.make("d_camproad", D, "营地下山路"),
        Entity.make("d_cliff", D, "断崖", oneway="jianhu"),
        Entity.make("d_cave", D, "玉璧旁的石缝", hidden=True, night_only=True, clue="yubi"),
        Entity.make("d_stonedoor", D, "石门"),
        Entity.make("d_tunnel", D, "山腹隧道", oneway="lancang"),
        # ---- 陈设 ----
        Entity.make("swordrack", S, "兵器架"),
        Entity.make("yubi", S, "无量玉璧"),
        Entity.make("statue", S, "玉像"),
        Entity.make("putuan", S, "蒲团"),
        # ---- 物件 ----
        Entity.make("mink", It, "闪电貂", small=True, weapon=True, venom=True, edge=0.45),
        Entity.make("antidote", It, "解药", small=True, cures="poisoned"),
        Entity.make("sword", It, "长剑", weapon=True, edge=0.2),
        Entity.make("yijing", It, "易经", small=True),
        Entity.make("scroll_bm", It, "北冥神功帛卷", small=True, hidden=True, teaches="absorb", difficulty=4),
        Entity.make("scroll_lb", It, "凌波微步帛卷", small=True, hidden=True, teaches="evasion", difficulty=3),
        # ---- 人物 ----
        Entity.make("duanyu", H, "段誉", martial=0.0, agility=0.5, alertness=0.4),
        Entity.make("mawude", H, "马五德", martial=0.35, agility=0.4, alertness=0.5),
        Entity.make("zuozimu", H, "左子穆", martial=0.8, agility=0.6, alertness=0.7),
        Entity.make("xinshuangqing", H, "辛双清", martial=0.75, agility=0.6, alertness=0.7),
        Entity.make("gongguangjie", H, "龚光杰", martial=0.5, agility=0.6, alertness=0.5),
        Entity.make("ganguanghao", H, "干光豪", martial=0.5, agility=0.5, alertness=0.6),
        Entity.make("geguangpei", H, "葛光佩", martial=0.4, agility=0.5, alertness=0.6),
        Entity.make("zhongling", H, "钟灵", martial=0.25, agility=0.8, alertness=0.8),
        Entity.make("sikongxuan", H, "司空玄", martial=0.7, agility=0.5, alertness=0.7),
        Entity.make("shennong", H, "神农帮帮众", martial=0.4, agility=0.5, alertness=0.6),
    ]


_DOORS = {
    "d_gate": ("shandao", "hall"), "d_corridor": ("hall", "houyuan"), "d_backgate": ("houyuan", "houshan"),
    "d_path": ("houshan", "yading"), "d_trail": ("houshan", "camp"), "d_camproad": ("camp", "shandao"),
    "d_cliff": ("yading", "jianhu"), "d_cave": ("jianhu", "shidong"), "d_stonedoor": ("shidong", "langhuan"),
    "d_tunnel": ("langhuan", "lancang"),
}


def _relations() -> list[Relation]:
    R = Relation
    rels = [R(d, Rel.CONNECTS, p) for d, ends in _DOORS.items() for p in ends]
    placement = {
        "swordrack": "hall", "yubi": "jianhu", "statue": "langhuan", "putuan": "langhuan",
        "mink": "zhongling", "antidote": "zhongling", "sword": "swordrack", "yijing": "duanyu",
        "scroll_bm": "putuan", "scroll_lb": "putuan",
        "duanyu": "hall", "mawude": "hall", "zuozimu": "hall", "xinshuangqing": "hall", "gongguangjie": "hall",
        "ganguanghao": "hall", "geguangpei": "hall", "zhongling": "hall", "sikongxuan": "camp", "shennong": "shandao",
    }
    rels += [R(e, Rel.AT, h) for e, h in placement.items()]
    rels += [R("zhongling", Rel.OWNS, "mink"), R("zhongling", Rel.OWNS, "antidote"), R("duanyu", Rel.OWNS, "yijing")]
    return rels


def _layout(*doors: str) -> tuple[Fact, ...]:
    facts = [Fact(Proposition.rel(d, Rel.CONNECTS, p)) for d in doors for p in _DOORS[d]]
    if "d_cliff" in doors:
        facts.append(Fact(Proposition.attr("d_cliff", "oneway", "jianhu")))   # 本门弟子都知道断崖爬不上来
    return tuple(facts)


def _allies(me: str, *groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(p for g in groups for p in g if p != me))


# ============================================================
#  角色
#  voice 腔调、knows 谈资只写公开的一面（主持人之声据此写台词）；persona 与目标才是心里的打算。
#  chatty ∈ [0, 1] 越大越爱找人搭话；temper ∈ [-1, 1] 越大越受不得激，越小越能忍。
#
#  盟友按门户划分（allies 决定替谁出头、救治谁、守地与灭口时放过谁）：
#  - 东宗护东宗、西宗护西宗：西宗掌门不替东宗弟子出头——龚光杰挨了钟灵的貂，出手的只有他师父左子穆；
#  - 两位掌门彼此互为盟友：东西二宗面和心不和，但外人欺到本派掌门头上，另一宗掌门不会坐视（一致对外只到掌门这一层）；
#  - 干光豪、葛光佩只认彼此与神农帮：二人入夜私奔去投神农帮，同门撞见照样灭口；
#    神农帮也把他们当自己人——投到营里的人不会被当作擅闯营地的外人。
#  开场冲突链（寻衅 → 放貂 → 护短 → 搜出解药 → 救治）只需要左子穆护着龚光杰，不需要西宗掌门出手
# ============================================================


def _profiles() -> dict[str, Profile]:
    return {
        # ---- 玩家：目标只驱动提示（/hint、“我该做什么”），玩家不跑策略 ----
        # ESCAPE 取其“抵达”的达成语义：离开无量山、到澜沧江畔（第一幕终）；“途中灭口”只是 NPC 策略的做法，与玩家无关。
        # DEFEND 钟灵：别连累替你出头的人。“保命”没有合适的目标类型（DEFEND 自己是误用），交给提示与叙述
        "duanyu": Profile(
            "duanyu", "书生", "大理段氏子弟，饱读诗书，生性仁厚，最厌习武；离家出走，随马五德上山观剑",
            goals=(Goal(GoalKind.ESCAPE, home="lancang"), Goal(GoalKind.DEFEND, person="zhongling", weight=0.5)),
            is_player=True,
            voice="书生腔，文绉绉的爱掉书袋，自称“在下”；待人谦和有礼，认准了的道理却迂执不改",
            knows="大理人氏，读过不少经史子集，尤精《易经》，说起卦象来头头是道；不会武功，也不想学",
            intro="大理来的书生，姓段，随马五德上山观剑，手无缚鸡之力",
            chatty=0.0, temper=-0.6),
        # ---- 东宗 ----
        "gongguangjie": Profile(
            "gongguangjie", "东宗弟子", "东宗弟子，骄横好胜；方才比剑得胜，却被一个书生当众嗤笑，恼羞成怒",
            goals=(Goal(GoalKind.HOSTILE, person="duanyu", until="wounded"),), allies=_allies("gongguangjie", EAST),
            voice="骄横刻薄，好出言讥讽，开口便是“酸秀才”“小子”，句句夹枪带棒，受不得半点顶撞",
            knows="无量剑东西二宗每五年比剑一次，胜的一宗入主剑湖宫五年；今日他替东宗出场，赢了西宗的对手",
            intro="无量剑东宗掌门左子穆门下的弟子，今日比剑替东宗出场赢了一场，性子骄横",
            chatty=0.3, temper=0.8),
        "zuozimu": Profile(
            "zuozimu", "东宗掌门", "无量剑东宗掌门，多疑而护短，门下弟子吃了亏必要讨回",
            allies=_allies("zuozimu", EAST, ("xinshuangqing",)),
            voice="阴沉多疑，说话绵里藏针：客客气气里夹着试探与威压，护短，从不轻易认错",
            knows="无量剑原有东、北、西三宗，北宗早已式微，东西二宗五年一比剑；后山是本派禁地；"
                  "门中老辈传说，月明之夜剑湖畔的无量玉璧上现过仙人舞剑的影子，他从没亲眼见过，只当是个传闻",
            intro="无量剑东宗掌门，东宗这五年坐镇剑湖宫",
            chatty=0.2, temper=0.5),
        "ganguanghao": Profile(
            "ganguanghao", "东宗弟子",
            "东宗弟子，与西宗葛光佩私下相好；神农帮围山，二人暗中与神农帮通了声气，打算入夜后私奔去投神农帮，最怕被人撞见",
            goals=(Goal(GoalKind.ESCAPE, home="camp", not_before=ELOPE),), allies=_allies("ganguanghao", LOVERS, SHENNONG),
            voice="心虚躲闪，说话含糊敷衍、眼神游移，旁人多问两句便岔开话头",
            knows="东宗弟子，师从左子穆；后山是本派禁地，后山崖顶下去便是断崖，只能下、不能上",
            intro="无量剑东宗弟子，左子穆的徒弟",
            chatty=0.1, temper=0.3),
        # ---- 西宗 ----
        "xinshuangqing": Profile(
            "xinshuangqing", "西宗掌门",
            "无量剑西宗掌门，冷傲寡言，与东宗面和心不和；东宗弟子吃了亏她乐得袖手，外人欺到本派掌门头上却不会坐视",
            allies=_allies("xinshuangqing", WEST, ("zuozimu",)),
            voice="冷峻寡言，惜字如金，开口多是短短一两句，不假辞色",
            knows="东西二宗同出无量剑一脉，五年一比剑，胜者入主剑湖宫；这回西宗又输了一场",
            intro="无量剑西宗掌门，一位冷傲的女子",
            chatty=0.05, temper=0.4),
        "geguangpei": Profile(
            "geguangpei", "西宗弟子", "西宗女弟子，与干光豪私订终身，入夜后便要随他逃去投神农帮",
            goals=(Goal(GoalKind.ESCAPE, home="camp", not_before=ELOPE),), allies=_allies("geguangpei", LOVERS, SHENNONG),
            voice="心虚躲闪，说话细声细气、欲言又止，被人盯着便低头不语",
            knows="西宗女弟子，师父是辛双清；后山是本派禁地，寻常弟子不得擅入",
            intro="无量剑西宗的女弟子，辛双清的徒弟",
            chatty=0.05, temper=-0.2),
        # ---- 局外人 ----
        "zhongling": Profile(
            "zhongling", "少女", "万劫谷少女，天真娇憨，胆大好事；见那书生挨打，心中不平；腰间藏着一只闪电貂",
            goals=(Goal(GoalKind.DEFEND, person="duanyu"),),
            voice="俏皮天真、口无遮拦，说话脆生生的，爱叫段誉“书呆子”；高兴了拍手笑，不高兴便噘嘴，对谁都没大没小",
            knows="家住万劫谷，是偷偷溜出来玩的；养着一只闪电貂，快得像一道闪电，咬人有毒，最听她的话",
            intro="不知打哪儿来的小姑娘，方才坐在梁上看热闹，身上带着一只貂",
            chatty=0.7, temper=0.2),
        "mawude": Profile(
            "mawude", "宾客", "普洱老武师，做茶叶生意，为人圆滑，不愿惹事",
            voice="圆滑和气，逢人先赔三分笑，爱打圆场，张口“和气生财”，闭口“大家都是朋友”",
            knows="普洱做茶叶生意的，与各派都有些交情；无量剑东西二宗每五年在剑湖宫比剑，胜的一宗入主剑湖宫五年，"
                  "这是多年的老规矩；听说神农帮已把无量山围了，山道也给封了",
            intro="普洱的老武师，做茶叶生意，与各派都有些交情",
            chatty=0.4, temper=-0.5),
        # ---- 神农帮 ----
        "sikongxuan": Profile(
            "sikongxuan", "神农帮帮主",
            "神农帮帮主，精于药理毒物，率众围住无量山，要强占后山采药；已许了无量剑中来投的人入夜到营中",
            goals=(Goal(GoalKind.GUARD, home="camp"),), allies=_allies("sikongxuan", SHENNONG, LOVERS),
            voice="蛮横粗豪，嗓门洪亮，说话不留余地，动辄拿毒药吓人，自称“老夫”",
            knows="神农帮以采药制药为业，精研药性毒物；无量山后山多生奇花异草；帮众奉他之命把守山道，谁也别想下山",
            intro="神农帮帮主，精于药理毒物",
            chatty=0.3, temper=0.7),
        "shennong": Profile(
            "shennong", "神农帮帮众", "神农帮帮众，奉命把守下山的道路，不许任何人离开",
            goals=(Goal(GoalKind.GUARD, home="shandao"),), allies=_allies("shennong", SHENNONG, LOVERS),
            voice="粗鲁，张口便骂，粗声大气，三句不离“帮主有令”",
            knows="奉司空玄帮主之命把守山道，一个人也不许下山；神农帮的人个个识得毒草",
            intro="神农帮的帮众",
            chatty=0.2, temper=0.6),
    }


# ============================================================
#  结局：抵达即落幕
#  只有一个：山道那头没有“山下”这个地点——拿山道当结局，等于一出宫门就收场。世界里有了山下，再加“守卫散了、从山道下山”
# ============================================================

ENDINGS = (Ending("river", "第一幕终 · 澜沧江畔", "lancang", EPILOGUE_RIVER),)


def build_wuliang(seed: int = 7) -> Scenario:
    state = WorldState.build(seed, START, _entities(), _relations())

    def past(facts: tuple[Fact, ...]):
        return replace(make_percept(state, Modality.SCENE, facts=facts), tick=START - 30)

    def now_seen(agent: str):
        return replace(scene_percept(state, agent), tick=START - 1)

    palace = _layout("d_gate", "d_corridor", "d_backgate", "d_path", "d_trail", "d_cliff")
    swords = (Fact(Proposition.rel("swordrack", Rel.AT, "hall")), Fact(Proposition.rel("sword", Rel.AT, "swordrack")))
    legend = (Fact(Proposition.rel("yubi", Rel.AT, "jianhu")),)     # 本门都知道剑湖畔有面无量玉璧（显影的事只是传闻）
    priors: dict[str, tuple] = {a: (past(palace + swords + legend), now_seen(a)) for a in SECT}
    priors["duanyu"] = (now_seen("duanyu"),)
    priors["mawude"] = (past(_layout("d_gate", "d_corridor")), now_seen("mawude"))
    priors["zhongling"] = (past(_layout("d_gate")), now_seen("zhongling"))
    camp = _layout("d_camproad", "d_trail", "d_gate")
    # 帮主派帮众去守山道：彼此知道对方守在哪里
    priors["sikongxuan"] = (past(camp + (Fact(Proposition.rel("shennong", Rel.AT, "shandao")),)), now_seen("sikongxuan"))
    priors["shennong"] = (past(camp + (Fact(Proposition.rel("sikongxuan", Rel.AT, "camp")),)), now_seen("shennong"))
    return Scenario("wuliang", state, _profiles(), priors, setting=SETTING, lore=LORE, aliases=ALIASES, hints=HINTS,
                    style=STYLE, guide=GUIDE, endings=ENDINGS, secrets=SECRETS, guide_at=GUIDE_AT, common_words=COMMON_WORDS)
