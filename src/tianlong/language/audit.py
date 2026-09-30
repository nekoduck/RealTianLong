"""
[INPUT]: 依赖 language/render 的 RenderPlan / Violation / 词法积木（_mentions / _lexical / _negated / _quotes / _mask / _clauses /
         _clause_before）与 PRONOUNS / LIKENESS（比方），language/deeds 的 _sentence / _hedged（找句子、问句与假设），
         language/quotes 的 _subject / _selves / PERCEPTION（小句的主语、感知动词），language/scene 的 SceneBrief / Sky / MOON_*
[OUTPUT]: 对外提供 audit()（硬事实审计：一句里的易手、被制者的肢体动作与天色），check_possession() / check_affordance() / check_sky()，
          词表 TAKES / GIVES / BODY / BODY_EXCLUSIONS / MOON_ORBS / MOON_GLOW / RISING / SUNS / WATCH（感知与期盼）/ GAZE、OBJECT_GAP
[POS]: language 的硬事实审计，由 gate.violations 调用，与 render.check()、deeds.check_deeds()、quotes.check_quotes() 同一条路。
       deeds 问“提到的人里有没有一个做成了”，这里问“施动者（小句的主语）是谁”——“钟灵伸手从你手里抽过了北冥神功”，
       你手里确有北冥神功不等于钟灵拿到了它。施动者：关键词所在小句往前第一个有主语的小句的主语（“便从你手里抽过”的你是宾语，
       主语省略、接着往前找）；感知/期盼动词之后的人才是内嵌小句的施动者（“你看着龚光杰转身走开”“钟灵只盼你快些走”）；
       回溯不越过感知动词后的内嵌小句（“你眼睁睁看着左子穆拂袖而去，走到殿外”）与自带非人主语的小句（“脚步声渐渐走远”）。三项：
       - possession：抽过/接过/夺过/抽出/拿起……与塞给/递给……这类易手说法，施动者须本回合确有 TAKE（受赠算 receive）或 GIVE，
         拿取类另认“东西本来就在他身上”（掏出自己的易经）；查看（INSPECT）只让人看见东西在哪、内核里不让任何东西易手，不算数；
         东西只认宾语位置（紧跟其后或“把/将 + 东西”在前；“接过话头说起易经的来历”“收回盯着易经的目光”不算）
       - affordance：本回合被制（plan.afflicted 里 subdued）的人不许有肢体动作（拍手、起身、走、扑……），开口、眼神、神情、笑都可以；
         本回合先动过手脚、随后才被制住的人（plan.done 与 brief.astir）不查
       - sky：brief.sky 给了天色才查（None = 旧版，不查）——月出之前没有月亮，月已在天上不许再升一次，入夜没有日头
       宽松原则：施动者找不到、是代词、“被”字句、问句与假设（想/要/险些）、否定（没能夺过、站不起来）都不判；
       引语里的话不查（交给台词闸门）；天色另认“还没/不见/等到月亮出来”这类说没有、说将来的与“早已沉下”这类说已落的
       （“还没落下”“尚未落尽”“快要落下”是还在，照拦）、比方里的（“白得像月光”“阳光般的笑容”）、lore 原文逐字有的小句。
       局限（如实）：主语靠词法近似——注视之后紧挨着关键词的人（“钟灵盯着你拍手”）说不清就不判，非人主语只认物件名与
       声/风/浪/貂这类字；东西只认本小句里宾语位置点名的；“弯成两道新月”这类不带“像/般”的比方照拦
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re

from tianlong.language.deeds import _hedged, _sentence
from tianlong.language.quotes import PERCEPTION, _selves, _subject
from tianlong.language.render import (
    _CLAUSE,
    LIKENESS,
    LIKENESS_WINDOW,
    PRONOUNS,
    RenderPlan,
    Violation,
    _clause_before,
    _clauses,
    _lexical,
    _mask,
    _mentions,
    _negated,
    _quotes,
)
from tianlong.language.scene import MOON_NONE, MOON_UP, SceneBrief, Sky

# ============================================================
#  词表：易手、肢体动作、天色（词条是正则片段，多数就是字面词；拿不准的不收）
# ============================================================

# 施动者把东西拿到自己手里：须本回合确有 take / receive，或东西本来就在他身上
TAKES: tuple[str, ...] = ("抽过", "接过", "夺过", "抢过", "拿过", "取过", "夺下", "夺走", "抢走", "抽走", "抽出", "拿起", "揣进",
                          "收回", "[抽夺抢拿]了?过去")
# 施动者把东西交到别人手里：须本回合确有 give
GIVES: tuple[str, ...] = ("塞给", "递给", "交给", "送给", "抛给", "扔给", "递到", "塞到", "交到")
# 被制的人做不出的肢体动作；心跳、吓一跳、扑哧一笑、扑面而来、走神说的不是他动了手脚
BODY: tuple[str, ...] = ("拍手", "拍案", "拍了拍", "抬手", "抬起手", "举手", "挥手", "摆手", "招手", "拱手", "伸手", "叉腰",
                         "起身", "站起", "坐起", "翻身", "跳", "蹦", "扑", "走", "跑", "奔", "跃", "踢", "跺脚", "转身", "迈步", "挥")
BODY_EXCLUSIONS: tuple[str, ...] = ("心头一跳", "心中一跳", "心里一跳", "心下一跳", "眼皮一跳", "眉心一跳", "心跳", "吓了一跳",
                                    "吓一跳", "扑哧", "扑嗤", "扑面", "扑鼻", "扑簌", "走神", "走眼", "走漏", "走火", "走样", "走调",
                                    "走廊", "走道", "奔涌", "奔流", "雀跃", "跃然", "跃跃欲试", "发挥", "指挥", "挥之不去")
OBJECT_GAP = 3                          # 易手说法与它的宾语之间至多隔几个字（“抽过了那卷北冥神功”），人名不计
_UNABLE = ("不", "没")                  # 紧跟的“不/没”是可能补语的否定：“站不起来”“走不动”“跳不起身”
# 月亮本身（会升起的那个）与月的光色（照在哪里都行，不会“再升一次”）；“月下”多是 lore 里的成词，不收
MOON_ORBS: tuple[str, ...] = ("月亮", "明月", "圆月", "满月", "弯月", "新月", "残月", "皓月", "月轮", "冰轮")
MOON_GLOW: tuple[str, ...] = ("月光", "月色", "月华", "月影", "月辉")
RISING: tuple[str, ...] = ("升了?起", "升上", "东升", "初升", "探了?出", "爬了?上", "露了?出", "钻了?出", "浮了?出", "冒了?出",
                          "跃了?出")
# 夜里的日头（黄昏另议：晚霞、余晖、暮色拿不准，不收）
SUNS: tuple[str, ...] = ("日头", "太阳", "阳光", "夕阳", "斜阳", "落日", "残阳", "烈日", "朝阳", "旭日", "日光", "艳阳")
# 说没有、说将来、说它早已落下的：“还不见月亮”“等到月亮出来”“月亮还没出来”“夕阳早已沉下山去”；
# 否定或“快要”后接落、沉、下山（“夕阳还没落下”“月亮快要落下去了”）说的是它此刻还在天上，照拦
_ABSENT_BEFORE = re.compile(r"(?:不见|没有|没|未见|无|等|待|盼|直到|看不到|瞧不见)[^，。！？；]{0,4}$")
_NOT_OUT = re.compile(r"^[^，。！？；]{0,4}?(?:还没有?|尚未|未|没有?|不见|快要|就要|将要|要)[^，。！？；]{0,2}?(?:出来|出|升|露)")
_SET = re.compile(r"^([^，。！？；]{0,4}?)(?:落|沉|下山|隐|散)")
_STILL = re.compile(r"(?:没有?|未|不|要)$")
# 比方里的月光、落日（“白得像月光”“阳光般的笑容”“如同落日”）不是天上的实物；比方收住了（“像月光一样洒在湖面上的月光”）之后照查
_LIKE_AFTER = ("般", "一般", "一样", "似的")
_LIKE_CLOSED = re.compile(r"一样|一般|般|似的")
_ALREADY = re.compile(r"(?:早已|已经|早|已)$")     # “月亮早已升起”说的是它此刻在天上，不是再升一次


# ============================================================
#  施动者：关键词所在小句往前，第一个有主语的小句的主语——不越过感知/期盼动词，也不越过自带非人主语的小句
# ============================================================

# 感知、期盼之后的人是内嵌小句的主语：“你看着龚光杰转身走开”走的是龚光杰，“钟灵只盼你快些走”走的是你；
# 注视（GAZE）之后紧挨着关键词的人说不清是谁在动（“钟灵盯着你拍手”），“只见钟灵拍手”“听见钟灵起身”是钟灵
GAZE: tuple[str, ...] = ("看着", "望着", "盯着", "瞧着", "瞅着")
WATCH: tuple[str, ...] = (*PERCEPTION, *GAZE, "眼看", "眼见", "见", "盼", "等")
# 小句自带的非人主语（开头一两个限定字之后）：“脚步声走远”“一阵风扑来”——动作不是前一小句那个人做的
_THING_HEAD = re.compile(r"^(?:[一那这几两][阵只声股道片团个条]?)?[^，。！？；]{0,3}?[声风浪雾烟影水雨云鸟兽马蛇鱼貂]")
_LEADS_OFF = re.compile(r"^(?:[一那这几两][阵只声股道片团个条卷柄本]?)?$")


def _watched(masked: str, a: int, stop: int, forms: set[str]) -> tuple[bool, str | None, bool]:
    """[a, stop) 里最后一个感知/期盼动词之后提到的人：(有没有内嵌小句, 那个人的称呼, 动词是不是注视)。
    动词之后空着或只有“状”（见状、看着，）不算内嵌；之后有字却没有人（只听见脚步声）是内嵌的非人主语，称呼为 None。"""
    seg = masked[a:stop]
    hits = [(k, w) for w in WATCH for k in [seg.rfind(w)] if k >= 0]
    if not hits:
        return False, None, False
    k, w = max(hits, key=lambda h: (h[0] + len(h[1]), len(h[1])))
    tail = seg[k + len(w):]
    if tail.strip(_CLAUSE) in ("", "状"):
        return False, None, False
    found = _mentions(tail, forms)
    return True, (found[-1][1] if found else None), w in GAZE


def _own_thing(masked: str, a: int, stop: int, things: set[str]) -> bool:
    """小句开头（限定字之后）是物件名或声、风、浪、貂这类非人的东西：它自己就有主语。"""
    seg = masked[a:stop].strip(_CLAUSE)
    lead = next((k for k, _ in _mentions(seg, things)), None)
    return bool(_THING_HEAD.match(seg)) or (lead is not None and bool(_LEADS_OFF.match(seg[:lead])))


def _agent(masked: str, i: int, plan: RenderPlan) -> str | None:
    """i 处的动作是谁做的（本名）：从关键词所在的小句往前，取第一个有主语的小句的主语（quotes._subject：宾语位置、
    带“的”的称呼、“你”在小句中间都不算——“便从你手里抽过”的主语省略了，接着往前找）；关键词所在的小句只看它之前。
    “你看着龚光杰转身走开”“钟灵只盼你快些走”：感知/期盼动词之后、关键词之前提到的人（龚光杰、你）才是施动者。
    不猜、返回 None：关键词所在的小句里有“被”或主语是代词；内嵌小句没有人（“只听见脚步声走远”）或注视之后的人紧挨着
    关键词（“钟灵盯着你拍手”谁拍手说不清；“只见钟灵拍手”是钟灵）；往前找时碰到自带非人主语的小句（“脚步声渐渐走远”），
    或前一小句是“看着某人做某事”（“你眼睁睁看着左子穆拂袖而去，走到殿外”）。只看着某人（“钟灵看着你，一把抽过”）照常取主语。"""
    people = {**dict(plan.people), "你": plan.viewer_name}
    selves = _selves(plan)
    forms = {*people, *selves, *PRONOUNS}
    things = {g for g, _ in plan.goods}
    lo, hi = _sentence(masked, i)
    for a, b in reversed([(a, b) for a, b in _clauses(masked, lo, hi) if a <= i]):
        stop, own = min(b, i), b > i
        if own and "被" in masked[a:stop]:
            return None                                 # “被”字句：动作未必是主语做的（“钟灵被左子穆拎着走”）
        embedded, who, gaze = _watched(masked, a, stop, forms)
        bare = who is not None and masked[a:stop].rstrip(_CLAUSE).endswith(who)    # “看着你”：你之后没有别的字
        if embedded and own:                            # “你看着龚光杰转身”：施动者是内嵌小句的主语；“盯着你拍手”说不清
            return None if who is None or who in PRONOUNS or (bare and gaze) else people.get(who, who)
        if embedded and not bare:
            return None                                 # 前一小句“看着左子穆拂袖而去”：接着做的是谁说不清
        if not _mentions(masked[a:stop], forms):
            if _own_thing(masked, a, stop, things):
                return None                             # 自带非人主语：“你动弹不得，脚步声渐渐走远”
            continue
        who = _subject(masked, forms, selves, a, stop, embedded=own)
        if who is None:
            continue                                    # 只有宾语位置的人（“便从你手里”）：主语省略，接着往前找
        return None if who in PRONOUNS else people.get(who, who)
    return None


def _clause_of(masked: str, i: int) -> tuple[int, int]:
    lo, hi = _sentence(masked, i)
    return next(((a, b) for a, b in _clauses(masked, lo, hi) if a <= i < b), (i, i))


def _unable(masked: str, i: int, w: str) -> bool:
    """说的不是已经发生的事：否定（没能夺过）、可能补语的否定（站不起来）、问句与假设（想拍手、险些夺过）。"""
    return _negated(masked, i) or masked[i + len(w):].startswith(_UNABLE) or _hedged(masked, i)


# ============================================================
#  三项审计
# ============================================================


def _handled(masked: str, i: int, w: str, goods: dict[str, str], persons: set[str]) -> set[str]:
    """i 处的易手说法点名的东西（本名）：宾语位置才算——紧跟其后（隔着“了那卷/一柄/钟灵手中的”这类，人名不计、
    至多 OBJECT_GAP 个字，不隔别的易手说法），或“把/将 + 东西”在它前面；带“的”的（“说起易经的来历”“盯着易经的目光”）
    是别的名词的定语，不是被拿的东西。"""
    a, b = _clause_of(masked, i)
    out: set[str] = set()
    for k, g in _mentions(masked[a:b], goods):
        k += a
        if masked[k + len(g):].startswith("的"):
            continue
        gap = masked[i + len(w):k]
        bare = len(gap) - sum(len(f) for _, f in _mentions(gap, persons))
        after = k >= i + len(w) and bare <= OBJECT_GAP and not _lexical(gap, (*TAKES, *GIVES))
        before = k < i and any(m in masked[max(a, k - OBJECT_GAP):k] for m in ("把", "将"))
        if after or before:
            out.add(goods[g])
    return out


def check_possession(masked: str, plan: RenderPlan, since: int = 0) -> list[Violation]:
    """易手说法对不上内核：拿取类的施动者本回合没有 take / receive、本小句点名的东西也没一件本来就在他身上；
    交付类的施动者本回合没有 give。本小句不点名东西（修辞里的茶碗）不查。"""
    goods = dict(plan.goods)
    persons = {*dict(plan.people), "你", *PRONOUNS}
    done = set(plan.done)
    out: list[Violation] = []
    for words, ops in ((TAKES, ("take", "receive")), (GIVES, ("give",))):
        for i, w in _lexical(masked, words):
            if i < since or _unable(masked, i, w):
                continue
            named = _handled(masked, i, w, goods, persons)
            who = _agent(masked, i, plan) if named else None
            if who is None or any((who, op) in done for op in ops):
                continue
            if words is TAKES and any((who, g) in plan.holdings for g in named):
                continue                                # 本来就在他身上：掏出自己的易经
            out.append(Violation("possession", f"{who}:{w}:{'/'.join(sorted(named))}"))
    return out


def check_affordance(masked: str, plan: RenderPlan, astir: tuple[str, ...] = (), since: int = 0) -> list[Violation]:
    """被制的人写出了肢体动作；开口、眼神、神情、笑不在词表里。本回合做成过动手脚的事（plan.done）或先动过手脚
    （astir：挥拳打去、随即被点了穴道）的人不算——被制也许是这之后的事。"""
    moved = {n for n, op in plan.done if op not in ("tell", "ask", "wait", "inspect")} | set(astir)
    held = {n for n, st in plan.afflicted if st == "subdued"} - moved
    if not held:
        return []
    out: list[Violation] = []
    for i, w in _lexical(masked, BODY, BODY_EXCLUSIONS):
        if i < since or _unable(masked, i, w):
            continue
        who = _agent(masked, i, plan)
        if who in held:
            out.append(Violation("affordance", f"{who}:{w}"))
    return out


def _likened(masked: str, i: int, w: str) -> bool:
    """i 处的天色词在比方里：紧跟“般/一样/似的”，或前面 LIKENESS_WINDOW 字内有“像/如同/仿佛”且比方还没收住。"""
    if masked[i + len(w):].startswith(_LIKE_AFTER):
        return True
    head = _clause_before(masked, i, LIKENESS_WINDOW)
    like = list(LIKENESS.finditer(head))
    return bool(like) and not _LIKE_CLOSED.search(head[like[-1].end():])


def check_sky(text: str, masked: str, plan: RenderPlan, sky: Sky, since: int = 0) -> list[Violation]:
    """月出之前没有月亮（月光月色也没有）；月已在天上，月亮本身不许再升起、探出、爬上；入夜没有日头。
    只有 moon=rising（本回合正是月出）才许写升起。lore 原文（出处与见过的外观描写）里逐字有的小句不查。"""
    lore = [plan.source, *(t for _, t in plan.scenery)]
    out: list[Violation] = []

    def judged(i: int, w: str) -> bool:
        a, b = _clause_of(masked, i)
        clause = text[a:b].strip()
        return (i >= since and not _hedged(masked, i) and not _likened(masked, i, w)
                and not any(clause and clause in t for t in lore))

    def absent(i: int, w: str) -> bool:
        a, b = _clause_of(masked, i)
        after = masked[i + len(w):b]
        gone = _SET.match(after)
        return bool(_ABSENT_BEFORE.search(masked[a:i]) or _NOT_OUT.match(after)
                    or (gone and not _STILL.search(gone.group(1))))

    if sky.moon == MOON_NONE:
        out += [Violation("sky", f"moon:{w}") for i, w in _lexical(masked, (*MOON_ORBS, *MOON_GLOW))
                if judged(i, w) and not absent(i, w)]
    elif sky.moon == MOON_UP:
        for i, w in _lexical(masked, RISING):
            a, b = _clause_of(masked, i)
            if (_mentions(masked[a:b], MOON_ORBS) and judged(i, w) and not _negated(masked, i)
                    and not _ALREADY.search(masked[a:i])):
                out.append(Violation("sky", f"rise:{w}"))
    if sky.night:
        out += [Violation("sky", f"sun:{w}") for i, w in _lexical(masked, SUNS) if judged(i, w) and not absent(i, w)]
    return out


def audit(text: str, plan: RenderPlan, brief: SceneBrief, since: int = 0) -> list[Violation]:
    """硬事实审计：易手、被制者的肢体动作、天色（brief.sky 为 None 时不查）。since：只查从该位置起的命中（此前的句子已验收）。"""
    masked = _mask(text, _quotes(text))
    found = check_possession(masked, plan, since) + check_affordance(masked, plan, brief.astir, since)
    if brief.sky is not None:
        found += check_sky(text, masked, plan, brief.sky, since)
    return list(dict.fromkeys(found))
