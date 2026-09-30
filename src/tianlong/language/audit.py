"""
[INPUT]: 依赖 language/render 的 RenderPlan / Violation / 词法积木（_mentions / _lexical / _negated / _quotes / _mask / _clauses）
         与 PRONOUNS，language/deeds 的 _sentence / _hedged（找句子、问句与假设），language/quotes 的 _subject / _selves（小句的主语），
         language/scene 的 SceneBrief / Sky / MOON_*
[OUTPUT]: 对外提供 audit()（硬事实审计：一句里的易手、被制者的肢体动作与天色），check_possession() / check_affordance() / check_sky()，
          词表 TAKES / GIVES / BODY / BODY_EXCLUSIONS / MOON_ORBS / MOON_GLOW / RISING / SUNS
[POS]: language 的硬事实审计，由 gate.violations 调用，与 render.check()、deeds.check_deeds()、quotes.check_quotes() 同一条路。
       deeds 问“提到的人里有没有一个做成了”，这里问“施动者（小句的主语）是谁”——“钟灵伸手从你手里抽过了北冥神功”，
       你手里确有北冥神功不等于钟灵拿到了它。三项：
       - possession：抽过/接过/夺过/抽出/拿起……与塞给/递给……这类易手说法，施动者须本回合确有 TAKE（受赠算 receive）或 GIVE，
         拿取类另认“东西本来就在他身上”（掏出自己的易经）；查看（INSPECT）只让人看见东西在哪、内核里不让任何东西易手，不算数
       - affordance：本回合被制（plan.afflicted 里 subdued）的人不许有肢体动作（拍手、起身、走、扑……），开口、眼神、神情、笑都可以；
         本回合先动过手脚、随后才被制住的人（plan.done 与 brief.astir）不查
       - sky：brief.sky 给了天色才查（None = 旧版，不查）——月出之前没有月亮，月已在天上不许再升一次，入夜没有日头
       宽松原则：施动者找不到、是代词、“被”字句、问句与假设（想/要/险些）、否定（没能夺过、站不起来）都不判；
       引语里的话不查（交给台词闸门）；天色另认“还没/不见/等到月亮出来”这类说没有、说将来的，与 lore 原文逐字有的小句。
       局限（如实）：主语靠词法近似，“你眼睁睁看着钟灵抽过……”会认成你（只会放过，不会误杀）；东西只认本小句里点名的
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re

from tianlong.language.deeds import _hedged, _sentence
from tianlong.language.quotes import _selves, _subject
from tianlong.language.render import (
    PRONOUNS,
    RenderPlan,
    Violation,
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
                          "收回")
# 施动者把东西交到别人手里：须本回合确有 give
GIVES: tuple[str, ...] = ("塞给", "递给", "交给", "送给", "抛给", "扔给", "递到", "塞到", "交到")
# 被制的人做不出的肢体动作；心跳、吓一跳、扑哧一笑、扑面而来、走神说的不是他动了手脚
BODY: tuple[str, ...] = ("拍手", "拍案", "拍了拍", "抬手", "抬起手", "举手", "挥手", "摆手", "招手", "拱手", "伸手", "叉腰",
                         "起身", "站起", "跳", "蹦", "扑", "走", "跑", "奔", "跃", "踢", "跺脚", "转身", "迈步", "挥")
BODY_EXCLUSIONS: tuple[str, ...] = ("心头一跳", "心中一跳", "心里一跳", "心下一跳", "眼皮一跳", "眉心一跳", "心跳", "吓了一跳",
                                    "吓一跳", "扑哧", "扑嗤", "扑面", "扑鼻", "扑簌", "走神", "走眼", "走漏", "走火", "走样", "走调",
                                    "走廊", "走道", "奔涌", "奔流", "雀跃", "跃然", "跃跃欲试", "发挥", "指挥", "挥之不去")
_UNABLE = ("不", "没")                  # 紧跟的“不/没”是可能补语的否定：“站不起来”“走不动”“跳不起身”
# 月亮本身（会升起的那个）与月的光色（照在哪里都行，不会“再升一次”）；“月下”多是 lore 里的成词，不收
MOON_ORBS: tuple[str, ...] = ("月亮", "明月", "圆月", "满月", "弯月", "新月", "残月", "皓月", "月轮", "冰轮")
MOON_GLOW: tuple[str, ...] = ("月光", "月色", "月华", "月影", "月辉")
RISING: tuple[str, ...] = ("升了?起", "升上", "东升", "初升", "探了?出", "爬了?上", "露了?出", "钻了?出", "浮了?出", "冒了?出",
                          "跃了?出")
# 夜里的日头（黄昏另议：晚霞、余晖、暮色拿不准，不收）
SUNS: tuple[str, ...] = ("日头", "太阳", "阳光", "夕阳", "斜阳", "落日", "残阳", "烈日", "朝阳", "旭日", "日光", "艳阳")
# 说没有、说将来、说它早已落下的：“还不见月亮”“等到月亮出来”“月亮还没出来”“夕阳早已沉下山去”
_ABSENT_BEFORE = re.compile(r"(?:不见|没有|没|未见|无|等|待|盼|直到|看不到|瞧不见)[^，。！？；]{0,4}$")
_ABSENT_AFTER = re.compile(r"^[^，。！？；]{0,4}?(?:还没|尚未|未|没|不见|快要|就要|将要|要|落|沉|下山|隐|散)")
_ALREADY = re.compile(r"(?:早已|已经|早|已)$")     # “月亮早已升起”说的是它此刻在天上，不是再升一次


# ============================================================
#  施动者：关键词所在小句往前，第一个提到人的小句的主语
# ============================================================


def _agent(masked: str, i: int, plan: RenderPlan) -> str | None:
    """i 处的动作是谁做的（本名）：从关键词所在的小句往前，取第一个提到人的小句的主语（quotes._subject：宾语位置、带“的”的称呼、
    “你”在小句中间都不算）；关键词所在的小句只看它之前，感知动词之后的称呼优先（“你看见钟灵抽过……”）。
    关键词所在的小句里有“被”、主语是代词或找不到主语：不猜，返回 None（前面小句的“钟灵被点了穴道，却拍手”照旧是钟灵）。"""
    people = {**dict(plan.people), "你": plan.viewer_name}
    selves = _selves(plan)
    forms = {*people, *selves, *PRONOUNS}
    lo, hi = _sentence(masked, i)
    for a, b in reversed([(a, b) for a, b in _clauses(masked, lo, hi) if a <= i]):
        stop = min(b, i)
        if not _mentions(masked[a:stop], forms):
            continue
        if stop == i and "被" in masked[a:stop]:
            return None                                 # “被”字句：动作未必是主语做的（“钟灵被左子穆拎着走”）
        who = _subject(masked, forms, selves, a, stop, embedded=stop == i)
        return None if who is None or who in PRONOUNS else people.get(who, who)
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


def check_possession(masked: str, plan: RenderPlan, since: int = 0) -> list[Violation]:
    """易手说法对不上内核：拿取类的施动者本回合没有 take / receive、本小句点名的东西也没一件本来就在他身上；
    交付类的施动者本回合没有 give。本小句不点名东西（修辞里的茶碗）不查。"""
    goods = dict(plan.goods)
    done = set(plan.done)
    out: list[Violation] = []
    for words, ops in ((TAKES, ("take", "receive")), (GIVES, ("give",))):
        for i, w in _lexical(masked, words):
            if i < since or _unable(masked, i, w):
                continue
            a, b = _clause_of(masked, i)
            named = {goods[g] for _, g in _mentions(masked[a:b], goods)}
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


def check_sky(text: str, masked: str, plan: RenderPlan, sky: Sky, since: int = 0) -> list[Violation]:
    """月出之前没有月亮（月光月色也没有）；月已在天上，月亮本身不许再升起、探出、爬上；入夜没有日头。
    只有 moon=rising（本回合正是月出）才许写升起。lore 原文（出处与见过的外观描写）里逐字有的小句不查。"""
    lore = [plan.source, *(t for _, t in plan.scenery)]
    out: list[Violation] = []

    def judged(i: int, w: str) -> bool:
        a, b = _clause_of(masked, i)
        clause = text[a:b].strip()
        return i >= since and not _hedged(masked, i) and not any(clause and clause in t for t in lore)

    def absent(i: int, w: str) -> bool:
        a, b = _clause_of(masked, i)
        return bool(_ABSENT_BEFORE.search(masked[a:i]) or _ABSENT_AFTER.search(masked[i + len(w):b]))

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
