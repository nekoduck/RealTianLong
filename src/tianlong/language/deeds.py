"""
[INPUT]: 依赖 language/render 的 RenderPlan / Violation / 词法积木（_mentions / _lexical / _negated / _short_of / _quotes / _mask /
         _clauses / _clause_after）与词表（STATUS_LEXICON / STATUS_EXCLUSIONS / DENIALS / PRONOUNS / ARRIVAL_WINDOW）
[OUTPUT]: 对外提供 check_deeds()（人事闸门：状态落在谁身上、谁做成了什么、东西在谁手里、玩家此刻在哪），
          词表 BODILY / DEED_LEXICON / DEED_LICENSE / POSSESSION / POSITION_VERBS / DEPARTURE_VERBS / MOODS
[POS]: language 的人事闸门，与 render.check()、quotes.check_quotes() 并用。check() 只问“这个状态、这个地点本回合有没有”，
       这里再问“落在谁身上”：状态词、动作结果词（拍中/打倒/拿起/抽出/得手）、持有说法（握在你手中）、站位（已身在/站在）
       与离开（溜出/出了），都要找到说的是谁——同一句里、从关键词所在小句往前，第一个提到人的小句里的称呼
       （状态词所在的小句整句都算，动作只看关键词之前；“被”字句取“被”之后的施动者）——再与内核结算对照：
       受伤的是你不是龚光杰、没挡开就不是拍中、没拿到就不在手里、没走成就还在大殿。
       宽松原则（宁可放过也不错杀好句子）：小句里有他/她这类代词、或“被”后面不是人，就不猜；提到的人里有一个对得上就放行；
       问句与“若/想/要/不妨/险些”等假设、意图说法，“原以为/莫非/难道”这类旧认知与疑问不查；否认只在提到的人个个都确有伤毒被制时才算。
       局限（如实）：只做词法近似，主谓关系复杂的长句可能认错人；“做成了什么”按人与操作记，不分对谁出手；
       持有与站位只查玩家自己；引语里的话交给台词闸门
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable

from tianlong.language.render import (
    ARRIVAL_WINDOW,
    DENIALS,
    PRONOUNS,
    STATUS_EXCLUSIONS,
    STATUS_LEXICON,
    RenderPlan,
    Violation,
    _clause_after,
    _clauses,
    _lexical,
    _mask,
    _mentions,
    _negated,
    _quotes,
    _short_of,
)

# ============================================================
#  词表：谁做成了什么、东西在谁手里、人在哪
# ============================================================

BODILY: tuple[str, ...] = ("wounded", "poisoned", "subdued")    # 否认它们（“毫发无损”“并未受伤”）即与内核相悖
# 动作结果：说出口就是断言做成了（“拍中”不是“拍向”，“拿起”不是“伸手去拿”）
DEED_LEXICON: dict[str, tuple[str, ...]] = {
    "attack": ("击中", "打中", "拍中", "刺中", "砍中", "劈中", "踢中", "命中", "打倒", "击倒", "打翻", "撂倒",
               "打伤", "刺伤", "砍伤", "击伤"),
    "take": ("拿起", "拿到", "取下", "夺过", "夺下", "抢到", "抢过", "捡起", "拾起", "到手", "抽出", "拔出", "掏出", "取出"),
    "gain": ("得手",),
}
# 哪些做成了的操作许可这些说法；拿取类另认“东西本来就在他身上”（掏出自己的易经）
DEED_LICENSE: dict[str, frozenset[str]] = {
    "attack": frozenset({"attack"}),
    "take": frozenset({"take", "inspect", "receive"}),
    "gain": frozenset({"attack", "take", "inspect"}),
}
# 东西在玩家手里的说法（词条是正则片段）
POSSESSION: tuple[str, ...] = (r"[握攥捏抓拿揣]在了?你?的?[手掌怀][中里心上]?", r"(?:落|到)[在入]?了?你的?[手掌怀]",
                               r"在你的?[手掌怀][中里心上]")
POSITION_VERBS: tuple[str, ...] = ("身在", "站在", "置身", "立在", "身处", "已在", "坐在")
DEPARTURE_VERBS: tuple[str, ...] = ("出了", "走出", "溜出", "离开", "退出", "逃出", "奔出", "冲出", "步出", "跨出", "离了")
PLACE_GAP = 1                 # 站位/离开的动词与地点之间至多隔这么多字（“已站在后院”“溜出了大殿”）
# 假设、意图、差一点：说的不是已经发生的事（词条是正则片段：“要穴”“想必”不算）
MOODS: tuple[str, ...] = ("若", "如果", "倘若", "要是", r"想(?![必来])", r"要(?![穴害紧])", "欲", "打算", "准备", "不妨", "可以",
                          "何不", "不如", "该", "能否", "试图", "企图", "妄图", "险些", "差点", "几乎", "正待", "未及",
                          "以为", "只道", "还当", "莫非", "难道")      # 原以为/莫非：说的是心里的旧认知与疑问
_STOPS = "。！？!?\n"


# ============================================================
#  找人：关键词说的是谁
# ============================================================


def _sentence(masked: str, i: int) -> tuple[int, int]:
    lo = max(masked.rfind(c, 0, i) for c in _STOPS) + 1
    hi = min((k for c in _STOPS if (k := masked.find(c, i)) >= 0), default=len(masked))
    return lo, hi


def _who(masked: str, i: int, forms: Iterable[str], people: dict[str, str], whole: bool) -> list[str] | None:
    """[i, …) 处的关键词说的是谁（本名）。同一句里从关键词所在的小句往前，取第一个提到人的小句里的全部称呼；
    whole=True 时关键词所在的小句整句都算（“受了伤的龚光杰”），否则只看它之前（施动者在前）。
    “被”字句取“被”之后的人；有代词、或“被”后面不是人：说不清是谁，返回 None。"""
    lo, hi = _sentence(masked, i)
    parts = _clauses(masked, lo, hi)
    own = next((k for k, (a, b) in enumerate(parts) if a <= i < b), None)
    if own is None:
        return []
    forms = tuple(forms)
    for k in range(own, -1, -1):
        a, b = parts[k]
        seg = masked[a:b] if whole or k != own else masked[a:i]
        passive = seg.rfind("被") if not whole else -1
        found = [w for _, w in _mentions(seg[passive + 1:], forms)]
        if passive >= 0 and not found:
            return None
        if found:
            return None if any(w in PRONOUNS for w in found) else [people.get(w, w) for w in found]
    return []


def _hedged(masked: str, i: int) -> bool:
    """问句、假设与意图（“你是溜出大殿，还是……？”“若能一掌打倒他”“险些击中”）：不是已经发生的事。"""
    lo, hi = _sentence(masked, i)
    return masked[hi:hi + 1] in ("？", "?") or bool(_lexical(masked[lo:i], MOODS))


def _denied(masked: str, i: int, w: str) -> bool:
    return _negated(masked, i) or _short_of(masked, i, w)


# ============================================================
#  人事闸门
# ============================================================


def check_deeds(text: str, plan: RenderPlan, known_names: Iterable[str] = ()) -> tuple[Violation, ...]:
    """与 check() 并用；引语里的话不在这里查（交给台词闸门）。
    - 状态落错人（“龚光杰受了内伤”而受伤的是你）或否认确有的伤毒被制（“你却毫发无损”“你并未受伤”）：status
    - 没做成的动作说成做成（拍中、打倒、拿起、抽出、得手），没拿到的东西说成在你手里：outcome
    - 没走成却已身在别处（“你已身在后院”），或没走却已离开此地（“你溜出大殿”）：teleport"""
    masked = _mask(text, _quotes(text))
    people = {**dict(plan.people), "你": plan.viewer_name}
    persons = (*people, *PRONOUNS)
    goods = dict(plan.goods)
    done = set(plan.done)
    afflicted = set(plan.afflicted)
    me = plan.viewer_name
    mine = {i for h, i in plan.holdings if h == me}
    out: list[Violation] = []

    # ---- 1. 状态落在谁身上 ----
    for status, words in STATUS_LEXICON.items():
        if status not in plan.statuses:
            continue                                    # 本回合没有的状态由 check() 拦
        for i, w in _lexical(masked, words, STATUS_EXCLUSIONS):
            who = _who(masked, i, persons, people, whole=True)
            if not who:
                continue
            if not _denied(masked, i, w):
                if not any((n, status) in afflicted for n in who):
                    out.append(Violation("status", f"{status}:{'/'.join(who)}:{w}"))
            elif status in BODILY and all((n, status) in afflicted for n in who):
                out.append(Violation("status", f"denied:{'/'.join(who)}:{w}"))
    for i, w in _lexical(masked, DENIALS):
        who = _who(masked, i, persons, people, whole=True)
        if who and all(any((n, st) in afflicted for st in BODILY) for n in who) and not _negated(masked, i):
            out.append(Violation("status", f"denied:{'/'.join(who)}:{w}"))

    # ---- 2. 动作结果：谁做成了什么 ----
    for group, words in DEED_LEXICON.items():
        for i, w in _lexical(masked, words):
            if _denied(masked, i, w) or _hedged(masked, i):
                continue
            who = _who(masked, i, persons, people, whole=False)
            if not who or any((n, op) in done for n in who for op in DEED_LICENSE[group]):
                continue
            if group == "take":
                named = {goods[g] for _, g in _mentions(_clause_after(masked, i, 2 * ARRIVAL_WINDOW), goods)}
                if not named or any((n, g) in plan.holdings for n in who for g in named):
                    continue                            # 没点名东西（修辞里的茶碗），或本来就在他身上
            out.append(Violation("outcome", f"{group}:{'/'.join(who)}:{w}"))

    # ---- 3. 东西在玩家手里 ----
    if (me, "take") not in done:
        for i, w in _lexical(masked, POSSESSION):
            if _denied(masked, i, w) or _hedged(masked, i):
                continue
            who = [me] if "你" in w else _who(masked, i, persons, people, whole=False)
            if who and set(who) == {me}:
                lo, _ = _sentence(masked, i)
                clause = max((a for a, _ in _clauses(masked, lo, i + len(w)) if a <= i), default=lo)
                out += [Violation("outcome", f"hold:{goods[g]}") for _, g in _mentions(masked[clause:i], goods)
                        if goods[g] not in mine]

    # ---- 4. 玩家此刻在哪：没走成就不在别处，没走就没离开 ----
    if plan.here:
        everything = set(known_names) | plan.names | plan.aliases | plan.hidden | plan.places
        moved = (me, "move") in done
        for i, v in _mentions(masked, (*POSITION_VERBS, *DEPARTURE_VERBS)):
            if _hedged(masked, i):
                continue
            hits = _mentions(_clause_after(masked, i + len(v), ARRIVAL_WINDOW), everything)
            if not hits or hits[0][0] > PLACE_GAP or hits[0][1] not in plan.places:
                continue
            place = hits[0][1]
            who = _who(masked, i, persons, people, whole=False)
            if not who or set(who) != {me}:
                continue                                # 只查玩家自己：别人站在哪由内核的清单说了算
            elsewhere = v in POSITION_VERBS and place not in plan.arrivals | plan.here and v + place not in plan.source
            left = v in DEPARTURE_VERBS and not moved and place in plan.here
            if elsewhere or left:
                out.append(Violation("teleport", v + place))
    return tuple(dict.fromkeys(out))
