"""
[INPUT]: 依赖 core 的 Op / Social / derive_seed，language/render 的 RenderPlan，language/scene 的 VoiceLine / SceneBrief
[OUTPUT]: 对外提供 system_prompt()（主持人之声的系统提示）、scene_prompt()（一回合的用户提示词）、SOCIAL_LABELS / SOCIAL_PHRASES
          （言语行为交给模型的说法与模板措辞）、render_voice()（一句 NPC 言语的确定性模板）、
          in_words() / lapse_line()（钟点换成时辰文字、一段等待之后的时辰）、CLOCK_ANY（正文里的钟点数字）/ TIMED（正文已交代过时辰）、
          grams()（去掉标点空白后的 n 字片段：复述、谈资说过没有、台词重复都用它）、SAID_SHOWN / REPEAT
[POS]: language/narrator 的措辞层：主持人之声交给模型的全部文字——系统提示（第二人称、80~250 字、台词归属、只许点名可点名的、
       不替玩家开口、不写钟点、停在钩子上）与用户提示（最近三段正文每段末尾约 300 字且钟点换成时辰、玩家原话、事实清单、
       身体状况与身边人的伤、天色、干等时的钩子、意料之外、没人接的话、要说出口的话（按叙述者给的顺序；腔调/谈资/来历/近来经历/回应/可点名）、
       本回合开口的人最近说过的原话（至多 SAID_SHOWN 句，与这回合要说的三字片段重合 ≥REPEAT 的加注“换个说法”）、
       初见外观、眼下处境、收幕、已写好的开头），以及没有模型时台词的模板措辞。提示词只是请求，闸门在 narrator 里验收；
       模型只见时辰文字，不见钟点数字（钟点的样子与闸门同一份 CLOCK_ANY）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from tianlong.core import Op, Social, derive_seed
from tianlong.language.render import RenderPlan
from tianlong.language.scene import SceneBrief, VoiceLine

RECENT_KEEP = 3            # 提示词里最近几回合的正文：只留最后几段
RECENT_CHARS = 300         # 每段只留末尾这么多字
SAID_SHOWN = 3             # 本回合开口的人最近说过的原话：提示词里至多列这么多句（新的在后）
REPEAT = 0.6               # 旧话与这回合要说的原话/说法三字片段重合（按较短的一方算）到这个比例：加注“换个说法”

# ============================================================
#  主持人之声的系统提示：第二人称、有限长度、台词归属、不替玩家开口、停在钩子上
# ============================================================

_GM = (
    "你是一部武侠文字冒险游戏的主持人，为玩家讲述本回合刚刚发生的事。\n"
    "1. 用第二人称“你”称呼玩家；写一段 80~250 字的中文，生动而简练，文风见下。只输出正文，不加标题与说明。\n"
    "2. 只写给定的事实与要说出口的话：不得添加任何新的人物、物品、事件、伤势、承诺或结论。程度照原样：受伤不等于被制住，"
    "略有所得不等于学成，发现了东西不等于拿到手。玩家的输入只表明意图与姿态，成败一律以事实为准。\n"
    "3. “要说出口的话”每一句都写成对白，格式为 某某道：“……”，措辞合乎说话者的腔调与言语行为；"
    "有“说法”的须如实转达、不多不少；闲话只可说他的谈资、他自己近来的经历与眼前看得见的事。\n"
    "4. 某人的台词里只许提到列给他的“可点名”的名字、他说话的对象和玩家，此外任何人与物都不许提。\n"
    "5. 绝不替玩家说新的话、生新的念头或做任何决定；玩家本回合的原话可以照引，一字不改。\n"
    "6. 换着说法写：不要重复最近几段正文的开头与句式，不要用“你按兵不动”“静观其变”之类的套话。\n"
    "7. 不写数字钟点，时辰与天色一律用文字描述。\n"
    "8. 结尾自然收住，停在一个钩子上：某人的问话或动作、一声响动、一个显露出来的代价、一处引人注意的细节。"
    "不要替玩家列选项，不要用“你是……还是……？”这种句式（最近几段正文用过的收尾方式就换一种），不替玩家选。"
    "不编造征兆、突然的静默、来历不明的影子；钩子只落在本回合给出的事实、台词、看点或眼前可做的事上。"
)


def system_prompt(setting: str = "", style: str = "") -> str:
    """系统提示 = 主持规矩 + 场景的世界前提与文风。"""
    return _GM + (f"\n世界：{setting}" if setting else "") + (f"\n文风：{style}" if style else "")


# 言语行为的中文说法（交给模型）与模板措辞（无模型时的台词引子；{to} 是听者，没有听者时是“众人”）
SOCIAL_LABELS: dict[Social, str] = {
    Social.GREET: "打招呼", Social.THANK: "道谢", Social.APOLOGIZE: "赔罪", Social.PLEAD: "求情", Social.PRAISE: "称赞",
    Social.THREATEN: "威胁", Social.TAUNT: "挑衅讥讽", Social.INSULT: "辱骂", Social.REFUSE: "回绝", Social.AGREE: "附和",
    Social.JOKE: "说笑打趣", Social.COMFORT: "安慰", Social.EXPLAIN: "解释", Social.CHALLENGE: "叫阵",
    Social.COMMAND: "喝令", Social.FAREWELL: "告辞", Social.REMARK: "随口一说", Social.SUBMIT: "服软",
}
SOCIAL_PHRASES: dict[Social, tuple[str, ...]] = {
    Social.GREET: ("笑嘻嘻地跟{to}打招呼", "拱手向{to}见礼", "朝{to}点头招呼"),
    Social.THANK: ("向{to}连声道谢", "朝{to}拱手称谢"),
    Social.APOLOGIZE: ("陪着笑脸向{to}告罪", "向{to}连连赔礼"),
    Social.PLEAD: ("向{to}苦苦求情", "央求{to}"),
    Social.PRAISE: ("对{to}赞不绝口", "向{to}大加夸赞"),
    Social.THREATEN: ("沉下脸来威吓{to}", "按剑向{to}厉声威吓"),
    Social.TAUNT: ("斜眼讥讽{to}", "冷笑着挖苦{to}"),
    Social.INSULT: ("指着{to}破口大骂", "向{to}恶声喝骂"),
    Social.REFUSE: ("摇头回绝{to}", "一口回绝了{to}"),
    Social.AGREE: ("连连点头附和{to}", "向{to}点头称是"),
    Social.JOKE: ("笑嘻嘻地打趣{to}", "跟{to}说笑"),
    Social.COMFORT: ("温言劝慰{to}", "柔声安慰{to}"),
    Social.EXPLAIN: ("慢条斯理地对{to}讲", "向{to}解释"),
    Social.CHALLENGE: ("冷笑着向{to}叫阵", "踏上一步向{to}邀斗"),
    Social.COMMAND: ("厉声喝令{to}", "沉声吩咐{to}"),
    Social.FAREWELL: ("向{to}拱手告辞", "朝{to}拱了拱手作别"),
    Social.REMARK: ("随口对{to}说", "自顾自地嘀咕"),
    Social.SUBMIT: ("向{to}低头服软", "朝{to}连连作揖"),
}
_OP_PHRASES: dict[str, tuple[str, ...]] = {Op.TELL.value: ("对{to}说", "对{to}道"), Op.ASK.value: ("问{to}", "向{to}问道")}

# 钟点：提示词里换成时辰文字，正文里出现即丢句（“第1日”“19:00”“19点20分”“晚上七点二十分”；“一点半点”不算）
_CLOCK = re.compile(r"第\s*(\d+)\s*[日天]\s*(\d{1,2})\s*[:：]\s*(\d{2})")
_NUM = "零一二两三四五六七八九十"
CLOCK_ANY = re.compile(rf"第\s*\d+\s*[日天]|\d{{1,2}}\s*[:：]\s*\d{{2}}|\d{{1,2}}\s*[点时](?:\s*\d{{1,2}}\s*分|钟|整|半)?"
                       rf"|[{_NUM}]{{1,3}}\s*点\s*(?:[{_NUM}]{{1,3}}\s*分|钟|整|半(?!点))"
                       rf"|(?:早上|上午|中午|下午|晚上|凌晨|夜里|傍晚)[{_NUM}]{{1,3}}点")
_BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
_SKY = ("夜半", "深夜", "黎明前", "破晓", "清晨", "上午", "正午", "午后", "下午", "傍晚", "入夜", "夜深")
TIMED = re.compile(f"[{_BRANCHES}]时|{'|'.join(_SKY)}")     # 正文已经交代过时辰


def grams(text: str, n: int = 3) -> frozenset[str]:
    """去掉标点空白后的 n 字片段：判断复述、判断谈资说过没有、判断台词重复，都用它。"""
    t = re.sub(r"[^\w]", "", text)
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def _span(minutes: int) -> str:
    return ("一小会儿" if minutes < 20 else "两三刻工夫" if minutes < 50 else "约莫半个时辰" if minutes < 90
            else "一个多时辰")


def lapse_line(lapse: str, since: str = "") -> str:
    """一段等待之后的时辰：跨了时辰说“（不觉已是入夜戌时）”；还在同一个时辰里说“（不觉过了两三刻工夫）”——
    开场已是酉时，等了半个时辰再说“到了傍晚酉时”，读来像时间倒流。"""
    a, b = _CLOCK.search(since), _CLOCK.search(lapse)
    if a and b:
        start, end = (int(m.group(1)) * 1440 + int(m.group(2)) * 60 + int(m.group(3)) for m in (a, b))
        if 0 <= end - start < 240 and (int(a.group(2)) + 1) // 2 == (int(b.group(2)) + 1) // 2:
            return f"（不觉过了{_span(end - start)}）"
    return f"（不觉已是{in_words(lapse)}）"


def in_words(text: str) -> str:
    """“第1日 19:00” → “入夜戌时”：模型只见时辰文字，不见钟点数字。"""
    def say(m: re.Match[str]) -> str:
        k = ((int(m.group(2)) + 1) // 2) % 12
        return f"{_SKY[k]}{_BRANCHES[k]}时"
    return CLOCK_ANY.sub("", _CLOCK.sub(say, text))


# ============================================================
#  台词的模板：说话者 + 言语行为措辞 + 原话（或说法）；措辞按语义输入确定地轮换
# ============================================================


def render_voice(line: VoiceLine, salt: str = "") -> str:
    """“龚光杰冷笑着向你叫阵：“你笑什么？””；没有原话也没有说法的闲话只写言语行为（“钟灵笑嘻嘻地跟你打招呼。”）。
    salt 让同一句话在不同的上下文里换个说法（会话用上一回合的正文），相同输入永远得到相同文字。"""
    words = line.template or line.claim
    options = (SOCIAL_PHRASES.get(line.social) if line.social else None) or _OP_PHRASES.get(line.op, _OP_PHRASES["tell"])
    pick = derive_seed("voice", line.speaker, line.social.value if line.social else "", words, salt) % len(options)
    phrase = options[pick].format(to=line.listener_name or "众人")
    return f"{line.speaker_name}{phrase}：“{words}”" if words else f"{line.speaker_name}{phrase}。"


# ============================================================
#  一回合的用户提示词
# ============================================================


def scene_prompt(brief: SceneBrief, command: str, when: Sequence[str], facts: Sequence[str], looks: Sequence[str],
                 plan: RenderPlan, known: frozenset[str], lead: str = "") -> str:
    parts: list[str] = []
    recent = [in_words(p).strip() for p in brief.recent[-RECENT_KEEP:] if p.strip()]
    if recent:
        recent = [p if len(p) <= RECENT_CHARS else "……" + p[-RECENT_CHARS:] for p in recent]
        rows = [f"【{i}】{p}" for i, p in enumerate(recent, 1)]
        parts.append("最近几段正文（旧→新，只作接续，不要重复其句式）：\n" + "\n".join(rows))
    if command:
        parts.append(f"玩家的输入：{command}")
    if brief.player_line:
        parts.append(f"玩家本回合说出口的话或做出的姿态（可原样照引，一字不改）：{brief.player_line}")
    empty = "（除下列言语外无事发生）" if brief.lines else "（无事发生：写眼前的光景与身边的人此刻的样子，不要编出新的事）"
    parts.append("本回合玩家感知到的事实：\n" + "\n".join([*when, *(list(facts) or [empty])]))
    if brief.present:
        around = f"{brief.present[0]}" + (f"，身边有{'、'.join(brief.present[1:])}" if len(brief.present) > 1 else "，身边没有别人")
        parts.append(f"玩家以为自己此刻在：{around}")
    if brief.sky is not None:
        parts.append(f"天色（日头、月亮只照这个写，别写早了或晚了）：{brief.sky.text}")
    if brief.condition:
        parts.append(f"玩家自己的身体状况（记在心里，不要写得像没事人，也不要加重；只在这一步吃力——走动攀爬、出手、挨打——"
                     f"或刚受伤时带一笔，别每回合都写疼）：{brief.condition}")
    if brief.hurt:
        parts.append("身边的人（玩家以为的伤势；写他们的举动时别像没事人，也不要加重；不必每回合都写到伤）："
                     + "；".join(brief.hurt))
    if brief.hooks:
        parts.append("玩家在等，眼前却有可做的事（" + " / ".join(brief.hooks) + "）：可以把他的目光引向其中一处，"
                     "只写看得见的样子，不点破会有什么结果，不替他决定")
    if brief.notes:
        parts.append("本回合出乎玩家意料之处（写出他的觉察与惊讶，不要替他下结论，也不要替别人解释原因）：\n"
                     + "\n".join(f"- {n}" for n in brief.notes))
    if brief.unanswered:
        parts.append(f"玩家的话还没人接：{brief.unanswered}没有回答你（不要替他编答案；写出他没顾上回答、或没有理会的样子）")
    if brief.lines:
        # 可点名只列真能通过闸门的：本回合计划里的名字、清单里原样有的名字、闸门不认识的名字
        universe = known | plan.hidden
        speakable = plan.names | plan.aliases

        def fits(n: str) -> bool:
            return n in speakable or n in plan.source or n not in universe or n in lines_text
        lines_text = "\n".join(x for vl in brief.lines for x in (vl.knows, vl.lately, vl.about) if x)
        parts.append("要说出口的话（按这个顺序，逐句写成对白）：\n"
                     + "\n".join(_describe(i, vl, fits) for i, vl in enumerate(brief.lines, 1)))
    parts += _said_before(brief)
    if looks:
        parts.append("玩家初次看清的人与物（仅作外观描写的依据）：\n" + "\n".join(looks))
    if brief.stakes:
        parts.append(f"眼下的处境（写到这里，停在钩子上）：{brief.stakes}")
    if brief.closing:
        parts.append("这是这一幕的最后一段：写出抵达此地的画面与此刻的心绪，收在余韵上；"
                     "不要再抛出任何选择、去向或问题（这一条压过“停在钩子上”）。")
    if lead:
        parts.append(f"开头一句已经写好，玩家已经看到了：{lead}\n"
                     "从下一句接着写：不要复述这一句，也不要再交代玩家这一步做成没有，直接写旁人的反应、言语与周遭。")
    return "\n\n".join(parts)


def _describe(i: int, vl: VoiceLine, fits: Callable[[str], bool]) -> str:
    """一句要说出口的话交给模型的样子：谁对谁、什么言语行为、说法还是闲话、腔调、谈资、回应什么、可点名什么。"""
    names = sorted(n for n in vl.may_name if n != vl.speaker_name and fits(n))
    if vl.act:                          # 动手时顺口喝的一声：给人物一副嗓子，不给事实
        return "\n".join([f"{i}. {vl.speaker_name}{vl.act}时，可以顺口喝一声（一两句，只说这一下的事，不说也行）",
                          *([f"   腔调：{vl.voice}"] if vl.voice else []),
                          "   台词里可点名：" + ("、".join(names) if names else "（除对手与玩家外，谁也不提）")])
    ask = vl.op == Op.ASK.value
    act = (SOCIAL_LABELS.get(vl.social) if vl.social else None) or ("问话" if ask else "说话")
    body = (f"说法“{vl.claim}”——须如实转达，不多不少" if vl.claim
            else "回话——先接住玩家的话头，只可说谈资里的掌故、他自己近来的经历、他所知的来历与眼前人人看得见的事，不得夹带别的事实"
            if vl.answering else
            "闲话——只可说谈资里的掌故、他自己近来的经历与眼前人人看得见的事，不得夹带别的事实")
    if vl.template:
        body += f"；原话“{vl.template}”——可换措辞，意思不变"
    rows = [f"{i}. {vl.speaker_name}对{vl.listener_name or '众人'}（{act}）：{body}"]
    if vl.voice:
        rows.append(f"   腔调：{vl.voice}")
    if vl.knows:
        rows.append(f"   谈资（他还没说过的；与眼下无关就不提，提也只挑一条）：{vl.knows}")
    if vl.about:
        rows.append(f"   玩家问到的人，他所知的来历（据此回答）：{vl.about}")
    if vl.lately:
        rows.append(f"   他自己近来的经历（{'玩家问起他怎么来的，就照这个回答' if vl.answering else '可以借这句话说起，不必都说'}）："
                    f"{vl.lately}")
    if vl.answering:
        rows.append(f"   回应的是：“{vl.answering}”")
    else:
        rows.append("   （不是在回应玩家的某句话：不要写成是在接玩家的话头）")
    rows.append("   台词里可点名：" + ("、".join(names) if names else "（除说话对象与玩家外，谁也不提）"))
    return "\n".join(rows)


def _said_before(brief: SceneBrief) -> list[str]:
    """本回合要开口的人最近说过的原话（至多 SAID_SHOWN 句）：与他这回合要说的原话或说法重合 ≥REPEAT 的，加注换个说法。
    他这回合只是顺口喝一声（act）或没有台词，就不列。"""
    out: list[str] = []
    for name, said in brief.said_before:
        mine = [vl for vl in brief.lines if vl.speaker_name == name]
        if not mine or not said:
            continue
        now = [g for vl in mine if (g := grams(vl.template or vl.claim or ""))]
        rows = [f"- “{old}”" + ("（与这回合要说的几乎一样：换个说法）" if _echoes(grams(old), now) else "")
                for old in said[-SAID_SHOWN:]]
        out.append(f"{name}最近说过的话（别重复）：\n" + "\n".join(rows))
    return out


def _echoes(was: frozenset[str], now: Sequence[frozenset[str]]) -> bool:
    """旧话与这回合要说的某一句三字片段重合（按较短的一方算）≥ REPEAT。"""
    return bool(was) and any(len(was & g) >= REPEAT * min(len(was), len(g)) for g in now)
