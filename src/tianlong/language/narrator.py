"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Social / is_night / derive_seed，language/templates 的 Names / render_percept，
         language/llm 的 LLMClient / LLMUnavailable，language/scene 的 VoiceLine / SceneBrief / TextSink，
         language/render 的 fact_lines / build_plan / check / restated_hearsay / sentence_ends / Violation / Rendered / RenderStatus，
         language/deeds 的 check_deeds，language/quotes 的 check_quotes / voiced，language/lead 的 lead_line / restates
[OUTPUT]: 对外提供 Narrator（narrate_scene() 主持人之声：流式生成、逐句过闸门、通过即交付；narrate_rendered() / narrate()
          以空 SceneBrief 委托之；secrets 是场景的秘密词表）、render_voice()（一句 NPC 言语的确定性模板）、SOCIAL_PHRASES / SOCIAL_LABELS、
          MAX_DROPS、MAX_CHARS、fact_lines()（再导出）、lore_keys()
[POS]: language 的输出层（主持人之声）。输入只有玩家自己的感知与会话交来的 SceneBrief（要替 NPC 说出口的话、最近几回合正文、
       玩家原话、眼下的钩子），不是世界真相；台词本身算出处（说话者与听者可点名，原话与说法照搬不算违规）。
       模板先写成事实清单，清单里听见的言语换成带言语行为的台词（“龚光杰冷笑着向你叫阵：……”；只看见的耳语照旧），措辞按语义输入确定地轮换。
       先声（lead_line：玩家自己这一步的结果，确定的句子，照样过闸门）两种用法：lead_after>0（默认 LEAD_AFTER 秒）只在模型
       迟迟不交付第一句时顶上、自成一段——模型不知道它，照常铺陈这一步，悬念留给模型快的回合；lead_after=0 立即交付，
       模型被告知开头已写好、清单里不再列玩家自己的行动；None 不用。先声之后模型开头 ECHO_WINDOW 句里复述它的
       （三字片段重合过半；立即模式另加 restates() 的同一动作）悄悄略过，夹带了错的照样丢句记账；收尾补模板时先声讲过的行不再重复。
       读流在常驻线程里边读边计时（_deadline）。随后逐句流式生成：每句对“已交付的文字 + 这一句”跑 check()、逐句传闻 restated_hearsay()、人事闸门 check_deeds()
       与台词闸门 check_quotes()，再查钟点数字（含“19点20分”“七点二十分”）；通过即经 on_text 交付，违规即丢弃并记下；
       交付满 MAX_CHARS 字即停止读流。收尾：传闻有没有归属整段查；台词没说出口、玩家自己的行动与后果、冲着玩家来的事、
       听见的话一个参与者都没提的，补上模板行、状态记为 gated_fallback（违规 omitted）——一行写出的后果里的东西（翻出的帛卷）
       也算参与者，玩家的原话或姿态照着写了出来也算讲到；一句都没通过、丢满 MAX_DROPS 句或传闻没有归属，补上模板
       （已交付过正文的，换行后只补正文没讲到的：必讲之事照上，别人之间的事与所见一个参与者都没提的才补，空空的所见不补；
       正文交代过时辰就不再补）。
       模型中途失败：已交付的留着，补上模板。补上的模板行读起来是句子（缺句末标点的补“。”，时辰用文字）；没有模型时的模板照旧。
       Rendered.text 恒等于 on_text 收到的全部文字首尾相接，Rendered.dropped 是丢掉的句数。
       提示词只是请求，闸门才是验收：第二人称、80~250 字、台词写成“某某道：“……””、不替玩家开口、停在钩子上、不写钟点；
       最近正文只留最后三段、每段末尾约 300 字，钟点换成时辰文字；世界前提与文风来自场景
       前后照应（SceneBrief：玩家的身体状况、出乎他意料之处、眼前的地点与人、说话者近来的经历）写进提示词且算出处；闲话只可说谈资、近来的经历与眼前的事；
       无事发生时有模型就写眼前的光景；到了结局收在余韵上
       先声之后，别的必讲之事讲到没有只看模型自己交付的正文；有台词时引语里的名字与状态词交给台词闸门按说话者查。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from tianlong.core import Modality, Op, Percept, Social, derive_seed, is_night
from tianlong.language.deeds import check_deeds
from tianlong.language.lead import lead_line, restates
from tianlong.language.llm import LLMClient, LLMUnavailable
from tianlong.language.quotes import check_quotes, voiced
from tianlong.language.render import (
    QUOTE_CLOSE,
    SENTENCE_ENDS,
    Rendered,
    RenderPlan,
    RenderStatus,
    Violation,
    build_plan,
    check,
    fact_lines,
    restated_hearsay,
    sentence_ends,
)
from tianlong.language.scene import SceneBrief, TextSink, VoiceLine
from tianlong.language.templates import Names, render_percept

log = logging.getLogger(__name__)

__all__ = ["MAX_DROPS", "SOCIAL_LABELS", "SOCIAL_PHRASES", "Narrator", "fact_lines", "lore_keys", "render_voice"]

MAX_DROPS = 2              # 丢满这么多句就不再相信这段生成：停止读流，补上模板
MAX_CHARS = 600            # 交付满这么多字就停止读流（提示词要 80~250 字）：啰嗦的模型也有个头
RECENT_KEEP = 3            # 提示词里最近几回合的正文：只留最后几段
RECENT_CHARS = 300         # 每段只留末尾这么多字
ECHO = 0.5                 # 模型的一句与先声的三字片段重合过半：是在复述先声，悄悄略过（不算违规）
ECHO_WINDOW = 2            # 只在模型开头这么多句里找复述：后文再提到同一件事是正常的接续
LEAD_AFTER = 1.5           # 模型这么多秒还没交付一句，先声顶上（0 = 立即交付并告诉模型开头已写好；None = 不用先声）

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
    "8. 结尾停在一个钩子上——某人的问话、一个显露出来的代价、或一个摆在玩家面前的选择——停在那里，不替玩家选。"
)

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
_CLOCK_ANY = re.compile(rf"第\s*\d+\s*[日天]|\d{{1,2}}\s*[:：]\s*\d{{2}}|\d{{1,2}}\s*[点时](?:\s*\d{{1,2}}\s*分|钟|整|半)?"
                        rf"|[{_NUM}]{{1,3}}\s*点\s*(?:[{_NUM}]{{1,3}}\s*分|钟|整|半(?!点))"
                        rf"|(?:早上|上午|中午|下午|晚上|凌晨|夜里|傍晚)[{_NUM}]{{1,3}}点")
_BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
_SKY = ("夜半", "深夜", "黎明前", "破晓", "清晨", "上午", "正午", "午后", "下午", "傍晚", "入夜", "夜深")
_TIMED = re.compile(f"[{_BRANCHES}]时|{'|'.join(_SKY)}")     # 正文已经交代过时辰


def _in_words(text: str) -> str:
    """“第1日 19:00” → “入夜戌时”：模型只见时辰文字，不见钟点数字。"""
    def say(m: re.Match[str]) -> str:
        k = ((int(m.group(2)) + 1) // 2) % 12
        return f"{_SKY[k]}{_BRANCHES[k]}时"
    return _CLOCK_ANY.sub("", _CLOCK.sub(say, text))


_VISUAL = frozenset({Modality.SELF, Modality.SIGHT, Modality.SCENE})


def lore_keys(viewer: str, percepts: Sequence[Percept], lore: Mapping[str, str]) -> list[str]:
    """本回合真正映入眼帘、且有外观描写的实体：身处的地点、在场的人与物、眼前的通道与事件的参与者。
    外观只由亲眼所见触发——听人说起的、隔墙听见的、门那头的地点都只有名字（草图 seen=False），不描写。
    夜里优先取 "id@night" 变体（月下的玉璧不同于白日的玉璧）。"""
    keys: list[str] = []
    for p in percepts:
        if p.modality not in _VISUAL:
            continue
        night = is_night(p.tick)
        in_view = {viewer}
        for f in p.facts:
            if f.holds and f.prop.predicate == "AT":
                in_view.update((f.prop.subject, str(f.prop.value)))
            elif f.holds and f.prop.predicate == "CONNECTS":
                in_view.add(f.prop.subject)          # 门本身在眼前；门那头的地点不算
        if p.event is not None:
            in_view.update(x for x in (p.event.actor, p.event.target, p.event.obj) if x)
        for sk in p.sketches:
            if sk.id == viewer or sk.id not in in_view or not sk.seen:
                continue
            key = f"{sk.id}@night" if night and f"{sk.id}@night" in lore else sk.id
            if key in lore and key not in keys:
                keys.append(key)
    return keys


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


def _with_lines(plan: RenderPlan, brief: SceneBrief) -> RenderPlan:
    """会话交来的上下文本身有出处：台词的说话者与听者可以点名，原话、说法与说话者近来的经历算出处；
    前后照应（玩家的身体状况、本回合的意外与变化、眼前的地点与人）同样算出处，其中的状态说出来不算升级，
    玩家自己身上的伤毒被制算落在对的人身上。"""
    names = {n for vl in brief.lines for n in (vl.speaker_name, vl.listener_name) if n and n != "你"}
    names |= set(brief.present)
    said = [x for vl in brief.lines for x in (vl.template, vl.claim) if x]       # 谈资与近来经历只在他自己的引语里算数（台词闸门）
    said += [x for x in (brief.condition, *brief.notes, "、".join(brief.present)) if x]
    if not said and not names and not brief.statuses:
        return plan
    return replace(plan, names=plan.names | names, source="\n".join([plan.source, *said]),
                   statuses=plan.statuses | brief.statuses, afflicted=(*plan.afflicted, *brief.afflicted))


_TALK = (Op.TELL.value, Op.ASK.value)


@dataclass(frozen=True, slots=True)
class _Row:
    """模板的一行：事实清单的一行，或替 NPC 说出口的一句台词。"""
    text: str
    voice: VoiceLine | None = None
    keys: tuple[str, ...] = ()     # 事实行的参与者（名与别称，观察者除外）：正文提到其一即算讲到了
    must: bool = False             # 必须讲到：玩家自己的行动与后果、冲着玩家来的事、听见的话
    mine: bool = False             # 玩家自己的行动：先声已经讲过，收尾补模板时不再重复
    said: str = ""                 # 玩家自己的原话或姿态：正文照着写了出来（三字片段重合过半）也算讲到了


def _must(p: Percept, viewer: str) -> bool:
    ev = p.event
    if ev is None:
        return False
    if p.modality == Modality.SELF:
        return ev.kind != Op.WAIT.value or bool(ev.utterance)     # 干等不必交代
    return p.modality == Modality.SPEECH or (p.modality == Modality.SIGHT and ev.target == viewer)


def _keys(p: Percept, viewer: str, names: Names, aliases: Mapping[str, Sequence[str]], line: str) -> tuple[str, ...]:
    """参与者之外，这一行写出来的后果与所见里的东西（“发现蒲团上藏着北冥神功帛卷……”）也算：正文写到帛卷就是讲到了。"""
    ev = p.event
    table = {sk.id: sk.name for sk in p.sketches} | {eid: sk.name for eid, sk in names.items()}
    ids = [x for x in ((ev.actor, ev.target, ev.obj) if ev is not None else ()) if x and x != viewer]
    ids += [f.prop.subject for f in p.facts if f.prop.subject != viewer and table.get(f.prop.subject, "\0") in line]
    return tuple(dict.fromkeys(n for x in ids for n in (table.get(x), *aliases.get(x, ())) if n))


def _scene_lines(plan: RenderPlan, viewer: str, percepts: Sequence[Percept], names: Names, brief: SceneBrief,
                 salt: str, aliases: Mapping[str, Sequence[str]],
                 familiar: frozenset[str] = frozenset()) -> tuple[list[_Row], frozenset[str]]:
    """事实清单里听见的 NPC 言语换成带言语行为的台词（同一说话者按先后、原话相同的优先对应），清单里对不上的台词补在最后；
    只看见在耳语、没听见内容的那一行照旧留着。返回 (模板的各行, 被台词取代的清单行)。"""
    pending: dict[str, list[int]] = {}
    for k, vl in enumerate(brief.lines):
        pending.setdefault(vl.speaker, []).append(k)
    spoken: dict[str, tuple[str, str | None]] = {}
    facts: dict[str, Percept] = {}
    for p in percepts:
        if p.modality == Modality.SCENE:
            facts.setdefault("你看到：" + render_percept(p, names, viewer, me="你", familiar=familiar), p)
            continue
        line = render_percept(p, names, viewer, me="你", familiar=familiar)
        facts.setdefault(line, p)
        ev = p.event
        if ev is not None and p.modality == Modality.SPEECH and ev.kind in _TALK and ev.actor in pending:
            spoken.setdefault(line, (ev.actor, ev.utterance))
    rows: list[_Row] = []
    covered: set[str] = set()
    for text in plan.lines:
        who, said = spoken.get(text, (None, None))
        if who is not None and pending.get(who):
            queue = pending[who]
            k = next((k for k in queue if brief.lines[k].template == said), queue[0])
            queue.remove(k)
            covered.add(text)
            rows.append(_Row(render_voice(brief.lines[k], salt), voice=brief.lines[k]))
        elif (p := facts.get(text)) is not None:
            mine = p.modality == Modality.SELF and p.event is not None and p.event.actor == viewer
            rows.append(_Row(text, keys=_keys(p, viewer, names, aliases, text), must=_must(p, viewer), mine=mine,
                             said=(p.event.utterance or "") if mine and p.event is not None else ""))
        else:
            rows.append(_Row(text))
    left = {k for ks in pending.values() for k in ks}
    rows += [_Row(render_voice(vl, salt), voice=vl) for k, vl in enumerate(brief.lines) if k in left]
    return rows, frozenset(covered)


def _prose(line: str) -> str:
    """模板行交给玩家时读起来像句子：没有句末标点的补个“。”；整行括注（外观描写、时辰）照旧。"""
    line = line.rstrip()
    if not line or line[-1] in SENTENCE_ENDS + QUOTE_CLOSE + "’』\"" or (line[0] == "（" and line[-1] == "）"):
        return line
    return line + "。"


def _spoken(row: _Row, text: str, said: frozenset[str]) -> bool:
    """这句台词正文里讲到了没有：有原话或说法的，须有一段归到说话者名下的引语（或名字后紧跟言说动词）；闲话提到其人即可。"""
    vl = row.voice
    assert vl is not None
    return vl.speaker_name in said if (vl.template or vl.claim) else vl.speaker_name in text


def _named(text: str, keys: Iterable[str]) -> bool:
    """正文提到了其中之一：四字以上的名字取头尾两字以上也算（“凌波微步帛卷”说成“帛卷”“凌波微步”）。"""
    return any(k in text or (len(k) >= 4 and any(k[:n] in text or k[-n:] in text for n in range(2, len(k))))
               for k in keys)


def _missing(rows: Sequence[_Row], text: str, plan: RenderPlan, brief: SceneBrief, dropped: int,
             strict: bool = False) -> list[_Row]:
    """模型正文漏掉的必讲之事：台词没说出口、玩家自己的行动与后果、冲着玩家来的事、听见的话一个参与者都没提
    （没有参与者可提的，只在丢过句子时算漏；玩家的原话或姿态照着写了出来不算漏）。
    strict：正文没通过闸门、要补模板时，别人之间的事（有参与者可提的）一个都没提也补；没有参与者的所见清单不补。"""
    said = voiced(text, plan, brief)
    out: list[_Row] = []
    for r in rows:
        if r.voice is not None:
            ok = _spoken(r, text, said)
        elif r.must or (strict and r.keys):
            ok = (_named(text, r.keys) if r.keys else dropped == 0) or _echoes(r.said, text)
        else:
            ok = True
        if not ok:
            out.append(r)
    return out


def _grams(text: str, n: int = 3) -> frozenset[str]:
    t = re.sub(r"[^\w]", "", text)
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def _echoes(said: str, text: str) -> bool:
    """玩家的原话或姿态在正文里照着写了出来：它的三字片段过半出现在正文里。"""
    mine = _grams(said)
    return bool(mine) and len(mine & _grams(text)) >= ECHO * len(mine)


def _deadline(pieces: Iterator[str], after: float, on_late: Callable[[], None], pool: ThreadPoolExecutor) -> Iterator[str]:
    """边读流边计时：读流放到常驻线程，after 秒一到就调用 on_late 一次（由它自己看要不要让先声顶上），其余原样转交。
    调用方中途不读了（丢满、写够长、出错），读流线程在下一段字到达时关掉底层的流。"""
    box: queue.Queue = queue.Queue()
    stop = threading.Event()

    def pump() -> None:
        try:
            for piece in pieces:
                if stop.is_set():
                    break
                box.put(("chunk", piece))
            else:
                box.put(("end", None))
        except Exception as e:  # noqa: BLE001 —— 原样交给读的一方
            box.put(("error", e))
        finally:
            close = getattr(pieces, "close", None)
            if callable(close):
                close()

    pool.submit(pump)
    due = time.monotonic() + after
    fired = False
    try:
        while True:
            if not fired and time.monotonic() >= due:
                fired = True
                on_late()
            try:
                kind, value = box.get(timeout=None if fired else max(0.0, due - time.monotonic()))
            except queue.Empty:
                continue
            if kind == "chunk":
                yield value
            elif kind == "end":
                return
            else:
                raise value
    finally:
        stop.set()


# ============================================================
#  流式交付与逐句闸门
# ============================================================


class _Delivery:
    """交付给玩家的全部文字：每一段都经 on_text 流出，Rendered.text 就是它们首尾相接。"""

    def __init__(self, sink: TextSink | None) -> None:
        self.sink = sink
        self.parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def emit(self, piece: str) -> None:
        if not piece:
            return
        self.parts.append(piece)
        if self.sink is not None:
            self.sink(piece)


class _Gate:
    """逐句验收：通过的句子立刻交付，违规的丢弃并记下；丢满 MAX_DROPS 句即停。"""

    def __init__(self, plan: RenderPlan, brief: SceneBrief, known: frozenset[str], command: str,
                 out: _Delivery) -> None:
        self.plan, self.brief, self.known, self.command, self.out = plan, brief, known, command, out
        # 有台词时，引语里的名字与状态词交给台词闸门按说话者查（他认识的、他的说法谈资与近来经历里有的才许）
        self.quoted = frozenset(n for vl in brief.lines for n in (*vl.may_name, vl.speaker_name)) if brief.lines else None
        self.buf = ""
        self.dropped = 0
        self.violations: list[Violation] = []
        self.lead = ""                          # 已交付的先声
        self._lead_grams: frozenset[str] = frozenset()
        self._echo: Callable[[str], bool] = lambda _: False
        self._judged = 0                        # 先声之后模型交来的句数

    def admit(self, lead: str, echo: Callable[[str], bool], sep: str = "") -> bool:
        """先声：内核结果写成的确定句子，照样过一遍闸门（不计丢句）；通过即交付（sep 接在后面：迟到的先声自成一段）。
        此后模型开头 ECHO_WINDOW 句里复述它的（三字片段重合过半，或 echo 判定是同一动作）悄悄略过，不算违规。"""
        if not lead or self.lead or self._found(lead, self.out.text + lead, self.out.text):
            return False
        self.out.emit(lead + sep)
        self.lead, self._lead_grams, self._echo = lead + sep, _grams(lead), echo
        self._judged = 0
        return True

    def feed(self, piece: str) -> bool:
        """吃进一段增量，交付其中完整的句子；返回 False 表示已丢满，不必再读。"""
        self.buf += piece
        return self._drain(final=False, tail=False)

    def finish(self, tail: bool = True) -> None:
        """流结束：结尾那句没有句末标点也照样验收（tail=False 用于中途失败：半句话不交付）。"""
        self._drain(final=True, tail=tail)
        self.buf = ""

    def _drain(self, final: bool, tail: bool) -> bool:
        cut = 0
        for end in sentence_ends(self.buf, final):
            self._judge(self.buf[cut:end])
            cut = end
            if self.dropped >= MAX_DROPS:
                self.buf = ""
                return False
        self.buf = self.buf[cut:]
        if tail and self.buf.strip():
            self._judge(self.buf)
            self.buf = ""
        return self.dropped < MAX_DROPS

    def _judge(self, raw: str) -> None:
        body = raw.strip()
        if not body:
            return
        before = self.out.text
        piece = ("\n" if before and "\n" in raw[:len(raw) - len(raw.lstrip())] else "") + body
        self._judged += 1
        found = self._found(piece, before + piece, before)
        if not found and self.lead and self._judged <= ECHO_WINDOW:      # 违规的照样记账；干净的复述悄悄略过
            grams = _grams(body)
            if (grams and len(grams & self._lead_grams) / len(grams) >= ECHO) or self._echo(body):
                log.info("模型复述了先声，略过: %s", body)
                return
        if found:
            self.dropped += 1
            self.violations += found
            log.info("叙述句未通过闸门，丢弃: %s %s", body, found)
            return
        self.out.emit(piece)

    def _found(self, piece: str, text: str, before: str) -> list[Violation]:
        found = [v for v in check(text, self.plan, self.known, self.quoted) if v.kind != "hearsay"]   # 传闻有没有归属：收尾整段查
        found += restated_hearsay(piece, text, self.plan)                                  # 这一句替传闻作保：当场丢
        found += check_deeds(text, self.plan, self.known)
        found += check_quotes(text, self.brief, self.plan, self.known, command=self.command, since=len(before))
        found += [Violation("clock", m.group(0)) for m in _CLOCK_ANY.finditer(piece)]
        return found


# ============================================================
#  叙述者
# ============================================================


class Narrator:
    """setting 给出世界前提与文风；lore 是实体外观描写，只在玩家看见该实体时、且仅首次看见时拿来润色；
    aliases 是场景别称全表，只供闸门使用：识别“走进大殿”这类以别称说出的抵达，并拒绝本回合未出场实体的别称（“神仙姐姐”）。"""

    def __init__(self, llm: LLMClient | None = None, setting: str = "", lore: Mapping[str, str] | None = None,
                 style: str = "", aliases: Mapping[str, Sequence[str]] | None = None, secrets: Sequence[str] = (),
                 lead_after: float | None = LEAD_AFTER) -> None:
        self.llm = llm
        self.lead_after = lead_after
        self._pump: ThreadPoolExecutor | None = None      # 迟到先声的读流线程（常驻：模型客户端的每线程长连接才用得上）
        self.setting = setting
        self.style = style
        self.lore = dict(lore or {})
        self.aliases = dict(aliases or {})
        self.secrets = tuple(secrets)

    def narrate(self, viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
                fresh: Sequence[str] = (), command: str = "", lapse: str = "", known: Iterable[str] = ()) -> str:
        return self.narrate_rendered(viewer, percepts, names, show_scene, fresh, command, lapse, known).text

    def narrate_rendered(self, viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
                         fresh: Sequence[str] = (), command: str = "", lapse: str = "",
                         known: Iterable[str] = ()) -> Rendered:
        """没有要替 NPC 说的话、也没有最近正文时的叙述：以空 SceneBrief 委托 narrate_scene()。"""
        return self.narrate_scene(viewer, percepts, names, brief=SceneBrief(), show_scene=show_scene, fresh=fresh,
                                  command=command, lapse=lapse, known=known)

    def narrate_scene(self, viewer: str, percepts: Sequence[Percept], names: Names, *, brief: SceneBrief,
                      show_scene: bool = False, fresh: Sequence[str] = (), command: str = "", lapse: str = "",
                      known: Iterable[str] = (), on_text: TextSink | None = None,
                      familiar: Iterable[str] = ()) -> Rendered:
        """familiar 是玩家此前已知下落的东西（再翻出来不算“发现”）；command 是玩家原话（让“跳下断崖”读起来像跳，成败仍以清单为准）；lapse 是一段等待之后的时辰，排在事实之前；
        known 是闸门用来拒绝的名字全集（玩家认识的 + 场景全部实体）；on_text 收到每一段交付的文字（通过闸门即交付）。"""
        known, familiar = frozenset(known), frozenset(familiar)
        looks = [self.lore[k] for k in fresh if k in self.lore]
        passed = [f"（不觉已是{lapse}）"] if lapse else []
        plan = _with_lines(build_plan(viewer, percepts, names, show_scene, looks, "".join(passed), self.aliases,
                                      self.secrets, familiar), brief)
        rows, covered = _scene_lines(plan, viewer, percepts, names, brief, brief.recent[-1] if brief.recent else "",
                                     self.aliases, familiar)
        scene = [r.text for r in rows]
        out = _Delivery(on_text)
        if not scene and not looks and (self.llm is None or not brief.present):
            out.emit("\n".join(["时间悄悄过去，什么也没有发生。", *passed]))
            return Rendered(out.text, RenderStatus.TEMPLATE)
        plain = "\n".join(passed + scene + [f"（{x}）" for x in looks])
        if self.llm is None:
            out.emit(plain)
            return Rendered(out.text, RenderStatus.TEMPLATE)

        # ---- 先声：玩家自己这一步的结果。lead_after=0 立即交付并告诉模型开头已写好（模型从下一句接着写）；
        #      lead_after>0 只在模型迟迟不交付第一句时顶上（模型不知道它，照常铺陈这一步——悬念留给模型快的回合） ----
        gate = _Gate(plan, brief, known, command, out)
        lead = lead_line(percepts, names, viewer, familiar) if self.lead_after is not None else ""
        if lead and self.lead_after is not None and self.lead_after <= 0:
            gate.admit(lead, lambda x: restates(x, percepts, names, viewer))
        # ---- 模型：逐句生成、逐句过闸门，通过即交付（先声讲过的玩家自己的行动不再列给模型，免得它照着再讲一遍） ----
        told = {r.text for r in rows if r.mine} if gate.lead else set()
        facts = [x for x in plan.lines if x not in covered and x not in told]
        prompt = self._prompt(brief, command, lapse, facts, looks, plan, known, gate.lead)
        failed = False
        pieces = self._pieces(prompt, self._system())
        if lead and not gate.lead:
            def late() -> None:
                if not out.text:
                    log.info("模型 %.1f 秒没交付一句，先声顶上", self.lead_after)
                    gate.admit(lead, lambda _: False, sep="\n")
            pieces = _deadline(pieces, self.lead_after or 0.0, late, self._pool())
        try:
            for piece in pieces:
                if not gate.feed(piece) or len(out.text) >= MAX_CHARS:
                    break                                   # 丢满了，或已经写够长：不再读流（没写完的半句不交付）
            else:
                gate.finish()
        except LLMUnavailable as e:
            log.warning("叙述生成失败，已交付 %d 字，补上事实清单: %s", len(out.text), e)
            failed = True
            gate.finish(tail=False)
        finally:
            pieces.close()

        # ---- 收尾：传闻归属整段查；漏掉的必讲之事补上模板行；没通过的补模板（已讲到的台词与时辰不重复） ----
        violations = list(gate.violations)
        if failed:
            status = RenderStatus.LLM_UNAVAILABLE
        elif not out.text[len(gate.lead):].strip():            # 模型一句也没交付（先声不算）
            status = RenderStatus.GATED_FALLBACK
            violations = violations or [Violation("empty", "")]
        else:
            hearsay = [v for v in check(out.text, plan, known) if v.kind == "hearsay"]
            violations += hearsay
            status = RenderStatus.GATED_FALLBACK if gate.dropped >= MAX_DROPS or hearsay else RenderStatus.LLM
        when = [f"（不觉已是{_in_words(lapse)}）"] if lapse else []
        body = out.text[len(gate.lead):]              # 模型交付的正文：先声里点过的名字不替别的事作证
        rows = [r for r in rows if not (gate.lead and r.mine)]      # 玩家自己的行动由先声讲过
        if status == RenderStatus.LLM:
            missing = _missing(rows, body, plan, brief, gate.dropped)
            if missing:
                out.emit("\n" + "\n".join(_prose(r.text) for r in missing))
                violations.append(Violation("omitted", "；".join(r.text for r in missing)))
                status = RenderStatus.GATED_FALLBACK
                log.info("叙述漏掉了必讲之事，补上模板行: %s", [r.text for r in missing])
        elif out.text:                                # 已交付过正文：只补正文没讲到的（一句正文都没有就整张清单）
            untold = _missing(rows, body, plan, brief, gate.dropped, strict=True) if body.strip() else rows
            tail = ([] if _TIMED.search(out.text) else when) + [r.text for r in untold]
            tail = tail if rows or gate.lead else [f"（{x}）" for x in looks]
            if tail:
                out.emit("\n" + "\n".join(_prose(x) for x in tail))
        elif scene or looks:
            out.emit("\n".join(_prose(x) for x in [*when, *scene, *(f"（{x}）" for x in looks)]))
        else:
            out.emit("\n".join(["时间悄悄过去，什么也没有发生。", *passed]))     # 安静的回合、模型又没交出话来
        if status == RenderStatus.GATED_FALLBACK:
            log.info("叙述未通过语义闸门，补上事实清单: %s", violations)
        return Rendered(out.text, status, tuple(dict.fromkeys(violations)), gate.dropped)

    # ------------------------------------------------------------
    #  提示词
    # ------------------------------------------------------------

    def _pool(self) -> ThreadPoolExecutor:
        if self._pump is None:
            self._pump = ThreadPoolExecutor(max_workers=2, thread_name_prefix="narrate")
        return self._pump

    def _pieces(self, prompt: str, system: str) -> Iterator[str]:
        """有流式接口就边生成边交付，没有就一次拿全文（同样逐句过闸门）。"""
        stream = getattr(self.llm, "stream", None)
        if callable(stream):
            yield from stream(prompt, system=system)
        else:
            yield self.llm.generate(prompt, system=system, temperature=0.7)

    def _system(self) -> str:
        return _GM + (f"\n世界：{self.setting}" if self.setting else "") + (f"\n文风：{self.style}" if self.style else "")

    def _prompt(self, brief: SceneBrief, command: str, lapse: str, facts: Sequence[str], looks: Sequence[str],
                plan: RenderPlan, known: frozenset[str], lead: str = "") -> str:
        parts: list[str] = []
        recent = [_in_words(p).strip() for p in brief.recent[-RECENT_KEEP:] if p.strip()]
        if recent:
            recent = [p if len(p) <= RECENT_CHARS else "……" + p[-RECENT_CHARS:] for p in recent]
            rows = [f"【{i}】{p}" for i, p in enumerate(recent, 1)]
            parts.append("最近几段正文（旧→新，只作接续，不要重复其句式）：\n" + "\n".join(rows))
        if command:
            parts.append(f"玩家的输入：{command}")
        if brief.player_line:
            parts.append(f"玩家本回合说出口的话或做出的姿态（可原样照引，一字不改）：{brief.player_line}")
        when = [f"（不觉已是{_in_words(lapse)}）"] if lapse else []
        empty = "（除下列言语外无事发生）" if brief.lines else "（无事发生：写眼前的光景与身边的人此刻的样子，不要编出新的事）"
        parts.append("本回合玩家感知到的事实：\n" + "\n".join(when + (list(facts) or [empty])))
        if brief.present:
            around = f"{brief.present[0]}" + (f"，身边有{'、'.join(brief.present[1:])}" if len(brief.present) > 1 else "，身边没有别人")
            parts.append(f"玩家以为自己此刻在：{around}")
        if brief.condition:
            parts.append(f"玩家自己的身体状况（不要写得像没事人一样，也不要加重）：{brief.condition}")
        if brief.notes:
            parts.append("本回合出乎玩家意料之处（写出他的觉察与惊讶，不要替他下结论，也不要替别人解释原因）：\n"
                         + "\n".join(f"- {n}" for n in brief.notes))
        if brief.lines:
            # 可点名只列真能通过闸门的：本回合计划里的名字、清单里原样有的名字、闸门不认识的名字
            universe = known | plan.hidden
            speakable = plan.names | plan.aliases

            def fits(n: str) -> bool:
                return n in speakable or n in plan.source or n not in universe or n in lines_text
            lines_text = "\n".join(x for vl in brief.lines for x in (vl.knows, vl.lately) if x)
            parts.append("要说出口的话（按先后，逐句写成对白）：\n"
                         + "\n".join(_describe(i, vl, fits) for i, vl in enumerate(brief.lines, 1)))
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
    ask = vl.op == Op.ASK.value
    act = (SOCIAL_LABELS.get(vl.social) if vl.social else None) or ("问话" if ask else "说话")
    body = (f"说法“{vl.claim}”——须如实转达，不多不少" if vl.claim
            else "闲话——只可说谈资里的掌故、他自己近来的经历与眼前人人看得见的事，不得夹带别的事实")
    if vl.template:
        body += f"；原话“{vl.template}”——可换措辞，意思不变"
    rows = [f"{i}. {vl.speaker_name}对{vl.listener_name or '众人'}（{act}）：{body}"]
    if vl.voice:
        rows.append(f"   腔调：{vl.voice}")
    if vl.knows:
        rows.append(f"   谈资：{vl.knows}")
    if vl.lately:
        rows.append(f"   他自己近来的经历（可以借这句话说起，不必都说）：{vl.lately}")
    if vl.answering:
        rows.append(f"   回应的是：“{vl.answering}”")
    else:
        rows.append("   （不是在回应玩家的某句话：不要写成是在接玩家的话头）")
    names = sorted(n for n in vl.may_name if n != vl.speaker_name and fits(n))
    rows.append("   台词里可点名：" + ("、".join(names) if names else "（除说话对象与玩家外，谁也不提）"))
    return "\n".join(rows)
