"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / routes_between，core 的 Op / Manner / Kind / Rel / Fact / Proposition / Social /
         signature_error，language/command 的 ACTION_WORDS / SOCIAL_WORDS / GESTURE_WORDS / action_hits / analyze / clarify /
         ParsedCommand / Mention，language/pose 的 pose_of / witness，language/llm 的 LLMClient / LLMUnavailable / parse_json
[OUTPUT]: 对外提供 MoveKind、Parsed（含语态结构、等待时长、这句话的类别、多步行动与问主持人的原话）、
          IntentParser（语态闸门 → 规则解析 → 受约束的 LLM 语义解析）、rule_parse()、normalize()，
          以及解释器复用的规则机件 mentions() / held_items() / exits() / leave_here() / leaving() / invalid() / manner_of() /
          speech_manner() / wait_length() / speech_line() / unwrap_line() / unsaid() / pose_of() / wield_problem()
[POS]: language 的输入解析；把玩家自由文本变成结构化候选行动。先由 command.analyze() 判定语态：只有单一、肯定、即时的指令
       才走规则解析；否定、条件、转述、复合、疑问交给 LLM（它也必须声明语态与主体），仍不确定就追问、不推进时间。
       LLM 失败或回复不成形时绝不回退到未经语义确认的候选。可引用的实体只来自玩家自己的认知图；解析结果仍要回到 kernel 结算。
       规则层修掉的误判：“打招呼/打量”不是动手，“大喊救命/用易经挡住脸”不是施用，“走出大殿”不是走进大殿，
       点名了不认识的秘籍就不悄悄改读手里那本，原话只引说出口的那句（没有可解析的命题就是闲话，不再反问“告诉谁什么”），
       姿态过 pose.witness()（夹带的拿取/研读落空就照实说落空，不让姿态把它吞掉），言语在原话之外“拔出长剑”同样要真在手里，
       “拿出勇气/亮出身份”不是掏东西；磕头只在此地有神像、蒲团一类可拜的陈设时才是伏地细看，其余是当众服软的姿态；
       normalize()/exits()/leave_here() 可指定出发地（多步计划从上一步的终点算起）
       先叫人再说话（句首人名紧跟逗号或冒号、此人在眼前）整句是说给他的原话，在语态分析之前认出。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import routes_between
from tianlong.core import Fact, Kind, Manner, Op, Proposition, Rel, Social
from tianlong.core.grammar import signature_error
from tianlong.language.command import (
    ACTION_WORDS,
    GESTURE_WORDS,
    SOCIAL_WORDS,
    Mention,
    ParsedCommand,
    SpeechMode,
    action_hits,
    analyze,
    clarify,
)
from tianlong.language.llm import LLMClient, LLMUnavailable, parse_json
from tianlong.language.pose import pose_of, witness

log = logging.getLogger(__name__)

Aliases = Mapping[str, Sequence[str]]


class MoveKind(StrEnum):
    """玩家这句话是哪一类：决定会话怎样推进时间。"""

    ACT = "act"            # 内核行动（1~3 步，后续步骤在 followups）
    SAY = "say"            # 对人说话：TELL/ASK，命题可选，原话在 utterance、言语行为在 candidate.social
    GESTURE = "gesture"    # 看得见的姿态：带姿态的 WAIT（utterance 是不带主语的动作短语）
    ASK_GM = "ask_gm"      # 场外问主持人（“我该做什么”“我身上有什么”）：不推进时间，只用玩家自己的认知作答
    META = "meta"          # 元指令（提示、回顾）：不推进时间
    UNCLEAR = "unclear"    # 听不懂：追问（主持层尽量少用）


@dataclass(frozen=True, slots=True)
class Parsed:
    candidate: Candidate | None
    utterance: str | None = None
    clarification: str | None = None   # 解析不了时给玩家的追问
    source: str = "rules"
    repeat: int = 1                    # 等待的分钟数（“等一炷香”= 30）
    until: str | None = None           # 等到某个时刻（"night"）：会话层按时钟换算
    command: ParsedCommand | None = None   # 语态结构：否定/条件/转述等非即时语态不会产生候选
    kind: MoveKind = MoveKind.ACT
    followups: tuple[Candidate, ...] = ()  # 多步行动的后续步骤（第一步是 candidate），逐 tick 执行、失败即止
    question: str | None = None        # ASK_GM：玩家问主持人的话；META：元指令名（hint / recap / beliefs）


# ============================================================
#  词表：时长、方式、追问与判别用的小词
# ============================================================

# 等待时长（分钟）；“等到天黑”交给会话层按时钟换算
_DURATIONS: tuple[tuple[str, int], ...] = (
    ("一个时辰", 120), ("半个时辰", 60), ("一炷香", 30), ("一盏茶", 15), ("一会", 10), ("片刻", 5),
)
_UNTIL_NIGHT = ("天黑", "入夜", "晚上", "夜里", "月亮")
_CAREFUL = ("悄悄", "小心", "轻轻", "偷偷", "藏", "低声", "小声", "耳语", "附耳")
_ROUGH = ("用力", "粗暴", "猛", "狠狠")
_WHISPER = ("悄悄", "偷偷", "低声", "小声", "轻声", "耳语", "附耳")   # 言语的“小心”只有耳语一种：“说小心点”不是耳语
_MANNER_ONLY = ("偷偷",)             # 只表方式的叠词：里面的“偷”不是拿
_ASKS = {
    Op.TAKE: "你想拿什么？", Op.PUT: "你想把什么放到哪里？", Op.GIVE: "你想把什么交给谁？",
    Op.UNLOCK: "你想用什么打开哪扇门？", Op.LOCK: "你想用什么锁上哪扇门？", Op.MOVE: "你想去哪里？你知道怎么走过去吗？",
    Op.TELL: "你想对谁说？", Op.ASK: "你想问谁？", Op.INSPECT: "你想查看什么？",
    Op.ATTACK: "你想对谁出手？", Op.STUDY: "你想研读什么？", Op.USE: "你想把什么用在谁身上？",
}
_HELP = "没听懂。直接说想做的事或想说的话就好，比如：环顾四周 / 问身边的人话 / 去某处 / 等一会儿"   # 不点具体地名，不剧透
_GENERIC = frozenset((*_ASKS.values(), _HELP))
_CLAIMING = frozenset({Op.TAKE, Op.PUT, Op.GIVE, Op.STUDY, Op.USE, Op.UNLOCK, Op.LOCK})   # 落空时说得出缺了什么的操作
_SOCIAL = dict(SOCIAL_WORDS)
_GESTURE = dict(GESTURE_WORDS)
_KOWTOW = ("磕头", "叩首", "跪拜")
_SHRINES = ("像", "蒲团", "拜垫", "垫", "龛", "牌位", "灵位", "香案", "供桌", "神台", "祭坛", "佛")   # 可以对着磕头的陈设
_SEARCH = ("搜", "检查", "找找", "search")     # 对人：搜身；其余查看词对人只是打量（姿态），不是搜身
_STRONG_USE = ("服下", "服用", "喂", "敷", "use")  # “用/救”单字定不了施用：“大喊救命”“用易经挡住脸”
_EXIT = ("离开", "走出", "溜出", "逃出", "退出", "出去")
_LEAVING = (*_EXIT, "冲出", "跑出", "奔出", "闯出", "逃离")   # 说了要离开此地（解释器据此才替人挑出路）
_GOTO = ("去", "前往", "到", "回")              # 明说要去某处：剩下的字才当作不认识的地名
_WHERE = ("在哪", "哪里", "哪儿", "何处", "下落", "什么地方", "去哪", "去了哪")
_AT_LINK = re.compile(r"^(?:就|正|还|也|已经|现在|已)?(不|没有|没)?(?:在|放在|藏在|留在|落在)$")
_WIELD = ("掏出", "拔出", "亮出", "拿出", "取出", "摸出", "抽出")   # 声称手里有：必须真在身上
# “拿出勇气”“亮出身份”是说法，不是从身上掏出一样东西
_INTANGIBLE = ("勇气", "胆量", "胆子", "身份", "本事", "本领", "看家本领", "真本事", "气势", "架势", "威风", "诚意", "诚心",
               "真心", "决心", "骨气", "耐心", "气概", "风度", "笑容", "笑脸", "全力", "浑身解数", "派头", "态度", "魄力", "精神")
_PUNCT = frozenset("，,。．.；;！!？?、：:\"'“”‘’「」『』（）() 　…~～")
# 数“还剩什么没被认出来”时忽略的虚字与泛称（“研读那本书”= 手里那本；“研读北冥神功”剩下的是一个不认识的名字）
_FILLERS = tuple(sorted((
    "一会儿", "一会", "一下", "一番", "片刻", "起来", "出来", "下来", "进来", "过来", "手里", "手中", "身上", "怀里", "怀中",
    "秘籍", "经书", "帛卷", "卷轴", "东西", "那本", "这本", "那个", "这个", "仔细", "认真", "好好",
    *_CAREFUL, *_ROUGH, "的", "了", "着", "过", "把", "将", "那", "这", "本", "卷", "个", "吧", "呢", "啊", "呀", "吗",
    "一", "书", "它", "再", "先", "来", "在", "上", "里", "中", "向", "对", "朝", "跟", "和", "地", "得", "要", "想",
), key=len, reverse=True))
# 引语前的包装：听者之后至多一个言语行为词（“赔罪”“打个招呼”），再是说话动词（“说：”“道：”）——剥掉之后才是说出口的话；
# 说话动词之后的“谢谢你”是原话，不再剥
_SOCIAL_LEADS = tuple(sorted(_SOCIAL, key=len, reverse=True))
_LEADS = tuple(sorted((
    "说道", "说", "讲", "道", "问道", "喊道", "喊", "叫道", "答道", "回道", "笑道", "低声", "小声", "轻声", "悄悄",
    "一声", "一句", "几句", "一下",
), key=len, reverse=True))
_QUOTES = (("“", "”"), ("「", "」"), ("『", "』"), ('"', '"'))
_NO_WORDS = frozenset({"问题", "一个问题", "几个问题", "些问题", "一件事", "件事", "事情", "几句话", "句话", "话"})


def _unclear(message: str, command: ParsedCommand | None = None, source: str = "rules") -> Parsed:
    return Parsed(None, clarification=message, source=source, command=command, kind=MoveKind.UNCLEAR)


# ============================================================
#  规则机件：实体提及、手里的东西、出路、方式、剩余文字（解释器的快路径同样使用）
# ============================================================


def _aliases(name: str, kind: Kind) -> tuple[str, ...]:
    out = [name]
    if kind == Kind.SURFACE and len(name) >= 2 and name[-1] in "面子":
        out.append(name[:-1])  # 桌面 → 桌
    return tuple(out)


def mentions(text: str, store: BeliefStore, extra: Aliases | None = None) -> list[Mention]:
    """按出现位置排序的实体提及；同一位置取最长名字。“我/自己”指玩家自己。别称只对玩家认识的实体生效。"""
    found: dict[int, Mention] = {}
    for eid, sk in store.entities.items():
        for alias in (*_aliases(sk.name, sk.kind), *((extra or {}).get(eid, ()))):
            start = text.find(alias)
            while start != -1:
                prev = found.get(start)
                if prev is None or len(alias) > prev.length:
                    found[start] = Mention(start, len(alias), eid, sk.kind)
                start = text.find(alias, start + 1)
    for word in ("自己", "我"):
        if word in text:
            pos = text.find(word)
            found.setdefault(pos, Mention(pos, len(word), store.owner, Kind.PERSON))
    return [found[k] for k in sorted(found)]


def held_items(store: BeliefStore) -> list[str]:
    """玩家认为自己身上带着的物件（按 ID 排序）。"""
    me = store.owner
    return [i for i, sk in sorted(store.entities.items()) if sk.kind == Kind.ITEM and store.location_of(i) == me]


def exits(store: BeliefStore, here: str | None = None) -> list[tuple[str, str]]:
    """玩家认为能从此地（或指定的出发地）走出去的路：(门, 门那头的地点)；确知单向走不通的不算。"""
    here = store.location_of(store.owner) if here is None else here
    out: list[tuple[str, str]] = []
    if here is None:
        return out
    for door, sk in sorted(store.entities.items()):
        if sk.kind != Kind.DOOR:
            continue
        ends = {str(b.prop.value) for b in store.positives(door, Rel.CONNECTS.value)}
        others = sorted(ends - {here})
        if here in ends and len(others) == 1 and door in routes_between(store, here, others[0]):
            out.append((door, others[0]))
    return out


def _curative(store: BeliefStore, item: str) -> bool:
    return any(b.prop.value not in (None, "none", "", False) for b in store.positives(item, "attr.cures"))


def invalid(c: Candidate, store: BeliefStore) -> str | None:
    """用玩家已知的实体种类检查行动语法（与 kernel 同一把尺子）。"""
    def kind_of(eid: str) -> Kind | None:
        sk = store.sketch(eid)
        return sk.kind if sk else None

    return signature_error(c.op, kind_of, c.target, c.obj, c.topic)


def manner_of(t: str) -> Manner:
    return Manner.CAREFUL if any(w in t for w in _CAREFUL) else (
        Manner.ROUGH if any(w in t for w in _ROUGH) else Manner.NORMAL)


def speech_manner(t: str) -> Manner:
    return Manner.CAREFUL if any(w in t for w in _WHISPER) else Manner.NORMAL


def leaving(t: str) -> bool:
    """这句话说了要离开此地（“溜出大殿”“冲出去”）：只有这时才替人挑一条出路。"""
    return any(w in t for w in _LEAVING)


def wait_length(t: str) -> tuple[int, str | None]:
    if any(w in t for w in _UNTIL_NIGHT):
        return 1, "night"
    # 只认紧挨“分”的十进制数字（\d 即 Unicode 十进制数字，int() 都认得）；“²”“①”不是，退回按说法估
    hit = re.search(r"(\d+)\s*分", t)
    if hit is not None:
        return max(1, int(hit.group(1))), None
    return next((m for word, m in _DURATIONS if word in t), 1), None


def _residue(t: str, ms: Sequence[Mention], hits: Sequence[tuple[int, int, Op]]) -> str:
    """去掉认出来的实体、行动词、虚字与标点之后还剩下的字：非空说明句子里点了一个不认识的名字。"""
    masked = {i for m in ms for i in range(m.pos, m.pos + m.length)} | {i for a, b, _ in hits for i in range(a, b)}
    rest = "".join(ch for i, ch in enumerate(t) if i not in masked and ch not in _PUNCT)
    for w in _FILLERS:
        rest = rest.replace(w, "")
    return rest


def _clean_name(t: str, residue: str) -> str | None:
    """剩下的字只有在原句里连成一片、且在句尾或标点前自然收住时，才当作一个名字说出口（“研读北冥神功”）；
    否则（“学着龚光杰的样子比划两下”）宁可说“这样东西”，也不说出半个词。"""
    i = t.find(residue)
    end = i + len(residue)
    body = t.rstrip("".join(_PUNCT))
    if not 2 <= len(residue) <= 6 or i == -1 or not (end >= len(body) or t[end] in _PUNCT):
        return None
    return residue


def _first_quote(t: str) -> str | None:
    for left, right in _QUOTES:
        a = t.find(left)
        b = t.find(right, a + 1) if a != -1 else -1
        if a != -1 and b != -1:
            return t[a + 1:b].strip()
    return None


def _strip_leads(s: str, social: bool = True) -> str:
    s = s.lstrip("".join(_PUNCT))
    act = next((w for w in _SOCIAL_LEADS if s.startswith(w)), None) if social else None
    s = s[len(act):] if act else s
    while True:
        s = s.lstrip("".join(_PUNCT))
        lead = next((w for w in _LEADS if s.startswith(w)), None)
        if lead is None:
            return s
        s = s[len(lead):]


def _as_line(s: str, ask: bool) -> str | None:
    s = s.strip().strip("".join(_PUNCT - set("？?！!。…")))
    if not s:
        return None
    if ask and not s.endswith(("？", "?", "吗", "呢", "么")):
        s += "？"
    return s


def speech_line(text: str, store: BeliefStore, listener: str | None, aliases: Aliases | None = None,
                ask: bool = False) -> str | None:
    """玩家真正说出口的那句：有引号取引号里的；否则取听者之后、剥掉“说/道/打个招呼”之类包装的文字。
    只有客套没有话（“跟钟灵打个招呼”）、听者只是定语（“拍拍钟灵的肩膀安慰她”）就是 None——不拿指令原文冒充原话。"""
    t = text.strip()
    quoted = _first_quote(t)
    if quoted is not None:
        return _as_line(quoted, ask)
    at = next((m for m in mentions(t.lower(), store, aliases) if m.eid == listener), None)
    if at is not None:
        rest = t[at.pos + at.length:]
        if rest.startswith("的"):
            return None
    else:
        rest = re.sub(r"^我?(?:对|跟|和|向|同|给)?(?:问|告诉)?(?:她们|他们|那人|对方|她|他)?", "", t)
    rest = _strip_leads(rest)
    return None if rest.rstrip("".join(_PUNCT)) in _NO_WORDS else _as_line(rest, ask)


def unwrap_line(line: str, names: Sequence[str] = (), ask: bool = False) -> str | None:
    """模型给的原话再剥一次包装：“对钟灵说：……”“问左子穆这是哪里”只留说出口的话；呼语（“钟灵，你好”）原样保留。"""
    s = line.strip()
    quoted = _first_quote(s)
    if quoted is not None:
        return _as_line(quoted, ask)
    m = re.match(r"^[^：:“”「」\"]{0,12}?(?:说道?|问道?|讲|道|喊道?)[：:]", s)
    if m is None and names:
        who = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        m = re.match(rf"^我?(?:(?:对|跟|和|向|同)(?:{who})(?:说道?|问道?|讲|道)?|(?:问|告诉)(?:{who}))[，,：:]?", s)
    if m is None:
        return _as_line(s, ask)
    rest = _strip_leads(s[m.end():], social=False)
    core = rest.rstrip("".join(_PUNCT))
    return None if core in _NO_WORDS or core in _SOCIAL else _as_line(rest, ask)


def unsaid(text: str, line: str | None = None) -> str:
    """原话之外、玩家自己动手的那截输入：第一个言语动词（问/告诉/说/赔罪……）或引语冒号、引号之前，再去掉原话本身。
    “我拔出长剑喝道：滚开”里声称拔剑的是这一截；“问龚光杰：你敢拔出长剑吗”“告诉钟灵快掏出你的貂”里拔剑、掏貂只是说出口的话。"""
    t = text.strip()
    quote = re.search(r"[：:“「『\"]", t)
    cuts = [quote.start()] if quote else []
    cuts += [a for a, _, op in action_hits(t) if op in (Op.TELL, Op.ASK)][:1]
    head = t[:min(cuts)] if cuts else t
    core = line.rstrip("".join(_PUNCT)) if line else ""
    return head.replace(core, "") if core else head


_NAME_STOP = "，,。！!？?来去了着向对朝往给在上里刺砍劈斩挥戳扎 "     # 兵刃之后紧跟的出招动词也是词界（“拔出倚天剑刺向他”）


def _unknown_name(rest: str) -> str | None:
    """“掏出”之后那个不认识的东西叫什么：None = 后面不是名字；"" = 是个名字但词界说不准（宁可不说，也不说出半个词）。
    只有在标点、虚字或句尾自然收住（不是被长度截断）才采用，“怀里的闪电貂”取“的”之后那段。"""
    m = re.match(rf"[^{_NAME_STOP}]{{1,8}}", rest)
    if m is None:
        return None
    end = m.end()
    if end < len(rest) and rest[end] not in _NAME_STOP:
        return ""                          # 被长度截断：“大理段氏的金牌，喝令……”
    name = m.group(0).split("的")[-1]
    return name if 2 <= len(name) <= 6 else ""


def wield_problem(text: str, store: BeliefStore, aliases: Aliases | None = None) -> str | None:
    """“掏出/拔出/亮出……”声称手里有某物：认识但不在身上、或根本不认识，就是编造前提——给出场内的否定。"""
    t = text.strip().lower()
    ms = {m.pos: m for m in mentions(t, store, aliases)}
    held = set(held_items(store))
    for w in _WIELD:
        i = t.find(w)
        while i != -1:
            k = t.rfind("从", max(0, i - 8), i)
            source = ms.get(k + 1) if k != -1 else None
            if source is not None and source.kind != Kind.ITEM and source.eid != store.owner:
                i = t.find(w, i + 1)      # “从兵器架上取出长剑”是拿，不是声称手里本来就有
                continue
            j = i + len(w)
            while j < len(t) and t[j] in "了一把本卷只柄支根":
                j += 1
            if t[j:].lstrip("点些儿").startswith(_INTANGIBLE):
                i = t.find(w, i + 1)      # “拿出勇气”“亮出身份”：是说法，不是掏出一样东西
                continue
            m = ms.get(j)
            if m is not None:
                sk = store.sketch(m.eid)
                if m.kind == Kind.ITEM and m.eid not in held and sk is not None:
                    return f"你身上并没有{sk.name}。"
            else:
                name = _unknown_name(t[j:])
                if name is not None:
                    return f"你身上并没有{name}。" if name else "你身上并没有这样东西。"
            i = t.find(w, i + 1)
    return None


# ============================================================
#  规则解析：语态闸门之后，关键词定操作，实体提及定角色
#  言语类最先判定（“告诉守卫我去港口”里的“去”不是移动）
# ============================================================


def rule_parse(text: str, store: BeliefStore, aliases: Aliases | None = None) -> Parsed:
    """先叫人再说话（“钟灵，你怎么也来了？”）是说给他听的话；句里还带着别的动作（“钟灵，把解药给我”）就先交常规解析，
    它解不出玩家自己的动作，才当作说给他听的话。其余交 _rule_parse。"""
    called, mixed = _vocative(text.strip(), mentions(text.strip().lower(), store, aliases), store)
    if called is not None and not mixed:
        return called
    parsed = _rule_parse(text, store, aliases)
    return called if called is not None and parsed.candidate is None else parsed


def _rule_parse(text: str, store: BeliefStore, aliases: Aliases | None = None) -> Parsed:
    """语态闸门：非即时语态不产生候选。即时指令按优先级尝试每个命中关键词的操作，返回第一个角色齐全的解析
    （“揣进兜里”的“进”不该赢过“揣”）；都不成时给出最具体的那句场内追问。"""
    t = text.strip().lower()
    ms = mentions(t, store, aliases)
    command = analyze(text, ms, store.owner)
    only = {i for w in _MANNER_ONLY for a in _find_all(t, w) for i in range(a, a + len(w))}
    hits = [h for h in action_hits(t) if h[0] not in only]
    ops = [op for op, _ in ACTION_WORDS if any(h[2] == op for h in hits)]
    if not ops:
        return _unclear(clarify(command) if command.mode == SpeechMode.QUOTED else _HELP, command)
    if not command.immediate:
        return _unclear(clarify(command), command)
    attempts = [_parse_as(op, text, t, store, ms, hits, aliases) for op in ops]
    chosen = next((p for p in attempts if p.candidate is not None), None)
    if chosen is not None and chosen.kind == MoveKind.GESTURE:
        # “我跪下捡起北冥神功帛卷”：拿取/研读落空就照实说落空，不让“跪下”把它吞成一个姿态
        failed = (p for o, p in zip(ops, attempts, strict=True)
                  if o in _CLAIMING and p.candidate is None and p.clarification not in _GENERIC)
        chosen = next(failed, chosen)
    if chosen is None:
        chosen = next((p for p in attempts if p.clarification not in _GENERIC), attempts[0])
    return replace(chosen, command=command)


_CALL = re.compile(r"^(?:喂|哎|嘿|诶|咳)?[，,\s]*")
_ASKING = ("？", "?", "吗", "呢", "么", "吧？")


def _vocative(text: str, ms: Sequence[Mention], store: BeliefStore) -> tuple[Parsed | None, bool]:
    """开口先叫人（“钟灵，你怎么也来了？”“喂，龚兄，看招！”）：句首的人名后紧跟逗号或冒号，后面整句都是说给他的原话。
    只认玩家以为就在眼前的人；叫到的人不在眼前、或逗号后什么也没说，返回 (None, False)。
    第二项：后面那句里还有说话之外的动作词（“把解药给我”），交给调用方先试常规解析。"""
    lead = _CALL.match(text)
    at = lead.end() if lead else 0
    who = next((m for m in ms if m.pos == at and m.kind == Kind.PERSON and m.eid != store.owner), None)
    if who is None:
        return None, False
    rest = text[at + who.length:]
    if not rest or rest[0] not in "，,：:！!":
        return None, False
    line = rest[1:].strip()
    here = store.location_of(store.owner)
    if not line or here is None or store.location_of(who.eid) != here:
        return None, False
    low = line.lower()
    hits = action_hits(low)
    social = next((_SOCIAL[low[a:b]] for a, b, _ in hits if low[a:b] in _SOCIAL), None)
    op = Op.ASK if line.endswith(_ASKING) else Op.TELL
    parsed = Parsed(Candidate(op, who.eid, social=social), line, source="rules", kind=MoveKind.SAY,
                    command=analyze(text, ms, store.owner))
    return parsed, bool({o for _, _, o in hits} - {Op.TELL, Op.ASK})


def _find_all(t: str, w: str) -> list[int]:
    out, i = [], t.find(w)
    while i != -1:
        out.append(i)
        i = t.find(w, i + 1)
    return out


def _first(ms: Sequence[Mention], kinds: tuple[Kind, ...], after: int = -1, exclude: tuple[str, ...] = ()) -> Mention | None:
    return next((m for m in ms if m.pos > after and m.kind in kinds and m.eid not in exclude), None)


def _gesture(text: str, social: Social | None, store: BeliefStore, aliases: Aliases | None,
             source: str = "rules") -> Parsed:
    pose = pose_of(text, (store.sketch(store.owner).name,) if store.sketch(store.owner) else ())
    problem = wield_problem(pose, store, aliases)
    if problem is not None:
        return _unclear(problem, source=source)
    kept, why = witness(pose, social, store, mentions(pose.lower(), store, aliases), aliases, said=text)
    if kept is None:
        return _unclear(why or _HELP, source=source)
    return Parsed(Candidate(Op.WAIT, social=social), kept, source=source, kind=MoveKind.GESTURE)


def _parse_as(op: Op, text: str, t: str, store: BeliefStore, ms: list[Mention],
              hits: list[tuple[int, int, Op]], aliases: Aliases | None) -> Parsed:
    manner = manner_of(t)
    me = store.owner
    here = store.location_of(me)
    words = tuple(t[a:b] for a, b, o in hits if o == op)

    def pick(*kinds: Kind, exclude: tuple[str, ...] = ()) -> str | None:
        hit = _first(ms, kinds, exclude=exclude)
        return hit.eid if hit else None

    held = held_items(store)
    residue = _residue(t, ms, hits)

    def in_hand() -> str | None:
        # 没点名物品时默认手里唯一的那件——前提是句子里再没有别的名字（“研读北冥神功”不该悄悄读成易经）
        return held[0] if len(held) == 1 and not residue else None

    def lacking(template: str) -> Parsed:
        if not residue:
            return _unclear(_ASKS[op])
        return _unclear(template.format(_clean_name(t, residue) or "这样东西"))

    target = obj = None
    social: Social | None = None
    if op == Op.TAKE:
        target = pick(Kind.ITEM)
        if target is None:
            return lacking("你并没见到{}。")
    elif op == Op.PUT:
        obj = pick(Kind.ITEM) or in_hand()
        target = pick(Kind.SURFACE, Kind.PLACE) or here
        if obj is None:
            return lacking("你身上并没有{}。")
    elif op == Op.GIVE:
        target, obj = pick(Kind.PERSON, exclude=(me,)), pick(Kind.ITEM)
    elif op in (Op.UNLOCK, Op.LOCK):
        target = pick(Kind.DOOR)
        obj = pick(Kind.ITEM) or in_hand()
    elif op == Op.INSPECT:
        person = pick(Kind.PERSON, exclude=(me,))
        if any(w in _KOWTOW for w in words):
            if person is not None or not _shrine_here(store, here, aliases):
                return _gesture(text, Social.SUBMIT, store, aliases)   # 向人磕头、当众磕头：服软的姿态，不是搜查
            target, social = here, Social.SUBMIT                       # 原著路线：在玉像、蒲团前磕头，伏在地上才看清蒲团
        elif any(w in _SEARCH for w in words):
            target = pick(Kind.PLACE, Kind.SURFACE, Kind.PERSON, exclude=(me,)) or here
        else:
            target = pick(Kind.PLACE, Kind.SURFACE)
            if target is None and person is not None:
                return _gesture(text, None, store, aliases)             # 看一个人是打量，不是搜身
            target = target or here
    elif op == Op.MOVE:
        return _parse_move(t, store, ms, words, manner, residue)
    elif op == Op.ATTACK:
        target = pick(Kind.PERSON, exclude=(me,))
    elif op == Op.STUDY:
        target = pick(Kind.ITEM) or in_hand()
        if target is None:
            return lacking("你身上并没有{}。")
    elif op == Op.USE:
        obj, target = pick(Kind.ITEM), pick(Kind.PERSON, exclude=(me,))
        strong = any(w in _STRONG_USE for w in words)
        if not strong and target is None and (obj is None or residue):
            return _unclear(_ASKS[op])     # 只有“用/救”一个字：说不清是把什么用在谁身上
        default = in_hand()
        if obj is None and not strong and default is not None and not _curative(store, default):
            default = None                 # “救钟灵”只在手里那件认得是药时才默认用它，不拿易经去救人
        obj, target = obj or default, target or me
    elif op in (Op.TELL, Op.ASK):
        return _parse_speech(op, text, t, store, ms, words, aliases)
    elif op == Op.WAIT:
        gesture = next((w for w in words if w in _GESTURE), None)
        if gesture == "打量" and pick(Kind.PERSON, exclude=(me,)) is None:
            return Parsed(Candidate(Op.INSPECT, here), source="rules") if here else _unclear(_ASKS[Op.INSPECT])
        if gesture is not None:
            return _gesture(text, _GESTURE[gesture], store, aliases)
    cand = normalize(Candidate(op, target, obj, manner, None, social), store)
    if invalid(cand, store):
        return _unclear(_ASKS.get(op, "请说得具体一些。"))
    if op == Op.WAIT:
        repeat, until = wait_length(t)
        return Parsed(cand, source="rules", repeat=repeat, until=until)
    return Parsed(cand, source="rules")


def _name(store: BeliefStore, eid: str | None) -> str:
    sk = store.sketch(eid or "")
    return sk.name if sk else "那里"


def _shrine_here(store: BeliefStore, here: str | None, aliases: Aliases | None) -> bool:
    """玩家以为此地有可以对着磕头的陈设（神像、蒲团、牌位……）：兵器架前磕头只是服软，不是伏地细看。"""
    return here is not None and any(
        sk.kind == Kind.SURFACE and store.location_of(eid) == here
        and any(w in n for n in (sk.name, *(aliases or {}).get(eid, ())) for w in _SHRINES)
        for eid, sk in store.entities.items())


def leave_here(store: BeliefStore, manner: Manner = Manner.NORMAL, here: str | None = None) -> Parsed:
    """离开此地（或指定的出发地）：只知道一条出路就走它；有几条就问走哪条（列出玩家知道的路）。"""
    ways = exits(store, here)
    if len(ways) == 1:
        door, dest = ways[0]
        return Parsed(Candidate(Op.MOVE, dest, door, manner), source="rules")
    if not ways:
        return _unclear("你不知道从哪里出去。")
    listed = "、".join(f"{_name(store, d)}（通往{_name(store, p)}）" for d, p in ways)
    return _unclear(f"你想从哪里出去？你知道的路有：{listed}。")


def _parse_move(t: str, store: BeliefStore, ms: list[Mention], words: tuple[str, ...], manner: Manner,
                residue: str) -> Parsed:
    here = store.location_of(store.owner)
    hit = _first(ms, (Kind.PLACE, Kind.DOOR))
    target = hit.eid if hit else None
    door = _first(ms, (Kind.DOOR,))
    obj = door.eid if door is not None and hit is not None and hit.kind == Kind.PLACE else None
    if any(w in _EXIT for w in words) and (target is None or target == here):
        return leave_here(store, manner)
    cand = normalize(Candidate(Op.MOVE, target, obj, manner), store)
    if not invalid(cand, store):
        return Parsed(cand, source="rules")
    if target is not None and target == here:
        return _unclear(f"你已经在{_name(store, here)}了。")
    if hit is not None and hit.kind == Kind.PLACE:
        return _unclear(f"你不知道该怎么去{_name(store, target)}。")
    if target is None and residue and any(w in _GOTO for w in words):
        return _unclear(f"你不知道该怎么去{residue}。")
    return _unclear(_ASKS[Op.MOVE])


def _parse_speech(op: Op, text: str, t: str, store: BeliefStore, ms: list[Mention], words: tuple[str, ...],
                  aliases: Aliases | None) -> Parsed:
    """言语：有听者就是 TELL/ASK（命题只在说清“某某在某处/在哪”时才有），原话只引说出口的那句；
    没有听者的言语行为（“大喊救命”）是当众的姿态。原话之外的“我拔出长剑”同样要真在手里。"""
    me = store.owner
    social = next((_SOCIAL[w] for w in words if w in _SOCIAL), None)
    listener = _first(ms, (Kind.PERSON,), exclude=(me,))
    if listener is None:
        if social is not None:
            return _gesture(text, social, store, aliases)
        return _unclear(_ASKS[op])
    cand = Candidate(op, listener.eid, None, speech_manner(t), _topic(op, t, ms, listener), social)
    if invalid(cand, store):
        return _unclear(_ASKS[op])
    line = speech_line(text, store, listener.eid, aliases, ask=op == Op.ASK)
    problem = wield_problem(unsaid(text, line), store, aliases)
    if problem is not None:
        return _unclear(problem)
    return Parsed(cand, line, source="rules", kind=MoveKind.SAY)


def _topic(op: Op, t: str, ms: list[Mention], listener: Mention) -> Fact | None:
    subject = _first(ms, (Kind.ITEM, Kind.PERSON), after=listener.pos, exclude=(listener.eid,))
    if subject is None:
        return None
    end = subject.pos + subject.length
    if op == Op.ASK:
        asks_where = any(w in t[end:end + 8] for w in _WHERE)
        return Fact(Proposition.rel(subject.eid, Rel.AT, None), True) if asks_where else None
    value = _first(ms, (Kind.PLACE, Kind.SURFACE, Kind.PERSON), after=subject.pos)
    link = _AT_LINK.match(t[end:value.pos].strip()) if value is not None else None
    if value is None or link is None:
        return None
    return Fact(Proposition.rel(subject.eid, Rel.AT, value.eid), link.group(1) is None)


def normalize(c: Candidate, store: BeliefStore, here: str | None = None) -> Candidate:
    """MOVE 绑定一条玩家自己知道的路：“朝那扇门走” = 经这扇门去门那边的地点；“去某地” = 经玩家认为连通的门
    （认为没锁的优先）；点名的门玩家并不认为通往那里，就换一条认为连通的。目的地就是脚下（“走进大殿”而人已在大殿）
    不算移动：经某扇门离开此地的，改成去门那头；否则目的地留空。玩家不知道怎么去，路线就留空——语法检查会追问，
    内核不会替他从真实地图里挑一条暗道。here 缺省为玩家以为自己所在之处；多步计划传入上一步的终点。"""
    if c.op != Op.MOVE:
        return c
    here = store.location_of(store.owner) if here is None else here
    sk = store.sketch(c.target or "")
    if sk is not None and sk.kind == Kind.DOOR:
        others = [b.prop.value for b in store.positives(sk.id, Rel.CONNECTS.value) if b.prop.value != here]
        if len(others) == 1:
            return Candidate(Op.MOVE, str(others[0]), sk.id, c.manner, None, c.social)
        return c
    if c.target is not None and c.target == here:
        door = store.sketch(c.obj or "")
        if door is not None and door.kind == Kind.DOOR:
            return normalize(Candidate(Op.MOVE, door.id, None, c.manner, None, c.social), store, here)
        return Candidate(Op.MOVE, None, None, c.manner, None, c.social)
    if c.target is not None and here is not None:
        routes = routes_between(store, here, c.target)
        ends = {b.prop.value for b in store.positives(c.obj, Rel.CONNECTS.value)} if c.obj else set()
        if c.obj is not None and {here, c.target} <= ends:
            return c
        if routes:
            return Candidate(Op.MOVE, c.target, routes[0], c.manner, None, c.social)
        return Candidate(Op.MOVE, c.target, None, c.manner, None, c.social)
    return c


# ============================================================
#  LLM 解析（旧接口，会话仍在用）：只给它玩家认识的实体表，输出受 JSON Schema 约束，再用同一把语法尺子校验
#  主持层的解释器见 language/interpret.py
# ============================================================

_SYSTEM = (
    "你是文字游戏的指令解析器。把玩家的中文输入解析成一个结构化行动。"
    "先判断语态 mode：immediate=玩家此刻要亲自做的一件事；negated=否定（不做）；conditional=带条件、计划或斟酌；"
    "narrative=叙述别人的举动或已经发生的事；quoted=行动只出现在引语里；compound=多件事；question=询问能否。"
    "actor=行动主体：player=玩家本人，other=别人，unknown=说不清。只有 immediate 且 actor=player 才会被执行。"
    "只能引用实体表里的 id；做不到或意图不明就把 op 设为 unknown 并给出 clarification。"
    "target=行动直接作用的对象（拿的物品、去的地点、开的门、说话的对象、查看的东西）；"
    "obj=工具或被递交/放置的物品（开锁的钥匙、放下或交出的东西）；move 的 obj 是走的那扇门（不确定就填 null）；"
    "不适用的字段填 null。"
    "manner: careful=小心/悄悄/藏，rough=粗暴/用力，否则 normal。"
    "tell/ask 的语义内容用 topic_subject/topic_value 表示“subject 在 value”（ask 时 topic_value 留空）。"
)


def _schema(ids: list[str]) -> dict:
    ref = {"type": "string", "enum": ids, "nullable": True}
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": [m.value for m in SpeechMode]},
            "actor": {"type": "string", "enum": ["player", "other", "unknown"]},
            "op": {"type": "string", "enum": [o.value for o in Op] + ["unknown"]},
            "target": ref, "obj": ref, "topic_subject": ref, "topic_value": ref,
            "topic_holds": {"type": "boolean"},
            "manner": {"type": "string", "enum": [m.value for m in Manner]},
            "clarification": {"type": "string"},
        },
        # 角色字段全部 required（允许 null）：否则模型会干脆省略 target
        "required": ["mode", "actor", "op", "target", "obj", "manner", "topic_subject", "topic_value", "topic_holds",
                     "clarification"],
    }


def _table(store: BeliefStore) -> str:
    rows = []
    for eid, sk in sorted(store.entities.items()):
        loc = store.location_of(eid)
        where = f"，你认为在 {loc}" if loc else ""
        rows.append(f"- {eid}：{sk.name}（{sk.kind.value}{where}）")
    return "\n".join(rows)


class IntentParser:
    def __init__(self, llm: LLMClient | None = None, prefer_llm: bool = False, aliases: Aliases | None = None) -> None:
        self.llm = llm
        self.prefer_llm = prefer_llm
        self.aliases = {k: tuple(v) for k, v in (aliases or {}).items()}

    def parse(self, text: str, store: BeliefStore) -> Parsed:
        ruled = rule_parse(text, store, self.aliases)
        command = ruled.command
        immediate = command is not None and command.immediate
        if self.llm is None or (ruled.candidate is not None and not self.prefer_llm):
            return ruled
        try:
            got = self._llm_parse(text, store, command)
        except LLMUnavailable as e:
            log.warning("LLM 解析不可用: %s", e)
            got = None
        if got is not None:
            return got
        # 回退只允许经语态闸门确认过的即时解析；否定/条件/转述绝不因模型失败而被执行
        return ruled if immediate else _unclear(ruled.clarification or _HELP, command)

    def _llm_parse(self, text: str, store: BeliefStore, command: ParsedCommand | None) -> Parsed | None:
        assert self.llm is not None
        ids = sorted(store.entities)
        prompt = f"你是 {store.owner}。你认识的实体：\n{_table(store)}\n\n玩家输入：{text}"
        data = parse_json(self.llm.generate(prompt, system=_SYSTEM, schema=_schema(ids), temperature=0.0))
        if not isinstance(data, dict):
            return None                         # 回复不成形：同模型失败，只回退经语态确认的规则解析
        mode = data.get("mode", SpeechMode.UNCLEAR.value)
        op = next((o for o in Op if o.value == data.get("op")), None)
        if op is None or mode != SpeechMode.IMMEDIATE.value or data.get("actor") != "player":
            fallback = clarify(command) if command is not None and not command.immediate else "请说得具体一些。"
            note = data.get("clarification")
            return _unclear(note.strip() if isinstance(note, str) and note.strip() else fallback, command, "llm")
        known = set(ids)

        def ref(key: str) -> str | None:
            v = data.get(key)
            return v if isinstance(v, str) and v in known else None

        topic = None
        subject = ref("topic_subject")
        if subject:
            topic = Fact(Proposition.rel(subject, Rel.AT, ref("topic_value")), bool(data.get("topic_holds", True)))
        manner = next((m for m in Manner if m.value == data.get("manner")), Manner.NORMAL)
        cand = normalize(Candidate(op, ref("target"), ref("obj"), manner, topic), store)
        if command is not None and cand.op in command.negated:
            # 规则层看见了对这个行动的否定：模型说“照做”也不行
            return _unclear(clarify(command), command, "llm")
        if invalid(cand, store):
            return None
        if cand.op in (Op.TELL, Op.ASK):
            line = speech_line(text, store, cand.target, self.aliases, ask=cand.op == Op.ASK)
            return Parsed(cand, line, source="llm", command=command, kind=MoveKind.SAY)
        return Parsed(cand, source="llm", command=command)
