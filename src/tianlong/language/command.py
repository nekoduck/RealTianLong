"""
[INPUT]: 依赖 core 的 Op / Kind / Social
[OUTPUT]: 对外提供 ACTION_WORDS（操作关键词表）、SOCIAL_WORDS（言语行为词 → Social，归入 TELL）、
          GESTURE_WORDS（看得见的姿态词 → Social，归入 WAIT）、SpeechMode、Span、Mention、ParsedCommand、
          action_hits()（去掉引语与被长词覆盖的短词之后的全部行动词命中）、analyze()、clarify()
[POS]: language 的语态分析层：在“选哪个操作”之前先回答“这句话是不是玩家此刻要做的一件事”。
       否定（我不攻击守卫）、条件（如果……才）、计划与斟酌（再决定是否）、转述（守卫刚刚攻击了我）、引语、复合指令、
       疑问都被识别为非即时语态——规则快路径只接受明确的单一、肯定、即时指令，其余交给受约束的语义解析或追问澄清。
       言语行为词与姿态词也是行动词：“打招呼”盖住“打”、“救命”盖住“救”、“坐下”盖住“下”，于是也受否定与条件约束。
       parser 在此之上做实体与角色绑定；本模块不引用任何实体表以外的知识
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from tianlong.core import Kind, Op, Social

# ============================================================
#  言语行为与姿态：没有命题的言语、看得见的姿态，各带一个社交含义（修辞层，不是事实）
#  长词在前无关紧要——命中时被更长命中覆盖的短词一律丢弃
# ============================================================

_S = Social
SOCIAL_WORDS: tuple[tuple[str, Social], ...] = (
    ("打招呼", _S.GREET), ("打个招呼", _S.GREET), ("招呼", _S.GREET), ("问好", _S.GREET), ("问候", _S.GREET),
    ("见礼", _S.GREET), ("行礼", _S.GREET), ("寒暄", _S.GREET), ("搭话", _S.GREET), ("搭讪", _S.GREET),
    ("攀谈", _S.GREET), ("说话", _S.GREET), ("聊聊", _S.GREET), ("聊天", _S.GREET), ("闲聊", _S.GREET),
    ("聊聊天", _S.GREET), ("聊会儿天", _S.GREET), ("聊几句", _S.GREET), ("说说话", _S.GREET), ("说会儿话", _S.GREET),
    ("道谢", _S.THANK), ("谢谢", _S.THANK), ("多谢", _S.THANK), ("致谢", _S.THANK), ("感谢", _S.THANK),
    ("赔罪", _S.APOLOGIZE), ("赔礼", _S.APOLOGIZE), ("赔不是", _S.APOLOGIZE), ("道歉", _S.APOLOGIZE),
    ("认错", _S.APOLOGIZE), ("对不起", _S.APOLOGIZE),
    ("求饶", _S.PLEAD), ("讨饶", _S.PLEAD), ("求情", _S.PLEAD), ("求救", _S.PLEAD), ("呼救", _S.PLEAD),
    ("救命", _S.PLEAD),
    ("称赞", _S.PRAISE), ("夸奖", _S.PRAISE), ("恭维", _S.PRAISE), ("奉承", _S.PRAISE), ("久仰", _S.PRAISE),
    ("威胁", _S.THREATEN), ("恐吓", _S.THREATEN), ("恫吓", _S.THREATEN),
    ("挑衅", _S.TAUNT), ("讥讽", _S.TAUNT), ("嘲笑", _S.TAUNT), ("挖苦", _S.TAUNT), ("取笑", _S.TAUNT),
    ("辱骂", _S.INSULT), ("大骂", _S.INSULT), ("痛骂", _S.INSULT), ("骂", _S.INSULT),
    ("拒绝", _S.REFUSE), ("回绝", _S.REFUSE),
    ("答应", _S.AGREE), ("附和", _S.AGREE),
    ("开玩笑", _S.JOKE), ("说笑", _S.JOKE), ("打趣", _S.JOKE),
    ("安慰", _S.COMFORT), ("劝解", _S.COMFORT), ("劝慰", _S.COMFORT),
    ("解释", _S.EXPLAIN), ("辩解", _S.EXPLAIN), ("分辩", _S.EXPLAIN), ("讲道理", _S.EXPLAIN),
    ("叫阵", _S.CHALLENGE), ("邀战", _S.CHALLENGE), ("挑战", _S.CHALLENGE),
    ("呵斥", _S.COMMAND), ("喝令", _S.COMMAND),
    ("告辞", _S.FAREWELL), ("告别", _S.FAREWELL), ("道别", _S.FAREWELL), ("辞行", _S.FAREWELL),
    ("服软", _S.SUBMIT), ("认输", _S.SUBMIT), ("示弱", _S.SUBMIT),
)
# 姿态：不带主语的动作短语即原话（带姿态的 WAIT），在场的人看得见；None = 没有特别的社交含义
GESTURE_WORDS: tuple[tuple[str, Social | None], ...] = (
    ("打量", None), ("坐下", None), ("喝茶", None), ("喝口茶", None), ("喝了口茶", None), ("伸懒腰", None),
    ("拱手", _S.GREET), ("抱拳", _S.GREET), ("作揖", _S.GREET), ("鞠躬", _S.GREET),
    ("点头", _S.AGREE), ("摇头", _S.REFUSE), ("叹气", _S.REMARK), ("叹了口气", _S.REMARK),
    ("冷笑", _S.TAUNT), ("哈哈大笑", _S.JOKE), ("跪下", _S.SUBMIT), ("下跪", _S.SUBMIT),
    ("发呆", None), ("发了会儿呆", None), ("发了一会儿呆", None), ("出神", None), ("愣神", None), ("沉思", None),
    ("闭目养神", None), ("伸了个懒腰", None),
)

# ============================================================
#  操作关键词：按优先级排列（言语类最先——“告诉守卫我去港口”里的“去”不是移动）
# ============================================================

ACTION_WORDS: tuple[tuple[Op, tuple[str, ...]], ...] = (
    (Op.ASK, ("问", "打听", "ask")),
    (Op.TELL, ("告诉", "说", "tell", *(w for w, _ in SOCIAL_WORDS))),
    (Op.UNLOCK, ("开锁", "解锁", "打开", "unlock")),
    (Op.LOCK, ("锁上", "上锁", "lock")),
    (Op.ATTACK, ("出手", "动手", "还手", "攻击", "偷袭", "一掌", "出招", "揍", "打", "attack")),
    (Op.STUDY, ("研读", "修习", "参详", "参悟", "练", "读", "学", "study")),
    (Op.USE, ("服下", "服用", "喂", "敷", "救", "用", "use")),
    (Op.GIVE, ("交给", "递给", "给", "give")),
    (Op.PUT, ("放", "藏", "put", "hide")),
    (Op.TAKE, ("拿", "取", "捡", "偷", "揣", "拾", "抓", "take", "grab")),
    (Op.INSPECT, ("查看", "检查", "搜", "看看", "环顾", "观察", "找找", "端详", "磕头", "叩首", "跪拜",
                  "张望", "望了望", "望一望", "看了看", "看一看", "瞧瞧", "瞧了瞧", "瞅瞅",
                  "inspect", "search", "look")),
    (Op.MOVE, ("去", "走", "前往", "进", "回", "到", "跳", "爬", "钻", "下", "离开", "走出", "溜出", "逃出", "退出",
               "出去", "go", "move")),
    (Op.WAIT, ("等", "休息", "歇", "wait", *(w for w, _ in GESTURE_WORDS))),
)
_SPEECH_OPS = frozenset({Op.TELL, Op.ASK})


class SpeechMode(StrEnum):
    IMMEDIATE = "immediate"        # 明确的单一、肯定、即时指令：唯一允许直接提交的形态
    NEGATED = "negated"            # 否定作用域覆盖了行动词：“我不攻击守卫”
    CONDITIONAL = "conditional"    # 条件、计划、斟酌：“如果……才”“再决定是否”
    NARRATIVE = "narrative"        # 转述或叙述：主体不是玩家，或讲的是已经发生的事
    QUOTED = "quoted"              # 行动词只出现在引语里
    COMPOUND = "compound"          # 多个行动分句：一次只做一件事
    QUESTION = "question"          # 询问可行性而非下令：“能拿钥匙吗”
    UNCLEAR = "unclear"            # 双重否定等规则无法可靠判定的形态


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Mention:
    """文本里对一个已知实体的提及（位置、长度、实体、种类）。"""

    pos: int
    length: int
    eid: str
    kind: Kind


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """一句输入的语态结构：同一句话的“了解、描述、计划、实际执行”在这里被分开。"""

    text: str
    mode: SpeechMode
    subject: str | None                 # 行动主体实体 ID；None = 未明说（祈使句默认玩家）
    negated: tuple[Op, ...] = ()        # 落在否定作用域里的操作
    condition: Span | None = None       # 条件/计划/斟酌标记
    clauses: tuple[Span, ...] = ()      # 含行动词的分句
    action: Span | None = None          # 主行动词
    confidence: float = 1.0

    @property
    def immediate(self) -> bool:
        return self.mode == SpeechMode.IMMEDIATE


# ============================================================
#  词表：否定、条件、时间、顺承、疑问
# ============================================================

_NEGATORS = ("没有", "不要", "不用", "不必", "不再", "禁止", "不", "别", "没", "未", "勿", "莫", "甭", "休")
# 否定词与行动词之间允许夹着的情态/副词（“我绝不会出手”“千万别打他”“不许动手”）
_FILLERS = ("打算", "准备", "愿意", "应该", "可以", "千万", "一定", "想", "要", "会", "能", "敢", "愿", "必", "该",
            "肯", "得", "再", "先", "就", "也", "都", "还", "绝", "决", "并", "可", "许", "准", "地", "敢于", "忍心")
_CONDITIONS = ("如果", "假如", "要是", "若是", "倘若", "假使", "万一", "的话", "除非", "只要", "一旦", "是否",
               "要不要", "决定", "考虑", "打算", "计划", "看情况", "视情况")
_PAST = ("刚才", "刚刚", "方才", "已经", "曾经", "之前", "昨天", "昨晚", "先前", "早先")
_SEQUENCERS = ("然后", "接着", "之后", "随后", "并且", "而且", "同时", "顺便", "顺手", "再")
_CLAUSE_BREAK = re.compile(r"[，,。．；;！!？?\n]")
_QUOTES = (("“", "”"), ("「", "」"), ("『", "』"), ('"', '"'), ("‘", "’"))
_PREPOSITIONS = ("向", "对", "朝", "跟", "和", "与", "给", "把", "将", "找", "同", "往", "冲", "替", "帮", "从", "在", "问",
                 "被", "让", "叫", "请")
_POSSESSIVE = ("的", "身上", "手里", "手中", "那里", "那儿", "这里", "处")
_MODAL_Q = ("能不能", "可不可以", "行不行", "该不该", "是不是", "能否", "可否", "可以", "能")
_Q_END = ("吗", "么", "？", "?")


# ============================================================
#  分析
# ============================================================


def _quoted(t: str) -> list[tuple[int, int]]:
    spans = []
    for left, right in _QUOTES:
        start = t.find(left)
        while start != -1:
            end = t.find(right, start + 1)
            if end == -1:
                break
            spans.append((start, end + 1))
            start = t.find(left, end + 1)
    return spans


def action_hits(text: str) -> list[tuple[int, int, Op]]:
    """全部行动词命中（位置、终点、操作），按位置排序：引语里的不算，被更长命中覆盖的短词丢弃。"""
    t = text.strip().lower()
    return _hits(t, {i for a, b in _quoted(t) for i in range(a, b)})


def _hits(t: str, masked: set[int]) -> list[tuple[int, int, Op]]:
    """全部行动词命中（位置、终点、操作）；被更长命中覆盖的短词丢弃（“打开”里的“打”不算动手）。"""
    raw = [(i, i + len(w), op) for op, words in ACTION_WORDS for w in words
           for i in _find_all(t, w)]
    raw.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, Op]] = []
    for h in raw:
        if any(k[0] <= h[0] and h[1] <= k[1] and k != h for k in kept):
            continue
        kept.append(h)
    return [h for h in kept if h[0] not in masked]


def _find_all(t: str, w: str) -> list[int]:
    out, i = [], t.find(w)
    while i != -1:
        out.append(i)
        i = t.find(w, i + 1)
    return out


_PRONOUNS = ("他们", "她们", "他", "她", "它", "人家")


def _negations(t: str, start: int, mentions: Sequence[Mention]) -> int:
    """从行动词向前回溯：只跨过情态/副词与介词短语（“不会对龚光杰动手”），数否定词的个数（“不得不去”为二，负负得正）。"""
    ends = {m.pos + m.length: m.pos for m in mentions}
    i, count = start, 0
    while i > 0:
        j = ends.get(i)
        if j is None:
            j = next((i - len(w) for w in _PRONOUNS if i - len(w) >= 0 and t.startswith(w, i - len(w))), None)
        prep = None if j is None else next(
            (w for w in _PREPOSITIONS if j - len(w) >= 0 and t.startswith(w, j - len(w))), None)
        if j is not None and prep is not None:
            i = j - len(prep)
            continue
        tok = next((w for w in (*_NEGATORS, *_FILLERS) if i - len(w) >= 0 and t.startswith(w, i - len(w))), None)
        if tok is None:
            break
        if tok in _NEGATORS:
            count += 1
        i -= len(tok)
    return count


def _content_ranges(t: str, hits: list[tuple[int, int, Op]]) -> list[tuple[int, int]]:
    """言语动词之后的内容（直到顺承词）是“说的话”，其中的行动词不是玩家的行动。"""
    out = []
    for _, e, op in hits:
        if op not in _SPEECH_OPS:
            continue
        stop = min((i for w in _SEQUENCERS for i in _find_all(t, w) if i >= e), default=len(t))
        out.append((e, stop))
    return out


def _clause_of(t: str, pos: int) -> tuple[int, int]:
    breaks = [m.start() for m in _CLAUSE_BREAK.finditer(t)]
    breaks += [i for w in _SEQUENCERS for i in _find_all(t, w)]
    left = max((b for b in breaks if b < pos), default=-1) + 1
    right = min((b for b in breaks if b > pos), default=len(t))
    return left, right


def _subject(t: str, action_pos: int, clause_start: int, mentions: Sequence[Mention], owner: str) -> str | None:
    """行动词之前、同一分句里、不带介词也不作定语的人物提及就是主语。"""
    subj = None
    for m in mentions:
        if not (clause_start <= m.pos < action_pos) or m.kind != Kind.PERSON:
            continue
        before = t[max(0, m.pos - 2):m.pos]
        after = t[m.pos + m.length:m.pos + m.length + 2]
        if any(before.endswith(p) for p in _PREPOSITIONS) or any(after.startswith(p) for p in _POSSESSIVE):
            continue
        subj = m.eid
    return subj


def analyze(text: str, mentions: Sequence[Mention], owner: str) -> ParsedCommand:
    """判定一句输入的语态。mentions 由 parser 按玩家认知给出（“我/自己”指玩家本人）。"""
    t = text.strip().lower()
    quotes = _quoted(t)
    masked = {i for a, b in quotes for i in range(a, b)}
    hits = _hits(t, masked)
    content = _content_ranges(t, hits)

    def in_content(pos: int) -> bool:
        return any(a <= pos < b for a, b in content)

    acts = [h for h in hits if not in_content(h[0])]
    if not acts:
        mode = SpeechMode.QUOTED if quotes and _hits(t, set()) else SpeechMode.UNCLEAR
        return ParsedCommand(text, mode, None, confidence=0.0)

    first = acts[0]
    c0, c1 = _clause_of(t, first[0])
    subject = _subject(t, first[0], c0, mentions, owner)
    clauses = sorted({_clause_of(t, h[0]) for h in acts})
    spans = tuple(Span(a, b, t[a:b]) for a, b in clauses)
    action = Span(first[0], first[1], t[first[0]:first[1]])
    neg_counts = {h: _negations(t, h[0], mentions) for h in acts}
    negated = tuple(dict.fromkeys(h[2] for h, n in neg_counts.items() if n % 2 == 1))
    condition = next((Span(i, i + len(w), w) for w in _CONDITIONS for i in _find_all(t, w) if i not in masked), None)
    if condition is None:
        condition = _wait_then(t, acts)

    def cmd(mode: SpeechMode, confidence: float) -> ParsedCommand:
        return ParsedCommand(text, mode, subject, negated, condition, spans, action, confidence)

    if condition is not None:
        return cmd(SpeechMode.CONDITIONAL, 0.9)       # 条件从句里常有别人作主语：“如果守卫攻击我，我才还手”
    if subject is not None and subject != owner:
        return cmd(SpeechMode.NARRATIVE, 0.9)
    if any(w in t for w in _PAST):
        return cmd(SpeechMode.NARRATIVE, 0.7)
    if len(clauses) > 1:
        return cmd(SpeechMode.COMPOUND, 0.8)
    if negated:
        return cmd(SpeechMode.NEGATED, 0.9)
    if any(n and n % 2 == 0 for n in neg_counts.values()):
        return cmd(SpeechMode.UNCLEAR, 0.4)
    if first[2] not in _SPEECH_OPS and any(w in t for w in _MODAL_Q) and t.endswith(_Q_END):
        return cmd(SpeechMode.QUESTION, 0.7)
    return cmd(SpeechMode.IMMEDIATE, 1.0)


def _wait_then(t: str, acts: list[tuple[int, int, Op]]) -> Span | None:
    """“等守卫走了就拿钥匙”：等 + 就/才/再 + 后续行动 = 条件（姿态词虽归 WAIT，却不是“等”）。"""
    for s, _, op in acts:
        if op != Op.WAIT or not t.startswith("等", s):
            continue
        for w in ("就", "才", "再"):
            j = t.find(w, s + 1)
            if j != -1 and any(h[0] > j for h in acts):
                return Span(s, j + 1, t[s:j + 1])
    return None


# ============================================================
#  追问：非即时语态不推进时间，只告诉玩家为什么没有照做
# ============================================================

_CLARIFY: dict[SpeechMode, str] = {
    SpeechMode.NEGATED: "明白，你不打算那样做。那你此刻想做什么？",
    SpeechMode.CONDITIONAL: "条件还没成立，先不替你行动。等时机到了，直接说你要做什么。",
    SpeechMode.NARRATIVE: "这听起来像在叙述别人的举动或已经发生的事，而不是你此刻要做的。你想做什么？",
    SpeechMode.QUOTED: "引号里的话要对谁说？试试：告诉某人……",
    SpeechMode.COMPOUND: "一次只做一件事。你想先做哪一件？",
    SpeechMode.QUESTION: "试不试由你决定——想做就直接说。",
    SpeechMode.UNCLEAR: "没太听明白，能换个说法吗？",
}


def clarify(cmd: ParsedCommand) -> str:
    return _CLARIFY.get(cmd.mode, "请说得具体一些。")
