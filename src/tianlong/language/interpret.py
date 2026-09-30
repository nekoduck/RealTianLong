"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / believed_place，core 的 Op / Kind / Manner / Social / Fact / Proposition / Rel /
         RELATIONS / OP_SIGNATURES / SKILLS，language/command 的 SpeechMode / ParsedCommand / action_hits / analyze / clarify，
         cognition/navigation 的 route_to，language/parser 的 MoveKind / Parsed / IntentParser / rule_parse / normalize 与规则机件（mentions / held_items / exits /
         invalid / leave_here / leaving / manner_of / speech_manner / wait_length / speech_line / unwrap_line / unsaid /
         wield_problem），language/pose 的 pose_of / witness / own_words / MIN_NAME，language/llm 的 LLMClient / LLMUnavailable /
         parse_json
[OUTPUT]: 对外提供 Interpreter（interpret(text, me, recent) → Parsed：GM 前缀与元指令 → 场外问题 → “跟上/跟着 + 认识的人”
          （走到玩家以为他在的地方，刚看见他从哪道门走的就走那道门）→ 高精度规则快路径（0 次模型调用）
          → 快模型一次 JSON（act / say / gesture / ask_gm / unclear，并声明语态与主体）→ 逐项校验；没有模型时退回 IntentParser，
          模型失败或回复不成形只退回规则解析、绝不再调一次模型；场景的 kowtow_ticks > 1 时，对着可拜的陈设叩首（解析成伏地细看）
          展开为“叩首 × (k-1) + 细看”的多 tick 计划，每步都是带字的姿态，旧版 kowtow_ticks=1 原样；prompt() 给评测与测试看模型收到的原文）、
          MAX_TABLE / RECENT_CHARS / MAX_TOKENS
[POS]: language 的主持层解释器（设计 §4.1）：听懂玩家的任何话，但模型的输出只是意图——引用的每个实体都必须是玩家认识的、
       种类要对、MOVE 按玩家的地图补全路线（多步计划从上一步的终点算起；目的地是脚下时，只有说了“离开/溜出”才替人挑出路）、
       声称在手里的东西必须真在（玩家以为的）身上（言语只查原话之外的那截）；多步计划在第一个不成立的步骤处截断，
       一步都不成立就给场内说法、不推进时间（编造前提的防线：变不出没有的秘籍与武功，从不回退到未经确认的候选）。
       玩家嘴里的话只能是玩家打出来的字：模型给的原话不是原文的一段就改从原文里取，没说具体的话就只有言语行为；
       命题的主语与处所都得是原文点了名的。姿态过 pose.witness()。回显给玩家的只有玩家自己写过的名字：模型的 missing
       不在原文里就用不点名的说法，模型写的追问（reply）点了玩家不认识的名字（场景别称或会话交来的名字全集 universe）就换成
       固定追问。否定与非即时语态照旧拦住：规则看见的否定，模型说“照做”也不行；明确的假设（如果/万一/等……就）
       与别人作主语的转述，模型说“即时”也不替玩家动手或摆姿态；问号结尾的整句命令不走快路径。
       模型只看玩家认识的实体（点名与在场的优先，至多 MAX_TABLE 个）与最近两段正文（消解“她/那人”），绝不看世界真相
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import believed_place, route_to
from tianlong.core import OP_SIGNATURES, RELATIONS, SKILLS, Fact, Kind, Manner, Op, Proposition, Rel, Social
from tianlong.language.command import (
    ACTION_WORDS,
    Mention,
    ParsedCommand,
    SpeechMode,
    action_hits,
    analyze,
    clarify,
)
from tianlong.language.llm import LLMClient, LLMUnavailable, parse_json
from tianlong.language.parser import (
    IntentParser,
    MoveKind,
    Parsed,
    exits,
    held_items,
    invalid,
    leave_here,
    leaving,
    manner_of,
    mentions,
    normalize,
    rule_parse,
    speech_line,
    speech_manner,
    unsaid,
    unwrap_line,
    wait_length,
    wield_problem,
)
from tianlong.language.pose import MIN_NAME, own_words, pose_of, witness

log = logging.getLogger(__name__)

MAX_TABLE = 40        # 提示词里至多列这么多实体（点名的、在场的优先）
RECENT_CHARS = 120    # 每段最近正文只留末尾这么多字：够消解“她/那人”，提示词不至于变长
MAX_TOKENS = 300      # 一次结构化解释的输出上限

_NO_ALIASES: Mapping[str, Sequence[str]] = MappingProxyType({})
_E = TypeVar("_E", bound=Enum)


def _unclear(message: str, command: ParsedCommand | None = None, source: str = "rules") -> Parsed:
    return Parsed(None, clarification=message, source=source, command=command, kind=MoveKind.UNCLEAR)


def _enum(cls: type[_E], value: Any) -> _E | None:
    try:
        return cls(value)
    except (ValueError, TypeError):
        return None


# ============================================================
#  前缀、元指令与场外问题：不需要模型，也不推进时间
# ============================================================

_GM = re.compile(r"^\s*(?:gm|ooc)\s*[:：]\s*", re.IGNORECASE)
_META: dict[str, tuple[str, ...]] = {
    "hint": ("hint", "提示"),
    "recap": ("recap", "回顾", "前情"),
    "beliefs": ("beliefs", "认知", "所知"),
}
_META_HELP = "可用的指令：/hint 提示、/recap 前情回顾、/beliefs 你所知道的。"
_OOC = tuple(re.compile(p) for p in (
    r"(?:那|那么)?(?:我|咱们?)?(?:现在|接下来|下一步|此刻)?(?:该|应该|应当|要|能|可以|还能|还可以)(?:做|干)(?:点|些)?(?:什么|啥)(?:好|才好)?",
    r"(?:我|咱们?)?(?:现在|此刻|这会儿?)?是?(?:在|身在|到了)(?:哪|哪里|哪儿|什么地方|何处)了?",
    r"我?(?:身上|手里|手上|怀里|包里)(?:都|还)?(?:有|带着|带了|拿着)(?:些)?(?:什么|啥)(?:东西)?",
    r"(?:那|那么|现在|这下|这可)?我?(?:该|可|要)?怎么办(?:才好)?",
    r"有?(?:什么|啥)?(?:提示|线索)吗?",
    r"(?:我的)?(?:目标|任务)是?(?:什么|啥)",
    r"我?(?:下一步|接下来)(?:该|要)?(?:去哪|去哪里|去哪儿|做什么|干什么)",
    r"(?:那|那么)?我?(?:现在|接下来|下一步)?(?:该|要|应该)?往哪(?:儿|里|边)?(?:走|去)(?:才好|呢)?",
))


def _meta(text: str) -> Parsed | None:
    if not text.startswith(("/", "／")):
        return None
    body = text[1:].strip()
    word = body.split(maxsplit=1)[0].lower() if body else ""
    key = next((k for k, names in _META.items() if word in names), None)
    if key is None:
        return _unclear(_META_HELP)
    return Parsed(None, kind=MoveKind.META, question=key)


def _ooc(text: str) -> bool:
    s = re.sub(r"[\s？?！!。，,…~～]+", "", text)
    s = re.sub(r"(?:呢|啊|呀|吧|嘛)+$", "", s)
    return any(p.fullmatch(s) for p in _OOC)


# ============================================================
#  规则快路径：只收整句命令，句中每个实义词都得有着落；单字关键词（打、下、用、说……）从不在此单独成立
#  先把认出来的实体换成占位符（P 地点 D 通道 S 陈设 I 物件 H 人 M 我），整句匹配白名单，再交给 rule_parse 定角色
# ============================================================

_SLOT = {Kind.PLACE: "P", Kind.DOOR: "D", Kind.SURFACE: "S", Kind.ITEM: "I", Kind.PERSON: "H"}
_KOWTOW = "磕头|叩首|跪拜|磕几个头|跪下磕头|跪地磕头|(?:向|对|朝)[SIP](?:磕头|叩首|跪拜|磕几个头|拜了几拜)"
_FAST: tuple[tuple[Op, re.Pattern[str]], ...] = tuple(
    (op, re.compile(rf"(?:M(?:要|就|这就|先)?)?(?:{body})"))
    for op, body in (
        (Op.INSPECT, "环顾四周|环顾|环视四周|看看四周|四下看看|四处看看|四下张望|看看周围|观察四周|查看四周|打量四周|四下打量"),
        (Op.WAIT, "等待|等等|等一等|等一下|等一会儿?|等片刻|稍等片刻?|等一炷香|等一盏茶|等半个时辰|等一个时辰|等到?天黑|"
                  "等到入夜|等到晚上|原地等待|休息一会儿?|歇一会儿?|歇息片刻"),
        (Op.MOVE, "(?:去|前往|走到|走进|进入|进|回到|钻进|钻入|爬进|跳下|爬下|攀下|穿过|走过)[PD]|(?:往|朝)[PD](?:走|去)"),
        (Op.STUDY, "(?:研读|修习|参详|参悟|钻研|翻阅|阅读)(?:一下|一会儿?)?I"),
        (Op.TAKE, "(?:拿起|拿走|拿|捡起|拾起|取下|取走|抓起)(?:S(?:上|里)的?)?I"),
        (Op.ATTACK, "(?:向|对|朝)H(?:出手|动手|出招|进攻)|(?:攻击|偷袭)H"),
        (Op.INSPECT, "(?:查看|检查|仔细查看|端详|察看|细看|搜查|搜)(?:一下|一番)?[PS]|(?:搜|搜查)H(?:的身|身上)?"),
        (Op.INSPECT, _KOWTOW),        # 此地有神像、蒲团：伏地细看（原著路线）
        (Op.WAIT, _KOWTOW),           # 否则是当众服软的姿态：rule_parse 按玩家以为此地有什么定夺
        (Op.ASK, "问H[IHM](?:在哪里?|在哪儿|在何处|去哪了|去哪儿了|去了哪里)"),
        (Op.TELL, "告诉H[IHMS](?:不|没)?在[PSH](?:上|里)?"),
    )
)
# 跟上认识的人：走到玩家以为他在的地方（刚看见他从哪道门走的就走那道门）
_FOLLOW = re.compile(r"(?:M(?:要|就|这就|赶紧|快|先)?)?(?:跟上|跟着|跟紧|紧跟|追上|跟随|随)H(?:去|走|过去|一起走)?")
# 拉着认识的人走：玩家自己走（被拉着的人跟不跟，由他自己决定）
_TOW = re.compile(r"(?:M(?:要|就|赶紧|快)?)?(?:拉着|拽着|带着|扶着|领着)H(?:快|赶紧)?(?:逃去|逃到|躲去|躲到|跑去|溜去|去|到)P")
_BOWS = ("磕头", "叩首", "跪拜", "拜了几拜")
_NOISE = re.compile(r"[\s，,。．.；;！!？?、…~～]+")
_ASKING = ("？", "?")            # 问号结尾的整句命令（“攻击龚光杰？”）是犹豫还是下令，交给模型判语态


def _skeleton(t: str, ms: Sequence[Mention], owner: str) -> str:
    out, i = [], 0
    for m in sorted(ms, key=lambda m: (m.pos, -m.length)):
        if m.pos < i:
            continue                      # 被更早、更长的提及盖住（“剑湖宫大殿”里的“大殿”）
        out += [t[i:m.pos], "M" if m.eid == owner else _SLOT[m.kind]]
        i = m.pos + m.length
    out.append(t[i:])
    return _NOISE.sub("", "".join(out))


# ============================================================
#  快模型：一次 generate，JSON Schema 约束；提示词只含玩家认识的实体与最近两段正文
# ============================================================

_SYSTEM = (
    "你是武侠文字游戏的输入解释器。玩家用第一人称说一句话，判断它属于哪一类，只输出 JSON。\n"
    "kind：\n"
    "- act：玩家亲手做的 1~3 个步骤（steps，按先后）。op：move 去某地或穿过某门、take 拿起、put 放下、give 交给、"
    "unlock 开锁、lock 上锁、inspect 仔细查看地点陈设或搜人身、attack 动手、study 研读手中秘籍、"
    "use 把手中物品用在某人（含自己）身上、wait 等待。"
    "例：“拿起长剑向龚光杰刺去”= take 长剑、attack 龚光杰；离开当前地点（“溜出大殿”）= move 经一条出路到那头："
    "target 填那头的地点，obj 填那扇门。\n"
    "- say：对一个人说话或发问。listener=听者；speech=tell（说）或 ask（问）；line=真正说出口的话，"
    "去掉“对某某说：”之类的包装，没说具体的话就留空；topic_* 只在断言或询问“某人/某物在某处”时填"
    "（问时 topic_value 为 null，断言“不在”时 topic_holds=false），否则为 null。\n"
    "- gesture：别人看得见的姿态或举动（坐下喝茶、拱手、叹气），line=不带主语的动作短语，如“坐下来喝了口茶”，"
    "只写玩家自己的动作，不写结果、伤势、别人的反应。"
    "向人磕头、当众磕头服软是 gesture，social=submit；对着神像、蒲团磕头是伏在地上细看眼前 = act：inspect 当前地点。\n"
    "- ask_gm：跳出故事问主持人（我该做什么、我在哪、身上有什么、怎么玩）。\n"
    "- unclear：实在听不懂才用，reply 写一句简短的场内追问。\n"
    "social：说话或姿态的社交含义，没有就填 none。\n"
    "mode：immediate=玩家此刻亲自要做（多步也算）；negated=否定；conditional=带条件或计划；"
    "narrative=讲别人或已发生的事；question=问能不能做。actor：player=玩家本人，other=别人，unknown=说不清。\n"
    "只能用实体表里的 id；“她/他/那人”按最近发生的事指认；用不上的字段填 null 或空串。\n"
    "missing：玩家声称拿着、掏出、使用、研读、会用的东西，或要去的地方、要找的人，却不在实体表里（或并不在你身上）的，"
    "写它的原文名；寻常陈设（茶、椅子）不算，没有就留空。\n"
    "manner：careful=悄悄/小心，rough=用力/粗暴，否则 normal。"
)
_MODES = (SpeechMode.IMMEDIATE, SpeechMode.NEGATED, SpeechMode.CONDITIONAL, SpeechMode.NARRATIVE, SpeechMode.QUESTION)
_KINDS = ("act", "say", "gesture", "ask_gm", "unclear")
_KIND_CN = {Kind.PERSON: "人", Kind.PLACE: "地点", Kind.SURFACE: "陈设", Kind.ITEM: "物件", Kind.DOOR: "通道"}
_FIELDS = ("kind", "mode", "actor", "steps", "listener", "speech", "line", "social", "topic_subject", "topic_value",
           "topic_holds", "missing", "reply")


def _schema(ids: Sequence[str]) -> dict[str, Any]:
    ref: dict[str, Any] = {"type": "string", "nullable": True}
    if ids:
        ref["enum"] = list(ids)
    step = {
        "type": "object",
        "properties": {"op": {"type": "string", "enum": [o.value for o in Op]}, "target": ref, "obj": ref,
                       "manner": {"type": "string", "enum": [m.value for m in Manner]}},
        "required": ["op", "target", "obj", "manner"],
    }
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(_KINDS)},
            "mode": {"type": "string", "enum": [m.value for m in _MODES]},
            "actor": {"type": "string", "enum": ["player", "other", "unknown"]},
            "steps": {"type": "array", "items": step, "maxItems": 3},
            "listener": ref,
            "speech": {"type": "string", "enum": ["tell", "ask"]},
            "line": {"type": "string"},
            "social": {"type": "string", "enum": [s.value for s in Social] + ["none"]},
            "topic_subject": ref, "topic_value": ref, "topic_holds": {"type": "boolean"},
            "missing": {"type": "string"},
            "reply": {"type": "string"},
        },
        # 全部 required（允许 null / 空串）：否则模型会干脆省略角色字段
        "required": list(_FIELDS),
    }


# ============================================================
#  校验后的场内说法：不回显模型编出来的 id（它可能恰好是玩家不该知道的真实实体）
# ============================================================

_NEGATED = clarify(ParsedCommand("", SpeechMode.NEGATED, None))
_VAGUE = "你一时不知从何下手。"
_IN_HAND = {Op.STUDY: "target", Op.GIVE: "obj", Op.PUT: "obj", Op.USE: "obj", Op.UNLOCK: "obj", Op.LOCK: "obj"}
_TO_OTHERS = (Op.ATTACK, Op.INSPECT, Op.GIVE, Op.TELL, Op.ASK)


def _role_kind(op: Op, role: str) -> Kind | None:
    kinds = getattr(OP_SIGNATURES[op], role) or frozenset()
    for k in (Kind.PERSON, Kind.PLACE, Kind.ITEM, Kind.DOOR, Kind.SURFACE):
        if k in kinds:
            return k
    return None


def _lack(op: Op | None, name: str | None, role: Kind | None = None) -> str:
    if op == Op.MOVE or role in (Kind.PLACE, Kind.DOOR):
        return f"你不知道该怎么去{name}。" if name else "你不知道该怎么去那里。"
    if role == Kind.PERSON or (role is None and op in (Op.ATTACK, Op.TELL, Op.ASK, Op.GIVE)):
        return f"你并不认识什么{name}。" if name else "你不知道说的是谁。"
    if op in (Op.TAKE, Op.INSPECT):
        return f"你并没见到{name}。" if name else "你不知道那是什么。"
    return f"你身上并没有{name}。" if name else "你身上并没有那样东西。"


# 功夫的样子：神功、掌法、指法、步法、剑法、心法……；兵刃与物件的样子：剑、刀、枪、鞭、扇、药、镖……（功夫里的“剑法”先算功夫）
_SKILLISH = re.compile(r"(?:功|掌|拳|指|步|法|诀|经|穴|神剑|剑气|内力)$")
_GEARISH = re.compile(r"(?:剑|刀|枪|棍|鞭|扇|锤|斧|钩|镖|针|药|粉|瓶|索|弓|箭)$")


def _unable(name: str | None) -> str:
    return f"你并不会{name}。" if name else "你并不会这样的功夫。"


def _text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _vetoed(ops: Sequence[Op], command: ParsedCommand | None) -> bool:
    """规则层看见了对这个行动的否定：模型说“照做”也不行。"""
    return command is not None and any(op in command.negated for op in ops)


# 明确的假设标记：规则看见了，模型说“即时”也不替玩家动手；“决定/打算/要不要”这类软标记仍听模型的
_HARD_IF = ("如果", "假如", "要是", "若是", "倘若", "假使", "万一", "一旦")


def _not_now(command: ParsedCommand, owner: str) -> bool:
    """规则确定这不是玩家此刻要做的事：明确的假设（含“等……就/才”，“等一会儿再去”只是先后），
    或别人作主语、玩家不在主语里的转述（“钟灵去拿长剑”算，“钟灵和我一起去后院”不算）。"""
    cond = command.condition
    if command.mode == SpeechMode.CONDITIONAL and cond is not None:
        return cond.text in _HARD_IF or (cond.text.startswith("等") and cond.text.endswith(("就", "才")))
    if command.mode != SpeechMode.NARRATIVE or command.subject in (None, owner) or command.action is None:
        return False
    t = command.text.strip().lower()
    left = max(t.rfind(ch, 0, command.action.start) for ch in "，,。；;！!？?") + 1
    return not any(w in t[left:command.action.start] for w in ("我", "自己"))


# ============================================================
#  解释器
# ============================================================


class Interpreter:
    """主持层的输入解释器。llm=None 即模板模式：退回 IntentParser 的规则解析（fallback）。
    universe：场景全部实体的名字（可选，会话交来），只用于拒绝——模型写的追问点了其中玩家不认识的，就不回显。"""

    def __init__(self, llm: LLMClient | None, aliases: Mapping[str, Sequence[str]] = _NO_ALIASES,
                 fallback: IntentParser | None = None, universe: Iterable[str] = (), kowtow_ticks: int = 1) -> None:
        """kowtow_ticks：对着可拜的陈设叩首占几个 tick（场景给出；1 = 旧版，一拜即伏地细看）。"""
        self.llm = llm
        self.aliases: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in aliases.items()}
        self.fallback = fallback or IntentParser(None, aliases=self.aliases)
        self.universe = frozenset(universe)
        self.kowtow_ticks = max(1, kowtow_ticks)

    def interpret(self, text: str, me: BeliefStore, recent: Sequence[str] = ()) -> Parsed:
        """recent：最近几段叙述正文（旧 → 新），只用来消解“她/那人”。"""
        raw = text.strip()
        if not raw:
            return _unclear("你想做什么？")
        meta = _meta(raw)
        if meta is not None:
            return meta
        gm = _GM.match(raw)
        if gm is not None:
            return Parsed(None, kind=MoveKind.ASK_GM, question=raw[gm.end():].strip() or raw)
        if _ooc(raw):
            return Parsed(None, kind=MoveKind.ASK_GM, question=raw)
        fast = self._follow(raw, me) or self._fast(raw, me)
        if fast is not None:
            return self._kowtow(fast, raw, me)
        if self.llm is None:
            return self._kowtow(self.fallback.parse(raw, me), raw, me)
        ms = mentions(raw.lower(), me, self.aliases)
        command = analyze(raw, ms, me.owner)
        prompt, ids = self.prompt(raw, me, recent, ms)
        try:
            data = parse_json(self.llm.generate(prompt, system=_SYSTEM, schema=_schema(ids), temperature=0.0,
                                                max_tokens=MAX_TOKENS))
        except LLMUnavailable as e:
            log.warning("解释器模型不可用，退回规则解析: %s", e)
            data = None
        if not isinstance(data, dict):
            # 只退回规则：会话交来的 fallback 可能带着同一个模型，再问一次就多等一轮超时，还绕过这里的校验
            return self._kowtow(rule_parse(raw, me, self.aliases), raw, me)
        return self._kowtow(self._decide(data, raw, me, command), raw, me)

    # ------------------------------------------------------------
    #  快路径
    # ------------------------------------------------------------

    def _fast(self, text: str, me: BeliefStore) -> Parsed | None:
        t = text.lower()
        skeleton = _skeleton(t, mentions(t, me, self.aliases), me.owner)
        ops = {op for op, pattern in _FAST if pattern.fullmatch(skeleton)}
        if not ops or (text.endswith(_ASKING) and ops != {Op.ASK}):
            return None
        parsed = rule_parse(text, me, self.aliases)
        if parsed.command is None or not parsed.command.immediate:
            return None
        if parsed.candidate is None:
            return parsed         # 整句命令认得出、只是做不到（已在此地、不知道路）：场内说法，不必问模型
        if parsed.candidate.op not in ops:
            return None
        problem = self._problem(parsed.candidate, me, set(held_items(me)))
        return _unclear(problem, parsed.command) if problem else parsed

    def _follow(self, text: str, me: BeliefStore) -> Parsed | None:
        """“跟上/跟着 + 认识的人”：走到玩家以为他在的地方——刚看见他从这里哪道门走的就走那道门，否则沿自己的地图走一步；
        “拉着某人逃去某地”：玩家自己去那里（被拉的人跟不跟由他自己）。"""
        t = text.lower()
        ms = mentions(t, me, self.aliases)
        skeleton = _skeleton(t, ms, me.owner)
        command = ParsedCommand(text, SpeechMode.IMMEDIATE, None)    # 整句只有“跟上某人”：没有否定、条件与转述的余地
        if _TOW.fullmatch(skeleton):
            place = next(m.eid for m in ms if m.kind == Kind.PLACE)
            cand = normalize(Candidate(Op.MOVE, place, None, manner_of(t)), me)
            why = self._problem(cand, me, set(held_items(me)), place)
            return _unclear(why, command) if why else Parsed(cand, source="rules", command=command)
        if not _FOLLOW.fullmatch(skeleton):
            return None
        who = next((m.eid for m in ms if m.kind == Kind.PERSON and m.eid != me.owner), None)
        if who is None:
            return None
        name, here, where = self._namer(me)(who), me.location_of(me.owner), believed_place(me, who)
        if where is None:
            return _unclear(f"你不知道{name}去了哪里。", command)
        if where == here:
            return _unclear(f"{name}就在你身边。", command)
        left = next((ep.event for ep in reversed(me.episodes) if ep.event.kind == Op.MOVE.value
                     and ep.event.actor == who and ep.event.place == here and ep.event.obj and ep.event.target), None)
        hop = (left.target, left.obj) if left is not None else route_to(me, where)
        cand = Candidate(Op.MOVE, hop[0], hop[1], manner_of(t)) if hop else None
        if cand is None or invalid(cand, me):
            return _unclear(f"你不知道该怎么跟上{name}。", command)
        return Parsed(cand, source="rules", command=command)

    def _kowtow(self, parsed: Parsed, text: str, me: BeliefStore) -> Parsed:
        """场景的叩首占几个 tick（kowtow_ticks > 1）且此地有可拜的陈设（解析成了伏地细看）：展开为叩首 × (k-1) 再细看，
        每一步都是带字的姿态；旧版 kowtow_ticks=1 原样返回。"""
        c = parsed.candidate
        if self.kowtow_ticks <= 1 or c is None or parsed.followups or c.op != Op.INSPECT:
            return parsed
        bowing = c.social == Social.SUBMIT or (c.target == me.location_of(me.owner) and any(w in text for w in _BOWS))
        if not bowing:
            return parsed
        bow = Candidate(Op.WAIT, social=Social.SUBMIT)
        pose = pose_of(text, self._names(me.owner, me)) or "磕头"
        return Parsed(bow, pose, source=parsed.source, command=parsed.command, kind=MoveKind.ACT,
                      followups=(*(bow,) * (self.kowtow_ticks - 2), c))

    # ------------------------------------------------------------
    #  提示词：身份、以为在哪、出路、玩家认识的实体表、最近两段正文、原话
    # ------------------------------------------------------------

    def prompt(self, text: str, me: BeliefStore, recent: Sequence[str] = (),
               ms: Sequence[Mention] | None = None) -> tuple[str, list[str]]:
        owner = me.owner
        here = me.location_of(owner)
        ids = self._pick(text, me, recent, mentions(text.lower(), me, self.aliases) if ms is None else ms)
        name = self._namer(me)
        lines = [f"你是{name(owner)}（{owner}）。" + (f"你以为自己在：{name(here)}（{here}）。" if here else "你不知道自己身在何处。")]
        held = held_items(me)
        lines.append("身上：" + ("、".join(f"{name(i)}（{i}）" for i in held) or "空无一物"))
        lines.append("会的武功：" + ("、".join(self._skills(me)) or "无"))
        ways = exits(me)
        lines.append("出路：" + ("；".join(f"{d} {name(d)} → {name(p)}（{p}）" for d, p in ways) or "不知道"))
        lines.append("实体表（id|名字|种类|你以为在哪）：")
        lines += [self._row(eid, me, name) for eid in ids]
        tail = [p.strip() for p in recent[-2:] if p and p.strip()]
        if tail:
            lines.append("最近发生：")
            lines += [("……" + p[-RECENT_CHARS:]) if len(p) > RECENT_CHARS else p for p in tail]
        lines.append(f"玩家输入：{text}")
        return "\n".join(lines), ids

    @staticmethod
    def _namer(me: BeliefStore):
        def name(eid: str | None) -> str:
            sk = me.sketch(eid or "")
            return sk.name if sk else "某处"
        return name

    def _pick(self, text: str, me: BeliefStore, recent: Sequence[str], ms: Sequence[Mention]) -> list[str]:
        """至多 MAX_TABLE 个玩家认识的实体：自己、原话点名的、此地与在场的、出路与门那头、最近正文提到的，其余按 ID。"""
        owner = me.owner
        here = me.location_of(owner)
        order: dict[str, None] = {}

        def add(eid: str | None) -> None:
            if eid is not None and eid in me.entities and len(order) < MAX_TABLE:
                order.setdefault(eid)

        add(owner)
        for m in ms:
            add(m.eid)
        add(here)
        for eid in sorted(me.entities):
            if here is not None and believed_place(me, eid) == here:
                add(eid)
        for door, dest in exits(me):
            add(door)
            add(dest)
        context = "".join(recent[-2:])
        if context:
            for eid, sk in sorted(me.entities.items()):
                if any(n and n in context for n in (sk.name, *self.aliases.get(eid, ()))):
                    add(eid)
        for eid in sorted(me.entities):
            add(eid)
        return list(order)

    def _row(self, eid: str, me: BeliefStore, name) -> str:
        sk = me.entities[eid]
        names = "/".join(dict.fromkeys((sk.name, *self.aliases.get(eid, ()))))
        if eid == me.owner:
            names += "（你自己）"
        if sk.kind == Kind.DOOR:
            ends = sorted(str(b.prop.value) for b in me.positives(eid, Rel.CONNECTS.value))
            where = ("连通" + "、".join(name(e) for e in ends)) if ends else "不知道通往哪里"
        elif sk.kind == Kind.PLACE:
            where = ""
        else:
            loc = me.location_of(eid)
            where = "你身上" if loc == me.owner else (name(loc) if loc else "不知道")
        return f"{eid}|{names}|{_KIND_CN[sk.kind]}|{where}"

    @staticmethod
    def _skills(me: BeliefStore) -> list[str]:
        """玩家自知学会的武功（内省），用教它的那卷帛书的名字称呼；认识的物件里没有就用技能键。"""
        out = []
        for skill in SKILLS:
            if me.holds(Proposition.attr(me.owner, skill, True)):
                books = [i for i, sk in sorted(me.entities.items()) if sk.kind == Kind.ITEM
                         and me.holds(Proposition.attr(i, "teaches", skill))]
                out.append(f"{me.entities[books[0]].name}所载" if books else skill)
        return out

    # ------------------------------------------------------------
    #  模型输出 → Parsed：先看类别与语态，再逐项校验
    # ------------------------------------------------------------

    def _decide(self, data: dict[str, Any], text: str, me: BeliefStore, command: ParsedCommand) -> Parsed:
        kind = data.get("kind") if data.get("kind") in _KINDS else "unclear"
        missing = _text(data.get("missing"), 12)
        # missing 仍是“编造前提”的信号，但只回显玩家自己写过的名字：模型从原著里补出来的名字可能正是玩家不该知道的
        said = missing if missing and missing.lower() in text.lower() else None
        if kind == "ask_gm":
            return Parsed(None, source="llm", command=command, kind=MoveKind.ASK_GM, question=text)
        if kind in ("act", "gesture", "unclear"):
            conjured = wield_problem(text, me, self.aliases)       # “我从怀里掏出北冥神功”：确定性地拦下编造的前提
            if conjured is not None:
                return _unclear(conjured, command, "llm")
        if kind == "unclear":
            reply = _text(data.get("reply"), 60)
            if reply and self._foreign(reply, me, text):
                reply = ""                                         # 模型写的追问点了玩家不认识的名字：换成固定追问
            fallback = clarify(command) if not command.immediate else "没太听明白，能换个说法吗？"
            return _unclear(_lack(self._hint(data, text), said) if missing else (reply or fallback), command, "llm")
        mode = _enum(SpeechMode, data.get("mode")) or SpeechMode.UNCLEAR
        if mode != SpeechMode.IMMEDIATE or data.get("actor") != "player":
            shown = SpeechMode.NARRATIVE if mode == SpeechMode.IMMEDIATE else mode   # 即时但主体不是玩家 = 叙述别人
            return _unclear(clarify(ParsedCommand(text, shown, None)), command, "llm")
        if kind == "act":
            return self._act(data, text, me, command, missing, said)
        if kind == "say":
            return self._say(data, text, me, command, missing, said)
        return self._gesture(data, text, me, command, missing, said)

    def _foreign(self, words: str, me: BeliefStore, text: str) -> bool:
        """模型写的话点了玩家不认识的实体（场景别称两字以上，或 universe 里的名字；玩家自己打出来的除外），或带着 id。
        先抹掉玩家认识的名字，免得被子串误伤（“剑湖宫”里的“剑湖”）。"""
        known = {n for eid, sk in me.entities.items() for n in (sk.name, *self.aliases.get(eid, ())) if n}
        hidden = {n for eid, names in self.aliases.items() if eid not in me.entities for n in names}
        hidden = {n for n in hidden | self.universe if len(n) >= MIN_NAME and n not in text} - known
        rest = words
        for n in sorted(known, key=len, reverse=True):
            rest = rest.replace(n, "\0")
        return bool(re.search(r"[A-Za-z_]{3,}", rest)) or any(n in rest for n in hidden)

    def _hint(self, data: dict[str, Any], text: str) -> Op | None:
        steps = data.get("steps") if isinstance(data.get("steps"), list) else []
        op = next((o for s in steps if isinstance(s, dict) for o in [_enum(Op, s.get("op"))] if o is not None), None)
        if op is not None:
            return op
        found = {h[2] for h in action_hits(text)}
        return next((o for o, _ in ACTION_WORDS if o in found), None)

    def _act(self, data: dict[str, Any], text: str, me: BeliefStore, command: ParsedCommand, missing: str,
             said: str | None) -> Parsed:
        raw = data.get("steps")
        steps = [s for s in raw if isinstance(s, dict)][:3] if isinstance(raw, list) else []
        if missing:
            op = self._hint(data, text)
            aimed = [s.get("target") for s in steps if isinstance(s.get("target"), str)]
            # 冲着认识的人使出不会的本事（“点了龚光杰的穴道”“用六脉神剑点倒他”）：缺的是功夫，不是人；
            # 缺的若是兵刃（“用倚天剑刺他”）仍是身上没有
            aims_known = any(me.sketch(t) is not None and me.sketch(t).kind == Kind.PERSON for t in aimed)
            if op == Op.ATTACK and aims_known:
                skill = not said or bool(_SKILLISH.search(said)) or not _GEARISH.search(said)
                return _unclear(_unable(said) if skill else _lack(Op.USE, said), command, "llm")
            return _unclear(_lack(op, said), command, "llm")
        plan, why = self._plan(steps, text, me)
        if not plan:
            return _unclear(why or _VAGUE, command, "llm")
        if _vetoed([c.op for c in plan], command):
            return _unclear(_NEGATED, command, "llm")
        if _not_now(command, me.owner) and any(c.op not in (Op.TELL, Op.ASK) for c in plan):
            return _unclear(clarify(command), command, "llm")     # “如果龚光杰攻击我，我就还手”“钟灵去拿长剑”
        first = plan[0]
        repeat, until = wait_length(text.lower()) if first.op == Op.WAIT else (1, None)
        speech = first.op in (Op.TELL, Op.ASK)
        line = self._own_words(data, text, me, first.target, first.op == Op.ASK) if speech else None
        kind = MoveKind.SAY if speech and len(plan) == 1 else MoveKind.ACT
        return Parsed(first, line, source="llm", repeat=repeat, until=until, command=command, kind=kind,
                      followups=tuple(plan[1:]))

    def _plan(self, steps: Sequence[dict[str, Any]], text: str, me: BeliefStore) -> tuple[list[Candidate], str | None]:
        """逐步校验：实体须是玩家认识的、种类对、路线按玩家的地图、要在手里的须在（计划里先拿起的也算）；
        第一个不成立的步骤截断整个计划，返回已成立的前缀与截断的理由。"""
        held = set(held_items(me))
        here = me.location_of(me.owner)                      # 计划里的“此地”：每走一步都从上一步的终点算起
        hinted = manner_of(text.lower())
        plan: list[Candidate] = []
        for raw in steps:
            op = _enum(Op, raw.get("op"))
            if op is None or (op == Op.WAIT and len(steps) > 1):
                return plan, None                            # 等待只能单独成为一个计划
            refs: dict[str, str | None] = {}
            for role in ("target", "obj"):
                value = raw.get(role)
                if value in (None, "", "null", "none"):
                    refs[role] = None
                elif isinstance(value, str) and value in me.entities:
                    refs[role] = value
                else:
                    return plan, _lack(op, None, _role_kind(op, role))
            manner = _enum(Manner, raw.get("manner")) or Manner.NORMAL
            cand = normalize(Candidate(op, refs["target"], refs["obj"], hinted if manner == Manner.NORMAL else manner),
                             me, here)
            if cand.op == Op.MOVE and cand.target is None and refs["target"] == here and leaving(text.lower()):
                out = leave_here(me, cand.manner, here)      # 说了“离开此地”却没说走哪扇门：只有一条出路就走它
                if out.candidate is None:
                    return plan, out.clarification
                cand = out.candidate
            why = self._problem(cand, me, held, refs["target"], here)   # 没说要离开、目的地又是脚下：“你已经在……了”
            if why is not None:
                return plan, why
            if cand.op == Op.TAKE and cand.target:
                held.add(cand.target)
            if cand.op in (Op.GIVE, Op.PUT) and cand.obj:
                held.discard(cand.obj)
            if cand.op == Op.MOVE and cand.target:
                here = cand.target
            plan.append(cand)
        return plan, None

    def _problem(self, c: Candidate, me: BeliefStore, held: set[str], asked: str | None = None,
                 here: str | None = None) -> str | None:
        name = self._namer(me)
        if invalid(c, me):
            if c.op == Op.MOVE:
                sk = me.sketch(asked or "")
                if asked is not None and asked == (me.location_of(me.owner) if here is None else here):
                    return f"你已经在{name(asked)}了。"
                return _lack(Op.MOVE, sk.name if sk is not None and sk.kind == Kind.PLACE else None)
            sig = OP_SIGNATURES[c.op]
            if sig.target is not None and c.target is None:
                return _lack(c.op, None, _role_kind(c.op, "target"))
            if sig.obj is not None and c.obj is None:
                return _lack(c.op, None, _role_kind(c.op, "obj"))
            return _VAGUE
        if c.op in _TO_OTHERS and c.target == me.owner:
            return "你不能对自己这样做。"
        role = _IN_HAND.get(c.op)
        need = getattr(c, role) if role else None
        if need is not None and need not in held:
            return f"你身上并没有{name(need)}。"
        return None

    def _names(self, eid: str | None, me: BeliefStore) -> tuple[str, ...]:
        sk = me.sketch(eid or "")
        return (sk.name, *self.aliases.get(sk.id, ())) if sk else ()

    def _own_words(self, data: dict[str, Any], text: str, me: BeliefStore, listener: str | None, ask: bool) -> str | None:
        """玩家嘴里的话只能是玩家打出来的字：模型给的原话（剥掉包装后）是原文的一段才收，否则改由规则从原文里取
        （speech_line 只会截原文）；没说具体的话（“和钟灵说话”“跟钟灵聊聊天”）就是 None，言语只带言语行为。"""
        line = unwrap_line(_text(data.get("line"), 80), self._names(listener, me), ask)
        if line and own_words(line, text):
            return line
        return speech_line(text, me, listener, self.aliases, ask=ask)

    def _say(self, data: dict[str, Any], text: str, me: BeliefStore, command: ParsedCommand, missing: str,
             said: str | None) -> Parsed:
        speech = Op.ASK if data.get("speech") == "ask" else Op.TELL
        raw = data.get("listener")
        sk = me.sketch(raw) if isinstance(raw, str) else None
        listener = sk.id if sk is not None and sk.kind == Kind.PERSON and sk.id != me.owner else None
        social = _enum(Social, data.get("social"))
        line = self._own_words(data, text, me, listener, speech == Op.ASK)
        conjured = wield_problem(unsaid(text, line), me, self.aliases)   # “我拔出长剑喝道：滚开”：原话之外的那截也要真
        if conjured is not None:
            return _unclear(conjured, command, "llm")
        if listener is None:
            if (raw not in (None, "", me.owner)) or missing:
                return _unclear(_lack(speech, said, Kind.PERSON), command, "llm")
            if line and not _vetoed([Op.WAIT, speech], command):
                # 没有听者的话是当众说的：看得见、听得见的姿态，不传递任何说法
                return Parsed(Candidate(Op.WAIT, social=social), f"说道：“{line}”", source="llm", command=command,
                              kind=MoveKind.GESTURE)
            return _unclear("你想对谁说？", command, "llm")
        named = {m.eid for m in mentions(text.lower(), me, self.aliases)}
        cand = Candidate(speech, listener, None, speech_manner(text.lower()), self._topic(data, me, speech, named), social)
        if invalid(cand, me):
            return _unclear(_VAGUE, command, "llm")
        if _vetoed([speech], command):
            return _unclear(_NEGATED, command, "llm")
        return Parsed(cand, line, source="llm", command=command, kind=MoveKind.SAY)

    @staticmethod
    def _topic(data: dict[str, Any], me: BeliefStore, speech: Op, named: set[str]) -> Fact | None:
        """命题只收“某某在某处”（问时只问下落），而且主语与处所都得是玩家原话里点了名的——模型不能替玩家编一句说法；
        不是玩家认识的、种类不对、没点名，命题作废，闲话照说。"""
        at = RELATIONS[Rel.AT]
        subject, value = data.get("topic_subject"), data.get("topic_value")
        s = me.sketch(subject) if isinstance(subject, str) else None
        if s is None or s.kind not in at.src_kinds or s.id not in named:
            return None
        if speech == Op.ASK:
            return Fact(Proposition.rel(s.id, Rel.AT, None), True)
        v = me.sketch(value) if isinstance(value, str) else None
        if v is None or v.kind not in at.dst_kinds or v.id == s.id or v.id not in named:
            return None
        holds = data.get("topic_holds")
        return Fact(Proposition.rel(s.id, Rel.AT, v.id), holds if isinstance(holds, bool) else True)

    def _gesture(self, data: dict[str, Any], text: str, me: BeliefStore, command: ParsedCommand,
                 missing: str, said: str | None) -> Parsed:
        if missing:
            return _unclear(_lack(None, said), command, "llm")
        own = self._names(me.owner, me)
        pose = pose_of(_text(data.get("line"), 80), own) or pose_of(text, own)
        if not pose:
            return _unclear("你想做什么？", command, "llm")
        problem = wield_problem(pose, me, self.aliases)
        if problem is not None:
            return _unclear(problem, command, "llm")
        if _vetoed([Op.WAIT], command):
            return _unclear(_NEGATED, command, "llm")
        if _not_now(command, me.owner):
            return _unclear(clarify(command), command, "llm")
        social = _enum(Social, data.get("social"))
        # 姿态只留看得见的那一截：不夹带结果、状态、别人的举动、不在手里的东西与不认识的名字
        kept, why = witness(pose, social, me, mentions(pose.lower(), me, self.aliases), self.aliases, self.universe, text)
        if kept is None:
            return _unclear(why or "你想做什么？", command, "llm")
        return Parsed(Candidate(Op.WAIT, social=social), kept, source="llm", command=command, kind=MoveKind.GESTURE)
