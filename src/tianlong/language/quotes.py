"""
[INPUT]: 依赖 language/render 的 RenderPlan / Violation / 词法积木（_mentions / _unsourced / _quotes / _mask / _clauses / _spans /
         _clause_after / _clause_before / _post_attribution / _persons）与词表（STATUS_LEXICON / STATUS_EXCLUSIONS / MIN_ALIAS / SPEECH_MARKS /
         PRONOUNS / POST_WINDOW / QUOTE_OPEN），language/scene 的 SceneBrief / VoiceLine
[OUTPUT]: 对外提供 check_quotes()（台词闸门：引语归属、替玩家开口、NPC 越界点名、凭空多出的说话者）、voiced()（正文里确有归属引语的说话者，
          供叙述者查台词是否讲到）、said_by()（正文里确凿归到本回合有台词的人名下、且本名就在这段引语引子里的原话，供会话记台词账本）、unspoken()（不是话的引语的起点，交给叙述闸门照叙述查）、
          introduces()（一句话里说话者自报了姓名：我叫/在下/本姑娘 + 名或带名的别称，我姓/在下姓/本姑娘姓 + 姓；相识账本与主持层共用）、
          recites()（正文里有一段引语照录了某句录入的原话，叙述者查 said 台词讲到没有）、
          台词词表 OBJECT_MARKERS / SUBJECT_LEADS / PERCEPTION / PLAYER_MIND / VOICED / SEQUENCE / PRETEND / DOUBTED / SOUNDS / SELF_INTRO /
          SURNAME_INTRO / NEGATIONS
[POS]: language 的台词闸门，与 render.check()、deeds.check_deeds() 并用。每段引语（含无引号的“某某道：……”与“某某说……”式转述）
       归到说话者：引子小句的主语、句首引语之后的“某某喝道”、上一段引语的说话者、上一句的主语（句首引语紧跟在谁的动作之后，
       读者就听成是谁说的——这也算确凿）；归到“你”名下的只能是玩家本回合的原话，NPC 只许点名自己认识的名字、只许说出计划里有的状态；
       说话者本回合只有录入的原话（VoiceLine.said，话即事实）时，归到他名下的引语须是其中一句的连续一段、或字二元组按原句的先后
       有六成对得上且否定字一个不添、对得上的那一截里一个不少（“我便不救他”“你们再为难”、挪前挪后都算另编），否则 fidelity；
       “她道：”这类代词引子接上一句的主语，他本回合只有录入的原话（或本回合的台词全是录入的原话）时照样查。
       刻着、写着、题作、绣着的字（“门楣上刻着四个字：“琅嬛福地””）是物件上的字，不归给任何人。
       不是话的引语不归给任何人，这里不查、交给叙述闸门照叙述查（unspoken）：一两个象声字后跟“地/的一声”（“嗒”的一声）、
       引子以“辨认得出/认出/读出”收尾（中间不点名人）且与玩家见过的外观描写逐字相同的字（蒲团绣字）、
       眼神里的问话（“那目光分明在问：”后跟无引号的话；“目光一沉问：“……””是开了口）、无引号冒号之后照着出处写的景；
       “龚光杰的声音远远传来：”里声音的主人是说话者；“你打定主意熬到天黑”只是复述玩家自己输入的意图（覆盖过半且收尾相同、
       不点名谁、不添先后说法），不算替他拿主意；引语里“当作/以为”之后的状态词是假设，不算说出的状态
       （“别以为我不知道……”“还当我瞧不出……”是反话，照查）。
       局限（如实）：代词不分男女，“钟灵瞪了段誉一眼。他道：”会接到钟灵身上——只会多丢一句（录入的原话随后照补），不会多交付一句；
       主语靠词法近似（宾语标记、“的”字结构、感知动词、“你”只在小句开头或承接词、状语之后才是主语），复杂句式可能归错；
       转述只认“说/告诉/提到/透露/低语”后面直接跟着的内容，且须找得到具名的说话者；替玩家起念头只认“决定/打定主意/心想”等少数说法
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tianlong.language.render import (
    _CLAUSE,
    _CLOSERS,
    _PUNCT,
    MIN_ALIAS,
    POST_WINDOW,
    PRONOUNS,
    QUOTE_OPEN,
    SPEECH_MARKS,
    STATUS_EXCLUSIONS,
    STATUS_LEXICON,
    RenderPlan,
    Violation,
    _asserted,
    _attributed,
    _clause_after,
    _clause_before,
    _clauses,
    _forms_of,
    _mask,
    _mentions,
    _persons,
    _post_attribution,
    _Quote,
    _quotes,
    _spans,
    _unsourced,
)
from tianlong.language.scene import SceneBrief, VoiceLine

# ============================================================
#  台词闸门：每段引语归到说话者，说话者只说得出他知道的；归到“你”名下的只能是玩家自己的原话
#  - 归属：引语前面紧挨着“道/说/问……”或冒号的，从引子所在的小句往前逐句找主语（同一句里没有就找上一句）；
#    句首的引语先看紧跟其后的“某某喝道”，再沿用上一段引语的说话者，最后取上一句的主语（这也算确凿）；
#    句中既无言说动词又无冒号的引语（匾额、称谓、强调）不归给任何人；“某某说……”的转述只归给引子小句里具名的主语
#  - 主语只在人里找（观察者认识的人、要说台词的人、你、他/她）：“钟灵抱着貂儿笑道”说话的是钟灵，不是貂儿；
#    NPC 的称呼要在小句开头几个字之内（“梁上的钟灵已笑道”），或紧跟在感知动词之后（“你听见钟灵笑道”）；
#    “你”只在小句开头、承接词或状语（“此刻/情急之下/慌乱中”）之后才是主语——“目光落在你脸上，冷笑道”说话的不是你；
#    紧跟在宾语标记之后（“指着你”“对龚光杰”“落在你”）或自己带着“的”（“你身旁的钟灵”）的称呼都不是主语
# ============================================================

# 称呼紧跟在这些词后面时是宾语，不是主语：“龚光杰指着你道”“马五德对龚光杰道”“龚光杰被钟灵一激，大声道”“走到你面前”
OBJECT_MARKERS: tuple[str, ...] = ("对着", "向着", "朝着", "冲着", "指着", "望着", "看着", "盯着", "瞪着", "瞧着", "看了", "望了",
                                   "瞥了", "瞧了", "瞪了", "打量", "拦住", "叫住", "喝住", "拉着", "扶着", "揪住", "抓住",
                                   "对", "向", "朝", "冲", "跟", "和", "与", "同", "给", "替", "被", "把", "将", "让", "叫",
                                   "请", "问", "劝", "骂", "瞪", "在", "到", "住", "着", "了")
# “你”前面只能是小句开头、这些承接词或状语，才算主语（“此刻你忙道”“情急之下你脱口而出”“慌乱中你连声道”）
SUBJECT_LEADS: tuple[str, ...] = ("于是", "此时", "这时", "当下", "随即", "之后", "然后", "只好", "只得", "终于", "当即",
                                  "此刻", "一时", "顿时", "登时", "霎时", "立时", "同时", "连忙", "急忙", "赶忙", "慌忙", "赶紧",
                                  "之下", "之中", "之间", "之际", "乱中", "急中", "慌中", "忙中", "暗中", "间",
                                  "便", "就", "才", "也", "又", "却", "而", "但", "则")
# 感知动词之后的称呼是内嵌小句的主语：“你听见钟灵笑道”说话的是钟灵
PERCEPTION: tuple[str, ...] = ("听见", "听到", "听得", "只听", "但听", "忽听", "看见", "瞧见", "只见", "但见", "望见")
PERCEPTION_WINDOW = 6
SUBJECT_WINDOW = 6            # NPC 的称呼离小句开头至多这么多字（“一旁的马五德”“梁上的钟灵已”）
POSSESSED = ("的", "身旁的", "身边的", "身后的", "面前的", "跟前的", "对面的")
# 声音的主人就是说话的人：“龚光杰的声音远远传来：……”里带“的”的称呼照样是主语（“你”不适用）
VOICED = ("的声音", "的嗓音", "的话音", "的语声", "的喝声", "的笑声", "的冷笑")
# 替玩家起念头、拿主意（词条是正则片段）：“你当即决定”“你心中暗想”“你在心中暗暗打定了主意”“你横下心来”；
# “由你决定”“你得尽快拿定主意”“你决定如何？”是把选择留给玩家，不算；“你心中想必”“想来”是揣测，不算
PLAYER_MIND: tuple[str, ...] = ("决定", r"打定了?主意", r"拿定了?主意", r"下定了?决心", "暗下决心", r"横[下了]心", "心想",
                                r"心[里中下]暗?想(?![必来])", "暗想", "心道", "寻思", "思忖", "盘算")
MIND_WINDOW = 6
MIND_PENDING = frozenset("须得要该需可妨能待还未没不难尚")
MIND_DEFER = frozenset("由看请等让待任凭听随该")
_MIND = re.compile(f"你([^{re.escape(_PUNCT)}]{{0,{MIND_WINDOW}}}?)((?:{'|'.join(PLAYER_MIND)}))")
# 复述自己的意图（输入“等到天黑”，正文“你打定主意熬到天黑”）：主意词之后的小句覆盖玩家输入过半的字二元组、
# 且以输入的末两字收尾，小句里既不点名谁、也没有输入之外的先后说法（SEQUENCE），就不算替他拿主意
# （“你打定主意等到天黑再去后山崖顶”“丢下钟灵独自等到天黑”“先杀了钟灵再等到天黑”照拦）
RESTATE = 0.5
SEQUENCE = re.compile(r"先|再|然后|随后|之后|接着|便|就(?![在这此])")
# 当作、以为（“我可要当你也被人点了穴道啦”）：引语里的状态词是打趣的假设，不断言谁被制；
# 前头有“别/莫/休/不”（“别以为我不知道……”）、后头紧跟“我”（“你还当我瞧不出……”）或中间有“不知/瞧不出”的，是反话，照查
PRETEND: tuple[str, ...] = ("要当", "只当", "还当", "当成", "当作", "当是", "以为", "只道", "还道")
_PRETEND = re.compile(f"(?<![别莫休不])(?<!不要)(?:{'|'.join(PRETEND)})(?!我)")
DOUBTED: tuple[str, ...] = ("不知", "不晓", "瞧不出", "看不出", "认不出", "不出来")
PRETEND_WINDOW = 12

# ============================================================
#  不是话的引语：声音、物件上的字、眼神、照着外观描写写的景
#  - 拟声：一两个象声字、收引号后紧跟“地/的一声/一声”（“嗒”的一声、“噗”地吐出瓜子壳）；“滚”“站住”不是象声字，照旧是话
#  - 物件上的字：引子以“辨认得出/认出/读出/看清/写道/绣着”收尾，且引语与玩家见过的某段外观描写逐字相同（编的字照旧归人）
#  - 眼神里的问话：“那目光分明在问：你笑什么？”——写的是眼神，没人开口
#  - 无引号的冒号之后是外观描写（“梁上传来一声轻笑：一个十六七岁的青衫少女坐在横梁上……”）：字二元组过六成出自出处
# ============================================================

SOUNDS = "嗒砰咚嗤哗噗咔啪嘭叮当嗖喀嚓铮锵吱呼嘶哧扑哈嘿呵嘻哼"
_SOUND = re.compile(f"[{SOUNDS}]{{1,2}}")
SOUND_TAILS: tuple[str, ...] = ("地", "的一声", "一声")
_READ = re.compile(r"(?:辨认得出|辨认出|认得出|认出|读出|看清|写道|写着|绣着|刻着)[^。！？；“”]{0,4}[：:，,]?$")
# 眼神里的问话只认“目光（分明）在问：”后跟无引号的话；“收回目光厉声问：”“神色一沉问：”是开了口
_GAZE = re.compile(r"(?:目光|眼神|眼光|眼色|神色|神情)(?:分明|似乎|像是|仿佛|都|也)?在问[：:]$")
LEAD_WINDOW = 16
SCENIC = 0.6
# 照录：说话者本回合只有录入的原话（VoiceLine.said）时，归到他名下的引语须有这么多字二元组按先后出自其中一句（截取连续一段照放），
# 否定字不许添、对得上的那一截里不许少——“你们不再为难”说成“你们再为难”、“我便救他”说成“我便不救他”都是另编
FIDELITY = 0.6
NEGATIONS = frozenset("不没别莫未非勿休无否")
# 自报姓名：“我叫钟灵”“在下段誉”——引子之后紧跟名或带名的别称；单一个姓只认“姓”字引子之后（“本姑娘姓钟”），
# “我是干什么吃的”“叫我干啥”“在下司马”里的头一个字不是姓
SELF_INTRO = ("我叫", "我是", "我姓", "在下", "本姑娘", "叫我")
SURNAME_INTRO = ("我姓", "在下姓", "本姑娘姓", "免贵姓")


@dataclass(frozen=True, slots=True)
class _Voice:
    names: frozenset[str]   # 可点名：may_name ∪ 说话者 ∪ 听者 ∪ 玩家
    source: str             # 说法与模板原话：其中本就有的名字与状态词算有出处


def introduces(words: str, forms: Iterable[str]) -> bool:
    """一句话里说话者自报了姓名：自报的引子（我叫/在下/本姑娘……）之后紧跟他的名或别称，“姓”字引子之后紧跟姓（名的头一个字）。"""
    full = {f for f in forms if f}
    surnames = {f[0] for f in full if len(f) >= 2}
    return any(words.startswith(h, i + len(lead)) for leads, heads in ((SELF_INTRO, full), (SURNAME_INTRO, surnames))
               for lead in leads for i in _at(words, lead) for h in heads)


def _at(text: str, word: str) -> list[int]:
    return [m.start() for m in re.finditer(re.escape(word), text)]


def _faithful(words: str, said: Sequence[str]) -> bool:
    """引语照着录入的原话说：是其中一句的连续一段；或换了几个字而大意不变（_close）。"""
    return any(words in s or _close(words, s) for s in said)


def recites(text: str, line: str) -> bool:
    """正文里有一段引语照录了这句录入的原话（叙述者查 said 台词讲到没有）。"""
    return any(_faithful(w, [_norm(line)]) for q in _quotes(text) if (w := _norm(q.words)))


def _close(words: str, line: str) -> bool:
    """换了几个字（“我便救他”说成“我就救他”）：字二元组按原句的先后有 FIDELITY 对得上（挪前挪后的不算），
    且否定字一个不添、对得上的那一截里一个不少——“你们再为难”“我便不救他”都是另编。"""
    pairs = [words[i:i + 2] for i in range(len(words) - 1)]
    hits, at = [], 0
    for p in pairs:
        k = line.find(p, at)
        if k >= 0:
            hits.append(k)
            at = k + 1
    if not pairs or len(hits) < FIDELITY * len(pairs):
        return False
    added = any(p not in line for p in pairs if NEGATIONS & set(p))
    dropped = any(line[i] in NEGATIONS and line[max(0, i - 1):i + 1] not in words and line[i:i + 2] not in words
                  for i in range(hits[0], hits[-1] + 2))
    return not added and not dropped


def _norm(s: str) -> str:
    """比对引语用：去掉标点与空白。"""
    return re.sub(r"[\W_]+", "", s)


def _subject(masked: str, forms: Iterable[str], selves: set[str], lo: int, hi: int,
             embedded: bool) -> str | None:
    """一个小句的主语：宾语位置与带“的”的称呼不算；embedded=True（引子所在的那一小句）时感知动词之后的称呼优先。"""
    found: list[tuple[str, bool]] = []
    for i, w in _mentions(masked[lo:hi], forms):
        k = lo + i
        before, after = masked[:k], masked[k + len(w):hi]
        voice = w not in selves and after.startswith(VOICED)       # “龚光杰的声音……”：声音的主人是主语
        if any(before.endswith(m) for m in OBJECT_MARKERS) or (after.startswith(POSSESSED) and not voice):
            continue
        if w in ("他", "她") and before.endswith("其"):
            continue
        head = masked[lo:k].strip(_CLAUSE)
        seen = any(v in masked[max(lo, k - PERCEPTION_WINDOW):k] for v in PERCEPTION)
        if w in selves:                               # “你”：小句开头、承接词或状语之后才是主语
            if not head or head.endswith(SUBJECT_LEADS):
                found.append((w, False))
        elif len(head) <= SUBJECT_WINDOW or seen:     # NPC：小句开头几个字之内，或紧跟感知动词
            found.append((w, seen))
    if embedded and any(seen for _, seen in found):
        return [w for w, seen in found if seen][-1]
    return found[0][0] if found else None


def _antecedent(masked: str, forms: Iterable[str], selves: set[str], spans: Sequence[tuple[int, int]],
                at: int) -> str | None:
    """at 处引语的代词接的是谁：本句引语之前、再往前一句里离得最近的、不是代词的主语。"""
    s = next((i for i, (a, b) in enumerate(spans) if a <= at < b), len(spans) - 1)
    lo = spans[max(0, s - 1)][0] if spans else 0
    return _speaker_in(masked, set(forms) - set(PRONOUNS), selves, lo, at)


def _speaker_in(masked: str, forms: Iterable[str], selves: set[str], lo: int, hi: int) -> str | None:
    """[lo, hi) 里离结尾最近的那个有主语的小句的主语（最后一小句是引子所在处）。"""
    parts = _clauses(masked, lo, hi)
    for n, (a, b) in enumerate(reversed(parts)):
        who = _subject(masked, forms, selves, a, b, embedded=n == 0)
        if who is not None:
            return who
    return None


# 刻在石上、写在纸上、绣在布上的字（“门楣上刻着四个字：“琅嬛福地””）：是物件上的字，不是谁说的话
_INSCRIBED = re.compile(r"(?:[刻写题绣印镌凿](?:着|有|的是|作)[^。！？；]{0,8}|[四八两几数]个?字)[：:，,]?$")


def _intro(lead: str) -> bool:
    """引语前面紧挨着的是不是“某某道：”式的引子。"""
    return lead.endswith(("：", ":", "，", ",", "一声")) or lead[-1:] in SPEECH_MARKS + "语曰"


def _attribute(text: str, masked: str, quotes: Sequence[_Quote], k: int, spans: Sequence[tuple[int, int]],
               forms: Iterable[str], selves: set[str],
               whos: Sequence[tuple[str | None, bool]]) -> tuple[str | None, bool]:
    """第 k 段引语的说话者（称呼）与归属是否确凿（有言说动词、冒号或紧挨着的上一句为证）。"""
    q = quotes[k]
    s = next((i for i, (a, b) in enumerate(spans) if a <= q.start < b), len(spans) - 1)
    a = spans[s][0]
    earlier = spans[s - 1] if s > 0 else None
    prev = quotes[k - 1] if k else None
    lo = max(a, prev.end) if prev else a
    if q.indirect:                                  # “某某说……”：只归给引子小句里的主语
        who = _speaker_in(masked, forms, selves, lo, q.start)
        return who, who is not None
    lead = text[lo:q.start].strip()
    if not lead:
        post = _clause_after(text, q.end, POST_WINDOW)
        if post and _post_attribution(text, q.end, q.end + len(post)):
            who = _subject(masked, forms, selves, q.end, q.end + len(post), embedded=True)
            if who is not None:
                return who, True
        if prev is not None and prev.start >= (earlier[0] if earlier else a):
            return whos[k - 1]                      # 接着上一段引语说
        # 句首引语紧跟在某人的动作之后（“左子穆脸色一沉。“够了。””）：读者就听成是他说的
        return (_speaker_in(masked, forms, selves, *earlier) if earlier else None), True
    if _INSCRIBED.search(lead):
        return None, False                          # 物件上的字：不归给任何人
    if _intro(lead):
        who = _speaker_in(masked, forms, selves, a, q.start)
        if who is None and earlier:
            who = _speaker_in(masked, forms, selves, *earlier)
        return who, True
    return None, False                              # 匾额、称谓、强调：不是谁说的话


def _judge(words: str, voices: Sequence[_Voice], who: str, universe: Iterable[str],
           statuses: frozenset[str]) -> list[Violation]:
    """引语里点名的实体与状态词，须在每一个 voices 的许可之内（归属不明时即取交集）；当作、以为之后的状态词是假设，不查。"""
    out: list[Violation] = []
    universe = tuple(universe)
    for v in voices:
        sourced = {n for _, n in _mentions(v.source, universe)}
        out += [Violation("quote_entity", f"{who}:{n}") for _, n in _mentions(words, universe)
                if n not in v.names and n not in sourced]
        for status, lexicon in STATUS_LEXICON.items():   # 他的出处里说过这个状态（“我被点了穴道”），换个说法照样许
            if status not in statuses and not _asserted(v.source, lexicon, STATUS_EXCLUSIONS):
                said = set(_unsourced(words, v.source, lexicon, STATUS_EXCLUSIONS))
                out += [Violation("quote_status", f"{who}:{status}:{w}")
                        for i, w in _asserted(words, lexicon, STATUS_EXCLUSIONS) if w in said and not _supposed(words, i)]
    return out


def _supposed(words: str, i: int) -> bool:
    """i 处的状态词是“要当/以为”之后的假设（“我可要当你也被人点了穴道啦”）；“别以为我不知道……”“还当我瞧不出……”是反话。"""
    head = _clause_before(words, i, PRETEND_WINDOW)
    m = next(reversed(list(_PRETEND.finditer(head))), None)
    return m is not None and not any(d in head[m.end():] for d in DOUBTED)


def _inert(text: str, q: _Quote, plan: RenderPlan) -> bool:
    """这段引语不是谁说的话：拟声、与见过的外观描写逐字相同的物件上的字、眼神里的问话、照着出处写的景。"""
    words, lead = _norm(q.words), text[max(0, q.start - LEAD_WINDOW):q.start]
    if q.indirect or not words:
        return False
    if _SOUND.fullmatch(words) and text[q.end:].startswith(SOUND_TAILS):
        return True
    if ((m := _READ.search(lead)) and not _mentions(m.group(0), _persons(plan)) and len(words) >= MIN_ALIAS
            and any(words in _norm(lore) for _, lore in plan.scenery)):     # “看清了你：……”看的是人，不是字
        return True
    bare = text[q.start] not in QUOTE_OPEN + '"'
    if bare and _GAZE.search(lead):
        return True
    if bare and text[q.start - 1:q.start] in ("：", ":"):                   # 无引号的冒号引语：照着出处写的景
        mine = _pairs(words)
        return bool(mine) and len(mine & _pairs(_norm(plan.source))) >= SCENIC * len(mine)
    return False


def _pairs(s: str) -> set[str]:
    """字二元组。"""
    return {s[i:i + 2] for i in range(len(s) - 1)}


def unspoken(text: str, plan: RenderPlan) -> frozenset[int]:
    """不是话的引语（起点）：叙述闸门 check() 把它们当叙述者自己的口吻查（眼神问话里的“你为何中了毒”照样是状态升级）。"""
    return frozenset(q.start for q in _quotes(text) if _inert(text, q, plan))


def _selves(plan: RenderPlan) -> set[str]:
    return {"你", *(f for f, n in plan.people if plan.viewer_name and n == plan.viewer_name)}


def _speakers(text: str, plan: RenderPlan, names: Iterable[str]
              ) -> tuple[list[_Quote], str, list[tuple[str | None, bool]], set[int]]:
    """全部引语、遮住引语后的正文、每段引语的（说话者称呼, 是否确凿）、不是话的引语（下标）。"""
    selves = _selves(plan)
    forms = {f for f, _ in plan.people} | set(names) | selves | set(PRONOUNS)
    quotes = _quotes(text)
    masked = _mask(text, quotes)
    spans = _spans(text)
    inert = {k for k, q in enumerate(quotes) if _inert(text, q, plan)}
    whos: list[tuple[str | None, bool]] = []
    for k in range(len(quotes)):
        whos.append((None, False) if k in inert else _attribute(text, masked, quotes, k, spans, forms, selves, whos))
    return quotes, masked, whos, inert


def voiced(text: str, plan: RenderPlan, brief: SceneBrief) -> frozenset[str]:
    """正文里确实开了口的人（本名）：有一段引语或转述归到他名下，或他的名字（别称）之后紧跟着言说动词。
    叙述者据此查要说的台词是否真的讲到了。"""
    people = dict(plan.people)
    quotes, _, whos, _ = _speakers(text, plan, (vl.speaker_name for vl in brief.lines))
    said = {people.get(w, w) for (w, _), q in zip(whos, quotes, strict=True) if w and _norm(q.words)}
    said |= {vl.speaker_name for vl in brief.lines if _attributed(text, *_forms_of(plan, vl.speaker_name))}
    return frozenset(said)


def said_by(text: str, brief: SceneBrief) -> tuple[tuple[str, str], ...]:
    """交付的正文里确凿归到本回合有台词的人名下的引语：(本名, 原话)，按先后。转述、不是话的引语、归属不明的都不算。
    只认 brief 里说话者的本名（会话手里没有计划，别称认不出）；所以还须他的本名就在这段引语自己的引子里
    （同一句里、上一段引语之后，或句首引语紧跟的“某某道”），或紧接着他上一段记下的引语说——
    “左掌门沉声喝道：”“一个清脆的声音道：”“段誉拱手道：”里认不出的人不会被往前找、记到龚光杰名下。宁可少记。"""
    names = {vl.speaker_name for vl in brief.lines}
    plan = RenderPlan("", (), frozenset(), frozenset(), frozenset(), frozenset(), (), (),
                      people=tuple((n, n) for n in sorted(names)))
    quotes, _, whos, inert = _speakers(text, plan, names)
    spans = _spans(text)
    out: list[tuple[str, str]] = []
    kept: set[int] = set()
    for k, (q, (who, sure)) in enumerate(zip(quotes, whos, strict=True)):
        if not (sure and who in names and k not in inert and not q.indirect and _norm(q.words)):
            continue
        a = next((a for a, b in spans if a <= q.start < b), 0)
        lead = text[max(a, quotes[k - 1].end if k else 0):q.start]
        named = who in lead or (not lead.strip() and who in _clause_after(text, q.end, POST_WINDOW))
        chained = k - 1 in kept and whos[k - 1][0] == who and not text[quotes[k - 1].end:q.start].strip()
        if named or chained:
            kept.add(k)
            out.append((who, q.words.strip()))
    return tuple(out)


def check_quotes(text: str, brief: SceneBrief, plan: RenderPlan, known_names: Iterable[str] = (), *,
                 command: str = "", since: int = 0) -> tuple[Violation, ...]:
    """台词闸门（与 check() 并用）：
    - 归到玩家（你/本名/别称）名下的引语与转述，去掉标点空白后须包含在玩家本回合的原话（brief.player_line 或 command）里，否则 puppet；
      引号之外替玩家起念头、拿主意（“你当即决定”“你心中暗想”）同样是 puppet，除非玩家自己的输入里就有这个词
    - 归到要说台词的 NPC 名下的：点名只许 may_name ∪ {说话者, 听者, 玩家} 与他那句说法/原话/谈资/近来经历里本就有的名字，否则 quote_entity；
      状态词只许本回合计划里有的状态、或他那句说法/原话/谈资/近来经历里本就肯定说出的，否则 quote_status
    - 确凿归到一个本回合既没开口、也没有台词的人名下：凭空多出的一句话，voice（句首引语沿用上一句主语也算确凿）
    - 归属不明（他/她、没找到说话者、匾额与称谓）：须同时满足所有说话者的许可（交集）；本回合没有台词时由 check() 把关；
      转述找不到具名的说话者就不查
    清单与外观描写里原样有的引语（别人的原话、匾额上的字）不另查。since：只查从该位置起的引语（流式时前面的句子已验收）。"""
    people = dict(plan.people)
    selves = _selves(plan)
    lines: dict[str, list[VoiceLine]] = {}
    for vl in brief.lines:
        lines.setdefault(vl.speaker_name, []).append(vl)
    voices = {name: _Voice(frozenset({name, *selves, *(n for vl in vls for n in (*vl.may_name, vl.listener_name) if n)}),
                           "\n".join(x for vl in vls for x in (vl.claim, vl.template, vl.knows, vl.lately, vl.about) if x))
              for name, vls in lines.items()}
    universe = (set(known_names) | plan.names | plan.aliases | plan.hidden | set(people) | set(voices)
                | {n for v in voices.values() for n in v.names}) - {"你"}
    said = [s for s in (_norm(brief.player_line or ""), _norm(command)) if s]
    verbatim = {name: [_norm(vl.template or "") for vl in vls] for name, vls in lines.items() if all(vl.said for vl in vls)}
    source = _norm(plan.source)
    quotes, masked, whos, inert = _speakers(text, plan, voices)
    spans = _spans(text)
    out: list[Violation] = []
    for k, (q, (who, sure)) in enumerate(zip(quotes, whos, strict=True)):
        words = _norm(q.words)
        if q.start < since or not words or k in inert:
            continue
        if who in selves:
            if not any(words in s for s in said):
                out.append(Violation("puppet", q.words.strip()))
            continue
        name = people.get(who, who) if who else None
        if q.indirect and (name is None or name in PRONOUNS):
            continue                                # 转述找不到具名的说话者：不猜
        if name not in voices and name in plan.speakers:
            continue                                # 确实开口、却没交来台词的人：由 check() 把关
        if name not in voices and sure and name and name not in PRONOUNS:
            out.append(Violation("voice", name))    # 别人的原话照搬到他嘴里也算：谁说了什么同样是事实
            continue
        if len(words) >= MIN_ALIAS and words in source:
            continue
        if name in voices:
            out += _judge(q.words, [voices[name]], name, universe, plan.statuses)
            if name in verbatim and not _faithful(words, verbatim[name]):
                out.append(Violation("fidelity", f"{name}:{q.words.strip()}"))   # 录入的原话另编一套词
        elif voices:
            out += _judge(q.words, list(voices.values()), "?", universe, plan.statuses)
            # “她道：”这类归属不明却确是开了口的：代词接的是上一句的主语（“梁上的青衫少女把瓷瓶一扬。她道：”），
            # 他本回合只有录入的原话，或本回合的台词全是录入的原话，照样须照录其中一句
            if sure:
                prior = _antecedent(masked, {f for f, _ in plan.people} | set(voices), selves, spans, q.start)
                pool = verbatim.get(people.get(prior, prior) if prior else "") or (
                    [x for v in verbatim.values() for x in v] if len(verbatim) == len(voices) else None)
                if pool is not None and not _faithful(words, pool):
                    out.append(Violation("fidelity", f"{prior or '?'}:{q.words.strip()}"))

    # ---- 替玩家起念头、拿主意 ----
    typed = f"{brief.player_line or ''}{command}"
    mine = _norm(brief.player_line or command)
    for m in _MIND.finditer(masked, since):
        gap, word = m.group(1), m.group(2)
        span = next((b for a, b in spans if a <= m.start() < b), len(text))
        asking = text[:span].rstrip().rstrip(_CLOSERS + '"')[-1:] in ("？", "?")
        deferred = m.start() > 0 and masked[m.start() - 1] in MIND_DEFER | MIND_PENDING   # “由你决定”“须你拿主意”
        if (MIND_PENDING & set(gap) or deferred or asking or word in typed
                or _restates(masked, m.end(), mine, {*universe, *PRONOUNS})):
            continue
        out.append(Violation("puppet", m.group(0)))
    return tuple(dict.fromkeys(out))


def _restates(masked: str, i: int, mine: str, names: Iterable[str]) -> bool:
    """主意词之后的小句只是玩家自己输入的意图换个说法：覆盖输入过半的字二元组、以输入的末两字收尾，
    且不点名任何人与物、不添输入里没有的先后说法（“先……再……”）。"""
    clause = _clause_after(masked, i, 40)
    said, typed = _norm(clause), _pairs(mine)
    return (bool(typed) and said.endswith(mine[-2:]) and len(typed & _pairs(said)) >= RESTATE * len(typed)
            and not _mentions(clause, names) and all(w.group() in mine for w in SEQUENCE.finditer(clause)))
