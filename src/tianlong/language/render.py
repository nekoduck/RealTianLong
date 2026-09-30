"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome / Kind / Rel / SKILLS / STATUS_ATTRS，language/templates 的 Names / render_percept
[OUTPUT]: 对外提供 fact_lines()（本回合允许讲的事实清单；familiar 里的东西再翻出来不算发现）、RenderPlan / build_plan()（渲染计划）、Violation / check()（叙述闸门）、
          restated_hearsay()（逐句查传闻：这一句替只闻其说的说法作保）、check_utterance()（对白闸门）、
          sentence_ends()（流式分句）、RenderStatus / Rendered（渲染结果与来源、丢句数）、
          词表 STATUS_LEXICON / DENIALS / COMMITMENT_WORDS / ARRIVAL_VERBS / ATTRIBUTION_VERBS / NEGATIONS / QUANTIFIERS / EXTRA_MARKERS
          及其排除表（STATUS_EXCLUSIONS / COMMITMENT_EXCLUSIONS / ATTRIBUTION_EXCLUSIONS / NOT_NEGATION）、MIN_ALIAS、
          SPEECH_MARKS / POST_MARKS / POST_WINDOW / PRONOUNS，以及台词闸门（quotes.py）与人事闸门（deeds.py）共用的词法积木
          （_mentions / _lexical / _negated / _quotes / _mask / _clauses……，含无引号的“某某道：……”与“某某说……”式转述）
[POS]: language 的“文字 ≠ 事实”闸门。提示词约束拦不住一次成功调用返回的错误非空文本，这里用确定性的词法检查拦：
       点名清单外的人与物（名或别称）、状态升级（受伤→被制、略有所得→学成，含“吐了一口血”“嘴唇发紫”“悟透”这类武侠说法）、
       瞬移、物品复制、编造承诺、把传闻说成叙述者确认的事实、场景秘密（私奔、投神农帮）——命中即丢句或回退确定模板。
       宁可错杀：误报只让这一句（或这一回合）的文字退回模板，世界结算不受任何影响。状态落在谁身上、谁做成了什么、
       玩家此刻在哪由 deeds.check_deeds() 查，引语归到谁、谁能说什么由 quotes.check_quotes() 查，渲染计划为它们记下
       谁身上有什么状态、谁做成了什么、玩家在哪、谁身上有什么东西、只闻其说的传闻里提到了谁。
       局限（如实）：只做词法比对，不做命题级语义理解——代词不参与点名检查，单字别称（“貂”）太泛、不作拒绝依据，
       两字以上的别称与名字同等对待，“石壁”“山道”这类泛称可能误报（误报只让文字退回模板）；
       传闻归属只检查“说话者（名或别称）之后有言说动词”，逐句只拦把说法里每个实体都说出来、而此前正文还没有归属的句子，
       换了说法的复述查不出；台词是否如实转达说法不查（交给提示词）；
       否定只认紧贴关键词的否定词（中间至多隔几个“有/能/曾/会/被”之类的虚字，双重否定算肯定），另认“离/距 + 关键词 + 还差/尚远”这种说没到的；
       数量只认“数词 + 量词 + 物品名”与“还有/另有/又……一 + 量词 + 物品名”的直接说法；秘密只认场景给出的词表；
       抵达动词的宾语只许本回合真的到达的地点或观察者此刻就在的地点；全角开引号遇半角收引号也算收，没收的引语到换行为止。
       出处优先：清单、外观描写与原话里本来就有的词、名字、数量与“抵达”说法，照搬不算违规
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache

from tianlong.core import SKILLS, STATUS_ATTRS, Kind, Modality, Op, Outcome, Percept, Rel
from tianlong.language.templates import Names, render_percept

# ============================================================
#  词表：闸门的全部语言知识集中在这里，改词表不改逻辑
#  - 词条是正则片段，多数就是字面词；词条里的“.”只匹配同一分句内的字（“受了.{0,2}伤”= 受了点伤 / 受了轻伤）
#  - 排除表：含关键词却不是那个意思的复合词（“控制住”不是被制住，“知道”不是开口说）；命中整个落在其中即不算
# ============================================================

# 状态 → 断言该状态的说法（受伤不等于被制住，略有所得不等于学成）
STATUS_LEXICON: dict[str, tuple[str, ...]] = {
    "subdued": ("点了穴", "点穴", "穴道被制", "穴道", "要穴", "点中", r"点.{0,4}穴", r"封住.{0,4}穴", "被制", "制住",
                "动弹不得", "动弹不了", "无法动弹", "动不了", r"(?:浑身|全身|半身|周身).{0,2}酸麻"),
    "poisoned": ("中了毒", "中毒", "毒发", "剧毒", "毒性发作", r"中了.{0,2}毒", r"身中.{0,2}毒",
                 r"(?:嘴唇|唇色|指甲|伤口|手背|面皮).{0,2}(?:发紫|泛紫|发黑|乌黑|青紫)"),
    "wounded": ("受了伤", "受伤", "负伤", "挂彩", "重伤", "轻伤", "内伤", "伤口", "伤势", "流血", "鲜血", r"受了.{0,2}伤",
                r"[吐呕咯喷].{0,3}血", r"嘴角.{0,4}血", "脸如金纸", "血流如注"),
    "mastered": ("融会贯通", "贯通", "学成", "练成", "大成", "学会", "练会", "练熟", "掌握", "尽数领悟", "悟透", "参透", "三昧",
                 r"(?:神功|功夫|武功|步法|心法).{0,2}已成"),
}
# 只收不含糊的武侠说法：“气得脸色发青”“头皮发麻”“一动不动”“僵住”“已成定局”“血丝”说的都不是伤毒被制学成，不收
STATUS_EXCLUSIONS: tuple[str, ...] = ("控制住", "克制住", "抑制住", "林中毒", "其中毒", r"贯通(?:前后|南北|东西)", "呕心沥血")
# 否认一个确有的状态（“你却毫发无损”）：与“并未受伤”同罪
DENIALS: tuple[str, ...] = ("毫发无损", "毫发无伤", "毫发未伤", "毫发不伤", "安然无恙", "分毫未损", "毫无损伤", "丝毫无损",
                            "完好无损")
# 承诺与誓言：引擎没有“承诺”这种行动，叙述与对白都不能凭空多出一句保证；“一定会”只在有人作保时才算（“这钥匙一定会有用”不算）
COMMITMENT_WORDS: tuple[str, ...] = ("答应", "保证", "承诺", "发誓", "起誓", "许诺", "应允", "誓要", r"(?:我|他|她|咱|俺)们?一定会")
COMMITMENT_EXCLUSIONS: tuple[str, ...] = (r"答应了?一声",)
# 抵达动词：其后同一分句内出现的地点必须是本回合真的到达的地方
ARRIVAL_VERBS: tuple[str, ...] = ("来到", "走进", "进入", "抵达", "到达", "回到", "走到", "踏入", "踏进", "钻进", "跳进", "到了",
                                  "跑到", "跑进", "冲进", "冲到", "奔到", "奔进", "赶到", "走入", "闯入", "闯进", "来至", "去了")
_NOT_ARRIVAL_BEFORE: dict[str, frozenset[str]] = {
    "到了": frozenset("看听见想找得遇碰感觉闻受收做办等迟猜料意顾说识注察"),   # “听到了”“想到了”不是抵达
    "去了": frozenset("过出离失死下上进回拿带送搬收"),                         # “过去了”“拿去了”不是谁抵达某处
}
ARRIVAL_WINDOW = 10
# 言语归属：复述传闻时，说话者名字之后应当跟着这些动词之一；单独的“道”只在后接冒号、引号或逗号时才是“说”
ATTRIBUTION_VERBS: tuple[str, ...] = ("说", "告诉", "讲", "透露", "提起", "提到", "低语", "喊", "称", "问", "答", "言道",
                                      r"道(?=[：:“\"，,])")
ATTRIBUTION_EXCLUSIONS: tuple[str, ...] = ("知道", "味道", "穴道", "栈道", "山道", "小道", "大道", "道路", "道理", "难道",
                                           "一言不发", "问题", "称呼", "答应", "说不定", "听说", "讲究")
ATTRIBUTION_WINDOW = 10
# 否定：只认紧贴关键词的否定词，中间至多隔几个虚字（“未能融会贯通”“并没有受伤”“没被点穴”）；
# 双重否定算肯定（“不得不答应”）；否定字只是前一个词的尾字时不算（“特别”“约莫”），“不慎受伤”“无量山中毒”里的否定字不紧贴关键词
NEGATIONS = frozenset("未没不无非别莫勿")
NEGATION_FILLERS = frozenset("有能曾会被再可得必法")
NEGATION_GAP = 3
NOT_NEGATION = frozenset({"特别", "分别", "区别", "告别", "离别", "个别", "类别", "派别", "性别", "约莫", "除非", "是非", "虚无",
                          "少不", "免不"})
# 数量：数词 + 量词 + 修饰（至多两字，或以“的”收尾的六字以内）+ 物品名；“又一/另一”意味着比现有的多一件；
# “一把钥匙”本身不多，同一分句前头说了“还有/另有/又/再”才是多出来的一件
QUANTIFIERS: dict[str, int] = {"好几": 2, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
                               "十": 10, "几": 2, "数": 2}
ANOTHER: tuple[str, ...] = ("又一", "另一")
EXTRA_MARKERS: tuple[str, ...] = ("还有", "另有", "也有", "另外", "又", "再")
EXTRA_WINDOW = 8
CLASSIFIERS = "把个只件枚支柄瓶卷本块串颗粒张根份对双"
# 别称至少两个字才作拒绝依据：单字别称（“貂”）太泛
MIN_ALIAS = 2
_PUNCT = "，。；！？、,.;!?：:“”\"'（）()\n"
_FREE = f"[^{re.escape(_PUNCT)}]"


# ============================================================
#  事实清单：本回合值得讲、也只允许讲的事
# ============================================================


def fact_lines(viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
               familiar: Container[str] = ()) -> list[str]:
    """本回合值得讲的事：事件感知全部讲；环顾只在移动/查看之后（或被要求时）讲。"""
    moved = any(
        p.modality == Modality.SELF and p.event and p.event.kind in (Op.MOVE.value, Op.INSPECT.value)
        and p.event.outcome == Outcome.SUCCESS
        for p in percepts
    )
    lines: list[str] = []
    for p in percepts:
        if p.modality == Modality.SCENE:
            if show_scene or moved:
                lines.append("你看到：" + render_percept(p, names, viewer, me="你", familiar=familiar))
        else:
            lines.append(render_percept(p, names, viewer, me="你", familiar=familiar))
    return list(dict.fromkeys(lines))      # 同一分钟里的三声响动，只说一次


# ============================================================
#  渲染计划：LLM 能说什么，由玩家本回合的感知决定
# ============================================================


@dataclass(frozen=True, slots=True)
class RenderPlan:
    viewer: str
    lines: tuple[str, ...]                    # 允许的事实清单（也是回退模板的正文）
    names: frozenset[str]                     # 允许点名的实体：本回合感知里出现的人、物、地点、通道 + 观察者自己
    statuses: frozenset[str]                  # 本回合感知里确实出现的状态：wounded / poisoned / subdued / mastered
    arrivals: frozenset[str]                  # 允许作为“抵达”宾语的地点（名与别称）：本回合真的有人走到了那里
    places: frozenset[str]                    # 观察者知道的全部地点（名与别称）：抵达动词之后只查这些
    items: tuple[tuple[str, int], ...]        # 本回合出现的物品名 → 件数
    hearsay: tuple[str, ...]                  # 传闻的说话者名：复述其说法必须带言语归属
    source: str = ""                          # 清单 + 外观描写 + 时辰原文：出现在其中的词算有出处
    commitment: bool = False                  # 本回合是否存在承诺；引擎没有承诺行动，恒为 False
    aliases: frozenset[str] = frozenset()     # 允许点名的实体的别称（“大殿”）：与名字同等对待
    hidden: frozenset[str] = frozenset()      # 其余实体的别称（两字以上）：只用于拒绝——“神仙姐姐”“营地”不许凭空出现
    viewer_name: str = ""                     # 观察者自己的名字（“段誉”）：引语归到他名下的，只能是玩家自己的原话
    people: tuple[tuple[str, str], ...] = ()  # 观察者认识的人：(称呼, 本名)，名字与别称都映射到本名——台词闸门据此给引语找说话者
    speakers: frozenset[str] = frozenset()    # 本回合确实开口、且观察者听见了（或自己说）的人（本名）：其余的人不许凭空多出一句台词
    afflicted: tuple[tuple[str, str], ...] = ()    # (本名, 状态)：谁身上确有哪个状态——状态词须落在对的人身上
    done: tuple[tuple[str, str], ...] = ()         # (本名, 操作)：本回合谁确实做成了什么（attack/take/move……；受赠记 receive）
    here: frozenset[str] = frozenset()             # 观察者此刻所在的地点（名与别称）；不知道时为空，站位与离开都不查
    goods: tuple[tuple[str, str], ...] = ()        # 观察者认识的物品：(称呼, 本名)
    holdings: tuple[tuple[str, str], ...] = ()     # (持有者本名, 物品本名)：亲眼所见、身上确有的东西
    claims: tuple[tuple[str, tuple[frozenset[str], ...]], ...] = ()   # 只闻其说的传闻：(说话者本名, 说法里每个实体的称呼)
    secrets: tuple[str, ...] = ()                  # 场景的秘密词表（正则片段）：计划与出处里没有就不许说


def build_plan(viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
               looks: Sequence[str] = (), lapse: str = "",
               aliases: Mapping[str, Sequence[str]] | None = None, secrets: Sequence[str] = (),
               familiar: Container[str] = ()) -> RenderPlan:
    """与 fact_lines() 同样的输入：观察者、本回合感知、观察者的名称表。looks/lapse 是一并交给 LLM 的外观描写与时辰；
    aliases 是场景别称全表（实体 ID → 别称）：本回合出场实体的别称可说，其余的只用于拒绝；secrets 是场景的秘密词表。"""
    aliases = aliases or {}
    table: dict[str, tuple[str, Kind]] = {sk.id: (sk.name, sk.kind) for p in percepts for sk in p.sketches}
    table.update({eid: (sk.name, sk.kind) for eid, sk in names.items()})
    ids: set[str] = {viewer}
    statuses: set[str] = set()
    arrived: set[str] = set()
    hearsay: list[str] = []
    talkers: set[str] = set()
    afflicted: set[tuple[str, str]] = set()
    done: set[tuple[str, str]] = set()
    here: set[str] = set()
    holdings: set[tuple[str, str]] = set()
    witnessed = {(f.prop, f.holds) for p in percepts if p.modality != Modality.SPEECH for f in p.facts}
    rumors: list[tuple[str, tuple[str, ...]]] = []
    for p in percepts:
        ev = p.event
        if ev is not None:
            ids.update(x for x in (ev.actor, ev.target, ev.obj, ev.place) if x)
            if ev.kind in (Op.TELL.value, Op.ASK.value) and ev.actor and p.modality in (Modality.SPEECH, Modality.SELF):
                talkers.add(ev.actor)            # 只看见在耳语、没听见内容的人不算开口
            if ev.kind == Op.MOVE.value and ev.outcome == Outcome.SUCCESS and ev.target:
                arrived.add(ev.target)
            if ev.outcome == Outcome.SUCCESS and ev.actor:
                done.add((ev.actor, ev.kind))
                if ev.kind == Op.GIVE.value and ev.target:
                    done.add((ev.target, "receive"))
            if ev.reason == "subdued" and ev.actor:
                statuses.add("subdued")          # “穴道被制，动弹不得”是失败原因，也是此刻的状态
                afflicted.add((ev.actor, "subdued"))
            if ev.outcome == Outcome.SUCCESS and ev.reason == "mastered" and ev.actor:
                statuses.add("mastered")
                afflicted.add((ev.actor, "mastered"))
        for f in p.facts:
            ids.add(f.prop.subject)
            if not f.prop.is_attr and isinstance(f.prop.value, str):
                ids.add(f.prop.value)
            if f.holds and f.prop.is_attr:
                key = f.prop.attr_key
                if key in STATUS_ATTRS:
                    statuses.add(key)
                    afflicted.add((f.prop.subject, key))
                elif key in SKILLS:
                    statuses.add("mastered")
                    afflicted.add((f.prop.subject, "mastered"))
            if f.holds and f.prop.predicate == Rel.AT.value and p.modality != Modality.SPEECH:
                if f.prop.subject == viewer:
                    here.add(str(f.prop.value))
                else:
                    holdings.add((str(f.prop.value), f.prop.subject))
        ids.update(sk.id for sk in p.sketches)
        if p.modality == Modality.SPEECH and p.facts and p.informant and p.informant != viewer:
            hearsay.append(p.informant)
            rumors += [(p.informant, (f.prop.subject, str(f.prop.value))) for f in p.facts
                       if not f.prop.is_attr and isinstance(f.prop.value, str) and (f.prop, f.holds) not in witnessed]

    def name(eid: str) -> str | None:
        return table[eid][0] if eid in table else None

    def with_aliases(eids: Iterable[str]) -> frozenset[str]:
        return frozenset(n for e in eids for n in (name(e), *aliases.get(e, ())) if n)

    def forms(eid: str) -> frozenset[str]:
        return with_aliases((eid,)) | ({"你"} if eid == viewer else set())

    def kinded(kind: Kind) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({(form, n) for e, (n, k) in table.items() if k == kind for form in (n, *aliases.get(e, ())) if form}))

    def named(pairs: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({(n, x) for e, x in pairs if (n := name(e))}))

    items = Counter(n for e in ids if (n := name(e)) and table[e][1] == Kind.ITEM)
    lines = fact_lines(viewer, percepts, names, show_scene, familiar)
    allowed = frozenset(n for e in ids if (n := name(e)))
    spoken = frozenset(a for e in ids for a in aliases.get(e, ()))
    return RenderPlan(
        viewer=viewer,
        lines=tuple(lines),
        names=allowed,
        statuses=frozenset(statuses),
        arrivals=with_aliases(arrived),
        places=with_aliases(e for e, (_, k) in table.items() if k == Kind.PLACE),
        items=tuple(sorted(items.items())),
        hearsay=tuple(dict.fromkeys(n for i in hearsay if (n := name(i)))),
        source="\n".join([*lines, *looks, lapse]),
        aliases=spoken,
        hidden=frozenset(a for al in aliases.values() for a in al
                         if len(a) >= MIN_ALIAS and a not in spoken and a not in allowed),
        viewer_name=name(viewer) or "",
        people=kinded(Kind.PERSON),
        speakers=frozenset(n for e in talkers if (n := name(e))),
        afflicted=named(afflicted),
        done=named(done),
        here=with_aliases(e for e in here if e in table and table[e][1] == Kind.PLACE),
        goods=kinded(Kind.ITEM),
        holdings=tuple(sorted({(h, i) for e, x in holdings if (h := name(e)) and (i := name(x))
                               and table[e][1] == Kind.PERSON and table[x][1] == Kind.ITEM})),
        claims=tuple((n, tuple(forms(e) for e in es)) for who, es in dict.fromkeys(rumors)
                     if (n := name(who)) and all(e in table for e in es)),
        secrets=tuple(secrets),
    )


# ============================================================
#  词法积木
# ============================================================


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str      # entity / status / teleport / duplicate / commitment / hearsay / secret / subject / empty；人事闸门另有 outcome，
    detail: str    # 台词闸门另有 puppet / voice / quote_entity / quote_status，叙述者另有 clock / omitted


def _mentions(text: str, words: Iterable[str]) -> list[tuple[int, str]]:
    """最长匹配扫描：“后山崖顶”不会再被数成“后山”，“剑湖宫大殿”不会再被数成别的短名。"""
    by_first: dict[str, list[str]] = {}
    for w in sorted({w for w in words if w}, key=lambda w: (-len(w), w)):
        by_first.setdefault(w[0], []).append(w)
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        hit = next((w for w in by_first.get(text[i], ()) if text.startswith(w, i)), None)
        if hit is None:
            i += 1
            continue
        out.append((i, hit))
        i += len(hit)
    return out


@cache
def _rx(pattern: str) -> re.Pattern[str]:
    """词条 → 正则：词条里的“.”只匹配同一分句内的字。"""
    return re.compile(pattern.replace(".", _FREE))


def _lexical(text: str, patterns: Iterable[str], exclusions: Iterable[str] = ()) -> list[tuple[int, str]]:
    """词表命中：同一起点取最长、命中互不重叠；整个落在排除复合词里的不算（“控制住”里的“制住”、“知道”里的“道”）。"""
    blocked = [m.span() for p in exclusions for m in _rx(p).finditer(text)]
    found = sorted((m.start(), -len(m.group(0)), m.group(0)) for p in patterns for m in _rx(p).finditer(text))
    out: list[tuple[int, str]] = []
    end = 0
    for start, neg, word in found:
        stop = start - neg
        if not word or start < end or any(b0 <= start and stop <= b1 for b0, b1 in blocked):
            continue
        out.append((start, word))
        end = stop
    return out


def _clause_before(text: str, i: int, width: int) -> str:
    start = max(0, i - width)
    for j in range(i - 1, start - 1, -1):
        if text[j] in _PUNCT:
            return text[j + 1:i]
    return text[start:i]


def _negated(text: str, i: int) -> bool:
    """从关键词往前数：紧贴着的否定词（隔着至多 NEGATION_GAP 个虚字）个数为奇数才算否定。"""
    count, gap, j = 0, 0, i - 1
    while j >= 0:
        ch = text[j]
        if ch in NEGATIONS and text[max(0, j - 1):j + 1] not in NOT_NEGATION:
            count, gap = count + 1, 0
        elif ch in NEGATION_FILLERS and gap < NEGATION_GAP:
            gap += 1
        else:
            break
        j -= 1
    return count % 2 == 1


_SHORT_OF = re.compile(r"^.{0,6}?(?:还差|尚差|差着|差得|还远|尚远|远着|隔着一层)")


def _short_of(text: str, i: int, word: str) -> bool:
    """“离融会贯通还差着一层”“距学成尚远”：离/距 + 关键词 + 还差/尚远，说的是没到。"""
    return i > 0 and text[i - 1] in "离距" and bool(_SHORT_OF.match(text[i + len(word):]))


def _asserted(text: str, patterns: Iterable[str], exclusions: Iterable[str] = ()) -> list[tuple[int, str]]:
    """文本里被肯定说出的词（去掉被否定的，与“离……还差”这类说没到的）。"""
    return [(i, w) for i, w in _lexical(text, patterns, exclusions) if not _negated(text, i) and not _short_of(text, i, w)]


def _arrivals(text: str) -> list[tuple[int, str]]:
    return [(i, v) for i, v in _mentions(text, ARRIVAL_VERBS)
            if not (v in _NOT_ARRIVAL_BEFORE and i > 0 and text[i - 1] in _NOT_ARRIVAL_BEFORE[v])]


def _clause_after(text: str, i: int, width: int) -> str:
    window = text[i:i + width]
    for k, ch in enumerate(window):
        if ch in _PUNCT:
            return window[:k]
    return window


def _quantified(text: str, item: str, count: int) -> list[tuple[int, str]]:
    """“两把钥匙”“另一把铜钥匙”“两把一模一样的钥匙”“桌下还有一把钥匙”：返回 (暗示的件数, 原文)。“又一个人拿起钥匙”不算。"""
    nums = "|".join(sorted([*QUANTIFIERS, *ANOTHER], key=lambda q: -len(q)))
    pat = re.compile(f"({nums}|一)[{CLASSIFIERS}](?:{_FREE}{{0,5}}的|{_FREE}{{0,2}}){re.escape(item)}")
    out: list[tuple[int, str]] = []
    for m in pat.finditer(text):
        q = m.group(1)
        if q == "一":
            if not any(w in _clause_before(text, m.start(), EXTRA_WINDOW) for w in EXTRA_MARKERS):
                continue
            implied = count + 1
        else:
            implied = count + 1 if q in ANOTHER else QUANTIFIERS[q]
        out.append((implied, m.group(0)))
    return out


def _attributed(text: str, *speaker: str) -> bool:
    """说话者（名或别称）之后不远处有言说动词：“守卫告诉你……”“钟灵笑道：……”；“守卫显然知道……”不算。"""
    verbs = [i for i, _ in _lexical(text, ATTRIBUTION_VERBS, ATTRIBUTION_EXCLUSIONS)]
    for i, n in _mentions(text, speaker):
        lo = i + len(n)
        if any(lo <= v < lo + ATTRIBUTION_WINDOW for v in verbs):
            return True
    return False


def _unsourced(text: str, source: str, patterns: Iterable[str], exclusions: Iterable[str] = ()) -> list[str]:
    """text 里肯定说出、而 source 里并未肯定说出的词。"""
    patterns, exclusions = tuple(patterns), tuple(exclusions)
    sourced = {w for _, w in _asserted(source, patterns, exclusions)}
    return [w for _, w in _asserted(text, patterns, exclusions) if w not in sourced]


# ============================================================
#  叙述闸门
# ============================================================


def check(text: str, plan: RenderPlan, known_names: Iterable[str] = ()) -> tuple[Violation, ...]:
    """known_names 是“可能被点名”的全集（玩家认识的 + 场景里所有实体），只用于拒绝：不在计划里的名字出现即违规。"""
    out: list[Violation] = []
    allowed = plan.names | plan.aliases
    universe = set(known_names) | allowed | plan.hidden
    sourced = {n for _, n in _mentions(plan.source, universe)}

    # ---- 1. 点名：清单外的人与物（名或别称）----
    for _, n in _mentions(text, universe):
        if n not in allowed and n not in sourced:
            out.append(Violation("entity", n))

    # ---- 2. 状态升级：本回合没有的状态不许出现 ----
    for status, words in STATUS_LEXICON.items():
        if status not in plan.statuses:
            out += [Violation("status", f"{status}:{w}") for w in _unsourced(text, plan.source, words, STATUS_EXCLUSIONS)]

    # ---- 3. 编造承诺 ----
    if not plan.commitment:
        out += [Violation("commitment", w)
                for w in _unsourced(text, plan.source, COMMITMENT_WORDS, COMMITMENT_EXCLUSIONS)]

    # ---- 4. 瞬移：抵达动词之后的地点必须是本回合真的到达的、或观察者此刻就在的（“走到大殿门口”“走到湖畔的钟灵”）；
    #      原话里转述的“回到港口”照搬不算；在全部名字上做最长匹配：“仓库大门”“后山小径”是门，不会被截成地点“仓库”“后山”
    everything = universe | plan.places
    for i, verb in _arrivals(text):
        hits = _mentions(_clause_after(text, i + len(verb), ARRIVAL_WINDOW), everything)
        if (hits and hits[0][1] in plan.places and hits[0][1] not in plan.arrivals | plan.here
                and verb + hits[0][1] not in plan.source):
            out.append(Violation("teleport", verb + hits[0][1]))

    # ---- 5. 物品复制（外观描写本就写着“插着几柄长剑”的，照此说不算）----
    for item, count in plan.items:
        ceiling = max([count, *(implied for implied, _ in _quantified(plan.source, item, count))])
        out += [Violation("duplicate", phrase) for implied, phrase in _quantified(text, item, count) if implied > ceiling]

    # ---- 6. 传闻不得变成叙述者确认的事实 ----
    out += [Violation("hearsay", who) for who in plan.hearsay if not _attributed(text, *_forms_of(plan, who))]

    # ---- 7. 场景的秘密（私奔、投神农帮）：计划与出处里没有，就是凭空泄露 ----
    out += [Violation("secret", w) for w in _unsourced(text, plan.source, plan.secrets)]
    return tuple(dict.fromkeys(out))


def _forms_of(plan: RenderPlan, who: str) -> tuple[str, ...]:
    return (who, *(f for f, n in plan.people if n == who and f != who))


def restated_hearsay(piece: str, text: str, plan: RenderPlan) -> tuple[Violation, ...]:
    """逐句查传闻：这一句把某条只闻其说的说法（说法里的每个实体都提到了）讲了出来，而到这句为止的正文（text）里
    还没有说话者的言语归属，这句就是叙述者在替传闻作保——流式交付前就丢，不等收尾。"""
    out: list[Violation] = []
    for who, entities in plan.claims:
        said = {n for _, n in _mentions(piece, set().union(*entities))}
        if entities and all(e & said for e in entities) and not _attributed(text, *_forms_of(plan, who)):
            out.append(Violation("hearsay", who))
    return tuple(dict.fromkeys(out))


# ============================================================
#  对白闸门：润色只能换说法，不能换内容
# ============================================================


def check_utterance(line: str, allowed: Iterable[str], universe: Iterable[str], plain: str,
                    subject_forms: Iterable[str]) -> tuple[Violation, ...]:
    """allowed：说话者、听者、话题主语与宾语的名字（与别称）；plain：模板说法（意图原文）；subject_forms：话题主语的可接受称呼。"""
    allowed = set(allowed)
    universe = set(universe) | allowed
    sourced = {n for _, n in _mentions(plain, universe)}
    out = [Violation("entity", n) for _, n in _mentions(line, universe) if n not in allowed and n not in sourced]
    out += [Violation("commitment", w) for w in _unsourced(line, plain, COMMITMENT_WORDS, COMMITMENT_EXCLUSIONS)]
    for status, words in STATUS_LEXICON.items():
        out += [Violation("status", f"{status}:{w}") for w in _unsourced(line, plain, words, STATUS_EXCLUSIONS)]
    forms = [f for f in subject_forms if f]
    if forms and not any(f in line for f in forms):
        out.append(Violation("subject", "/".join(forms)))
    return tuple(dict.fromkeys(out))


# ============================================================
#  分句：主持人之声逐句生成、逐句过闸门
# ============================================================

SENTENCE_ENDS = "。！？!?…"
QUOTE_OPEN = "“「"
QUOTE_CLOSE = "”」"
_CLOSERS = QUOTE_CLOSE + "’』）)"
_TRAILING = SENTENCE_ENDS + _CLOSERS
SPEECH_MARKS = "道说问答喝叫喊嚷骂叹笑吼斥应想"     # 言说动词的字：引语之前的“某某道/喝道/笑道/问/说/心想”
POST_MARKS = "道说问答喝叫喊嚷骂吼斥"               # 引语之后的归属只认这些（“满堂笑声”“你心想”不是归属）
POST_WINDOW = 16                                    # 引语之后的归属小句至多这么长（“你一边拱手一边连声赔笑道”）


def sentence_ends(text: str, final: bool = True) -> list[int]:
    """完整句子的结束位置（不含该位置）。句末标点只在引号外断句，其后紧跟的句末标点与收引号并入本句（“？！”“……”“。”」”）；
    引语以句末标点收尾时，紧跟其后的一小句若带言说动词（“你笑什么？”龚光杰喝道。），它是这段引语的归属，句子继续；
    否则引语就是一句（“……！”满堂哗然。）；换行总是断句。
    final=False（流还没完）：结尾处也许还会接着来标点、收引号或那一小句，先不算完，等下一段再定。"""
    ends: list[int] = []
    depth, ascii_open, i, n = 0, False, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            if text[ends[-1] if ends else 0:i].strip():
                ends.append(i)
            depth, ascii_open, i = 0, False, i + 1
            continue
        if ch in QUOTE_OPEN:
            depth += 1
        elif ch in QUOTE_CLOSE:
            depth = max(0, depth - 1)
        elif ch == '"' and depth > 0:
            depth -= 1                              # 全角开、半角收（“……！"）：当作收引号，不让一个错配吞掉后文
        elif ch == '"':
            ascii_open = not ascii_open             # 半角引号不分开合，按出现次序配对
        inside = depth > 0 or ascii_open
        closing = ch in QUOTE_CLOSE or ch == '"'
        quoted_end = closing and not inside and i > 0 and text[i - 1] in SENTENCE_ENDS
        if not quoted_end and not (ch in SENTENCE_ENDS and not inside):
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in _TRAILING:
            j += 1
        if j == n and not final:
            break                                   # 后面也许还有标点、收引号或正文：等下一段
        marked = any(c in SENTENCE_ENDS for c in text[i + 1:j])
        if quoted_end and not marked and j < n and not text[j].isspace() and text[j] not in QUOTE_OPEN:
            k = j
            while k < n and k - j < POST_WINDOW and text[k] not in _PUNCT and not text[k].isspace():
                k += 1
            if k == n and k - j < POST_WINDOW and not final:
                break                               # 引语后面那一小句还没写完：等下一段再定
            if _post_attribution(text, j, k):
                i = j                               # “……”龚光杰喝道：句子带着归属继续
                continue
        ends.append(j)
        i = j
    return ends


def _post_attribution(text: str, j: int, k: int) -> bool:
    """引语之后 [j, k) 这一小句是不是它的归属（“龚光杰喝道。”）；以冒号或开引号收尾的是下一段引语的引子（“钟灵笑道：“……””）。"""
    return any(c in POST_MARKS for c in text[j:k]) and text[k:k + 1] not in (*"：:", *QUOTE_OPEN, '"')


def _spans(text: str) -> list[tuple[int, int]]:
    cuts = [0, *sentence_ends(text), len(text)]
    return [(a, b) for a, b in zip(cuts, cuts[1:], strict=False) if b > a]


# ============================================================
#  引语的词法积木（台词闸门 quotes.py 与人事闸门 deeds.py 共用）：找出引语、遮住引语、切小句
# ============================================================

_CLAUSE = "，,；;：:、□ \t\n"
PRONOUNS: tuple[str, ...] = ("他们", "她们", "你们", "他", "她")
# 全角开引号遇到半角收引号也算收（“……！"），换行处没收的引语到此为止：一个错配的引号不该吞掉后面整段正文
_QUOTED = re.compile(r"“([^“”\"\n]*)(?:[”\"]|(?=\n)|$)|「([^「」\n]*)(?:」|(?=\n)|$)|\"([^\"\n]*)\"")
_BARE = re.compile(f"(?<=[{SPEECH_MARKS}])[：:](?!\\s*[“「\"])([^{re.escape(SENTENCE_ENDS)}\\n“”「」]+)")
# 无引号的转述：“干光豪低声说今夜要带她走”“你赔笑说小弟这就告辞”——说出口的内容同样是台词；
# “听说/据说/不说/说罢/说着/说时迟/说了些什么”之类不是转述
_INDIRECT = re.compile(r"(?<![听据虽再所传小解游劝演胡难不没别未分可有话按照如还])(?:说|告诉|提到|透露|低语)"
                       r"(?![：:“「\"了过话道起着完罢毕不出得清来一个些什声句明服笑的时])"
                       r"([^，,。！？!?\n“”「」：:；;][^。！？!?\n“”「」]+)")


@dataclass(frozen=True, slots=True)
class _Quote:
    start: int              # 开引号（无引号的“某某道：……”为冒号之后，转述为言说动词之后）
    end: int                # 收引号之后
    words: str
    indirect: bool = False  # “某某说……”式的转述：只归给具名的主语


def _quotes(text: str) -> list[_Quote]:
    found = [_Quote(m.start(), m.end(), next((g for g in m.groups() if g is not None), ""))
             for m in _QUOTED.finditer(text)]
    taken = [(q.start, q.end) for q in found]
    found += [_Quote(m.start() + 1, m.end(), m.group(1)) for m in _BARE.finditer(text)
              if not any(a <= m.start() < b for a, b in taken)]
    taken = [(q.start, q.end) for q in found]
    found += [_Quote(m.start(1), m.end(), m.group(1), indirect=True) for m in _INDIRECT.finditer(text)
              if not any(a <= m.start() < b or a <= m.start(1) < b for a, b in taken)]
    return sorted(found, key=lambda q: q.start)


def _mask(text: str, quotes: Sequence[_Quote]) -> str:
    """引语内容换成占位符（位置不变）：找说话者、找替玩家拿的主意、找状态落在谁身上，都只看引号之外。"""
    chars = list(text)
    for q in quotes:
        chars[q.start:q.end] = "□" * (q.end - q.start)
    return "".join(chars)


def _clauses(masked: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) 按逗号、分号、冒号与已遮住的引语切成小句。"""
    out: list[tuple[int, int]] = []
    start = lo
    for i in range(lo, hi + 1):
        if i == hi or masked[i] in _CLAUSE:
            if masked[start:i].strip():
                out.append((start, i))
            start = i + 1
    return out


# ============================================================
#  渲染结果：世界结算成功与文字生成成功分开记录
# ============================================================


class RenderStatus(StrEnum):
    TEMPLATE = "template"                # 没有模型（或无事可润色）：确定模板
    LLM = "llm"                          # 模型润色且通过闸门
    GATED_FALLBACK = "gated_fallback"    # 模型返回了文字，但被闸门拦下，回退模板
    LLM_UNAVAILABLE = "llm_unavailable"  # 模型调用失败，回退模板


@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    status: RenderStatus
    violations: tuple[Violation, ...] = field(default=())
    dropped: int = 0                     # 闸门丢掉的模型句子数（评测 G1 用；违规条数可能多于句数）
