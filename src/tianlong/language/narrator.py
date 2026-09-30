"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome / is_night，language/templates 的 Names / render_percept，
         language/llm 的 LLMClient / LLMUnavailable，language/scene 的 VoiceLine / SceneBrief / TextSink，
         language/render 的 fact_lines / build_plan / check / restated_hearsay / sentence_ends / Violation / Rendered / RenderStatus，
         language/deeds 的 check_deeds，language/quotes 的 check_quotes / voiced，language/lead 的 lead_line / restates，
         language/voice_prompt 的 system_prompt / scene_prompt / render_voice / lapse_line / CLOCK_ANY / TIMED（措辞层）
[OUTPUT]: 对外提供 Narrator（narrate_scene() 主持人之声：流式生成、逐句过闸门、通过即交付；narrate_rendered() / narrate()
          以空 SceneBrief 委托之；secrets 是场景的秘密词表）、MAX_DROPS、MAX_CHARS、LEAD_AFTER、lore_keys()、grams()（三字片段：复述与谈资说过没有都用它），
          再导出 render / voice_prompt 的 fact_lines()、render_voice()、SOCIAL_PHRASES / SOCIAL_LABELS 与 _when（= lapse_line）
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
       提示词只是请求，闸门才是验收：系统提示与用户提示由 voice_prompt 写成（第二人称、80~250 字、台词写成“某某道：“……””、
       不替玩家开口、停在钩子上、不写钟点；最近正文只留最后三段、每段末尾约 300 字，钟点换成时辰文字；世界前提与文风来自场景）；
       前后照应（SceneBrief：玩家的身体状况、出乎他意料之处、眼前的地点与人、说话者近来的经历）写进提示词且算出处；闲话只可说谈资、近来的经历与眼前的事；
       无事发生时有模型就写眼前的光景；到了结局收在余韵上；玩家的话没人接就照实写出没人接（不替他编答案）；
       回话先接住玩家的话头，谈资只给没说过的（会话记账），被问到的人附上说话者所知的来历；此地叫得出名字的东西（nearby）可以点名；
       四下看看而什么也没翻出来不是必讲之事（写出眼前的光景就是交代了）；等待还在同一个时辰里说过了多久（_when）；
       没有原话、没有说法、也不是回应玩家的闲话漏写了不补；近旁东西的别称同样可点名；结尾不替玩家列选项；
       玩家的姿态不是必讲之事；动手时顺口喝的一声（VoiceLine.act）可写可不写、没有模板；台词那句被丢，紧跟着的“她说完……”一并略过
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

from tianlong.core import Modality, Op, Outcome, Percept, is_night
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
from tianlong.language.voice_prompt import (
    CLOCK_ANY,
    SOCIAL_LABELS,
    SOCIAL_PHRASES,
    TIMED,
    render_voice,
    scene_prompt,
    system_prompt,
)
from tianlong.language.voice_prompt import lapse_line as _when

log = logging.getLogger(__name__)

__all__ = ["MAX_DROPS", "SOCIAL_LABELS", "SOCIAL_PHRASES", "Narrator", "fact_lines", "lore_keys", "render_voice"]

MAX_DROPS = 2              # 丢满这么多句就不再相信这段生成：停止读流，补上模板
MAX_CHARS = 600            # 交付满这么多字就停止读流（提示词要 80~250 字）：啰嗦的模型也有个头
ECHO = 0.5                 # 模型的一句与先声的三字片段重合过半：是在复述先声，悄悄略过（不算违规）
ECHO_WINDOW = 2            # 只在模型开头这么多句里找复述：后文再提到同一件事是正常的接续
LEAD_AFTER = 1.5           # 模型这么多秒还没交付一句，先声顶上（0 = 立即交付并告诉模型开头已写好；None = 不用先声）

_VISUAL = frozenset({Modality.SELF, Modality.SIGHT, Modality.SCENE})


def lore_keys(viewer: str, percepts: Sequence[Percept], lore: Mapping[str, str]) -> list[str]:
    """本回合真正映入眼帘、且有外观描写的实体：身处的地点、在场的人与物、眼前的通道与事件的参与者。
    外观只由亲眼所见触发——听人说起的、隔墙听见的、门那头的地点都只有名字（草图 seen=False），不描写。
    夜里优先取 "id@night" 变体（月下的玉璧不同于白日的玉璧）；观察者自己穿过一道有 "门@pass" 描写的通道（攀着藤萝落下断崖），
    也给出这段经过——下断崖不是走台阶，书生毫发无伤地“纵身一跃”读来不合理。"""
    keys: list[str] = []
    for p in percepts:
        ev = p.event
        if (p.modality == Modality.SELF and ev is not None and ev.actor == viewer and ev.kind == Op.MOVE.value
                and ev.outcome == Outcome.SUCCESS and f"{ev.obj}@pass" in lore and f"{ev.obj}@pass" not in keys):
            keys.append(f"{ev.obj}@pass")
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


def _with_lines(plan: RenderPlan, brief: SceneBrief, forms: Mapping[str, Sequence[str]] | None = None) -> RenderPlan:
    """会话交来的上下文本身有出处：台词的说话者与听者可以点名，原话、说法与说话者近来的经历算出处；
    前后照应（玩家的身体状况、本回合的意外与变化、眼前的地点与人、近旁叫得出名字的东西）同样算出处，其中的状态说出来不算升级，
    玩家自己与他以为身边的人身上的伤毒被制算落在对的人身上。forms 是名字 → 别称：近旁的东西说别称（“北冥神功”）同样可以。"""
    names = {n for vl in brief.lines for n in (vl.speaker_name, vl.listener_name) if n and n != "你"}
    names |= set(brief.present) | set(brief.nearby)
    also = frozenset(a for n in names for a in (forms or {}).get(n, ()))
    said = [x for vl in brief.lines for x in (vl.template, vl.claim) if x]   # 谈资、近来经历与来历只在他自己的引语里算数（台词闸门）
    said += [x for x in (brief.condition, *brief.notes, "、".join(brief.present)) if x]
    if not said and not names and not brief.statuses:
        return plan
    return replace(plan, names=plan.names | names, aliases=plan.aliases | also, source="\n".join([plan.source, *said]),
                   statuses=plan.statuses | brief.statuses, afflicted=(*plan.afflicted, *brief.afflicted))


_TALK = (Op.TELL.value, Op.ASK.value)
# 紧跟在一句台词后面、指着它说的话（“她说完拍着手直笑”“话音未落”）：台词被丢了，这句就没了着落
_AFTER_LINE = re.compile(r"^[^，。！？“”]{0,6}(?:说完|说罢|言罢|话音|话未说完|说着|一边说)")


@dataclass(frozen=True, slots=True)
class _Row:
    """模板的一行：事实清单的一行，或替 NPC 说出口的一句台词。"""
    text: str
    voice: VoiceLine | None = None
    keys: tuple[str, ...] = ()     # 事实行的参与者（名与别称，观察者除外）：正文提到其一即算讲到了
    must: bool = False             # 必须讲到：玩家自己的行动与后果、冲着玩家来的事、听见的话
    mine: bool = False             # 玩家自己的行动：先声已经讲过，收尾补模板时不再重复
    said: str = ""                 # 玩家自己的原话或姿态：正文照着写了出来（三字片段重合过半）也算讲到了


def _must(p: Percept, viewer: str, line: str = "") -> bool:
    ev = p.event
    if ev is None:
        return False
    if p.modality == Modality.SELF:
        if ev.kind == Op.INSPECT.value and ev.target == ev.place and "——" not in line:
            return False                    # 四下看看、什么也没翻出来：写出眼前的光景就是交代了，不必补“你仔细查看某处”
        return ev.kind != Op.WAIT.value         # 干等与姿态不必交代：玩家自己做的姿态他自己知道，漏写了不在钩子后面补一句
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
        if not vl.act:
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
            rows.append(_Row(text, keys=_keys(p, viewer, names, aliases, text), must=_must(p, viewer, text), mine=mine,
                             said=(p.event.utterance or "") if mine and p.event is not None else ""))
        else:
            rows.append(_Row(text))
    left = {k for ks in pending.values() for k in ks}
    rows += [_Row(render_voice(vl, salt), voice=vl) for k, vl in enumerate(brief.lines) if k in left and not vl.act]
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
            vl = r.voice                    # 没有原话、没有说法、也不是回应玩家的闲话：漏写了就算了，不补一句空洞的“某某打趣你”
            ok = _spoken(r, text, said) or vl.act or not (vl.template or vl.claim or vl.answering)
        elif r.must or (strict and r.keys):
            ok = (_named(text, r.keys) if r.keys else dropped == 0) or _echoes(r.said, text)
        else:
            ok = True
        if not ok:
            out.append(r)
    return out


def grams(text: str, n: int = 3) -> frozenset[str]:
    """去掉标点空白后的 n 字片段：判断复述、判断谈资说过没有，都用它。"""
    t = re.sub(r"[^\w]", "", text)
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def _echoes(said: str, text: str) -> bool:
    """玩家的原话或姿态在正文里照着写了出来：它的三字片段过半出现在正文里。"""
    mine = grams(said)
    return bool(mine) and len(mine & grams(text)) >= ECHO * len(mine)


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
        self._dropped_line = False              # 上一句是带台词被丢的：紧跟着的“她说完……”没了着落，一并略过

    def admit(self, lead: str, echo: Callable[[str], bool], sep: str = "") -> bool:
        """先声：内核结果写成的确定句子，照样过一遍闸门（不计丢句）；通过即交付（sep 接在后面：迟到的先声自成一段）。
        此后模型开头 ECHO_WINDOW 句里复述它的（三字片段重合过半，或 echo 判定是同一动作）悄悄略过，不算违规。"""
        if not lead or self.lead or self._found(lead, self.out.text + lead, self.out.text):
            return False
        self.out.emit(lead + sep)
        self.lead, self._lead_grams, self._echo = lead + sep, grams(lead), echo
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
            said = grams(body)
            if (said and len(said & self._lead_grams) / len(said) >= ECHO) or self._echo(body):
                log.info("模型复述了先声，略过: %s", body)
                return
        if found:
            self.dropped += 1
            self.violations += found
            self._dropped_line = "“" in body
            log.info("叙述句未通过闸门，丢弃: %s %s", body, found)
            return
        if self._dropped_line and _AFTER_LINE.match(body):
            self._dropped_line = False
            log.info("上一句台词被丢，接着它的这句也略过: %s", body)
            return
        self._dropped_line = False
        self.out.emit(piece)

    def _found(self, piece: str, text: str, before: str) -> list[Violation]:
        found = [v for v in check(text, self.plan, self.known, self.quoted) if v.kind != "hearsay"]   # 传闻有没有归属：收尾整段查
        found += restated_hearsay(piece, text, self.plan)                                  # 这一句替传闻作保：当场丢
        found += check_deeds(text, self.plan, self.known)
        found += check_quotes(text, self.brief, self.plan, self.known, command=self.command, since=len(before))
        found += [Violation("clock", m.group(0)) for m in CLOCK_ANY.finditer(piece)]
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
                      familiar: Iterable[str] = (), since: str = "") -> Rendered:
        """familiar 是玩家此前已知下落的东西（再翻出来不算“发现”）；command 是玩家原话（让“跳下断崖”读起来像跳，成败仍以清单为准）；lapse 是一段等待之后的时辰，排在事实之前；
        known 是闸门用来拒绝的名字全集（玩家认识的 + 场景全部实体）；on_text 收到每一段交付的文字（通过闸门即交付）；
        since 是这段等待开始时的时辰（还在同一个时辰里就说过了多久，不说“到了某时”）。"""
        known, familiar = frozenset(known), frozenset(familiar)
        looks = [self.lore[k] for k in fresh if k in self.lore]
        passed = [f"（不觉已是{lapse}）"] if lapse else []
        forms = {sk.name: tuple(self.aliases.get(eid, ())) for eid, sk in names.items()}
        plan = _with_lines(build_plan(viewer, percepts, names, show_scene, looks, "".join(passed), self.aliases,
                                      self.secrets, familiar), brief, forms)
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
        prompt = scene_prompt(brief, command, [_when(lapse, since)] if lapse else [], facts, looks, plan, known, gate.lead)
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
        when = [_when(lapse, since)] if lapse else []
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
            tail = ([] if TIMED.search(out.text) else when) + [r.text for r in untold]
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
    #  读流（提示词的措辞在 language/voice_prompt）
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
        return system_prompt(self.setting, self.style)
