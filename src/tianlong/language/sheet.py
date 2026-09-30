"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome / derive_seed，language/render 的 RenderPlan / SENTENCE_ENDS / QUOTE_CLOSE，
         language/scene 的 SceneBrief / VoiceLine，language/templates 的 Names / render_percept，language/voice_prompt 的 render_voice
[OUTPUT]: 对外提供 Row（模板的一行：事实行或台词）、ARRIVE（Row.op：来到眼前的走动）、节的名字 SELF / ANSWER / COME / CLASH / MOVE / SCENE、
          scene_rows()（事实清单换成模板各行：听见的言语换成台词，必讲与否、参与者一并记下）、Section（节目单的一节）、
          compose()（本回合要讲的事排成节目单）、voices()（要说出口的话按节目单排）、
          lines()（几行模板按节目单写出）、patches()（漏讲的必讲之事写成人话补句）、prose()（模板行读起来像句子）
[POS]: language 的节目单：叙述者（narrator）交给模型的事实清单、要说出口的话与模板回退都按它排。
       顺序：玩家这一步 → 冲着玩家的回答（回话的那人冲你说的话合成一节）→ 来到眼前的人（排在他动手之前）→ 按（施动者, 目标）
       合并的交手（同一对的几下合成一行“钟灵向龚光杰连出两下——龚光杰中了毒，又受了伤”，动手时顺口喝的一声挂在对应的交手上；
       带着录入原话的动作不是吆喝——那句话真说了，与冲着玩家的录入原话一样照台词排、模板照印）→
       离开的人（离开玩家所在地、此后玩家没挪地方、那人也没回来的必讲：人不会凭空消失）→ 景物；玩家换了地方就在“你来到……”处
       分段，原处见到的事排在前面。补句按 (措辞, salt) 派生的种子轮换说法，相同输入永远得到相同文字
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tianlong.core import Modality, Op, Outcome, Percept, derive_seed
from tianlong.language.render import QUOTE_CLOSE, SENTENCE_ENDS, RenderPlan
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.language.templates import Names, render_percept
from tianlong.language.voice_prompt import render_voice

SELF, ANSWER, COME, CLASH, MOVE, SCENE = "self", "answer", "come", "clash", "move", "scene"     # 节目单的节，按此先后
_ORDER = (SELF, ANSWER, COME, CLASH, MOVE, SCENE)
ARRIVE = "arrive"      # Row.op：来到观察者眼前的走动（玩家自己到了新地方也是）
_TALK = (Op.TELL.value, Op.ASK.value)


# ============================================================
#  模板的各行：事实清单的一行，或替 NPC 说出口的一句台词
# ============================================================


@dataclass(frozen=True, slots=True)
class Row:
    """模板的一行：事实清单的一行，或替 NPC 说出口的一句台词。"""
    text: str
    voice: VoiceLine | None = None
    keys: tuple[str, ...] = ()     # 事实行的参与者（名与别称，观察者除外）：正文提到其一即算讲到了
    must: bool = False             # 必须讲到：玩家自己的行动与后果、冲着玩家来的事、听见的话、离开玩家所在地的人
    mine: bool = False             # 玩家自己的行动：先声已经讲过，收尾补模板时不再重复
    said: str = ""                 # 玩家自己的原话或姿态：正文照着写了出来（三字片段重合过半）也算讲到了
    pair: tuple[str, str] = ("", "")   # (施动者, 目标) 的称呼（观察者是“你”）：节目单据此合并交手、挂上吆喝
    op: str = ""                   # 事件的操作（attack / move / ……）；所见清单是 "scene"，来到眼前的走动是 "arrive"


def _must(p: Percept, viewer: str, line: str = "") -> bool:
    ev = p.event
    if ev is None:
        return False
    if p.modality == Modality.SELF:
        if ev.kind == Op.INSPECT.value and ev.target == ev.place and "——" not in line:
            return False                    # 四下看看、什么也没翻出来：写出眼前的光景就是交代了，不必补“你仔细查看某处”
        return ev.kind != Op.WAIT.value         # 干等与姿态不必交代：玩家自己做的姿态他自己知道，漏写了不在钩子后面补一句
    return p.modality == Modality.SPEECH or (p.modality == Modality.SIGHT and ev.target == viewer)


def _arrives(p: Percept, viewer: str) -> bool:
    """来到观察者眼前：玩家自己走到了新地方，或旁人走进玩家所在之处。"""
    ev = p.event
    return (ev is not None and ev.kind == Op.MOVE.value and ev.outcome == Outcome.SUCCESS
            and (ev.actor == viewer or ev.place == ev.target))


def _leaves(p: Percept, viewer: str, came: Sequence[tuple[str, int]]) -> bool:
    """有人离开玩家所在之处（此后玩家自己没挪地方、那人也没回来）：必讲——漏写了，那人就在正文里凭空消失了；
    先走后回的不算，免得补句替还在眼前的人说“转身走了”。"""
    ev = p.event
    return (p.modality == Modality.SIGHT and ev is not None and ev.kind == Op.MOVE.value and ev.actor != viewer
            and ev.outcome == Outcome.SUCCESS and ev.place != ev.target
            and not any(a in (viewer, ev.actor) and t >= p.tick for a, t in came))


def _row(p: Percept, viewer: str, names: Names, aliases: Mapping[str, Sequence[str]], line: str,
         came: Sequence[tuple[str, int]]) -> Row:
    """事实行：参与者之外，这一行写出来的后果与所见里的东西（“发现蒲团上藏着北冥神功帛卷……”）也算——正文写到帛卷就是讲到了。"""
    ev = p.event
    table = {sk.id: sk.name for sk in p.sketches} | {eid: sk.name for eid, sk in names.items()}
    ids = [x for x in ((ev.actor, ev.target, ev.obj) if ev is not None else ()) if x and x != viewer]
    ids += [f.prop.subject for f in p.facts if f.prop.subject != viewer and table.get(f.prop.subject, "\0") in line]
    keys = tuple(dict.fromkeys(n for x in ids for n in (table.get(x), *aliases.get(x, ())) if n))
    if ev is None:                                  # 所见清单与只记下事实的感知
        return Row(line, keys=keys, op=SCENE if p.modality == Modality.SCENE else "")

    def call(x: str | None) -> str:
        return "你" if x == viewer else table.get(x, "") if x else ""
    mine = p.modality == Modality.SELF and ev.actor == viewer
    return Row(line, keys=keys, must=_must(p, viewer, line) or _leaves(p, viewer, came), mine=mine,
               said=(ev.utterance or "") if mine else "", pair=(call(ev.actor), call(ev.target or ev.obj)),
               op=ARRIVE if _arrives(p, viewer) else ev.kind)


def scene_rows(plan: RenderPlan, viewer: str, percepts: Sequence[Percept], names: Names, brief: SceneBrief,
               salt: str, aliases: Mapping[str, Sequence[str]], familiar: frozenset[str] = frozenset()) -> list[Row]:
    """事实清单里听见的 NPC 言语换成带言语行为的台词（同一说话者按先后、原话相同的优先对应），清单里对不上的台词补在最后；
    只看见在耳语、没听见内容的那一行照旧留着。"""
    pending: dict[str, list[int]] = {}
    for k, vl in enumerate(brief.lines):
        if not _shout(vl):
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
    came = [(p.event.actor, p.tick) for p in percepts if p.event is not None and _arrives(p, viewer)]
    rows: list[Row] = []
    for text in plan.lines:
        who, said = spoken.get(text, (None, None))
        if who is not None and pending.get(who):
            queue = pending[who]
            k = next((k for k in queue if brief.lines[k].template == said), queue[0])
            queue.remove(k)
            rows.append(_voiced(brief.lines[k], salt))
        elif (p := facts.get(text)) is not None:
            rows.append(_row(p, viewer, names, aliases, text, came))
        else:
            rows.append(Row(text))
    left = {k for ks in pending.values() for k in ks}
    return rows + [_voiced(vl, salt) for k, vl in enumerate(brief.lines) if k in left and not _shout(vl)]


def _shout(vl: VoiceLine) -> bool:
    """动手时顺口喝的一声（可说可不说、没有模板）；带着录入原话的动作不算——那句话真说了，照录、模板照印。"""
    return bool(vl.act) and not vl.said


def _voiced(vl: VoiceLine, salt: str) -> Row:
    return Row(render_voice(vl, salt), voice=vl, pair=(vl.speaker_name, vl.listener_name or ""), op=vl.op)


# ============================================================
#  节目单：玩家这一步 → 冲着玩家的回答 → 来到眼前的人 → 按对合并的交手 → 离开的人 → 景物
# ============================================================

_TIMES = "两三四五六七八九"


@dataclass(frozen=True, slots=True)
class Section:
    """节目单的一节：同一件事的几行（同一对的几下交手、回话那人冲你说的几句），以及挂在上面的吆喝。"""
    kind: str
    rows: tuple[Row, ...]
    pair: tuple[str, str] = ("", "")
    shouts: tuple[VoiceLine, ...] = ()

    @property
    def lines(self) -> tuple[str, ...]:
        """模板回退的各行：同一对的几下交手合成一行，其余一行一句。"""
        return _merged(self.rows, self.kind != SELF)

    @property
    def facts(self) -> tuple[str, ...]:
        """交给模型的事实行：台词另列在“要说出口的话”里，不在这里。"""
        return _merged(tuple(r for r in self.rows if r.voice is None), self.kind != SELF)

    @property
    def voices(self) -> tuple[VoiceLine, ...]:
        return (*(r.voice for r in self.rows if r.voice is not None), *self.shouts)


def _merged(rows: tuple[Row, ...], merge: bool) -> tuple[str, ...]:
    hits = [i for i, r in enumerate(rows) if merge and r.voice is None and r.op == Op.ATTACK.value]
    if len(hits) < 2:
        return tuple(r.text for r in rows)
    volley = _volley([rows[i] for i in hits])
    return tuple(volley if i == hits[0] else r.text for i, r in enumerate(rows) if i == hits[0] or i not in hits)


def _volley(hits: Sequence[Row]) -> str:
    """同一对的几下交手合成一行：“看见钟灵向龚光杰连出两下——龚光杰中了毒，又受了伤”（落空的那下照实说）。"""
    a, t = hits[0].pair
    tails: list[str] = []
    for r in hits:
        if "——" in r.text:
            tail = r.text.split("——", 1)[1]
        elif "，但没有成功" in r.text:
            tail = "一下落了空" + r.text.split("，但没有成功", 1)[1]
        else:
            continue
        if t and tail.startswith(t) and any(x.startswith(t) for x in tails):
            tail = "又" + tail[len(t):]
        tails.append(tail)
    n = _TIMES[len(hits) - 2] if len(hits) - 2 < len(_TIMES) else "数"
    head = "看见" if hits[0].text.startswith("看见") else ""
    return f"{head}{a}向{t}连出{n}下" + ("——" + "，".join(tails) if tails else "")


def _kind(r: Row, answering: set[tuple[str, str]], seen: set[str]) -> str:
    if r.mine:
        return SELF
    if r.op == ARRIVE:
        return MOVE if r.pair[0] in seen else COME      # 来到眼前的人排在他动手之前；先走后回的，回来仍排在走之后
    if r.op in (SCENE, MOVE):
        return r.op
    return ANSWER if r.pair in answering else CLASH


def compose(rows: Sequence[Row], brief: SceneBrief) -> tuple[Section, ...]:
    """本回合要讲的事排成节目单：玩家这一步 → 冲着玩家的回答（回话那人冲你说的话合成一节）→ 来到眼前的人 →
    按（施动者, 目标）合并的交手（动手时顺口喝的一声挂在对应的交手上）→ 离开的人 → 景物；同一节里按发生先后。
    玩家换了地方，就在“你来到……”处分段：在原处见到的事排在前面，免得读来像发生在新地方；段与段之间按先后。"""
    answering = {r.pair for r in rows if r.voice is not None and (r.voice.answering or r.voice.said) and r.pair[1] == "你"}
    groups: dict[tuple[int, str, object], list[Row]] = {}
    seg, seen = 0, set()
    for r in rows:
        if r.mine and r.op == ARRIVE and groups:
            seg, seen = seg + 1, set()
        kind = _kind(r, answering, seen)
        key = (seg, kind, None) if kind in (SELF, SCENE) else (seg, kind, r.pair if r.pair[0] else r.text)
        groups.setdefault(key, []).append(r)
        seen.update(x for x in r.pair if x)
    shouts = [vl for vl in brief.lines if _shout(vl)]
    out: list[Section] = []
    for (_, kind, key), group in sorted(groups.items(), key=lambda g: (g[0][0], _ORDER.index(g[0][1]))):
        pair = key if isinstance(key, tuple) else ("", "")
        hung = tuple(vl for vl in shouts if kind == CLASH and (vl.speaker_name, vl.listener_name) == pair)
        shouts = [vl for vl in shouts if vl not in hung]       # 同一对在两段里都交了手：吆喝只挂在头一节
        out.append(Section(kind, tuple(group), pair, hung))
    return tuple(out)


def voices(sheet: Sequence[Section], brief: SceneBrief) -> tuple[VoiceLine, ...]:
    """要说出口的话按节目单排：回答在前，吆喝挨着它的交手；节目单外的（对不上交手的吆喝）照旧排在后面。"""
    ordered = [vl for s in sheet for vl in s.voices]
    return (*ordered, *(vl for vl in brief.lines if not any(vl is x for x in ordered)))


def lines(rows: Sequence[Row], brief: SceneBrief) -> list[str]:
    """几行模板按节目单写出（同一对的交手合成一行）。"""
    return [x for s in compose(rows, brief) for x in s.lines]


# ============================================================
#  补句：漏讲的必讲之事写成人话，插在钩子之前
# ============================================================

_SEEN = ("看见", "听见", "听到")
_LINK = ("这当口，", "与此同时，", "其间，", "")
_LEAVE = ("{a}转身往{t}去了。", "{a}没再多留，朝{t}走了。", "{a}径自往{t}去了。")
_GONE = ("{a}转身走了。", "{a}没再多留，走开了。")


def prose(line: str) -> str:
    """模板行交给玩家时读起来像句子：没有句末标点的补个“。”；整行括注（外观描写、时辰）照旧。"""
    line = line.rstrip()
    if not line or line[-1] in SENTENCE_ENDS + QUOTE_CLOSE + "’』\"" or (line[0] == "（" and line[-1] == "）"):
        return line
    return line + "。"


def _pick(options: tuple[str, ...], text: str, salt: str) -> str:
    return options[derive_seed("patch", text, salt) % len(options)]


def _human(line: str, salt: str) -> str:
    """“看见钟灵猛地向龚光杰出手——龚光杰受了伤” → “与此同时，钟灵猛地向龚光杰出手——龚光杰受了伤。”"""
    pre = next((x for x in _SEEN if line.startswith(x)), None)
    if pre is None:
        return prose(line)
    body = line[len(pre):]
    return prose(_pick(_LINK, body, salt) + body)


def patches(missing: Sequence[Row], salt: str) -> list[str]:
    """漏讲的必讲之事写成人话补句（按节目单排、同一对的交手合成一句）：离开的人写成“某某转身往某处去了”，
    看见听见的去掉“看见/听见”换个承接语，台词与玩家自己的行动照录；措辞按 (这一句, salt) 派生的种子轮换。"""
    out: list[str] = []
    for s in compose(missing, SceneBrief()):
        if s.kind == MOVE:
            for r in s.rows:
                a, t = r.pair
                leaving = r.must and a and not r.mine
                out.append(_pick(_LEAVE if t else _GONE, r.text, salt).format(a=a, t=t) if leaving
                           else _human(r.text, salt))
        else:
            out += [_human(x, salt) for x in s.lines]
    return out
