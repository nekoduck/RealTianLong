"""
[INPUT]: 依赖 core 的 Percept / Modality / Outcome / Op / Rel / clock_label / is_night，scenarios 的 Beat / Scenario，
         language/scene 的 SceneBrief / Sky / MOON_*，language/voice_prompt 的 in_words（钟点换成时辰文字），persistence 的 TurnEnvelope
[OUTPUT]: 对外提供 recognize()（玩家本回合感知到了哪些看点）、stops_wait()（等待该不该在这个 tick 停下）、witnessed()（B1：看点编号集合）、
          lore_at()（外观描写按场景的时刻换成那一刻的样子：月出之后取 "id@moon"）、sky()（此刻的天色）、
          Staging / staging()（本回合的看点、写法卡、景观、许可词、天色与细节）、dress()（把它们交给 SceneBrief）
[POS]: runtime 的看点识别与舞台调度：真相只用于呈现——这里只读玩家自己的感知（已结算的事件、环顾所见），从不读世界真相，
       也从不进入任何 NPC 的 Situation（只由会话与评测脚本调用）。识别器由场景给出（Scenario.beats）：
       事件看点按行动/施动者/对象/物件/门/地点/言语行为/原因/得手后的状态/原话或姿态里的字匹配（缺省只认成功的）；景观看点（lore）只看环顾：
       玩家在该处、时钟已到、看得见那件陈设。once 的看点一局只让等待停一次（已停过的记在会话运行态 staged 里）。
       景观看点认得出的那一刻，会话经 lore_at() 交付的正是那段描写（月下玉璧的舞剑人影），看点与正文不相矛盾。
       staging() 给叙述者的一切都只在“真发生了、玩家也感知到了”时出现：事件看点须是成功的（没得手的一掌不是看点），
       景观看点须是那段描写本回合初次交付（lore_at 与 env.fresh 是唯一的一处换景：景观不另起第二套）；本回合的看点取认出的
       识别器里在场景表中最靠后的那个（表按剧情推进排）；写法卡来自认出的看点（Beat.stage）与卡自己的识别器（Card.cues）；
       细节：玩家这一步查看某处或某件陈设（INSPECT 成功），给出它的卡组里下一条还没给过的（facets 是一幕里已给过的细节原文，
       会话记账、随运行态往返），一回合至多一条；天黑透了、月亮还没出来时不给（看不出来）。天色 sky() 只在场景有月出时刻时给
       （旧版 None：天色不查）：夜按内核的夜（is_night），月亮在月出之前 none、月出那个 tick 所在的回合 rising、之后 up；
       玩家在看不见天的地方（Scenario.enclosed：石洞、石室）文字只报时辰、不提日月，夜与月相照旧交给审计
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace

from tianlong.core import Modality, Op, Outcome, Percept, Rel, clock_label, is_night
from tianlong.language.scene import MOON_NONE, MOON_RISING, MOON_UP, SceneBrief, Sky
from tianlong.language.voice_prompt import in_words
from tianlong.persistence import TurnEnvelope
from tianlong.scenarios import Beat, Scenario


def _from(beat: Beat, moments: Mapping[str, int]) -> int | None:
    """看点的时钟下限：整数照用，字符串按场景的时刻换算（场景没有这个时刻就永不成立）。"""
    if beat.clock_from is None or isinstance(beat.clock_from, int):
        return beat.clock_from
    return moments.get(beat.clock_from, 1 << 60)


def _where(p: Percept, player: str) -> str | None:
    return next((str(f.prop.value) for f in p.facts if f.holds and f.prop.subject == player
                 and f.prop.predicate == Rel.AT.value), None)


def matches(beat: Beat, p: Percept, player: str, moments: Mapping[str, int] | None = None) -> bool:
    """这条感知是否就是这个看点：景观只看环顾，事件只看感知到的那一件（看不清是谁的响动不算）。"""
    floor = _from(beat, moments or {})
    if floor is not None and p.tick < floor:
        return False
    if beat.lore:
        eid = beat.lore.split("@", 1)[0]
        return (p.modality == Modality.SCENE and (not beat.place or _where(p, player) in beat.place)
                and any(f.holds and f.prop.subject == eid for f in p.facts))
    ev = p.event
    if ev is None or p.modality == Modality.SCENE or ev.actor is None or beat.words not in (ev.utterance or ""):
        return False
    if beat.status and not any(f.holds and f.prop.is_attr and f.prop.attr_key == beat.status for f in p.facts):
        return False
    door_ok = beat.door is None or ev.obj == beat.door or any(f.prop.subject == beat.door for f in p.facts)
    return ((beat.op is None or ev.kind == beat.op.value) and (not beat.actors or ev.actor in beat.actors)
            and (not beat.target or ev.target in beat.target) and (beat.obj is None or ev.obj == beat.obj)
            and door_ok and (not beat.place or ev.place in beat.place)
            and (beat.social is None or ev.social == beat.social) and (beat.reason is None or ev.reason == beat.reason)
            and (not beat.success or ev.outcome == Outcome.SUCCESS))


def recognize(beats: Iterable[Beat], percepts: Iterable[Percept], player: str, clock: int | None = None,
              moments: Mapping[str, int] | None = None) -> tuple[Beat, ...]:
    """玩家这些感知里认得出的看点，每个编号只留第一条（按感知的先后）。clock 只作没有时刻的感知的兜底。"""
    beats = tuple(beats)
    found: dict[str, Beat] = {}
    for p in percepts:
        if clock is not None and not p.tick:
            p = Percept(clock, p.modality, p.event, p.facts, p.scopes, p.sketches, p.informant)
        for b in beats:
            if b.key not in found and matches(b, p, player, moments):
                found[b.key] = b
    return tuple(found.values())


def stops_wait(beats: Iterable[Beat], percepts: Iterable[Percept], player: str,
               moments: Mapping[str, int] | None = None, staged: Collection[str] = ()) -> bool:
    """等待在玩家感知到看点的那个 tick 停下；once 的看点停过一次（staged 里有它）就不再为它停。"""
    return any(not (b.once and b.key in staged) for b in recognize(beats, percepts, player, moments=moments))


def lore_at(keys: Iterable[str], clock: int, lore: Mapping[str, str], moments: Mapping[str, int]) -> list[str]:
    """外观描写按场景的时刻换成那一刻的样子（plan §4.5）：时钟过了月出，"id@night" 或 "id" 有 "id@moon" 的取月下那段；
    穿过通道的经过（"门@pass"）与没有月出时刻的场景（旧版）原样不动。"""
    moon = moments.get("moon")
    out: list[str] = []
    for k in keys:
        base, _, variant = k.partition("@")
        if moon is not None and clock >= moon and variant in ("", "night") and f"{base}@moon" in lore:
            k = f"{base}@moon"
        if k not in out:
            out.append(k)
    return out


def witnessed(beats: Iterable[Beat], percepts: Iterable[Percept], player: str,
              moments: Mapping[str, int] | None = None) -> frozenset[str]:
    """B1：这些感知里玩家目击了哪些看点（编号集合）。"""
    return frozenset(b.key for b in recognize(beats, percepts, player, moments=moments))


# ============================================================
#  天色：月出之前没有月亮，月出那一回合才许写它升起，之后月已在天上；夜按内核的夜
# ============================================================


def sky(clock: int, moments: Mapping[str, int], start: int | None = None, enclosed: bool = False) -> Sky | None:
    """clock 时刻的天色；start 是这一回合开始时的时钟（月出那个 tick 落在 (start, clock] 里就是 rising）。
    场景没有月出时刻（旧版、仓库）返回 None：叙述不给天色，审计也不查。
    enclosed：玩家在看不见天的地方（石洞、石室）——文字只报时辰、不提日月，夜与月相照旧交给审计。"""
    moon = moments.get("moon")
    if moon is None:
        return None
    when, night = in_words(clock_label(clock)), is_night(clock)
    if clock >= moments.get("dawn", 1 << 60):
        out = Sky(f"{when}，天边已经泛白", night, MOON_UP)
    elif clock < moon:
        out = Sky(f"{when}，{'天已黑透' if night else '天色将暗未暗'}，月亮还没出来", night, MOON_NONE)
    elif start is not None and start < moon:
        out = Sky(f"{when}，月亮正从东边的山头后升起", night, MOON_RISING)
    else:
        out = Sky(f"{when}，月已升过东边的山头", night, MOON_UP)
    return replace(out, text=f"{when}，四下不见天日") if enclosed else out


# ============================================================
#  舞台调度：本回合的看点、写法卡、景观、许可词、天色与细节（只读玩家本回合的感知）
# ============================================================


@dataclass(frozen=True, slots=True)
class Staging:
    focus: str | None = None                  # 本回合的看点（玩家口吻）
    cards: tuple[str, ...] = ()               # 写法卡编号（Scenario.cards 的键）
    spectacle: tuple[str, ...] = ()           # 景观 lore 的键（"yubi@moon"）
    allowed: frozenset[str] = frozenset()     # 额外许可词：只许点名
    sky: Sky | None = None
    details: tuple[str, ...] = ()             # 这回合给出的细节原文（至多一条）


def _settled(beats: Iterable[Beat], env: TurnEnvelope, sc: Scenario, player: str) -> tuple[Beat, ...]:
    """已结算、玩家感知到的识别器：事件的须成功（叫阵里没得手的那一掌也不算），景观的须是那段描写本回合初次交付。"""
    found = recognize((b if b.lore else replace(b, success=True) for b in beats), env.percepts, player, moments=sc.moments)
    return tuple(b for b in found if not b.lore or b.lore in env.fresh)


def staging(scenario: Scenario, env: TurnEnvelope, facets: Collection[str] = (), here: str | None = None) -> Staging:
    """本回合交给叙述者的舞台调度。facets：一幕里已经给过的细节原文（会话记账）；here：玩家（自以为）此刻身在何处。"""
    player = scenario.player or ""
    rank = {b.key: i for i, b in enumerate(scenario.beats)}          # 同一个看点有几条识别器：按最后一条排
    beats = _settled(scenario.beats, env, scenario, player)
    cues = {k for k, card in scenario.cards.items() if card.cues and _settled(card.cues, env, scenario, player)}
    stage = [k for b in beats for k in b.stage] + [k for k in scenario.cards if k in cues]
    focus = max(beats, key=lambda b: rank[b.key]) if beats else None
    looked = [ev.target for p in env.percepts if (ev := p.event) is not None and p.modality == Modality.SELF
              and ev.actor == player and ev.kind == Op.INSPECT.value and ev.outcome == Outcome.SUCCESS
              and ev.target in scenario.details]
    end = env.ticks[-1] if env.ticks else env.start_clock
    now = sky(end, scenario.moments, env.start_clock, here in scenario.enclosed)
    dark = now is not None and now.night and now.moon == MOON_NONE          # 天黑透了、月亮还没出来：看不出细节
    fresh = [] if dark else [d for t in looked[:1] for d in scenario.details[t] if d not in facets][:1]
    return Staging(focus.gloss if focus is not None else None, tuple(dict.fromkeys(k for k in stage if k in scenario.cards)),
                   tuple(b.lore for b in beats if b.lore), frozenset(a for b in beats for a in b.allowed), now, tuple(fresh))


def dress(brief: SceneBrief, st: Staging, lore: Mapping[str, str]) -> SceneBrief:
    """把舞台调度交给 SceneBrief：景观键换成原文。什么都没有（旧版）时原样返回，提示词逐字不变。"""
    if st == Staging():
        return brief
    return replace(brief, focus=st.focus, cards=st.cards, spectacle=tuple(lore[k] for k in st.spectacle if k in lore),
                   details=st.details, allowed=st.allowed, sky=st.sky)
