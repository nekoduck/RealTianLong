"""
[INPUT]: 依赖 core 的 Event / Fact / Kind / Modality / Op / Outcome / Percept / PerceivedEvent / Proposition / Rel / Social / WorldState /
         AddRelation / RemoveRelation / SetAttr / clock_label，core/attributes 的 true_value，core/profiles 的 Goal / GoalKind / Profile，
         cognition 的 BeliefStore / believed_place，kernel/perception 的 sketches_for，language/parser 的 MoveKind / Parsed，
         language/llm 的 LLMUnavailable，language/render 的 sentence_ends，
         language/scene 的 SceneBrief / VoiceLine，language/templates 的 SOCIAL_VERBS / render_event / render_experience / render_fact，
         persistence 的 TurnEnvelope，scenarios 的 Scenario，runtime/continuity 的 continuity / lately，runtime/talk 的 fresh
[OUTPUT]: 对外提供 gm_command()（元指令与“GM：”前缀）、companions()（玩家的同伴 = 自己人 + DEFEND 目标）、salient()（等待该不该被打断）、
          build_brief()（SceneBrief：要替 NPC 说出口的话——谈资只给没说过的、被问到的人附上来历——+ 前后照应 + 没人接的话 + 是否收幕）、
          self_view() / goal_text() / aside_prompt()（场外问答只用玩家自己的认知）、gated_stream()（场外回答逐句过名字闸门、边生成边交付）、
          closing_prompt()（终章只取玩家亲历）、leaked() / leaks()（名字闸门：玩家不认识的实体不许出现在模型写的文字里，玩家亲口说出的名字除外）、
          reveal()（终章的真相揭晓表）、is_ooc() / asks_direction()（场外还是故事里的自问、问没问方向）、
          META_HELP / ASIDE_SYSTEM（场外）/ ASIDE_INNER（故事里的自问：故事口吻、不标场外）/ ASIDE_TOKENS / CLOSING_SYSTEM
[POS]: runtime 的主持层纯函数：会话（session）的回合循环调用它们，它们只读传进来的认知、已落库的请求进度与事件日志，从不写任何东西
       （gated_stream 只经调用方给的回调交付文字）。
       给模型看的（场外问答、终章收束）只取玩家自己的认知与亲历；真相只出现在 reveal() 里——落幕之后、明确标作“真相”，
       按世界状态与玩家认知确定地生成（你以为的 vs 实际的：每个 NPC 的下落与伤/毒/被制，以及你没看见的动手、偷盗与潜逃）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

from tianlong.cognition import BeliefStore
from tianlong.cognition.navigation import believed_place
from tianlong.core import (
    AddRelation,
    Event,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    RemoveRelation,
    SetAttr,
    Social,
    WorldState,
    clock_label,
)
from tianlong.core.attributes import true_value
from tianlong.core.memories import MemoryRecord
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel.perception import sketches_for
from tianlong.language.llm import LLMUnavailable
from tianlong.language.parser import MoveKind, Parsed
from tianlong.language.render import sentence_ends
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.language.templates import SOCIAL_VERBS, render_event, render_experience, render_fact
from tianlong.persistence import TurnEnvelope
from tianlong.runtime.continuity import continuity, lately
from tianlong.runtime.talk import fresh
from tianlong.scenarios import Scenario

TALK = frozenset({Op.TELL.value, Op.ASK.value})
META_HELP = "可用的指令：/hint 提示、/recap 前情回顾、/beliefs 你所知道的；场外提问请以“GM：”开头。"
_META_WORDS = {"hint": ("hint", "提示"), "recap": ("recap", "回顾", "前情"), "beliefs": ("beliefs", "认知", "所知")}
_GM_PREFIX = re.compile(r"^\s*(?:gm|ooc)\s*[:：]\s*", re.IGNORECASE)
_GOALS = {GoalKind.PROTECT: "护住{item}", GoalKind.ACQUIRE: "弄到{item}", GoalKind.DELIVER: "把{item}交给{recipient}",
          GoalKind.GUARD: "守住{home}", GoalKind.HOSTILE: "对付{person}", GoalKind.ESCAPE: "设法去往{home}",
          GoalKind.DEFEND: "护着{person}"}
_STATUS = {"wounded": "受了伤", "poisoned": "中了毒", "subdued": "被点了穴道"}
MAX_UNSEEN = 20      # 真相揭晓里“你没看见的事”至多列这么多条
ASIDE_TOKENS = 200   # 场外问答只要两三句：给声音模型的输出上限

ASIDE_SYSTEM = (
    "你是一部中文武侠文字游戏的主持人，此刻在场外回答玩家的问题。只能依据给出的玩家自己的认知、目标、提示与最近的正文作答——"
    "那是他自己以为的，不一定是真相；你不知道、也不许猜测他不知道的人、物、地方与事。不剧透，不替他做决定，"
    "第二人称，两三句，不写钟点。"
)
ASIDE_INNER = (
    "你是一部中文武侠文字游戏的主持人。玩家在故事里自问了一句（不是跳出故事问你）：用故事里的口吻、第二人称，"
    "像他自己心里盘算、低头打量那样答两三句。只能依据给出的玩家自己的认知、目标、提示与最近的正文——那是他自己以为的，"
    "不一定是真相；不许说他“听说过”没人告诉过他的事，不许提他不知道的人、物、地方与事。不剧透，不替他做决定，不写钟点。"
)
# 故事里的自问要不要给提示：问到“怎么办、往哪走”才给，问身上有什么、这是哪里就只答所问
_DIRECTION = re.compile(r"怎么办|怎么做|该做什么|做什么好|往哪|去哪|怎么走|下一步|接下来|出路|怎么出去|提示|线索")
CLOSING_SYSTEM = (
    "你是一部中文武侠文字游戏的主持人。这一幕已经落幕，请用第二人称写一段 120~250 字的终章收束。"
    "只能取材于列出的玩家亲历之事，按给定的基调收束；不得添加玩家不知道的人物、事件或结论，不预告后事，不写钟点。"
)


# ============================================================
#  输入：元指令与场外问题（规则解析器不认它们；主持层解释器并入之前由这里兜住）
# ============================================================


def is_ooc(text: str) -> bool:
    """玩家明说了“GM：/OOC：”，是跳出故事问主持人；其余的问话（“我该怎么办？”）是故事里的自问。"""
    return _GM_PREFIX.match(text) is not None


def asks_direction(question: str) -> bool:
    return _DIRECTION.search(question) is not None


def gm_command(text: str) -> Parsed | None:
    raw = text.strip()
    if raw[:1] in ("/", "／"):
        body = raw[1:].strip()
        word = body.split(maxsplit=1)[0].lower() if body else ""
        key = next((k for k, words in _META_WORDS.items() if word in words), None)
        return (Parsed(None, kind=MoveKind.META, question=key) if key
                else Parsed(None, clarification=META_HELP, kind=MoveKind.UNCLEAR))
    gm = _GM_PREFIX.match(raw)
    if gm is not None:
        return Parsed(None, kind=MoveKind.ASK_GM, question=raw[gm.end():].strip() or raw)
    return None


# ============================================================
#  等待只被要紧的事打断
# ============================================================


def companions(prof: Profile) -> frozenset[str]:
    """玩家的同伴：自己人（allies）加上他要护着的人（DEFEND 目标）——与 NPC 的“替谁出头”同一口径。"""
    return frozenset(prof.allies) | {g.person for g in prof.goals if g.kind == GoalKind.DEFEND and g.person}


def salient(percepts: Iterable[Percept], player: str, friends: Iterable[str], here: str | None) -> bool:
    """冲着我来的（对我说话、对我动手、给我东西、搜我的身）、我自己的处境变了、当着我的面公开说话或做出有所指的姿态、
    当着我的面动手（打的是谁都算）、有人对我的同伴（friends，见 companions()）动手、有人进出我所在的地方。
    旁人的例行举动（查看、拿放）、远处的响动、耳语都不算。"""
    friends = frozenset(friends)
    for p in percepts:
        ev = p.event
        if ev is None or ev.actor == player or p.modality == Modality.SCENE:
            continue
        if ev.kind in TALK and ev.topic is None and ev.social in (Social.REMARK, Social.JOKE):
            continue                     # 闲扯（说笑、随口一句）不打断等待：话多的人不该让“等到天黑”原地打转
        if ev.target == player or any(player in (f.prop.subject, f.prop.value) for f in p.facts):
            return True
        if ev.place == here and ((ev.kind in TALK and p.modality == Modality.SPEECH)
                                 or ev.kind == Op.ATTACK.value
                                 or (ev.kind == Op.WAIT.value and ev.utterance
                                     and ev.social not in (None, Social.REMARK))):
            return True
        if ev.kind == Op.ATTACK.value and ev.target in friends:
            return True
        if ev.kind == Op.MOVE.value and p.modality == Modality.SIGHT and ev.outcome == Outcome.SUCCESS:
            return True
    return False


# ============================================================
#  叙述上下文：只用已落库的请求进度与此刻各人的认知拼出
# ============================================================


def build_brief(env: TurnEnvelope, me: BeliefStore, scenario: Scenario, beliefs_of: Callable[[str], BeliefStore],
                recent: Sequence[str], before: BeliefStore | None = None, closing: bool = False,
                told: Mapping[str, Collection[int]] | None = None,
                memories_of: Callable[[str], Sequence[MemoryRecord]] | None = None,
                hooks: Sequence[str] = ()) -> SceneBrief:
    """要替 NPC 说出口的话（本回合玩家听见的每一句 NPC 言语、看见的每一个 NPC 姿态）、最近几段正文、玩家原话，
    以及前后照应（continuity：玩家自己的身体状况、本回合的意外与变化、身在何处身边有谁；出人意料地出现的人带上他近来的经历）。
    耳语（只看见在交谈、没听见内容）不算：玩家没听见的话，叙述者也不该替它编出来。
    answering：NPC 冲着玩家、且在玩家开口（或冲他摆了姿态、赔了罪道了谢）之后说的话，带上玩家这一步作回话的由头。
    before 是本回合之前玩家的认知（重试补写时没有）；closing 表示这是这一幕的最后一段；
    told 是谈资账本（每个 NPC 已经说过的谈资条目）：只把还没说过的交给叙述者；memories_of 取某人自己的经历记录
    （早先是谁制住了谁、他挨过的那一下，短期经历里早已滚掉）；hooks 是行动建议，只在玩家干等、身边没人说话时交给叙述者。
    玩家问到的人（原话里点了名、说话者认识的），附上说话者所知的公开来历；玩家冲着谁说了话、他本回合却没接话，记在 unanswered；
    普通等待被身边的事打断，前后照应里添一句“你本想再等下去”。"""
    player = me.owner
    prof = scenario.profiles.get(player)
    recall = memories_of or (lambda _: ())
    ctx = continuity(env, me, before, companions(prof) if prof else (), recall(player))
    plan = (env.intent, *env.followups)
    mine = next((it for it in plan if it.utterance and it.op in (Op.TELL, Op.ASK, Op.WAIT)), None)
    # 玩家本回合冲人说的话、做的姿态或只有言语行为的客套（赔罪、道谢）：之后冲着他来的话都是在回应它
    acted = next((p for p in env.percepts if p.modality == Modality.SELF and p.event is not None
                  and p.event.actor == player and (p.event.kind in TALK or (p.event.kind == Op.WAIT.value
                                                                             and p.event.utterance))), None)
    cue = _cue(acted.event, me) if acted is not None else None
    lines = []
    for p in env.percepts:
        ev = p.event
        if ev is None or ev.actor in (None, player) or ev.actor not in scenario.profiles:
            continue
        talk = ev.kind in TALK and p.modality == Modality.SPEECH
        pose = ev.kind == Op.WAIT.value and p.modality == Modality.SIGHT and bool(ev.utterance)
        if talk or pose:
            answer = cue if ev.target == player and acted is not None and p.tick > acted.tick else None
            mind = beliefs_of(ev.actor)
            lines.append(_voice(ev, me, scenario, mind, answer,
                                lately(mind, ev.actor, p.tick, recall(ev.actor)) if ev.actor in ctx.newcomers else "",
                                (told or {}).get(ev.actor, ())))
    asked = (acted.event.target if acted is not None and acted.event.kind in TALK
             and acted.event.outcome == Outcome.SUCCESS else None)          # 话没说成（人不在）就谈不上没人接
    unanswered = None
    if asked and asked != player and asked in scenario.profiles and not any(
            vl.speaker == asked and vl.answering for vl in lines):
        sk = me.sketch(asked)
        unanswered = sk.name if sk is not None else None
    notes = ctx.notes
    if env.done and env.intent.op == Op.WAIT and not env.followups and not env.reaction \
            and len(env.ticks) < env.planned_ticks:
        notes = (*notes, "你本想再等下去，却被眼前的事打断了")
    return SceneBrief(tuple(lines), tuple(recent), mine.utterance if mine is not None else None,
                      condition=ctx.condition, notes=notes, present=ctx.present, statuses=ctx.statuses,
                      afflicted=ctx.afflicted, closing=closing, nearby=ctx.nearby, unanswered=unanswered, hurt=ctx.hurt,
                      hooks=tuple(hooks) if env.intent.op == Op.WAIT and not lines and not closing else ())


def _cue(ev: PerceivedEvent, me: BeliefStore) -> str | None:
    """玩家这一步给 NPC 的由头：说出口的原话；姿态写成“（你拱手作揖）”；只有言语行为的写成“（你向龚光杰赔不是）”。"""
    if ev.utterance:
        return ev.utterance if ev.kind in TALK else f"（你{ev.utterance}）"
    if ev.social is not None and ev.kind in TALK:
        who = me.sketch(ev.target).name if ev.target and me.sketch(ev.target) else "对方"
        return f"（你{SOCIAL_VERBS[ev.social].format(t=who)}）"
    return None


def _about(answering: str | None, who: str, mind: BeliefStore, scenario: Scenario) -> str:
    """玩家这句话里问到的人（名或别称；说话者自己除外），说话者认识的，附上他们的公开来历。"""
    if not answering:
        return ""
    rows = []
    for pid, prof in sorted(scenario.profiles.items()):
        sk = mind.sketch(pid)
        forms = (sk.name, *scenario.aliases.get(pid, ())) if sk is not None else ()
        if pid != who and prof.intro and any(f and f in answering for f in forms):
            rows.append(f"{sk.name}：{prof.intro}")
    return "；".join(rows)


def _voice(ev: PerceivedEvent, me: BeliefStore, scenario: Scenario, mind: BeliefStore,
           answering: str | None, lately_text: str = "", told: Collection[int] = ()) -> VoiceLine:
    """一句 NPC 言语：结构取自玩家的感知，说法与可点名的名字取自说话者自己的认知，腔调与谈资取自角色设定
    （谈资只给还没说过的），被问到的人附上说话者所知的来历。"""
    who = str(ev.actor)
    prof = scenario.profiles[who]
    listener = ("你" if ev.target == me.owner
                else me.sketch(ev.target).name if ev.target and me.sketch(ev.target) else None)
    claim = render_fact(ev.topic, mind.entities, who) if ev.topic is not None else None
    may = {sk.name for sk in mind.entities.values()}
    may |= {a for eid in mind.entities for a in scenario.aliases.get(eid, ())}
    sk = me.sketch(who)
    return VoiceLine(who, sk.name if sk else scenario.state.entity(who).name, listener, ev.kind, ev.social, claim,
                     ev.utterance, prof.voice, fresh(prof.knows, told), frozenset(may), answering, lately_text,
                     _about(answering, who, mind, scenario))


# ============================================================
#  场外问答与终章收束：模型只看玩家自己的认知与亲历
# ============================================================


def self_view(me: BeliefStore) -> list[str]:
    """玩家自己以为的处境：身在何处、身上带着什么、身边有谁、最近听到了什么（全部来自他自己的认知）。"""
    player = me.owner
    here = believed_place(me, player)
    rows = [f"你在{me.sketch(here).name}"] if here and me.sketch(here) else ["你不知道自己身在何处"]
    items = [sk.name for eid, sk in sorted(me.entities.items())
             if sk.kind == Kind.ITEM and me.location_of(eid) == player]
    people = [sk.name for eid, sk in sorted(me.entities.items())
              if sk.kind == Kind.PERSON and eid != player and here and believed_place(me, eid) == here]
    rows.append(f"身上带着{'、'.join(items)}" if items else "身上没带什么要紧的东西")
    rows.append(f"身边有{'、'.join(people)}" if people else "身边没有别人")
    heard = [render_experience(ep.modality, ep.event, me.entities, player, me="你")
             for ep in me.episodes if ep.modality == Modality.SPEECH][-3:]
    return rows + [f"听到过：{h}" for h in heard]


def goal_text(g: Goal, me: BeliefStore) -> str | None:
    """玩家的目标，用他认识的名字说；涉及他还不认识的人、物、地方就不提（免得借目标剧透）。"""
    refs = {"item": g.item, "home": g.home, "recipient": g.recipient, "person": g.person}
    if any(v is not None and not me.knows(v) for v in refs.values()):
        return None
    return _GOALS[g.kind].format(**{k: me.sketch(v).name if v else "" for k, v in refs.items()})


def aside_prompt(question: str, view: Sequence[str], persona: str, goals: Sequence[str], guide: Sequence[str],
                 recent: Sequence[str]) -> str:
    """guide 只给到下一条没给过的提示为止：更深的提示不进提示词，模型想剧透也无从剧透。"""
    parts = [f"玩家的问题：{question}", "玩家此刻的认知（他自己以为的）：\n" + "\n".join(f"- {v}" for v in view)]
    if persona:
        parts.append(f"玩家的身份：{persona}")
    if goals:
        parts.append("玩家的目标：" + "；".join(goals))
    if guide:
        parts.append("可以给的提示（由浅入深，至多用到最后一条）：\n" + "\n".join(f"{i}. {g}" for i, g in enumerate(guide, 1)))
    if recent:
        parts.append("最近的正文：\n" + "\n".join(f"【{i}】{t}" for i, t in enumerate(recent, 1)))
    return "\n\n".join(parts)


def gated_stream(pieces: Iterator[str], leaked_of: Callable[[str], Sequence[str]], emit: Callable[[str], None],
                 lead: str = "") -> tuple[str, list[str], bool]:
    """模型写的场外文字边生成边交付：逐句连同已交付的一起过 leaked_of()（名字闸门），通过即经 emit 交出（第一句前带上 lead）；
    一句不过即停、不再读流（后面的话多半接着它说）。模型中途失败同样停下，已交付的照旧算数，半句话不交付。
    返回 (交付的全部文字, 拦下那句点出的名字, 是否中途失败)；调用方据此决定补不补模板。"""
    parts: list[str] = []
    found: list[str] = []
    failed = False
    sentences = _sentences(pieces)
    try:
        for raw in sentences:
            body = raw.strip()
            if not body:
                continue
            gap = "\n" if parts and "\n" in raw[:len(raw) - len(raw.lstrip())] else ""
            piece = (gap if parts else lead) + body
            found = list(leaked_of("".join(parts) + piece))
            if found:
                break
            parts.append(piece)
            emit(piece)
    except LLMUnavailable:
        failed = True
    finally:
        sentences.close()
        close = getattr(pieces, "close", None)
        if close is not None:
            close()                                    # 提前收手：让流式连接及时关掉
    return "".join(parts), found, failed


def _sentences(pieces: Iterable[str]) -> Iterator[str]:
    """把流切成完整的句子（引号、省略号与流的边界同叙述者的分句器）；中途失败时先交出已完整的句子再抛出。"""
    buf, failed = "", None
    try:
        for chunk in pieces:
            buf += chunk
            cut = 0
            for end in sentence_ends(buf, final=False):
                yield buf[cut:end]
                cut = end
            buf = buf[cut:]
    except LLMUnavailable as e:
        failed = e
    cut = 0
    for end in sentence_ends(buf, final=True):
        yield buf[cut:end]
        cut = end
    if failed is not None:
        raise failed
    if buf[cut:].strip():
        yield buf[cut:]                                # 结尾那句没有句末标点也照样验收


def closing_prompt(tone: str, lived: Sequence[str], recent: Sequence[str]) -> str:
    parts = [f"终章基调：{tone}", "玩家亲历的事（按先后，第一人称记录）：\n" + "\n".join(f"- {t}" for t in lived)]
    if recent:
        parts.append("最近的正文：\n" + "\n".join(recent))
    return "\n\n".join(parts)


def leaked(text: str, me: BeliefStore, scenario: Scenario, said: str = "") -> list[str]:
    """名字闸门：模型写出的玩家不认识的实体名（名或两字以上的别称）。先抹掉玩家认识的名字，免得被子串误伤；
    said 是玩家自己的原话，他亲口说出的名字（只是那几个字，不连带同一实体的别的称呼）照样抹掉——复述它不算泄露。"""
    state = scenario.state

    def names(eid: str) -> list[str]:
        return [state.entity(eid).name, *(a for a in scenario.aliases.get(eid, ()) if len(a) >= 2)]

    known = {n for eid in me.entities if state.has_entity(eid) for n in names(eid)}
    known |= {n for eid in state.entities for n in names(eid) if said and n in said}
    for n in sorted(known, key=len, reverse=True):
        text = text.replace(n, "")
    return [n for eid in state.entities if eid not in me.entities for n in names(eid) if n in text]


def leaks(text: str, me: BeliefStore, scenario: Scenario, said: str = "") -> bool:
    return bool(leaked(text, me, scenario, said))


# ============================================================
#  真相揭晓：落幕之后才给玩家看，确定地由世界状态、事件日志与玩家认知生成
# ============================================================


def reveal(scenario: Scenario, st: WorldState, me: BeliefStore, events: Sequence[Event]) -> str:
    names = {sk.id: sk for sk in sketches_for(st, st.entities)}
    rows = ["—— 真相揭晓：你以为的 vs 实际的 ——"]
    for npc in scenario.npcs:
        rows.append(f"· {names[npc].name}：你以为{_thought(me, npc)}；实际{_actual(st, npc, names)}")
    unseen = _unseen(scenario, st, events, names)
    if unseen:
        more = len(unseen) - MAX_UNSEEN
        rows += ["—— 你没有看见的事 ——", *unseen[:MAX_UNSEEN]] + ([f"……另有 {more} 件"] if more > 0 else [])
    return "\n".join(rows)


def _thought(me: BeliefStore, npc: str) -> str:
    if not me.knows(npc):
        return "从没听说过此人"
    where = believed_place(me, npc)
    place = f"在{me.sketch(where).name}" if where and me.sketch(where) else "不知在哪"
    status = [w for k, w in _STATUS.items() if me.holds(Proposition.attr(npc, k, True))]
    return place + ("，" + "、".join(status) if status else "")


def _actual(st: WorldState, npc: str, names) -> str:
    where = st.target(npc, Rel.AT)
    status = [w for k, w in _STATUS.items() if true_value(st.entity(npc), k, st.clock)]
    return f"在{names[where].name if where in names else '某处'}，" + ("、".join(status) if status else "安然无恙")


def _unseen(scenario: Scenario, st: WorldState, events: Sequence[Event], names) -> list[str]:
    """玩家当时不在场的动手、偷盗、施用，以及潜逃者的去向。玩家在哪按事件日志里他自己的位移逐条推进（同一 tick 按结算顺序）。"""
    player = str(scenario.player)
    place = scenario.state.target(player, Rel.AT)
    escapers = {a for a, p in scenario.profiles.items() if any(g.kind == GoalKind.ESCAPE for g in p.goals)}
    fled: dict[str, int] = {}
    out: list[str] = []
    for e in events:
        seen = e.place == place or (e.op == Op.MOVE and e.intent.target == place)
        if e.actor != player and not seen and e.outcome != Outcome.REJECTED:
            theft = _theft(e, scenario.state)
            if e.op == Op.ATTACK or theft or (e.op == Op.USE and e.intent.target not in (None, e.actor)):
                where = names[e.place].name if e.place in names else "某处"
                text = theft or render_event(PerceivedEvent(e.op.value, e.place or "", e.actor, e.intent.target,
                                                            e.intent.obj, e.outcome, None, e.reason), names)
                after = _aftermath(e, names)
                out.append(f"· {clock_label(e.tick)}，{where}：{text}" + (f"（{after}）" if after else ""))
            elif e.op == Op.MOVE and e.outcome == Outcome.SUCCESS and e.actor in escapers:
                fled.setdefault(e.actor, e.tick)
        for c in e.changes:
            if isinstance(c, AddRelation) and c.rel.src == player and c.rel.type == Rel.AT:
                place = c.rel.dst
    for who, tick in sorted(fled.items(), key=lambda x: (x[1], x[0])):
        dest = st.target(who, Rel.AT)
        out.append(f"· {names[who].name}于{clock_label(tick)}悄悄动身，如今在{names[dest].name if dest in names else '某处'}")
    return out


def _theft(e: Event, base: WorldState) -> str | None:
    """从别人身上拿走东西（夺、搜走被制者之物）：变化里物件离开另一个人、落到行动者身上。"""
    gone = {c.rel.src: c.rel.dst for c in e.changes if isinstance(c, RemoveRelation) and c.rel.type == Rel.AT
            and c.rel.dst != e.actor and base.has_entity(c.rel.dst) and base.kind(c.rel.dst) == Kind.PERSON}
    for c in e.changes:
        if isinstance(c, AddRelation) and c.rel.type == Rel.AT and c.rel.dst == e.actor and c.rel.src in gone:
            name = lambda eid: base.entity(eid).name   # noqa: E731
            return f"{name(e.actor)}从{name(gone[c.rel.src])}身上拿走了{name(c.rel.src)}"
    return None


def _aftermath(e: Event, names) -> str:
    out = []
    for c in e.changes:
        if not isinstance(c, SetAttr) or c.entity not in names:
            continue
        who = names[c.entity].name
        if c.key == "wounded":
            out.append(f"{who}{'受了伤' if c.new else '伤势好转'}")
        elif c.key == "poisoned":
            out.append(f"{who}{'中了毒' if c.new else '毒解了'}")
        elif c.key == "subdued_until" and c.new and int(c.new) > e.tick:
            out.append(f"{who}被点了穴道")
    return "，".join(out)
