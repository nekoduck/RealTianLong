"""
[INPUT]: 依赖 tianlong.runtime 的 GameSession / gm / cli，tianlong.agents 的 Choice / Situation，tianlong.language 的 MoveKind / Parsed /
         Narrator / ScriptedLLM / LLMUnavailable / Rendered / Violation，tianlong.persistence 的 InMemoryWorldStore / RequestConflict，
         tianlong.scenarios 的 Scenario / Ending / build_wuliang，tianlong.core / kernel.perception 搭一个小世界
[OUTPUT]: 主持层回合循环验收：说话/姿态加一个反应 tick 且 NPC 的回话落在同一回合的感知与 SceneBrief 里；多步计划逐 tick 执行、
          失败即止而反应 tick 照旧；场外问答与元指令不推进时间、不落库，模型只看玩家自己的认知（泄露即回退模板），
          场外回答流式逐句过名字闸门、带 request_id 的重试原样返回；模型写的追问过名字闸门（玩家自己说出的名字不算）；
          等待只被要紧的事打断（当面动手、对同伴动手都算），计划与反应 tick 从不截短；多步计划中途崩溃的重试接在正确的一步上；
          重复投递抢先写下叙述时以落库的那一段为准；
          后台预算的 NPC 决策与顺序执行逐项相同、版本变了即作废；向量回忆只为声明 reads_memories 的策略而跑；最近正文随提交落库、读档恢复；抵达结局即落幕，
          终章以场景给的标题开头、带明确标注的真相揭晓（含受伤、玩家没看见的动手与潜逃）；叙述经 on_text 流式交付并记下首字耗时；
          命令行模板模式脚本化跑通到落幕；解释器按需接入（模块缺失时退回规则解析）
[POS]: tests 的主持层（设计 §2、§4.4）：证伪“说话之后 NPC 要等下一回合才开口”“失败了还接着砍”“问主持人也会过一分钟”
       “一点响动就把等待打断”“眼前打成一团也不停下”“后台预算改变了结果”“读档后忘了上一段”“终章看不到真相”
       “场外回答要等整段写完才看得到”“重试一次提示就多翻一条”“追问里说漏了嘴”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.agents.policy_kit import Choice, Situation  # noqa: E402
from tianlong.cognition import Candidate  # noqa: E402
from tianlong.core import (  # noqa: E402
    Entity,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    Proposition,
    Rel,
    Relation,
    Social,
    WorldState,
    at,
)
from tianlong.core.profiles import Goal, GoalKind, Profile  # noqa: E402
from tianlong.kernel.perception import make_percept, scene_percept  # noqa: E402
from tianlong.language.interpret import Interpreter  # noqa: E402
from tianlong.language.llm import LLMUnavailable, ScriptedLLM  # noqa: E402
from tianlong.language.narrator import Narrator  # noqa: E402
from tianlong.language.parser import MoveKind, Parsed  # noqa: E402
from tianlong.language.render import Rendered, RenderStatus, Violation  # noqa: E402
from tianlong.persistence import InMemoryWorldStore, RequestConflict  # noqa: E402
from tianlong.runtime import cli, gm  # noqa: E402
from tianlong.runtime.session import ENDED, GameSession  # noqa: E402
from tianlong.scenarios import Ending, Scenario, build_wuliang  # noqa: E402

# ============================================================
#  小世界：大殿 ─回廊─ 后院 ─小门─ 山道；大殿 ─侧门─ 营地
#  段誉、钟灵、龚光杰、干光豪在大殿；马五德（带伤）在营地，段誉只听说过他在那里；秘籍在营地，段誉不知道
# ============================================================

T0 = at(1, 9, 0)
HERO = "hero"
GUIDE = ("先稳住局面，别和龚光杰硬碰。", "后院那边清静些。", "山道通往山外。")


def _state() -> WorldState:
    P, S, It, D, H = Kind.PLACE, Kind.SURFACE, Kind.ITEM, Kind.DOOR, Kind.PERSON
    ents = [
        Entity.make("hall", P, "大殿"), Entity.make("yard", P, "后院"), Entity.make("road", P, "山道"),
        Entity.make("camp", P, "营地"),
        Entity.make("d_hall", D, "回廊"), Entity.make("d_road", D, "小门"), Entity.make("d_side", D, "侧门"),
        Entity.make("rack", S, "兵器架"),
        Entity.make("sword", It, "长剑", weapon=True), Entity.make("fan", It, "折扇", small=True),
        Entity.make("scroll", It, "秘籍", small=True),
        Entity.make(HERO, H, "段誉", martial=0.0, agility=0.5, alertness=0.5),
        Entity.make("ling", H, "钟灵", martial=0.25, agility=0.8, alertness=0.8),
        Entity.make("gong", H, "龚光杰", martial=0.5, agility=0.6, alertness=0.5),
        Entity.make("gan", H, "干光豪", martial=0.5, agility=0.5, alertness=0.6),
        Entity.make("ma", H, "马五德", martial=0.35, agility=0.4, alertness=0.5, wounded=True),
    ]
    R = Relation
    rels = [R("d_hall", Rel.CONNECTS, "hall"), R("d_hall", Rel.CONNECTS, "yard"),
            R("d_road", Rel.CONNECTS, "yard"), R("d_road", Rel.CONNECTS, "road"),
            R("d_side", Rel.CONNECTS, "hall"), R("d_side", Rel.CONNECTS, "camp"),
            R("rack", Rel.AT, "hall"), R("sword", Rel.AT, "rack"), R("fan", Rel.AT, "ling"),
            R(HERO, Rel.AT, "hall"), R("ling", Rel.AT, "hall"), R("gong", Rel.AT, "hall"), R("gan", Rel.AT, "hall"),
            R("ma", Rel.AT, "camp"), R("scroll", Rel.AT, "camp")]
    return WorldState.build(7, T0, ents, rels)


def scenario(endings: tuple[Ending, ...] = (), world_id: str = "gm-test") -> Scenario:
    st = _state()
    layout = tuple(Fact(Proposition.rel(d, Rel.CONNECTS, p))
                   for d, ends in (("d_hall", ("hall", "yard")), ("d_road", ("yard", "road")), ("d_side", ("hall", "camp")))
                   for p in ends)

    def past(facts):
        return replace(make_percept(st, Modality.SCENE, facts=facts), tick=T0 - 30)

    def now_seen(a):
        return replace(scene_percept(st, a), tick=T0 - 1)

    people = (HERO, "ling", "gong", "gan", "ma")
    priors = {a: (past(layout), now_seen(a)) for a in people}
    priors[HERO] = (past(layout + (Fact(Proposition.rel("ma", Rel.AT, "camp")),)), now_seen(HERO))
    profiles = {
        HERO: Profile(HERO, "书生", "大理段氏子弟，离家出走", is_player=True, goals=(Goal(GoalKind.ESCAPE, home="road"),)),
        "ling": Profile("ling", "少女", "万劫谷少女", voice="娇憨泼辣，说话像连珠炮", knows="万劫谷的规矩"),
        "gong": Profile("gong", "东宗弟子", "骄横好胜", voice="骄横"),
        "gan": Profile("gan", "东宗弟子", "打算私奔", goals=(Goal(GoalKind.ESCAPE, home="camp"),)),
        "ma": Profile("ma", "宾客", "普洱老武师", voice="圆滑"),
    }
    return Scenario(world_id, st, profiles, priors, setting="小世界", aliases={"ling": ("灵儿",)},
                    hints="你可以说任何想做的事，例如：环顾四周 / 去后院", guide=GUIDE, endings=endings)


OUT = Ending("out", "第一幕终·出山", "road", "劫后余生，江风浩荡")


# ============================================================
#  测试用的小策略与小解释器
# ============================================================


def _index(sit: Situation, op: Op, target: str | None = None) -> int | None:
    return next((i for i, c in enumerate(sit.candidates) if c.op == op and (target is None or c.target == target)), None)


class Script:
    """plan：时刻 → (操作, 目标)；speak：在这些时刻对段誉说一句闲话；reply：段誉上一刻当面对我说话或做了姿态就回话
    （True = 闲话，"claim" = 从候选集里挑一句带命题的 TELL 告诉他）；busy：没事就去查看这件东西（例行举动，也让自己每个 tick
    都在调度里）。闲话一律是 Choice.free，index 指向 WAIT。"""

    def __init__(self, plan=None, speak=(), reply=False, busy=None):
        self.plan, self.speak, self.reply, self.busy = dict(plan or {}), set(speak), reply, busy

    def choose(self, sit: Situation) -> Choice:
        wait = _index(sit, Op.WAIT)
        assert wait is not None
        step = self.plan.get(sit.now)
        if step is not None and (i := _index(sit, *step)) is not None:
            return Choice(i, f"按剧本：{step[0].value}")
        if self.reply == "claim" and self._addressed(sit) and (i := _index(sit, Op.TELL, HERO)) is not None:
            return Choice(i, "如实相告", social=Social.EXPLAIN)
        if sit.now in self.speak or (self.reply and self._addressed(sit)):
            return Choice(wait, "开口", free=Candidate(Op.TELL, HERO, social=Social.AGREE))
        if self.busy is not None and (i := _index(sit, Op.INSPECT, self.busy)) is not None:
            return Choice(i, "例行查看")
        return Choice(wait, "无事")

    @staticmethod
    def _addressed(sit: Situation) -> bool:
        for ep in sit.beliefs.episodes:
            ev = ep.event
            if ep.tick == sit.now - 1 and ev.actor == HERO and (
                    (ep.modality == Modality.SPEECH and ev.target == sit.agent)
                    or (ep.modality == Modality.SIGHT and ev.kind == Op.WAIT.value and ev.utterance)):
                return True
        return False


class Interp:
    """替身解释器：按原文查表，查不到就交给规则解析器；记下每次收到的最近正文。"""

    def __init__(self, table: dict[str, Parsed], fallback=None):
        self.table, self.fallback, self.calls = table, fallback, []

    def interpret(self, text, me, recent=()):
        self.calls.append((text, tuple(recent)))
        if text in self.table:
            return self.table[text]
        return gm.gm_command(text) or self.fallback.parse(text, me)


SAY = Parsed(Candidate(Op.TELL, "ling", social=Social.GREET), "姑娘好", source="llm", kind=MoveKind.SAY)
BOW = Parsed(Candidate(Op.WAIT, social=Social.SUBMIT), "拱手作揖", source="llm", kind=MoveKind.GESTURE)
SLASH = Parsed(Candidate(Op.TAKE, "sword"), source="llm", followups=(Candidate(Op.ATTACK, "gong"),))
GRAB = Parsed(Candidate(Op.TAKE, "fan"), source="llm", followups=(Candidate(Op.ATTACK, "gong"),))


def session(policies=None, table=None, store=None, endings=(), **kw) -> GameSession:
    """没点名的 NPC 一律闲着（Script()）：每个用例只让它关心的人动。"""
    sc = scenario(endings)
    s = GameSession(sc, store=store, policies={**{a: Script() for a in sc.npcs}, **(policies or {})}, **kw)
    if table is not None:
        s.interpreter = Interp(table, s.parser)
    return s


def _mine(events, op=None):
    return [e for e in events if e.actor == HERO and (op is None or e.op == op)]


# ============================================================
#  说话与姿态：多走一个反应 tick，NPC 当场回话，回话进入同一回合的感知与 SceneBrief
# ============================================================


def test_say_turn_takes_a_reaction_tick_and_the_reply_is_voiced_in_the_same_turn():
    s = session({"ling": Script(reply=True)}, {"姑娘好": SAY, "拱手作揖": BOW})
    r = s.turn("姑娘好", request_id="say")
    env = s.store.request(s.ref, "say")
    assert r.advanced and r.kind == MoveKind.SAY and s.authority.head().version == 2
    assert env.reaction and env.planned_ticks == 2 and env.versions == (1, 2)
    heard = [p for p in env.percepts if p.modality == Modality.SPEECH and p.event.actor == "ling"]
    assert heard and heard[0].event.target == HERO and heard[0].event.social == Social.AGREE, "回话在同一回合的感知里"
    [line] = r.brief.lines
    assert (line.speaker, line.speaker_name, line.listener_name, line.op) == ("ling", "钟灵", "你", "tell")
    assert line.social == Social.AGREE and line.claim is None and line.answering == "姑娘好"
    assert line.voice == "娇憨泼辣，说话像连珠炮" and line.knows == "万劫谷的规矩"
    assert {"段誉", "钟灵", "灵儿", "大殿"} <= line.may_name and "秘籍" not in line.may_name, "只许点名钟灵认识的"
    assert r.brief.player_line == "姑娘好" and r.brief.recent == ()
    assert _mine(r.events, Op.TELL)[0].intent.utterance == "姑娘好"

    bow = s.turn("拱手作揖")
    assert bow.kind == MoveKind.GESTURE and s.authority.head().version == 4, "姿态同样多走一个反应 tick"
    assert bow.brief.player_line == "拱手作揖" and [vl.answering for vl in bow.brief.lines] == [None]
    assert s.interpreter.calls[-1] == ("拱手作揖", (r.narration,)), "解释器拿到最近几段正文"


def test_npc_speech_is_never_voiced_by_a_model_inside_the_decision_graph():
    """即便有模型，NPC 带命题的台词也由模板说出（落库的原话是模板措辞），措辞留给叙述那一次调用。"""
    from tianlong.language.speaker import TemplateSpeaker
    calls = []
    llm = ScriptedLLM(lambda prompt, system, schema: calls.append(system) or "你点了点头。")
    s = session({"ling": Script(reply="claim")}, {"姑娘好": SAY}, llm=llm)
    r = s.turn("姑娘好")
    [said] = [e for e in r.events if e.actor == "ling" and e.op == Op.TELL]
    assert said.intent.topic is not None and said.intent.social == Social.EXPLAIN
    ling = s.beliefs("ling")
    assert said.intent.utterance == TemplateSpeaker().utter(s.scenario.profiles["ling"], Candidate.of(said.intent),
                                                            ling.entities)
    assert len(calls) == 1, "整个回合只有叙述调了一次模型"
    [line] = r.brief.lines
    assert line.claim and line.template == said.intent.utterance and line.social == Social.EXPLAIN


# ============================================================
#  多步计划：逐 tick 执行、失败即止，反应 tick 照旧
# ============================================================


def test_two_step_plan_runs_take_then_attack():
    s = session({}, {"拿起长剑向龚光杰刺去": SLASH})
    r = s.turn("拿起长剑向龚光杰刺去", request_id="slash")
    mine = _mine(r.events)
    assert [(e.op, e.intent.based_on) for e in mine] == [(Op.TAKE, 0), (Op.ATTACK, 1), (Op.WAIT, 2)]
    assert mine[0].outcome == Outcome.SUCCESS and mine[1].intent.target == "gong"
    env = s.store.request(s.ref, "slash")
    assert env.planned_ticks == 3 and env.reaction and env.done, "龚光杰在场：两步之后再加一个反应 tick"


def test_plan_stops_on_failure_but_keeps_the_reaction_tick():
    s = session({}, {"夺过折扇打龚光杰": GRAB})
    r = s.turn("夺过折扇打龚光杰", request_id="grab")
    mine = _mine(r.events)
    assert [(e.op, e.outcome) for e in mine] == [(Op.TAKE, Outcome.FAILURE), (Op.WAIT, Outcome.SUCCESS)]
    env = s.store.request(s.ref, "grab")
    assert env.followups == () and env.planned_ticks == 2 and env.versions == (1, 2), "截短的计划随提交落库"
    assert r.parsed.followups == GRAB.followups, "报告里仍是玩家原本的计划"


# ============================================================
#  场外问答与元指令：不推进时间、不落库；模型只看玩家自己的认知
# ============================================================


def _aside_llm(answer: str) -> ScriptedLLM:
    return ScriptedLLM(lambda prompt, system, schema: answer if system == gm.ASIDE_SYSTEM else "……")


def test_ask_gm_and_meta_never_advance_or_commit():
    llm = _aside_llm("不妨先去后院看看，那里清静。")
    s = session(llm=llm)
    head = s.authority.head()
    r = s.turn("GM：我该做什么", request_id="gm-1")
    assert not r.advanced and r.kind == MoveKind.ASK_GM and r.narration == "（场外）不妨先去后院看看，那里清静。"
    assert r.render.status == RenderStatus.LLM
    system, prompt = llm.prompts[-1]
    assert system == gm.ASIDE_SYSTEM and "玩家的问题：我该做什么" in prompt
    assert "你在大殿" in prompt and "钟灵" in prompt and "设法去往山道" in prompt and GUIDE[0] in prompt
    assert GUIDE[1] not in prompt, "提示只给到下一条，更深的不进提示词"
    assert "秘籍" not in prompt and "受了伤" not in prompt, "玩家不知道的真相不进提示词"
    hints = [s.turn("/hint").narration, s.turn("/hint").narration]
    assert hints == [f"提示：{GUIDE[0]}", f"提示：{GUIDE[1]}"]
    assert s.turn("/recap").narration == "（故事才刚开始。）"
    beliefs = s.turn("/beliefs")
    assert beliefs.kind == MoveKind.META and "[亲见" in beliefs.narration and "马五德在营地" in beliefs.narration
    after = s.authority.head()
    assert (after.version, after.clock) == (head.version, head.clock), "时间一分未动"
    assert s.store.request(s.ref, "gm-1") is None and s.store.session_state(s.ref) is None, "什么都没落库"
    s.turn("等待")
    assert s.store.session_state(s.ref)["hint"] == 2, "提示进度随下一次提交落库"


def test_ask_gm_answer_that_names_unknown_things_falls_back_to_the_template():
    s = session(llm=_aside_llm("营地里藏着一本秘籍，快去拿。"))
    r = s.turn("GM：我该做什么")
    assert r.render.status == RenderStatus.GATED_FALLBACK and "秘籍" not in r.narration
    assert r.narration.startswith("（场外）你在大殿") and f"提示：{GUIDE[0]}" in r.narration
    plain = session().turn("GM：我身上有什么")
    assert plain.render.status == RenderStatus.TEMPLATE and "身上没带什么要紧的东西" in plain.narration


class Streamer:
    """替身声音模型：按给定的分段流式吐字（None = 中途断线），把吐出的每一段记进 log，记下收到的 max_tokens。"""

    model = "streamer"

    def __init__(self, pieces, log):
        self.pieces, self.log, self.read, self.max_tokens = pieces, log, 0, None

    def generate(self, *a, **kw):
        raise AssertionError("场外问答应当走流式")

    def stream(self, prompt, *, system=None, max_tokens=None):
        assert system == gm.ASIDE_SYSTEM
        self.max_tokens = max_tokens
        for piece in self.pieces:
            if piece is None:
                raise LLMUnavailable("注入的断线")
            self.read += 1
            self.log.append(("model", piece))
            yield piece


def _ask(pieces, text="GM：我该怎么办"):
    log: list = []
    llm = Streamer(pieces, log)
    r = session(llm=llm).turn(text, on_text=lambda t: log.append(("text", t)))
    return r, llm, log


def test_ask_gm_answer_streams_sentence_by_sentence_through_the_name_gate():
    r, llm, log = _ask(["你眼下在大殿里。", "不妨先", "向龚光杰赔个不是。"])
    assert r.narration == "（场外）你眼下在大殿里。不妨先向龚光杰赔个不是。" and r.render.status == RenderStatus.LLM
    assert log.index(("text", "（场外）你眼下在大殿里。")) < log.index(("model", "向龚光杰赔个不是。")), "首句不等整段生成完就交付"
    assert llm.max_tokens == gm.ASIDE_TOKENS and r.first_text_ms is not None
    r, llm, log = _ask(["先稳住局面。", "营地里藏着秘籍。", "快去拿。", "别让人瞧见。"])
    shown = "".join(t for k, t in log if k == "text")
    assert r.narration == shown == "（场外）先稳住局面。", "点了玩家不认识的名字那句不交付"
    assert llm.read == 3, "拦下之后不再读流"
    assert r.render.status == RenderStatus.LLM and Violation("entity", "秘籍") in r.render.violations
    r, _, _ = _ask(["先稳住局面。", "别", None])
    assert r.narration == "（场外）先稳住局面。" and r.render.status == RenderStatus.LLM_UNAVAILABLE, "断线：已交付的算数，半句不交"
    r, _, _ = _ask([None])
    assert r.narration.startswith("（场外）你在大殿") and r.render.status == RenderStatus.LLM_UNAVAILABLE

    class GenerateOnly:
        model = "plain"
        kw: dict = {}

        def generate(self, prompt, **kw):
            GenerateOnly.kw = kw
            return "不妨先去后院看看。"

    r = session(llm=GenerateOnly()).turn("GM：我该做什么")
    assert r.narration == "（场外）不妨先去后院看看。" and GenerateOnly.kw["max_tokens"] == gm.ASIDE_TOKENS


def test_aside_retried_with_the_same_request_id_returns_the_same_result():
    llm = _aside_llm("不妨先去后院看看。")
    s = session(llm=llm)
    a, b = s.turn("/hint", request_id="h1"), s.turn("/hint", request_id="h1")
    assert a.narration == b.narration == f"提示：{GUIDE[0]}" and not a.replayed and b.replayed
    q = s.turn("GM：我该做什么", request_id="q1")
    calls = len(llm.prompts)
    streamed: list[str] = []
    again = s.turn("GM：我该做什么", request_id="q1", on_text=streamed.append)
    assert again.narration == q.narration and again.replayed and len(llm.prompts) == calls, "重试不再问一遍模型"
    assert streamed == [q.narration] and again.first_text_ms is not None
    with pytest.raises(RequestConflict):
        s.turn("/recap", request_id="h1")
    assert s.turn("/hint").narration == f"提示：{GUIDE[1]}", "重试没有多翻一条提示"
    s.turn("等待")
    assert s.store.session_state(s.ref)["hint"] == 2 and s.store.request(s.ref, "h1") is None, "不推进的回合照旧不落库"


def _interpreting(answers: dict[str, dict], bare: bool = False) -> GameSession:
    """真解释器 + 脚本快模型：按玩家原文回一份 JSON（缺的字段取默认）。
    bare：解释器不拿场景的名字全集与别称（它自己的回显闸门认不出陌生名字），只剩会话的名字闸门兜底。"""
    base = {"kind": "unclear", "mode": "immediate", "actor": "player", "steps": [], "listener": None, "speech": "tell",
            "line": "", "social": "none", "topic_subject": None, "topic_value": None, "topic_holds": True,
            "missing": "", "reply": ""}
    fast = ScriptedLLM(lambda p, s, sc: json.dumps({**base, **answers[p.rsplit("玩家输入：", 1)[1]]}, ensure_ascii=False))
    return session(fast_llm=fast, interpreter=Interpreter(fast) if bare else None)


def test_model_written_clarifications_pass_the_name_gate_but_the_players_own_words_do_not_count():
    study = [{"op": "study", "target": None, "obj": None, "manner": "normal"}]
    s = _interpreting({"我想找个清静地方练功": {"reply": "你是想去营地翻那本秘籍吗？"},
                       "我翻出怀里那本书来读": {"kind": "act", "steps": study, "missing": "秘籍"},
                       "我从怀里掏出秘籍": {"kind": "act", "steps": study, "missing": "秘籍"}})
    head = s.authority.head().version
    r = s.turn("我想找个清静地方练功")
    assert not r.advanced and "秘籍" not in r.narration, "解释器自己先把点了陌生名字的追问换掉"
    bare = _interpreting({"我想找个清静地方练功": {"reply": "你是想去营地翻那本秘籍吗？"}}, bare=True)
    r = bare.turn("我想找个清静地方练功")
    assert not r.advanced and "秘籍" not in r.narration and r.render.status == RenderStatus.GATED_FALLBACK
    assert Violation("entity", "秘籍") in r.render.violations, "解释器漏过的，会话的名字闸门兜住"
    r = s.turn("我翻出怀里那本书来读")
    assert not r.advanced and "秘籍" not in r.narration, "模型编的名字不回显"
    r = s.turn("我从怀里掏出秘籍")
    assert r.narration == "你身上并没有秘籍。" and r.render.status == RenderStatus.TEMPLATE, "玩家自己说出的名字照样复述"
    assert s.authority.head().version == head


# ============================================================
#  等待只被要紧的事打断；计划与反应 tick 从不截短
# ============================================================


def test_routine_acts_nearby_do_not_interrupt_a_wait():
    s = session({"gong": Script(busy="rack"), "ling": Script(busy="rack")})
    r = s.turn("等一会", request_id="wait")
    env = s.store.request(s.ref, "wait")
    assert any(p.modality == Modality.SIGHT and p.event.kind == Op.INSPECT.value for p in env.percepts), "确实看见了"
    assert s.authority.head().version == 10 and env.done and len(env.versions) == 10, "例行举动不打断等待"
    assert r.advanced


@pytest.mark.parametrize("who, policy, stop", [
    ("ling", Script(busy="rack", speak={T0 + 3}), 4),                          # 有人对我说话
    ("gan", Script(busy="rack", plan={T0 + 2: (Op.MOVE, "camp")}), 3),         # 有人离开我所在的地方
    ("gong", Script(busy="rack", plan={T0 + 1: (Op.ATTACK, HERO)}), 2),        # 有人对我动手
])
def test_salient_events_interrupt_a_wait(who, policy, stop):
    s = session({who: policy, "gong" if who != "gong" else "ling": Script(busy="rack")})
    s.turn("等一会", request_id="wait")
    env = s.store.request(s.ref, "wait")
    assert env.done and env.versions == tuple(range(1, stop + 1)), f"{who} 的举动在第 {stop} 个 tick 打断等待"


def test_a_fight_in_front_of_me_or_on_my_companion_interrupts_a_wait():
    """当面动手（打的是谁都算）即打断；同伴 = 自己人 + 玩家要护着的人（DEFEND 目标），与 NPC 替谁出头同一口径。"""
    s = session({"gong": Script(busy="rack", plan={T0 + 2: (Op.ATTACK, "ling")}), "ling": Script(busy="rack")})
    s.turn("等一会", request_id="wait")
    assert s.store.request(s.ref, "wait").versions == (1, 2, 3), "龚光杰当面向钟灵动手，第 3 个 tick 就停"
    guard = Profile(HERO, "书生", "护着钟灵", is_player=True, allies=("gan",), goals=(Goal(GoalKind.DEFEND, person="ling"),))
    assert gm.companions(guard) == {"gan", "ling"}
    assert gm.companions(scenario().profiles[HERO]) == frozenset()


def test_wuliang_wait_never_runs_past_a_fight_in_the_hall():
    """无量山大殿里段誉等着看：钟灵与东宗动起手来的那一刻就把回合交还玩家，不再一口气看到她被制住。"""
    s = GameSession(build_wuliang())
    s.intro()
    fights = 0
    for i in range(5):
        s.turn("等一会", request_id=f"w{i}")
        env = s.store.request(s.ref, f"w{i}")
        here = {e.intent.based_on for e in s.store.events(s.ref) if e.op == Op.ATTACK and e.place == "hall"
                and e.intent.based_on + 1 in env.versions}
        fights += len(here)
        assert here <= {env.versions[-1] - 1}, f"第 {i} 次等待越过了大殿里的动手：{sorted(here)}"
    assert fights >= 2, "开场确有人在大殿动手"


def test_planned_reaction_tick_is_never_cut_short():
    s = session({"gong": Script(speak={T0}), "ling": Script(reply=True)}, {"姑娘好": SAY})
    s.turn("姑娘好", request_id="say")
    env = s.store.request(s.ref, "say")
    assert env.versions == (1, 2), "第一个 tick 就有人冲我说话，反应 tick 照走"


def test_salient_rules_directly():
    from tianlong.core import PerceivedEvent, Percept
    here = "hall"

    def seen(kind, actor="gong", target=None, modality=Modality.SIGHT, place=here, **kw):
        return Percept(T0, modality, PerceivedEvent(kind, place, actor, target, **kw))

    assert not gm.salient([seen("inspect", target="rack")], HERO, (), here)
    assert not gm.salient([Percept(T0, Modality.SOUND, PerceivedEvent("noise", "yard"))], HERO, (), here)
    assert not gm.salient([seen("tell", target="ling")], HERO, (), here), "耳语：只看见在交谈"
    assert gm.salient([seen("tell", target="ling", modality=Modality.SPEECH, utterance="你好")], HERO, (), here)
    assert gm.salient([seen("tell", target=HERO, modality=Modality.SPEECH)], HERO, (), here)
    assert gm.salient([seen("attack", target="ling")], HERO, (), here), "当着我的面动手，打的是谁都算"
    assert not gm.salient([seen("attack", target="ling", place="yard")], HERO, (), here), "别处的旁人斗殴不算"
    assert gm.salient([seen("attack", target="ling", place="yard")], HERO, ("ling",), here), "有人对同伴动手"
    assert gm.salient([seen("wait", utterance="拔出长剑", social=Social.THREATEN)], HERO, (), here)
    assert not gm.salient([seen("wait", utterance="打了个哈欠", social=Social.REMARK)], HERO, (), here)
    assert gm.salient([seen("move", target="yard", outcome=Outcome.SUCCESS)], HERO, (), here)
    assert not gm.salient([seen("move", target="yard", outcome=Outcome.FAILURE)], HERO, (), here)
    hurt = Percept(T0, Modality.SIGHT, PerceivedEvent("use", here, "gong", "ling"),
                   facts=(Fact(Proposition.attr(HERO, "poisoned", True)),))
    assert gm.salient([hurt], HERO, (), here), "我自己的处境变了"


# ============================================================
#  请求幂等 + 多步计划：中途崩溃的重试接在正确的一步上
# ============================================================


def _fail_once(monkeypatch, obj, name, after=0):
    real, calls = getattr(obj, name), {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == after + 1:
            raise RuntimeError("注入的故障")
        return real(*a, **kw)

    monkeypatch.setattr(obj, name, flaky)


@pytest.mark.parametrize("parsed, ticks", [(SLASH, 3), (GRAB, 2)], ids=["take-attack", "failed-take"])
def test_crash_mid_plan_retry_continues_at_the_right_step(monkeypatch, parsed, ticks):
    text = "动手"
    reference = session({}, {text: parsed})
    ref_report = reference.turn(text, request_id="plan")
    store = InMemoryWorldStore()
    s = session({}, {text: parsed}, store=store)
    _fail_once(monkeypatch, s.indexer, "drain")                     # 第 1 个 tick 提交之后崩溃
    with pytest.raises(RuntimeError, match="注入的故障"):
        s.turn(text, request_id="plan")
    env = store.request(s.ref, "plan")
    assert env.versions == (1,) and not env.done
    retry = session({}, store=store).turn(text, request_id="plan")  # 新进程、没有解释器：步骤全由落库的进度推出
    assert not retry.replayed and store.head(s.ref).version == ticks
    assert [e.id for e in store.events(s.ref)] == [e.id for e in reference.store.events(reference.ref)]
    assert [(e.op, e.outcome) for e in _mine(retry.events)] == [(e.op, e.outcome) for e in _mine(ref_report.events)]
    assert retry.narration == ref_report.narration
    again = session({}, store=store).turn(text, request_id="plan")
    assert again.replayed and again.narration == ref_report.narration and store.head(s.ref).version == ticks


class Draft(Narrator):
    """替身主持人之声：写出固定的一稿；before 在第一次落笔之前跑一次（模拟重复投递恰在提交之后、叙述之前到达）。"""

    def __init__(self, base: Narrator, text: str, before=None):
        super().__init__(None, base.setting, base.lore, base.style, base.aliases)
        self.text, self.before = text, before

    def narrate_scene(self, viewer, percepts, names, *, brief, on_text=None, **kw):
        hook, self.before = self.before, None
        if hook is not None:
            hook()
        if on_text is not None:
            on_text(self.text)
        return Rendered(self.text, RenderStatus.TEMPLATE)


def test_duplicate_delivery_returns_and_remembers_the_first_recorded_narration():
    store = InMemoryWorldStore()
    a, b = session(store=store), session(store=store)
    other = {}
    b.narrator = Draft(b.narrator, "乙稿")
    a.narrator = Draft(a.narrator, "甲稿", before=lambda: other.setdefault("b", b.turn("等待", request_id="r")))
    mine = a.turn("等待", request_id="r")
    assert other["b"].narration == store.request(a.ref, "r").narration == "乙稿", "重复投递先写下了它那一稿"
    assert mine.narration == "乙稿" and a.session_state()["recent"][-1] == "乙稿", "同一请求只有一段正文：先写者为准"
    assert session(store=store).turn("等待", request_id="r").narration == "乙稿"


# ============================================================
#  后台预算：与顺序执行逐项相同；版本变了即作废；不推进的回合丢弃
# ============================================================

COMMANDS = ["等待", "GM：我该做什么", "/hint", "等待", "问钟灵长剑在哪", "/recap", "去后院", "等待", "等待"]


def _trace(pipeline: bool, monkeypatch):
    s = GameSession(build_wuliang(), pipeline=pipeline)                  # 各用各的内存存储，世界 ID 相同
    s.intro()
    calls = []
    real = s.orchestrator.decide
    monkeypatch.setattr(s.orchestrator, "decide", lambda npcs: calls.append(len(npcs)) or real(npcs))
    out = []
    for cmd in COMMANDS:
        r = s.turn(cmd)
        out.append((r.narration, r.advanced, r.kind, [(d.agent, d.intent, d.rationale) for d in r.deliberations]))
    events = [(e.op, e.actor, e.intent.target, e.outcome, e.changes, e.tick) for e in s.store.events(s.ref)]
    return out, events, s.session_state(), s.authority.head().fingerprint(), calls


def test_pipelined_decisions_are_identical_to_sequential(monkeypatch):
    piped, plain = _trace(True, monkeypatch), _trace(False, monkeypatch)
    assert piped[0] == plain[0], "每回合的叙述、类别与 NPC 决策逐项相同"
    assert piped[1] == plain[1] and piped[2] == plain[2] and piped[3] == plain[3]
    idle = sum(1 for _, advanced, _, _ in piped[0] if not advanced)
    assert idle == 3 and len(piped[4]) == len(plain[4]), "每个推进的回合都用上了后台预算；元指令与“GM：”一眼认得，不白算一份"


def test_stale_lookahead_is_discarded(monkeypatch):
    store = InMemoryWorldStore()
    s = session({"gong": Script(busy="rack")}, store=store)
    other = session({"gong": Script(busy="rack")}, store=store)
    calls = []
    real_decide = s.orchestrator.decide
    monkeypatch.setattr(s.orchestrator, "decide", lambda npcs: calls.append(
        next(iter(npcs.values())).port.version if npcs else None) or real_decide(npcs))
    real_parse = s._parse

    def racing(text, me):
        other.turn("等待")                                            # 解释期间另一个写入者推进了世界
        return real_parse(text, me)

    monkeypatch.setattr(s, "_parse", racing)
    r = s.turn("等待")
    assert calls == [0, 1], "后台按版本 0 预算过一次，作废后按版本 1 重算"
    assert {d.intent.based_on for d in r.deliberations} == {1} and _mine(r.events)[0].intent.based_on == 1


def test_recall_runs_only_for_policies_that_read_memories():
    class Recorder(Script):
        def __init__(self, reads: bool):
            super().__init__(busy="rack")
            self.reads_memories, self.seen = reads, []

        def choose(self, sit):
            self.seen.append(sit.memories)
            return super().choose(sit)

    reader, skipper = Recorder(True), Recorder(False)
    s = session({"gong": reader, "ling": skipper})
    for _ in range(3):
        s.turn("等待")
    assert any(reader.seen) and not any(skipper.seen), "向量回忆只为声明 reads_memories 的策略而跑"


# ============================================================
#  最近正文：叙述之后更新，随下一次提交落库，读档恢复
# ============================================================


def test_transcript_persists_and_is_restored_after_load():
    store = InMemoryWorldStore()
    s = session({}, store=store)
    s.intro()
    a, b = s.turn("等待"), s.turn("去后院")
    assert s.session_state()["recent"] == [a.narration, b.narration]
    assert store.session_state(s.ref)["recent"] == [a.narration], "b 的正文随下一次提交落库"
    again = session({}, {}, store=store)
    assert again.resumed and again.session_state()["recent"] == [a.narration]
    assert again.turn("/recap").narration == a.narration
    c = again.turn("等待")
    assert c.brief.recent == (a.narration,) and again.interpreter.calls[-1] == ("等待", (a.narration,))
    for _ in range(3):
        again.turn("等待")
    assert len(again.session_state()["recent"]) == 3, "只留最近三段"


# ============================================================
#  落幕与终章
# ============================================================


def test_ending_triggers_and_epilogue_reveals_the_truth():
    policies = {"gong": Script(busy="rack", plan={T0 + 1: (Op.ATTACK, "ling")}),
                "gan": Script(busy="rack", plan={T0 + 1: (Op.MOVE, "camp")})}
    s = session(policies, endings=(OUT,))
    assert s.turn("去后院").ending is None
    s.turn("等待")                                                    # 段誉在后院：大殿里动手、干光豪溜去营地，他都没看见
    r = s.turn("去山道")
    assert r.ending == OUT and s.ending == OUT
    version = s.authority.head().version
    later = s.turn("等待")
    assert not later.advanced and later.narration == ENDED and later.ending == OUT
    assert s.authority.head().version == version, "落幕之后不再推进"
    assert s.turn("/recap").narration, "元指令照常可用"
    text = s.epilogue()
    assert text.startswith("【第一幕终·出山】") and "—— 真相揭晓：你以为的 vs 实际的 ——" in text
    for name in ("钟灵", "龚光杰", "干光豪", "马五德"):
        assert f"· {name}：你以为" in text, name
    assert "· 马五德：你以为在营地；实际在营地，受了伤" in text, "受伤也在真相里"
    assert "—— 你没有看见的事 ——" in text and "龚光杰猛地向钟灵出手" in text
    assert "干光豪于第1日 09:01悄悄动身，如今在营地" in text
    assert GameSession(scenario((OUT,)), store=s.store).ending == OUT, "读档后仍是落幕状态"


def test_epilogue_heading_is_the_scenario_title_as_given():
    s = GameSession(build_wuliang())
    s.ending = s.scenario.endings[0]
    head = s.epilogue().split("\n", 1)[0]
    assert head == f"【{s.ending.title}】" and head.count("第一幕终") == 1, head


def test_epilogue_closing_passage_uses_only_the_players_experiences():
    seen = []

    def respond(prompt, system, schema):
        seen.append((system, prompt))
        return "江风拂面，你回头望了一眼大殿的方向。" if system and system.startswith(gm.CLOSING_SYSTEM) else "……"

    s = session({"gong": Script(busy="rack", plan={T0 + 1: (Op.ATTACK, "ling")})}, endings=(OUT,),
                llm=ScriptedLLM(respond))
    for cmd in ("去后院", "等待", "去山道"):
        s.turn(cmd)
    text = s.epilogue()
    closing = [p for sys_, p in seen if sys_ and sys_.startswith(gm.CLOSING_SYSTEM)]
    assert closing and "劫后余生" in closing[0] and "秘籍" not in closing[0] and "受了伤" not in closing[0]
    assert "出手" not in closing[0], "玩家没看见的动手不进终章提示词"
    assert text.index("江风拂面") < text.index("真相揭晓"), "收束在前，真相揭晓在后且明确标注"


# ============================================================
#  流式交付与首字耗时
# ============================================================


class StreamingNarrator(Narrator):
    """替身主持人之声：把模板正文拆成两段经 on_text 交付，并记下收到的 SceneBrief。"""

    def __init__(self, base: Narrator):
        super().__init__(None, base.setting, base.lore, base.style, base.aliases)
        self.briefs = []

    def narrate_scene(self, viewer, percepts, names, *, brief, show_scene=False, fresh=(), command="", lapse="",
                      known=(), on_text=None, familiar=()):
        self.briefs.append(brief)
        text = super().narrate_scene(viewer, percepts, names, brief=brief, show_scene=show_scene, fresh=fresh,
                                     command=command, lapse=lapse, known=known, familiar=familiar).text   # 模板正文（不经 on_text）
        half = max(1, len(text) // 2)
        for piece in (text[:half], text[half:]):
            on_text(piece)
        return Rendered(text, RenderStatus.TEMPLATE)


def test_on_text_streams_during_the_turn_and_first_text_is_timed():
    s = session({"ling": Script(reply=True)}, {"姑娘好": SAY})
    pieces: list[str] = []
    t0 = time.perf_counter()
    r = s.turn("等待", on_text=pieces.append)
    total = (time.perf_counter() - t0) * 1000
    assert pieces == [r.narration] and r.first_text_ms is not None and 0 <= r.first_text_ms <= total + 1
    s.narrator = StreamingNarrator(s.narrator)
    streamed: list[str] = []
    r = s.turn("姑娘好", on_text=streamed.append)
    assert len(streamed) == 2 and "".join(streamed) == r.narration, "主持人之声逐段交付"
    assert s.narrator.briefs[-1] is r.brief and r.brief.lines, "SceneBrief 交给了 narrate_scene"
    assert r.first_text_ms is not None and r.timings["narrate"] >= 0 and "interpret" in r.timings
    aside: list[str] = []
    q = s.turn("/hint", on_text=aside.append)
    assert aside == [q.narration] and q.first_text_ms is not None


# ============================================================
#  命令行：模板模式脚本化跑到落幕；解释器按需接入
# ============================================================


def test_cli_runs_a_scripted_session_to_the_ending(monkeypatch, capsys):
    monkeypatch.setitem(cli.SCENARIOS, "gmtest", lambda seed: scenario((Ending("yard", "第一幕终·后院", "yard", "清静"),)))
    lines = iter(["/hint", "GM：我该做什么", "/beliefs", "/recap", "/debug", "去后院"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert cli.main(["--world", "gmtest", "--llm", "none"]) == 0
    out = capsys.readouterr().out
    assert "你可以说任何想做的事" in out, "开场附上不剧透的输入示例"
    assert f"提示：{GUIDE[0]}" in out and "（场外）你在大殿" in out and "[亲见" in out and "（故事才刚开始。）" in out
    assert "first_text=" in out and "interpret=" in out, "/debug 显示分阶段耗时与首字耗时"
    assert "【第一幕终·后院】" in out and "真相揭晓" in out
    with pytest.raises(StopIteration):
        next(lines)                                                   # 落幕即退出：没有多读输入


def test_cli_and_session_wire_the_interpreter_with_the_fast_model():
    from tianlong.language.interpret import Interpreter
    sc = scenario()
    fast, voice = ScriptedLLM(lambda *a: "{}"), ScriptedLLM(lambda *a: "")
    got = cli.interpreter_for(fast, sc)
    assert isinstance(got, Interpreter) and got.llm is fast
    s = GameSession(sc, llm=voice, fast_llm=fast)
    assert isinstance(s.interpreter, Interpreter) and s.interpreter.llm is fast, "解释用快模型，叙述用叙述模型"
    assert GameSession(sc, llm=voice).interpreter.llm is voice, "没有快模型时解释也用叙述模型"
