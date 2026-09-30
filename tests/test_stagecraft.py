"""
[INPUT]: 依赖 runtime/staging 的 staging / dress / sky / Staging，scenarios 的 build_wuliang_commoner / build_wuliang，
         scenarios/tianlong/stagecraft 的 BEATS_C / CARDS_C / DETAILS_C，scenarios/tianlong/drives_c 的 HUNT / MOONRISE，
         kernel/perception 的 make_percept / scene_percept，language/narrator 的 Narrator，language/llm 的 ScriptedLLM，
         language/voice_prompt 的 system_prompt，persistence 的 TurnEnvelope，tests/test_commoner 的 stage（把人挪到某处、把时钟拨到某刻）
[OUTPUT]: 看点、写法卡、景观、细节与天色的验收（plan §7 M3）：
          only_settled（看点只在匹配的事件成功、且在玩家感知里时出现：同一回合的感知把那一掌改成落空，focus 为 None；
          玩家不在场，开场的叫阵与那一掌都不是他的看点；景观只在那段描写本回合初次交付时；火光与叫骂只认搜人时那几下、看点与写法卡同进同退，作罢后路过不算；
          替人求饶不是后院私语；许可词不是世界里实体的名字或别称；天黑透了、月亮还没出来时不给细节）、
          gate_clean（每一张写法卡、每一条细节、每一处景观：在它生效的场面里把卡文、细节原文或景观原文与许可词附在模板叙述后面，
          交给真实的叙述闸门，零违规、零丢句）、sky（月出之前 none、月出那一回合 rising、之后 up；旧版 None；石洞石室里只报时辰）、
          提示词的四段（看点、写法、眼前的景象、细节；卡文在系统提示的目录里；景观不再当初见外观列一遍）与旧版逐字不变
[POS]: tests 的舞台调度层：真相只用于呈现——看点只读玩家本回合的感知，写法只加修辞、不加事实
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tianlong.core import (
    Fact,
    Intent,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Proposition,
    Rel,
    Social,
    at,
)
from tianlong.kernel.perception import make_percept, scene_percept
from tianlong.language.llm import ScriptedLLM
from tianlong.language.narrator import Narrator
from tianlong.language.render import RenderStatus
from tianlong.language.scene import MOON_NONE, MOON_RISING, MOON_UP, SceneBrief
from tianlong.language.voice_prompt import system_prompt
from tianlong.persistence import TurnEnvelope
from tianlong.runtime import staging
from tianlong.scenarios import Scenario, build_wuliang, build_wuliang_commoner
from tianlong.scenarios.tianlong.drives_c import HUNT, MOONRISE
from tianlong.scenarios.tianlong.stagecraft import BEATS_C, CARDS_C, DETAILS_C

from .test_commoner import stage

ME = "ashun"
SC = build_wuliang_commoner(7)


def _env(percepts, start: int, end: int | None = None, fresh: tuple[str, ...] = ()) -> TurnEnvelope:
    end = start + 1 if end is None else end
    return TurnEnvelope("", "", Intent("", ME, Op.WAIT), 1, 0, start, versions=(1,), ticks=(end,),
                        percepts=tuple(percepts), fresh=fresh, done=True)


def _seen(sc: Scenario, place: str, op: Op, actor: str, target: str | None = None, obj: str | None = None,
          facts: tuple[Fact, ...] = (), modality: Modality = Modality.SIGHT, outcome: Outcome = Outcome.SUCCESS,
          utterance: str | None = None, social: Social | None = None):
    view = PerceivedEvent(op.value, place, actor, target, obj, outcome, None, None, utterance, social)
    return make_percept(sc.state, modality, view, facts, vantage=place)


def _at(where: dict[str, str], clock: int) -> Scenario:
    return stage(SC, where, clock)


def _status(who: str, attr: str) -> tuple[Fact, ...]:
    return (Fact(Proposition.attr(who, attr, True)),)


# ============================================================
#  每一张写法卡生效的场面（只放它的识别器所要求的那几样感知）
# ============================================================


def _contexts() -> dict[str, tuple[Scenario, TurnEnvelope]]:
    hall = SC
    torch = _at({ME: "houshan", "gongguangjie": "houshan", "duanyu": "houshan"}, HUNT + 5)
    lake = _at({ME: "jianhu"}, MOONRISE)
    shrine = _at({ME: "langhuan", "duanyu": "langhuan", "zhongling": "langhuan"}, at(1, 19, 50))
    return {
        "mink_strike": (hall, _env([_seen(hall, "hall", Op.ATTACK, "zhongling", "gongguangjie", "mink",
                                          _status("gongguangjie", "poisoned"))], at(1, 17, 43))),
        "subdue_style": (hall, _env([_seen(hall, "hall", Op.ATTACK, "zuozimu", "zhongling", None,
                                           _status("zhongling", "subdued"))], at(1, 17, 44))),
        "torch_search": (torch, _env([_seen(torch, "houshan", Op.WAIT, "gongguangjie",
                                            utterance="举着火把逼上一步，冷笑着堵住去路")], HUNT + 5)),
        "moon_wall": (lake, _env([scene_percept(lake.state, ME)], MOONRISE - 1, MOONRISE, fresh=("yubi@moon",))),
        "kowtow_count": (shrine, _env([_seen(shrine, "langhuan", Op.WAIT, "duanyu", utterance="对着玉像恭恭敬敬地磕头",
                                             social=Social.SUBMIT)], at(1, 19, 50))),
        "grab_and_miss": (shrine, _env([_seen(shrine, "langhuan", Op.WAIT, "zhongling",
                                              utterance="扑过去要抓段公子的袖子，却扑了个空：“不玩了不玩了！书呆子，你这是什么古怪步法？”")],
                                       at(1, 19, 50))),
    }


def _clean(sc: Scenario, env: TurnEnvelope, st: staging.Staging, extra: str) -> None:
    """把 extra 附在模板叙述后面当作模型的答案，交给真实的叙述闸门：须零违规、零丢句，extra 原样交付。"""
    names: dict = {}
    for p in env.percepts:
        for sk in p.sketches:
            names[sk.id] = sk
    brief = staging.dress(SceneBrief(), st, sc.lore)
    known = frozenset(e.name for e in sc.state.entities.values())
    args = dict(brief=brief, fresh=env.fresh, known=known, aliases=sc.gate_aliases)
    plain = Narrator(None, sc.setting, sc.lore, sc.style, sc.gate_aliases, sc.secrets)
    answer = plain.narrate_scene(ME, env.percepts, names, **args).text + "\n" + extra
    voice = Narrator(ScriptedLLM(lambda *_: answer), sc.setting, sc.lore, sc.style, sc.gate_aliases, sc.secrets,
                     lead_after=None, cards={k: c.text for k, c in sc.cards.items()})
    r = voice.narrate_scene(ME, env.percepts, names, **args)
    assert r.status == RenderStatus.LLM and not r.violations and r.dropped == 0, (extra, r.violations)
    assert all(x.strip() in r.text for x in extra.splitlines()), extra


# ============================================================
#  only_settled：看点只在事件成功、且在玩家感知里时出现
# ============================================================


def test_each_card_fires_in_its_scene():
    for key, (sc, env) in _contexts().items():
        st = staging.staging(sc, env)
        assert key in st.cards, key
    marten = staging.staging(*_contexts()["mink_strike"])
    assert marten.focus == "一道灰影扑上龚光杰，他中了貂毒" and marten.allowed == {"小貂"}
    moon = staging.staging(*_contexts()["moon_wall"])
    assert moon.spectacle == ("yubi@moon",) and moon.allowed == {"仙人"} and moon.sky.moon == MOON_RISING
    assert staging.staging(*_contexts()["subdue_style"]).focus is None, "写法卡的识别器不是看点：不计 B1、不给看点"


def test_only_settled_events_are_staged():
    sc, env = _contexts()["mink_strike"]
    missed = replace(env, percepts=tuple(replace(p, event=replace(p.event, outcome=Outcome.FAILURE)) for p in env.percepts))
    st = staging.staging(sc, missed)
    assert st.focus is None and st.cards == () and st.allowed == frozenset(), "貂没咬中：没有看点，也没有貂的写法"
    lake, moonlit = _contexts()["moon_wall"]
    again = staging.staging(lake, replace(moonlit, fresh=()))
    assert again.focus is None and again.spectacle == () and not again.cards, "月下玉璧只在那段描写初次交付时上演"


def test_torch_only_while_the_search_is_on():
    """火把与叫骂只认搜人时那几下（举火把逼近、堵住叫骂、向人打听），看点与写法卡同进同退；只路过、作罢摔火把都不算。"""
    sc = _at({ME: "houyuan", "gongguangjie": "houyuan", "duanyu": "houyuan"}, HUNT + 5)
    gg = "gongguangjie"
    searching = [_seen(sc, "houyuan", Op.TELL, gg, "duanyu", modality=Modality.SPEECH,
                       utterance="酸秀才，这回看你还往哪里逃！", social=Social.TAUNT),
                 _seen(sc, "houyuan", Op.ASK, gg, ME, modality=Modality.SPEECH, utterance="段誉在哪里？"),
                 _seen(sc, "houyuan", Op.WAIT, gg, utterance="举着火把逼上一步，冷笑着堵住去路")]
    for p in searching:
        st = staging.staging(sc, _env([p], HUNT + 5))
        assert st.focus == "火把与叫骂声追了过来" and st.cards == ("torch_search",), p.event
    quit_ = [_seen(sc, "houyuan", Op.MOVE, gg, "hall", "d_corridor"),
             _seen(sc, "houyuan", Op.WAIT, gg, utterance="把火把往地上一掼：“哼，算那酸秀才命大！”")]
    for p in quit_:
        st = staging.staging(sc, _env([p], HUNT + 30))
        assert st.focus is None and st.cards == (), ("作罢之后走回大殿只是路过：没有追逐", p.event)


def test_whisper_is_the_back_yard_murmur_only():
    """私语只认后院里那一对的闲话与低语的姿态：后山上替挑夫高声求饶（plan 分歧 4）不是私语。"""
    lovers = {"ganguanghao": "houshan", "geguangpei": "houshan"}
    for place in ("houshan", "houyuan"):
        sc = _at({ME: place, **{k: place for k in lovers}}, at(1, 19, 21))
        plea = _seen(sc, place, Op.TELL, "geguangpei", "ganguanghao", modality=Modality.SPEECH,
                     utterance="师哥，他一个挑茶的，饶了他吧！", social=Social.PLEAD)
        assert staging.staging(sc, _env([plea], at(1, 19, 21))).focus is None, place
    sc = _at({ME: "houyuan", **{k: "houyuan" for k in lovers}}, at(1, 18, 20))
    murmur = _seen(sc, "houyuan", Op.WAIT, "ganguanghao", utterance="凑在她耳边低声道：“……天黑了再走……”",
                   social=Social.REMARK)
    assert staging.staging(sc, _env([murmur], at(1, 18, 20))).focus == "后院里有人低声私语"


def test_allowed_words_are_not_world_names():
    """许可词只许点名：它若是世界里某个实体的名字或别称（“长剑”是兵器架上的真剑），闸门就不再查那件真东西。"""
    universe = {e.name for e in SC.state.entities.values()} | {a for al in SC.aliases.values() for a in al} | {
        a for al in SC.gate_aliases.values() for a in al}
    allowed = {w for b in BEATS_C for w in b.allowed} | {w for c in CARDS_C.values() for b in c.cues for w in b.allowed}
    assert allowed and not allowed & universe, allowed & universe


def _session(sc: Scenario, llm=None):
    pytest.importorskip("langgraph")                # 会话要 LangGraph：核心零依赖环境先跳过，再导入
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    return GameSession(sc, llm=llm, pipeline=False)


def test_focus_follows_the_players_percepts_in_a_session():
    s = _session(SC)
    s.intro()
    for i in range(8):
        r = s.turn("等待", request_id=f"t{i}")
        if "beating" in r.beats:
            break
    assert r.brief.focus == "段公子挨了龚光杰一掌", "同一回合认出几个看点：取表里靠后的那个"
    env = s.store.request(s.ref, r.request_id)
    flipped = tuple(replace(p, event=replace(p.event, outcome=Outcome.FAILURE))
                    if p.event is not None and p.event.kind == Op.ATTACK.value else p for p in env.percepts)
    assert staging.staging(s.scenario, replace(env, percepts=flipped)).focus is None, "那一掌落空：没有看点"
    away = _session(_at({ME: "houyuan"}, SC.state.clock))
    away.intro()
    assert all(away.turn("等待").brief.focus is None for _ in range(4)), "大殿里的事玩家没感知到：不是他的看点"


# ============================================================
#  gate_clean：写法、许可词、细节、景观在它们生效的场面里过得了叙述闸门
# ============================================================


def test_every_card_is_gate_clean_in_its_scene():
    for key, (sc, env) in _contexts().items():
        st = staging.staging(sc, env)
        extra = [sc.cards[key].text, *(f"{w}。" for w in sorted(st.allowed))]
        _clean(sc, env, st, "\n".join(extra))


def test_every_spectacle_is_gate_clean():
    lores = [b for b in BEATS_C if b.lore]
    assert [b.lore for b in lores] == ["yubi@moon"]
    sc, env = _contexts()["moon_wall"]
    st = staging.staging(sc, env)
    _clean(sc, env, st, "\n".join([sc.lore["yubi@moon"], *(sc.cards[k].text for k in st.cards),
                                   *(f"{w}。" for w in sorted(st.allowed))]))


@pytest.mark.parametrize("eid", sorted(DETAILS_C))
def test_every_detail_is_gate_clean(eid):
    kind = SC.state.entity(eid).kind
    place = eid if kind == Kind.PLACE else SC.state.target(eid, Rel.AT)
    sc = _at({ME: place}, SC.state.clock)
    looked = make_percept(sc.state, Modality.SELF, PerceivedEvent(Op.INSPECT.value, place, ME, eid, None, Outcome.SUCCESS),
                          vantage=place)
    env = _env([looked, scene_percept(sc.state, ME)], sc.state.clock)
    deck = DETAILS_C[eid]
    assert 3 <= len(deck) <= 5 and len(set(deck)) == len(deck)
    for i, detail in enumerate(deck):
        st = staging.staging(sc, env, deck[:i])
        assert st.details == (detail,), "给下一条还没给过的"
        _clean(sc, env, st, detail)
    assert staging.staging(sc, env, deck).details == (), "一幕之内不重复：给完就不再给"


def test_details_are_still_life():
    """细节只写看得见的静物：不点任何人与别处的名字（本处的名与别称除外），不写日月。"""
    persons = {e.name for e in SC.state.entities.values() if e.kind == Kind.PERSON}
    sky = ("月", "日头", "太阳", "阳光", "夕阳")
    for eid, deck in DETAILS_C.items():
        mine = {SC.state.entity(eid).name, *SC.aliases.get(eid, ())}
        others = {n for e, al in SC.gate_aliases.items() if e != eid for n in al if len(n) >= 2} | {
            e.name for e in SC.state.entities.values() if e.id != eid}
        for d in deck:
            assert not any(p in d for p in persons) and not any(w in d for w in sky), d
            assert not [n for n in others - mine if n in d], (d, [n for n in others - mine if n in d])


# ============================================================
#  天色：月出之前 none、月出那一回合 rising、之后 up；旧版 None
# ============================================================


def test_sky_by_the_clock():
    m = SC.moments
    dusk, dark = staging.sky(at(1, 18, 20), m), staging.sky(at(1, 19, 10), m)
    assert dusk.moon == MOON_NONE and not dusk.night and "月亮还没出来" in dusk.text and "酉时" in dusk.text
    assert dark.moon == MOON_NONE and dark.night and "天已黑透" in dark.text
    assert staging.sky(MOONRISE, m, MOONRISE - 30).moon == MOON_RISING, "月出那个 tick 落在这一回合里"
    assert staging.sky(MOONRISE + 5, m, MOONRISE).moon == MOON_UP and staging.sky(MOONRISE, m).moon == MOON_UP
    assert staging.sky(MOONRISE, {}) is None and staging.staging(build_wuliang(7), _env([], at(1, 18))).sky is None
    for place in SC.enclosed:                           # 山腹里看不见天：只报时辰，不提日月；月相照旧交给审计
        shut = staging.staging(SC, _env([], MOONRISE - 30, MOONRISE), here=place).sky
        assert shut.moon == MOON_RISING and not any(w in shut.text for w in ("月", "日头", "天边")), shut
    assert SC.enclosed == {"shidong", "langhuan"} and "月亮正从" in staging.staging(SC, _env([], MOONRISE - 30, MOONRISE),
                                                                                     here="jianhu").sky.text


def test_no_details_in_the_dark():
    """天黑透了、月亮还没出来：查看也看不出细节（湖底的小鱼、玉璧的倒影都是要光的）；有了月光或天没黑再给。"""
    for when, given in ((at(1, 19, 10), False), (MOONRISE + 5, True), (at(1, 18, 20), True)):
        sc = _at({ME: "jianhu"}, when)
        looked = make_percept(sc.state, Modality.SELF, PerceivedEvent(Op.INSPECT.value, "jianhu", ME, "yubi", None,
                                                                      Outcome.SUCCESS), vantage="jianhu")
        st = staging.staging(sc, _env([looked, scene_percept(sc.state, ME)], when), here="jianhu")
        assert bool(st.details) == given, (when, st.details)


def test_waiting_for_the_moon_rises_it_once():
    s = _session(_at({ME: "jianhu"}, at(1, 19, 30)))
    s.intro()
    r = s.turn("等到月亮出来")
    assert r.brief.sky.moon == MOON_RISING and r.brief.focus == "月光照上玉璧，壁上似有仙人舞剑"
    assert r.brief.spectacle == (SC.lore["yubi@moon"],) and r.brief.cards == ("moon_wall",)
    after = s.turn("等待").brief
    assert after.sky.moon == MOON_UP and after.spectacle == () and after.focus is None


# ============================================================
#  提示词：静态前缀带写法卡目录；四段只在有时出现；旧版逐字不变
# ============================================================


def test_prompt_carries_the_staging_and_legacy_is_unchanged():
    assert system_prompt("世界", "文风") == system_prompt("世界", "文风", {}), "没有写法卡：系统提示逐字不变"
    llm = ScriptedLLM(lambda *_: "")
    s = _session(_at({ME: "jianhu"}, at(1, 19, 30)), llm)
    s.intro()
    s.turn("等到月亮出来")
    system, prompt = llm.prompts[-1]
    assert "写法卡目录" in system and all(c.text in system for c in CARDS_C.values())
    assert "本回合的看点（围绕它写，其余一笔带过）：月光照上玉璧" in prompt and "写法（只加修辞，不加事实" in prompt
    assert "moon_wall" in prompt and "眼前的景象" in prompt and "月亮正从东边的山头后升起" in prompt
    looks = prompt.split("玩家初次看清的人与物", 1)[-1] if "玩家初次看清的人与物" in prompt else ""
    assert SC.lore["yubi@moon"] not in looks, "景观另起一段写，不再当初见外观列一遍"
    old = ScriptedLLM(lambda *_: "")
    legacy = _session(build_wuliang(7), old)
    legacy.intro()
    r = legacy.turn("环顾四周")
    assert r.brief.focus is None and r.brief.cards == () and r.brief.sky is None and r.brief.details == ()
    system, prompt = old.prompts[-1]
    assert "写法卡目录" not in system and not any(t in prompt for t in ("本回合的看点", "写法（", "眼前的景象", "多看出的"))
