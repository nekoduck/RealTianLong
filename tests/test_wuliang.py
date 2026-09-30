"""
[INPUT]: 依赖 tianlong.scenarios 的 build_wuliang，tianlong.scenarios.tianlong.wuliang 的 EAST / WEST / SHENNONG / LOVERS，
         tianlong.runtime 的 GameSession，tianlong.kernel 的 violations / space，tianlong.cognition 的 BeliefStore，
         tianlong.language.parser 的 rule_parse，tianlong.core.goals 的 GoalRegistry
[OUTPUT]: 天龙八部·无量山场景验收：世界自洽、开场冲突链自然涌现（先叫阵、不应才动手）、入夜私奔按约动身、
          玩家可循原著路线抵达琅嬛福地并学成凌波微步（开溜后被寻仇者追上、等待被打断也照样走得通）；
          主持层内容：每个角色都有腔调/谈资/话多/脾气且谈资只点名本人认识的实体、开场收在对峙的钩子上、段誉的目标已注册、
          逐级提示由浅入深而不列步骤、输入示例不剧透、结局是世界图里真走得到的地点、别称都解析得到、盟友按门户划分
          （西宗掌门不替东宗弟子出头、私奔的一对投到神农帮营中不挨打）、蒲团绣字只是外观不揭示藏物
[POS]: tests 的内容层；剧情不是写死的脚本，而是由规则与认知涌现——这里断言的是“会发生”，不是“按剧本发生”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections import deque

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.cognition import BeliefStore  # noqa: E402
from tianlong.core import Fact, Kind, Modality, Op, Proposition, Rel, at  # noqa: E402
from tianlong.core.goals import GoalRegistry  # noqa: E402
from tianlong.core.profiles import GoalKind  # noqa: E402
from tianlong.kernel import space, violations  # noqa: E402
from tianlong.kernel.perception import make_percept  # noqa: E402
from tianlong.language.parser import rule_parse  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import SCENARIOS, build_wuliang  # noqa: E402
from tianlong.scenarios.tianlong.wuliang import EAST, LOVERS, SHENNONG, WEST  # noqa: E402


def test_world_is_consistent():
    sc = build_wuliang()
    assert not violations(sc.state)
    assert sc.player == "duanyu" and "wuliang" in SCENARIOS
    assert set(sc.profiles) == {e.id for e in sc.state.of_kind(Kind.PERSON)}
    assert all(k.split("@")[0] in sc.state.entities for k in sc.lore), "外观描写只针对存在的实体"


def test_opening_conflict_chain_emerges():
    """龚光杰先叫阵、不应才寻衅 → 钟灵放貂 → 同门长辈护短、制住钟灵、搜出解药、救治弟子。"""
    s = GameSession(build_wuliang())
    events = []
    for _ in range(12):      # 招呼、叫阵、讥讽、旁人喝止都要占 tick：链条比从前长，但照样自然走完
        events += [(e.actor, e.op, e.intent.target, e.outcome.value) for e in s.turn("等待").events]
    assert ("gongguangjie", Op.ATTACK, "duanyu", "success") in events
    assert events.index(("gongguangjie", Op.TELL, "duanyu", "success")) < \
        events.index(("gongguangjie", Op.ATTACK, "duanyu", "success")), "先礼后兵：叫阵在前"
    assert ("zhongling", Op.ATTACK, "gongguangjie", "success") in events
    healers = [a for a in ("zuozimu", "xinshuangqing") if (a, Op.TAKE, "antidote", "success") in events]
    assert healers and (healers[0], Op.USE, "gongguangjie", "success") in events
    st = s.authority.head()
    assert not st.attr("gongguangjie", "poisoned") and st.target("antidote", Rel.AT) == healers[0]


def test_lovers_leave_at_the_appointed_hour():
    s = GameSession(build_wuliang())
    s.turn("去后院")
    moved = None
    for _ in range(200):        # 等待可能被身边的动静打断，逐次推进直到过了约定时辰
        if s.authority.head().clock > at(1, 19, 21):
            break
        r = s.turn("等到天黑") if s.authority.head().clock < at(1, 19, 0) else s.turn("等待")
        hit = [e for e in r.events if e.actor == "ganguanghao" and e.op == Op.MOVE]
        if hit:
            moved = hit[0].tick
            break
    assert moved == at(1, 19, 20), "约定的时辰一到就动身，而不是等到下一次闲置轮询"


def test_player_can_follow_the_canon_route_to_the_scrolls():
    s = GameSession(build_wuliang())
    for cmd in ["去后院", "往后山走", "去崖顶", "跳下断崖"]:
        s.turn(cmd)
    s.turn("查看玉璧")
    assert not s.beliefs("duanyu").knows("d_cave"), "白日里看不出玉璧的秘密"
    for _ in range(20):                                  # 被人打断（开溜后寻仇者追来、当面动手）则再等
        if s.authority.head().clock >= at(1, 19, 0):
            break
        s.turn("等到天黑")
    r = s.turn("查看玉璧")
    assert "暗道" in r.narration
    s.turn("钻进石缝")
    r = s.turn("进石门")
    assert "叩首千遍" in r.narration, "初见蒲团时读得到上面的绣字"
    assert not s.beliefs("duanyu").knows("scroll_lb"), "绣字只是外观：读到它不等于知道蒲团里藏着什么"
    for cmd in ["磕头", "拿凌波微步", "拿北冥神功"]:
        s.turn(cmd)
    for _ in range(3):
        r = s.turn("研读凌波微步")
    assert "豁然贯通" in r.narration and s.authority.head().attr("duanyu", "evasion") is True
    s.turn("钻进隧道")
    st = s.authority.head()
    assert st.target("duanyu", Rel.AT) == "lancang"
    assert st.target("duanyu", Rel.AT) in {e.place for e in s.scenario.endings}, "原著路线的终点就是第一幕的结局"
    assert {st.target("scroll_bm", Rel.AT), st.target("scroll_lb", Rel.AT)} == {"duanyu"}


def test_first_sight_descriptions_only_once():
    s = GameSession(build_wuliang())
    s.intro()
    r1 = s.turn("去后院")
    r2 = s.turn("去大殿")
    assert "厢房" in r1.narration and "锦幡" not in r2.narration, "大殿开场已描写过，不再重复"


# ============================================================
#  主持层内容：腔调、谈资、目标、提示、结局、别称、盟友
# ============================================================


def _named(sc, text: str) -> set[str]:
    """文本里点到的实体（名或别称，最长匹配：“剑湖畔”不被截成“剑湖”，“剑湖宫”不被截成剑湖畔的别称）。"""
    forms: dict[str, str] = {e.name: e.id for e in sc.state.entities.values()}
    forms.update({a: eid for eid, al in sc.aliases.items() for a in al if len(a) >= 2})
    words = sorted(forms, key=lambda w: (-len(w), w))
    found, i = set(), 0
    while i < len(text):
        hit = next((w for w in words if text.startswith(w, i)), None)
        if hit is None:
            i += 1
            continue
        found.add(forms[hit])
        i += len(hit)
    return found


def _known(sc, agent: str) -> set[str]:
    return set(BeliefStore(agent).revise_all(sc.priors[agent])[0].entities)


def test_every_character_has_a_voice_and_lore():
    sc = build_wuliang()
    for agent, p in sc.profiles.items():
        assert p.voice and p.knows, f"{agent} 缺腔调或谈资"
        assert 0.0 <= p.chatty <= 1.0 and -1.0 <= p.temper <= 1.0, agent
    prof = sc.profiles
    assert "书呆子" in prof["zhongling"].voice and (prof["zhongling"].chatty, prof["zhongling"].temper) == (0.7, 0.2)
    assert prof["gongguangjie"].temper == 0.8 and prof["mawude"].chatty == 0.4
    assert prof["zhongling"].chatty == max(p.chatty for p in prof.values()), "钟灵最爱搭话"
    assert prof["xinshuangqing"].chatty < 0.1, "辛双清冷峻寡言"
    assert "万劫谷" in prof["zhongling"].knows and "闪电貂" in prof["zhongling"].knows
    assert "比剑" in prof["mawude"].knows and "神农帮" in prof["mawude"].knows
    assert "仙人舞剑" in prof["zuozimu"].knows and "传闻" in prof["zuozimu"].knows, "玉璧显影只是他听来的传说"


def test_voice_and_lore_name_only_what_the_speaker_knows():
    """腔调与谈资会交给主持人之声写台词：点到的实体必须是此人开场就认识的——否则等于把世界真相塞进他嘴里；
    私奔的一对不会把自己的打算当谈资。"""
    sc = build_wuliang()
    for agent, p in sc.profiles.items():
        stray = _named(sc, p.voice + p.knows) - _known(sc, agent)
        assert not stray, f"{agent} 的腔调/谈资点名了他不认识的 {sorted(stray)}"
    assert "yubi" in _named(sc, sc.profiles["zuozimu"].knows), "左子穆的传闻说的是他知道在哪的那面玉璧"
    for agent in LOVERS:
        public = sc.profiles[agent].voice + sc.profiles[agent].knows
        assert not [w for w in ("私奔", "神农帮", "相好") if w in public], agent
        assert not (set(LOVERS) - {agent}) & _named(sc, public), "不当众提起对方"


def test_opening_ends_on_the_confrontation_hook():
    sc = build_wuliang()
    last = [s for s in sc.setting.split("。") if s][-1]
    assert "笑" in sc.setting and "龚光杰" in last and "你" in last, "开场停在龚光杰将要发作的一刻"
    assert "段誉" in sc.setting, "玩家知道自己是谁"


def test_player_goals_are_registered_and_point_at_the_ending():
    sc = build_wuliang()
    registry = GoalRegistry()
    for p in sc.profiles.values():
        registry.check(p.goals)                                 # 未注册或缺字段即报错
    goals = sc.profiles["duanyu"].goals
    assert {g.kind for g in goals} == {GoalKind.ESCAPE, GoalKind.DEFEND}
    escape = next(g for g in goals if g.kind == GoalKind.ESCAPE)
    assert escape.home in {e.place for e in sc.endings}, "离开无量山 = 抵达第一幕的结局"
    assert next(g for g in goals if g.kind == GoalKind.DEFEND).person == "zhongling", "别连累钟灵"


def test_guide_is_progressive_not_a_walkthrough():
    sc = build_wuliang()
    guide = sc.guide
    assert 4 <= len(guide) <= 7 and all(h.strip() for h in guide) and len(set(guide)) == len(guide)
    steps = re.compile(r"[/／→>]|然后|接着|第[一二三四五六七八九十\d]+步|\d|[;；]\s*再")
    assert not [h for h in guide if steps.search(h)], "只点方向，不列步骤"
    persons = {e.id for e in sc.state.of_kind(Kind.PERSON)}
    concrete = [bool(_named(sc, h) - persons) for h in guide]   # 点到了地方或器物
    assert not concrete[0] and concrete[-1], "先说处境，最后才落到具体的地方"
    assert concrete == sorted(concrete), "由浅入深：一旦落到具体的地方，后面的提示不再退回泛泛之谈"


def test_hints_are_examples_not_spoilers():
    sc = build_wuliang()
    assert " / " in sc.hints and "你可以说" in sc.hints
    assert _named(sc, sc.hints) <= _known(sc, "duanyu"), "输入示例只点段誉开场就认识的人与地方"
    assert not [w for w in ("玉璧", "断崖", "石缝", "凌波微步", "北冥", "研读", "磕头") if w in sc.hints], "不把通关步骤写进示例"


def _reachable(state, start: str) -> set[str]:
    """沿 CONNECTS 的全部通道（暗门、夜现之门都算，终归会显形）、按单向通道的方向，从 start 能走到的地点。"""
    seen, queue = {start}, deque([start])
    while queue:
        here = queue.popleft()
        for door, there in space.neighbors(state, here):
            if there not in seen and space.passable(state, door, there):
                seen.add(there)
                queue.append(there)
    return seen


def test_endings_are_real_places_reachable_from_the_start():
    sc = build_wuliang()
    start = sc.state.target("duanyu", Rel.AT)
    reachable = _reachable(sc.state, start)
    assert sc.endings and len({e.key for e in sc.endings}) == len(sc.endings)
    for e in sc.endings:
        assert sc.state.has_entity(e.place) and sc.state.kind(e.place) == Kind.PLACE, e
        assert e.place != start and e.place in reachable, e
        assert e.title and e.epilogue, e
    river = next(e for e in sc.endings if e.key == "river")
    assert river.place == "lancang" and "澜沧江" in river.title
    assert _reachable(sc.state, river.place) == {river.place}, "抵达即落幕：隧道只能出不能进，回不去"


def test_aliases_resolve():
    sc = build_wuliang()
    owners: dict[str, str] = {}
    names = {e.name for e in sc.state.entities.values()}
    for eid, al in sc.aliases.items():
        assert sc.state.has_entity(eid), eid
        for a in al:
            assert a not in names and owners.setdefault(a, eid) == eid, f"别称 {a} 不能与别的实体撞名"
    for eid in ("ganguanghao", "geguangpei", "sword", "langhuan", "d_gate", "putuan", "yijing", "duanyu"):
        assert sc.aliases.get(eid), f"{eid} 缺别称"
    assert _named(sc, "东宗弟子在剑湖宫中比剑") == {"hall"}, "“剑湖宫”是大殿，不是剑湖畔"

    store = BeliefStore("duanyu").revise_all(sc.priors["duanyu"])[0]
    for text, op, target in [("向干师兄出手", Op.ATTACK, "ganguanghao"), ("向葛姑娘出手", Op.ATTACK, "geguangpei"),
                             ("拿起青钢剑", Op.TAKE, "sword"), ("研读周易", Op.STUDY, "yijing"),
                             ("从宫门出去", Op.MOVE, "shandao"), ("查看剑架", Op.INSPECT, "swordrack")]:
        c = rule_parse(text, store, sc.aliases).candidate
        assert c is not None and (c.op, c.target) == (op, target), text
    # 开场还不认识的地方与陈设：认识之后，别称照样生效
    seen = make_percept(sc.state, Modality.SCENE, facts=(Fact(Proposition.rel("putuan", Rel.AT, "langhuan")),))
    later = BeliefStore("duanyu").revise_all([*sc.priors["duanyu"], seen])[0]
    for text, target in [("查看拜垫", "putuan"), ("查看琅嬛", "langhuan"), ("查看石室", "langhuan")]:
        c = rule_parse(text, later, sc.aliases).candidate
        assert c is not None and c.target == target, text


def test_alliances_follow_the_sect_split():
    sc = build_wuliang()
    allies = {a: set(p.allies) for a, p in sc.profiles.items()}
    assert not allies["xinshuangqing"] & {"gongguangjie", "ganguanghao"}, "西宗掌门不替东宗弟子出头"
    assert not allies["zuozimu"] & {"geguangpei"}, "东宗掌门也不管西宗弟子"
    assert {"gongguangjie", "ganguanghao"} <= allies["zuozimu"] and "geguangpei" in allies["xinshuangqing"]
    assert "xinshuangqing" in allies["zuozimu"] and "zuozimu" in allies["xinshuangqing"], "外人欺到掌门头上，两宗一致对外"
    sect = set(EAST) | set(WEST)
    assert sect == {a for a, p in sc.profiles.items() if "宗" in p.role}
    for lover in LOVERS:
        assert set(SHENNONG) <= allies[lover] and not (sect - set(LOVERS)) & allies[lover], \
            "私奔的一对只认彼此与神农帮：同门撞见照样灭口"
    for boss in SHENNONG:
        assert set(LOVERS) <= allies[boss], "神农帮把来投的人当自己人"


def test_western_master_stays_out_of_the_opening_brawl():
    s = GameSession(build_wuliang())
    events = []
    for _ in range(8):
        events += [(e.actor, e.op, e.intent.target) for e in s.turn("等待").events]
    assert ("zuozimu", Op.ATTACK, "zhongling") in events, "左子穆护着自己的弟子"
    assert not [e for e in events if e[0] == "xinshuangqing" and e[1] == Op.ATTACK], "辛双清袖手旁观"


def test_lovers_are_welcome_at_the_shennong_camp():
    """干葛二人投奔神农帮：到了营地，司空玄不把他们当擅闯的外人。"""
    s = GameSession(build_wuliang())
    hits = []
    for _ in range(60):
        if s.authority.head().clock >= at(1, 19, 45):
            break
        r = s.turn("等到天黑" if s.authority.head().clock < at(1, 19, 0) else "等一会")
        hits += [e for e in r.events if e.actor in SHENNONG and e.op == Op.ATTACK and e.intent.target in LOVERS]
    st = s.authority.head()
    assert {st.target(p, Rel.AT) for p in LOVERS} == {"camp"}, "二人按约投到了神农帮营地"
    assert not hits and not any(st.attr(p, "wounded") for p in LOVERS), "来投的人不挨打"
