"""
[INPUT]: 依赖 tianlong.language 的 render / deeds / narrator / speaker，tianlong.runtime.authority（真实内核产出的玩家感知），tianlong.scenarios
[OUTPUT]: 文字 ≠ 事实闸门验收 L01–L02：合法修辞放行；点名清单外实体（名或别称）、状态升级、瞬移、物品复制、编造承诺、传闻去归属
          各被拦下；人事闸门（伤落错人、否认确有的伤、没做成说成做成、没拿到说成在手、没走成却已在别处）；武侠状态说法；词法回归（松散否定、不连续说法、抵达动词、“还有一把”、伪言说动词）与误报回归（排除复合词、门名里的地名）；
          叙述者命中即回退模板且世界结算与文字结果分开记录；对白润色多出承诺/外人/丢了主语、或点名说话者没听说过的场景实体即回退模板
[POS]: tests 的语言输出层；验证“LLM 成功返回的错误非空文字不会被展示为已发生的事实”，且闸门不误伤正常的氛围与动作描写
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.cognition import Candidate
from tianlong.core import (
    EntitySketch,
    Fact,
    Intent,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
)
from tianlong.core.profiles import Profile
from tianlong.kernel.perception import sketches_for
from tianlong.language.llm import LLMUnavailable
from tianlong.language.narrator import Narrator
from tianlong.language.render import RenderStatus, build_plan, check
from tianlong.language.speaker import LLMSpeaker
from tianlong.scenarios import build_warehouse


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply, self.fail, self.calls = reply, fail, []

    def generate(self, prompt, *, system=None, schema=None, temperature=0.4):
        self.calls.append(prompt)
        if self.fail:
            raise LLMUnavailable("boom")
        return self.reply


KNOWN = frozenset(e.name for e in build_warehouse().state.entities.values())   # 场景全部实体名：只用于拒绝


def _player_view(authority, *specs):
    """在真实内核上行动，返回玩家本回合的感知与名称表。"""
    v = authority.head().version
    intents = [Intent(f"g{v}-{i}", s[0], s[1], *s[2:3], obj=s[3] if len(s) > 3 else None,
                      topic=s[4] if len(s) > 4 else None, based_on=v) for i, s in enumerate(specs)]
    r = authority.settle(intents)
    names = authority.store.beliefs(authority.ref, "player").entities
    return [o.percept for o in r.observations_of("player")], names


@pytest.fixture
def take_plan(authority):
    percepts, names = _player_view(authority, ("player", Op.TAKE, "key"))
    plan = build_plan("player", percepts, names)
    assert plan.lines == ("你拿起钥匙",)
    return plan


# ============================================================
#  L01：合法修辞放行，每一类错误各被拦下
# ============================================================


@pytest.mark.parametrize("text", [
    "你伸手从桌面上拿起钥匙，冰凉的铜齿硌着掌心。",
    "四下寂静，你把钥匙攥进手里。",
    "你悄悄拿起钥匙，心跳得厉害，却并没有受伤。",
    "你拿起钥匙，仓库里一片昏暗。",
])
def test_legal_paraphrases_pass(take_plan, text):
    assert check(text, take_plan, KNOWN) == ()


@pytest.mark.parametrize("text, kind", [
    ("你拿起钥匙，账簿就压在它下面。", "entity"),                 # 玩家不认识、本回合也没出现的实体
    ("你拿起钥匙，手指被划破，受了伤。", "status"),                # 凭空受伤
    ("你拿起钥匙，暗自发誓一定会把它还回去。", "commitment"),        # 编造承诺
    ("你拿起钥匙，转身走进内仓。", "teleport"),                    # 没有移动却抵达别处
    ("你拿起钥匙，又发现另一把钥匙。", "duplicate"),               # 物品复制
    ("你一口气拿起两把钥匙。", "duplicate"),
])
def test_each_violation_class_is_rejected(take_plan, text, kind):
    assert {v.kind for v in check(text, take_plan, KNOWN)} == {kind}


def test_actual_arrival_is_not_teleport(authority):
    percepts, names = _player_view(authority, ("player", Op.MOVE, "entrance", "door_main"))
    plan = build_plan("player", percepts, names)
    assert check("你推开仓库大门，来到仓库入口，海风扑面而来。", plan, KNOWN) == ()
    assert {v.kind for v in check("你推开仓库大门，径直来到港口。", plan, KNOWN)} == {"teleport"}


def test_hearsay_keeps_its_attribution(authority):
    _player_view(authority, ("guard", Op.MOVE, "warehouse", "door_main"))
    rumor = Fact(Proposition.rel("key", Rel.AT, "harbor"))
    percepts, names = _player_view(authority, ("guard", Op.TELL, "player", None, rumor))
    plan = build_plan("player", percepts, names)
    assert plan.hearsay == ("守卫",)
    for ok in ["守卫压低声音告诉你，钥匙在港口。", "守卫看了你一眼，说钥匙在港口。", "守卫道：“钥匙在港口。”"]:
        assert check(ok, plan, KNOWN) == (), ok
    assert {v.kind for v in check("钥匙在港口。", plan, KNOWN)} == {"hearsay"}, "转述不能变成叙述者确认的事实"


def _opening_plan():
    """无量山开场：段誉的初始认知、初见外观描写与场景别称全表；known 是场景全部实体名。"""
    from tianlong.cognition import BeliefStore
    from tianlong.language.narrator import lore_keys
    from tianlong.scenarios import build_wuliang
    sc = build_wuliang()
    prior = sc.priors["duanyu"]
    names = BeliefStore("duanyu").revise_all(prior)[0].entities
    looks = [sc.lore[k] for k in lore_keys("duanyu", prior, sc.lore)]
    plan = build_plan("duanyu", prior, names, show_scene=True, looks=looks, aliases=sc.aliases)
    return plan, {e.name for e in sc.state.entities.values()}


def test_rich_scene_prose_passes_but_additions_do_not():
    """无量山开场：满堂人物与外观描写（“插着几柄长剑”）照着写都放行；多走一步、多一句承诺、多一处伤都拦下。"""
    plan, known = _opening_plan()
    for ok in ["剑湖宫大殿里宾客满座，兵器架上插着好几柄长剑，钟灵坐在横梁上嗑着瓜子。",
               "比剑方罢，大殿里人声渐息，东西两宗弟子各自落座，你站在一旁看热闹。",
               "你又一次打量四周：左子穆目光阴沉，辛双清神情冷峻。"]:
        assert check(ok, plan, known) == (), ok
    assert {v.kind for v in check("你悄悄溜出大殿，走进琅嬛福地。", plan, known)} == {"entity"}
    assert {v.kind for v in check("左子穆答应收你为徒。", plan, known)} == {"commitment"}
    assert {v.kind for v in check("龚光杰受了伤，脸色铁青。", plan, known)} == {"status"}
    assert {v.kind for v in check("你起身走进后院。", plan, known)} == {"teleport"}, "别称说出的抵达也算"


def _wuxia_names() -> dict[str, EntitySketch]:
    return {s.id: s for s in (EntitySketch("duanyu", Kind.PERSON, "段誉"), EntitySketch("gong", Kind.PERSON, "龚光杰"),
                              EntitySketch("hall", Kind.PLACE, "剑湖宫大殿"),
                              EntitySketch("scroll", Kind.ITEM, "凌波微步帛卷"))}


def test_status_degree_is_preserved():
    names = _wuxia_names()
    hit = Percept(0, Modality.SIGHT, PerceivedEvent("attack", "hall", "gong", "duanyu", outcome=Outcome.SUCCESS),
                  (Fact(Proposition.attr("duanyu", "wounded", True)),), (), tuple(names.values()))
    plan = build_plan("duanyu", [hit], names)
    assert "wounded" in plan.statuses
    assert check("龚光杰猛地出手，你肩头一痛，受了伤。", plan) == ()
    assert {v.kind for v in check("龚光杰出手点了你的穴道，你动弹不得。", plan)} == {"status"}, "受伤不等于被制住"

    study = Percept(0, Modality.SELF, PerceivedEvent("study", "hall", "duanyu", "scroll", outcome=Outcome.SUCCESS,
                                                     reason="progress"), (), (), tuple(names.values()))
    plan = build_plan("duanyu", [study], names)
    assert "mastered" not in plan.statuses and "未能融会贯通" in plan.lines[0]
    assert check("你埋头研读凌波微步帛卷，若有所悟，可惜还未能融会贯通。", plan) == ()
    assert check("你埋头研读凌波微步帛卷，若有所悟，离融会贯通终究还差着一层。", plan) == (), "“离……还差”说的是没到"
    assert check("你埋头研读凌波微步帛卷，距学成尚远。", plan) == ()
    assert {v.kind for v in check("你研读片刻，豁然贯通，学成了这门步法。", plan)} == {"status"}, "略有所得不等于学成"


# ============================================================
#  词法回归：否定、常见说法、别称、归属、复制与误报
# ============================================================


@pytest.mark.parametrize("text", [
    "你拿起钥匙，却不慎受伤。",                    # “不慎”“猝不及防”“不幸”里的否定字不紧贴关键词
    "你拿起钥匙，猝不及防地受了伤。",
    "你拿起钥匙，不幸中了毒。",
    "你拿起钥匙，少不了受伤。",                    # “少不了”是肯定
    "你中了剧毒。", "你身中剧毒。", "你受了轻伤。", "你受了点伤。",   # 不连续的常见说法
])
def test_status_words_not_hidden_by_loose_negation_or_phrasing(take_plan, text):
    assert {v.kind for v in check(text, take_plan, KNOWN)} == {"status"}


@pytest.mark.parametrize("text, kind", [
    ("你拿起钥匙，不得不答应下来。", "commitment"),       # 双重否定算肯定
    ("你拿起钥匙，转身跑到内仓。", "teleport"),           # 常见抵达动词
    ("你拿起钥匙，转身冲进内仓。", "teleport"),
    ("你拿起钥匙，桌下还有一把钥匙。", "duplicate"),      # “还有/另有/又……一把”是多出来的一件
    ("你拿起钥匙，桌上另有一把钥匙。", "duplicate"),
    ("你拿起钥匙，又摸到一把钥匙。", "duplicate"),
])
def test_more_phrasings_are_rejected(take_plan, text, kind):
    assert {v.kind for v in check(text, take_plan, KNOWN)} == {kind}


@pytest.mark.parametrize("text", [
    "你勉强控制住颤抖的手，拿起钥匙。",               # “控制住”不是被制住
    "你拿起钥匙，想起林中毒蛇的传闻。",               # “林中毒蛇”不是中毒
    "你拿起钥匙，穿堂风贯通前后。",                   # 风贯通前后不是学成
    "你拿起钥匙，远处有人答应了一声。",               # 应了一声不是承诺
    "你拿起钥匙，这钥匙一定会有用。",                 # 预料不是作保
    "你拿起钥匙，身上并没有伤口。",
])
def test_lexicon_false_positives_pass(take_plan, text):
    assert check(text, take_plan, KNOWN) == ()


@pytest.mark.parametrize("move, text", [
    (("player", Op.MOVE, "entrance", "door_main"), "你走到仓库大门前，推门而出，来到仓库入口。"),
    (("player", Op.MOVE, "storeroom", "door_store"), "你走到仓库门前，用力推了推，门纹丝不动。"),   # 门锁着，没走成
])
def test_door_names_are_not_cut_into_place_names(authority, move, text):
    """“仓库大门”“仓库门”是门：最长匹配先吃掉整个名字，不会被截成地点“仓库”而误判为瞬移。"""
    percepts, names = _player_view(authority, move)
    assert check(text, build_plan("player", percepts, names), KNOWN) == ()


def test_hearsay_needs_a_real_speech_verb(authority):
    _player_view(authority, ("guard", Op.MOVE, "warehouse", "door_main"))
    rumor = Fact(Proposition.rel("key", Rel.AT, "harbor"))
    percepts, names = _player_view(authority, ("guard", Op.TELL, "player", None, rumor))
    plan = build_plan("player", percepts, names)
    for bad in ["守卫显然知道些什么。钥匙就在港口。", "守卫一言不发。钥匙就在港口。"]:
        assert {v.kind for v in check(bad, plan, KNOWN)} == {"hearsay"}, bad
    assert check("守卫笑道，钥匙在港口。", plan, KNOWN) == ()


def test_opening_rejects_aliases_negation_tricks_and_unknown_places():
    """别称与名字同等对待：段誉没见过、没听过的实体（“神仙姐姐”“北冥神功”）与地点（“营地”“江边”）不许凭空出现；
    “毫不犹豫”“无奈”“无量山”里的否定字不遮住关键词。"""
    plan, known = _opening_plan()
    for text in ["你心里想着神仙姐姐。", "你仿佛看见了北冥神功。", "你悄悄溜出大殿，来到营地。", "你悄悄溜出大殿，来到江边。",
                 "你悄悄溜出大殿，走到崖顶。", "你溜出大殿，来到湖畔。"]:
        assert {v.kind for v in check(text, plan, known)} == {"entity"}, text
    for text in ["左子穆毫不犹豫地答应收你为徒。", "左子穆无奈答应收你为徒。"]:
        assert {v.kind for v in check(text, plan, known)} == {"commitment"}, text
    assert {v.kind for v in check("你在无量山中毒已深。", plan, known)} == {"status"}
    assert check("大殿里人声渐息，左掌门目光阴沉。", plan, known) == (), "在场实体的别称照常可说"


def test_common_ways_to_say_mastery_and_subdual_are_rejected():
    names = _wuxia_names()
    hit = Percept(0, Modality.SIGHT, PerceivedEvent("attack", "hall", "gong", "duanyu", outcome=Outcome.SUCCESS),
                  (Fact(Proposition.attr("duanyu", "wounded", True)),), (), tuple(names.values()))
    plan = build_plan("duanyu", [hit], names)
    for text in ["龚光杰出手点了你的穴道。", "龚光杰一指点中你胸口要穴。"]:
        assert {v.kind for v in check(text, plan)} == {"status"}, text
    study = Percept(0, Modality.SELF, PerceivedEvent("study", "hall", "duanyu", "scroll", outcome=Outcome.SUCCESS,
                                                     reason="progress"), (), (), tuple(names.values()))
    plan = build_plan("duanyu", [study], names)
    for text in ["你研读凌波微步帛卷，片刻间便学会了上面的步法。", "你研读凌波微步帛卷，不知不觉已将步法练熟。"]:
        assert {v.kind for v in check(text, plan)} == {"status"}, text


@pytest.mark.parametrize("text, status", [
    ("龚光杰脸如金纸，嘴角溢出一丝血迹。", "wounded"), ("龚光杰吐了一口血。", "wounded"), ("龚光杰肩头血流如注。", "wounded"),
    ("你只觉头晕目眩，嘴唇发紫。", "poisoned"), ("你浑身酸麻，再也动不了。", "subdued"),
    ("你依图运气，神功已成。", "mastered"), ("你已然悟透了北冥神功。", "mastered"), ("你已得北冥神功三昧。", "mastered"),
])
def test_wuxia_ways_to_say_a_status_are_rejected(text, status):
    plan, known = _opening_plan()
    assert {v.detail.split(":")[0] for v in check(text, plan, known) if v.kind == "status"} == {status}, text


@pytest.mark.parametrize("text", ["龚光杰气得脸色发青。", "左子穆脸色发紫，显是怒极。", "你只觉头皮发麻。", "天色渐渐发黑。",
                                  "众弟子站在原地，一动不动。", "龚光杰僵住了。", "此事已成定局。", "龚光杰双眼布满血丝。"])
def test_ordinary_wuxia_prose_is_not_a_status(text):
    plan, known = _opening_plan()
    assert check(text, plan, known) == (), text


# ============================================================
#  人事闸门：状态落在谁身上、谁做成了什么、东西在谁手里、玩家此刻在哪（无量山真实内核）
# ============================================================


def _wuliang_turn(*intents, until=None, tries: int = 8):
    """在无量山上按 tick 结算同一组意图，直到 until(plan) 成立（内核确定，次数固定）；返回段誉的计划。"""
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority
    from tianlong.scenarios import build_wuliang
    sc = build_wuliang()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    for n in range(tries):
        v = auth.head().version
        r = auth.settle([Intent(f"w{n}-{i}", *x, based_on=v) for i, x in enumerate(intents)])
        names = auth.store.beliefs(auth.ref, "duanyu").entities
        plan = build_plan("duanyu", [o.percept for o in r.observations_of("duanyu")], names, aliases=sc.aliases)
        if until is None or until(plan):
            return plan, {e.name for e in sc.state.entities.values()}
    raise AssertionError("内核没有结算出想要的局面")


def _deeds(text, plan, known):
    from tianlong.language.deeds import check_deeds
    return {(v.kind, v.detail) for v in check_deeds(text, plan, known)}


def test_status_is_bound_to_the_person_who_has_it():
    """龚光杰打伤了段誉：受伤的是你。把伤安到龚光杰身上、或说你毫发无损，都与内核相悖。"""
    plan, known = _wuliang_turn(("gongguangjie", Op.ATTACK, "duanyu"), until=lambda p: "wounded" in p.statuses)
    assert ("段誉", "wounded") in plan.afflicted
    for bad, detail in [("龚光杰一掌拍来，你侧身让过，反手一推，龚光杰闷哼一声，受了内伤。", "wounded:龚光杰:受了内伤"),
                        ("你却毫发无损。", "denied:段誉:毫发无损"), ("你并未受伤。", "denied:段誉:受伤")]:
        assert _deeds(bad, plan, known) == {("status", detail)}, bad
    for ok in ["龚光杰猛地出手，你肩头一痛，受了伤。", "龚光杰反手一掌正中你肩头，你闷哼一声，受了伤。", "你肩头的伤口隐隐作痛。",
               "龚光杰一掌打中你，自己却毫发无损。", "他出手极快，正中她穴道。"]:
        assert _deeds(ok, plan, known) == set(), ok


def test_failed_attack_and_take_are_not_narrated_as_success():
    """段誉出手被化解：“拍中”就是说成了；没拿兵器架上的长剑，它就不在你手里。"""
    plan, known = _wuliang_turn(("duanyu", Op.ATTACK, "gongguangjie"))
    assert "你猛地向龚光杰出手，但没有成功（被对方化解了）" in plan.lines
    for bad, kind in [("你大喝一声，一掌拍中龚光杰胸口，他踉跄着连退三步。", "outcome"),
                      ("你抢上两步，从兵器架上抽出长剑，横在胸前。", "outcome"), ("长剑已握在你手中。", "outcome"),
                      ("转眼之间，你已身在后院。", "teleport"), ("你出了大殿，站在剑湖宫后院里。", "teleport")]:
        assert {k for k, _ in _deeds(bad, plan, known)} == {kind}, bad
    for ok in ["你一掌拍向龚光杰，却被他挡了开去。", "你从怀里掏出易经，挡在胸前。", "你站在大殿中央，一时手足无措。",
               "你是溜出大殿，还是留下赔罪？", "你若想溜出大殿，眼下正是时候。", "钟灵站在梁上拍手。", "你险些击中龚光杰。",
               "你未能得手。"]:
        assert _deeds(ok, plan, known) == set(), ok
    both, _ = _wuliang_turn(("duanyu", Op.ATTACK, "gongguangjie"), ("gongguangjie", Op.ATTACK, "duanyu"),
                            until=lambda p: "wounded" in p.statuses)
    assert _deeds("龚光杰一掌拍中你肩头，你受了伤。", both, known) == set(), "做成了的一方照常可说"
    assert ("outcome", "attack:段誉:拍中") in _deeds("你一掌拍中龚光杰胸口。", both, known), "按人记：他得手不等于你得手"


def test_move_that_failed_or_succeeded():
    failed, known = _wuliang_turn(("duanyu", Op.MOVE, "houshan", "d_path"))
    text = "你趁众人不备，悄悄溜出大殿，穿过回廊，不多时已站在后院的古井旁。"
    assert _deeds(text, failed, known) == {("teleport", "溜出大殿"), ("teleport", "站在后院")}
    moved, _ = _wuliang_turn(("duanyu", Op.MOVE, "houyuan", "d_corridor"))
    assert _deeds("你出了大殿，已站在剑湖宫后院的古井旁。", moved, known) == set()
    took, _ = _wuliang_turn(("duanyu", Op.TAKE, "sword"))
    assert _deeds("你抢上两步，从兵器架上抽出长剑，长剑已握在你手中。", took, known) == set()


# ============================================================
#  叙述者：闸门命中即回退模板；来源分开记录
# ============================================================


def test_narrator_reports_render_source(authority):
    percepts, names = _player_view(authority, ("player", Op.TAKE, "key"))
    args = ("player", percepts, names)
    assert Narrator().narrate_rendered(*args).status == RenderStatus.TEMPLATE
    good = Narrator(FakeLLM("你屏住呼吸，从桌面上拿起钥匙。")).narrate_rendered(*args, known=KNOWN)
    assert good.status == RenderStatus.LLM and good.text == "你屏住呼吸，从桌面上拿起钥匙。"
    bad = Narrator(FakeLLM("你拿起钥匙，又顺手摸到另一把钥匙，转身走进内仓。")).narrate_rendered(*args, known=KNOWN)
    assert bad.status == RenderStatus.GATED_FALLBACK and bad.text == "你拿起钥匙。", "回退的模板行读起来是句子"
    assert {v.kind for v in bad.violations} == {"duplicate", "teleport"} and bad.dropped == 1
    down = Narrator(FakeLLM(fail=True)).narrate_rendered(*args)
    assert down.status == RenderStatus.LLM_UNAVAILABLE and down.text == "你拿起钥匙。"
    assert Narrator(FakeLLM("  ")).narrate_rendered(*args).status == RenderStatus.GATED_FALLBACK
    assert Narrator(FakeLLM("你拿起两把钥匙。")).narrate(*args) == "你拿起钥匙。", "narrate() 仍返回文字"
    assert Narrator().narrate(*args) == "你拿起钥匙", "没有模型时的模板照旧"


def test_session_records_settlement_and_render_separately():
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession

    llm = FakeLLM("你拿起钥匙，账簿也一并揣进怀里，随即走进内仓。")
    s = GameSession(build_warehouse(), llm=llm)
    r = s.turn("拿走桌上的钥匙")
    assert r.advanced and s.authority.head().version == 1, "世界照常结算，且只结算一次"
    assert s.authority.head().target("key", Rel.AT) == "player"
    assert r.render.status == RenderStatus.GATED_FALLBACK and r.narration == "你拿起钥匙。"
    assert {v.kind for v in r.render.violations} >= {"entity", "teleport"}
    assert s.store.events(s.ref) == r.events, "文字被拦不会让行动重跑"


# ============================================================
#  L02：对白润色只能换说法，不能换内容
# ============================================================


@pytest.fixture
def speech():
    sc = build_warehouse()
    names = {s.id: s for s in sketches_for(sc.state, ["key", "player", "table", "guard", "captain"])}
    cand = Candidate(Op.TELL, "player", topic=Fact(Proposition.rel("key", Rel.AT, "table")))
    return Profile("guard", "guard", "守卫"), cand, names


@pytest.mark.parametrize("reply, expected", [
    ("钥匙嘛，就搁在桌面上。", "钥匙嘛，就搁在桌面上。"),              # 合法润色：原样采用
    ("钥匙在桌面上，我保证不告诉别人。", "钥匙在桌面上"),              # 多出承诺 → 模板
    ("钥匙在桌面上，船长吩咐我看着。", "钥匙在桌面上"),                # 点名话题之外的人 → 模板
    ("就在那边桌面上。", "钥匙在桌面上"),                            # 丢了话题主语 → 模板
    ("钥匙在桌面上，玩家受了伤。", "钥匙在桌面上"),                    # 多出状态 → 模板
])
def test_speaker_gate(speech, reply, expected):
    profile, cand, names = speech
    assert LLMSpeaker(FakeLLM(reply)).utter(profile, cand, names) == expected


def test_speaker_may_refer_to_itself_as_i():
    sc = build_warehouse()
    names = {s.id: s for s in sketches_for(sc.state, ["guard", "player", "warehouse"])}
    cand = Candidate(Op.TELL, "player", topic=Fact(Proposition.rel("guard", Rel.AT, "warehouse")))
    assert LLMSpeaker(FakeLLM("我一直守在仓库。")).utter(Profile("guard", "guard", "守卫"), cand, names) == "我一直守在仓库。"


def test_speaker_rejects_scenario_entities_the_speaker_never_heard_of(speech):
    """守卫的认知里没有账簿：润色点名场景里真实存在、却不在话题里的实体同样回退模板——否则它作为原话落库、被听者记住，
    叙述者再照搬时就成了“有出处”的名字。"""
    profile, cand, names = speech
    assert "ledger" not in names
    speaker = LLMSpeaker(FakeLLM("钥匙在桌面上，就压在账簿底下。"), universe=KNOWN)
    assert speaker.utter(profile, cand, names) == "钥匙在桌面上"


def test_speaker_universe_covers_scenario_names_and_aliases():
    from tianlong.scenarios import build_wuliang
    sc = build_wuliang()
    names = {s.id: s for s in sketches_for(sc.state, ["zhongling", "duanyu", "zuozimu", "hall"])}
    cand = Candidate(Op.TELL, "duanyu", topic=Fact(Proposition.rel("zuozimu", Rel.AT, "hall")))
    profile = Profile("zhongling", "zhongling", "钟灵")
    universe = {e.name for e in sc.state.entities.values()}

    def utter(reply):
        return LLMSpeaker(FakeLLM(reply), universe=universe, aliases=sc.aliases).utter(profile, cand, names)

    plain = utter("")
    assert utter("左子穆在大殿，北冥神功就藏在琅嬛福地。") == plain, "泄露场景秘密的地点与秘籍（名或别称）→ 模板"
    assert utter("左掌门就在大殿里呢。") == "左掌门就在大殿里呢。", "话题里的人与地点用别称说，照常采用"
