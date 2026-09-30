"""
[INPUT]: REQUEST_ITEM、实际内核、NPC 脚本决策、个人认知、SQLite 与特征编码
[OUTPUT]: 请求/同意/真实交付/使用分离；请求成功/失败先声；拒绝、暂缓和恢复；回应绑定、耳语隔离、受益人批指针与裁剪、模型前向、yielded 接续、旧记录和版本拒绝验收
[POS]: 请求型对话的完整闭环；不能用漂亮文字或 Social.AGREE 代替 GIVE/USE。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from dataclasses import replace

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.cognition import Candidate, candidates  # noqa: E402
from tianlong.cognition.view import belief_view  # noqa: E402
from tianlong.core import (  # noqa: E402
    Entity,
    Fact,
    Intent,
    Manner,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    Relation,
    Social,
    WorldState,
)
from tianlong.core.profiles import Goal, GoalKind  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.kernel.perception import make_percept, scene_percept  # noqa: E402
from tianlong.language.interpret import Interpreter  # noqa: E402
from tianlong.language.llm import ScriptedLLM  # noqa: E402
from tianlong.language.parser import MoveKind  # noqa: E402
from tianlong.persistence import SQLiteWorldStore, codec  # noqa: E402
from tianlong.runtime.choice_model import parsed_from, parsed_to  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.versions import IncompatibleSave, current_versions  # noqa: E402

from .test_suggest import Idle, choice_scene, scene_beliefs  # noqa: E402


def request_scene(*, beneficiary="player", pill_at="npc", protected=False):
    sc = choice_scene()
    entities = [Entity.make(e.id, e.kind, e.name, **{**dict(e.attrs), **({"poisoned": True} if e.id == beneficiary else {})})
                for e in sc.state.entities.values()]
    relations = [r for r in sc.state.relations if not (r.src == "pill" and r.type == Rel.AT)
                 and not (beneficiary == "hidden" and r.src == "hidden" and r.type == Rel.AT)]
    relations.append(Relation("pill", Rel.AT, pill_at))
    if beneficiary == "hidden":
        relations.append(Relation("hidden", Rel.AT, "hall"))
    state = WorldState.build(sc.state.seed, sc.state.clock, entities, relations)
    profiles = dict(sc.profiles)
    if beneficiary == "hidden":
        profiles["player"] = replace(profiles["player"], allies=("hidden",))
    if protected:
        profiles["npc"] = replace(profiles["npc"], goals=(Goal(GoalKind.PROTECT, "pill", home="npc"),))
    priors = {a: (scene_percept(state, a),) for a in profiles}
    priors["player"] += (make_percept(state, Modality.SELF, facts=(Fact(Proposition.attr("book", "teaches", "evasion")),)),)
    return replace(sc, state=state, profiles=profiles, priors=priors)


def session(sc, store=None):
    return GameSession(sc, store=store, policies={"hidden": Idle()}, pipeline=False)


def select(s, op):
    d = s.choices.current()
    c = next(c for c in d["choices"] if s.choices.resolve(d["decision_id"], c["id"]).parsed.candidate.op == op)
    return d["decision_id"], c["id"]


@pytest.mark.parametrize("beneficiary", ["player", "hidden"])
def test_request_agree_give_use_are_separate_real_events_and_resume(tmp_path, beneficiary):
    sc = request_scene(beneficiary=beneficiary)
    path = tmp_path / "request.sqlite3"
    s = session(sc, SQLiteWorldStore(path))
    decision, choice = select(s, Op.REQUEST_ITEM)
    p = s.choices.resolve(decision, choice).parsed
    assert p.candidate.beneficiary == beneficiary and p.kind == MoveKind.SAY
    assert parsed_from(parsed_to(p)) == p
    report = s.choose(decision, choice, "request")
    assert any(e.actor == "npc" and e.op == Op.TELL and e.intent.social == Social.AGREE for e in report.events)
    assert not any(e.op in (Op.GIVE, Op.USE) for e in report.events)
    assert s.authority.head().target("pill", Rel.AT) == "npc"
    assert s.authority.head().attr(beneficiary, "poisoned") is True
    assert not any(c.op == Op.USE and c.obj == "pill" for c in candidates(s.beliefs("player")))
    assert any(o.kind == "request_item" and o.state == "accepted" for o in s.beliefs("npc").obligations)
    assert s.beliefs("npc").yielded.get("player") is not None
    yielded = dict(s.beliefs("npc").yielded)
    saved_menu = s.choices.current()
    s.index.client.close()
    s = session(sc, SQLiteWorldStore(path))
    assert s.choices.current() == saved_menu
    assert dict(s.beliefs("npc").yielded) == yielded
    version = s.authority.head().version
    assert s.choose(decision, choice, "request").replayed
    assert s.authority.head().version == version
    decision, choice = select(s, Op.WAIT)
    report = s.choose(decision, choice, "delivery")
    assert any(e.actor == "npc" and e.op == Op.GIVE and e.outcome == Outcome.SUCCESS for e in report.events)
    assert s.authority.head().target("pill", Rel.AT) == "player"
    assert s.beliefs("player").location_of("pill") == "player"
    assert s.authority.head().attr(beneficiary, "poisoned") is True
    decision, choice = select(s, Op.USE)
    report = s.choose(decision, choice, "use")
    assert any(e.actor == "player" and e.op == Op.USE and e.intent.target == beneficiary for e in report.events)
    assert not s.authority.head().attr(beneficiary, "poisoned")
    s.index.client.close()


def test_npc_can_refuse_by_its_own_motive_without_menu_leaking_the_motive():
    good, guarded = request_scene(), request_scene(protected=True)
    a, b = session(good), session(guarded)
    # NPC 私密保护目标不同，按钮依旧完全相同；选择之后才知道对方会拒绝。
    assert a.choices.current() == b.choices.current()
    decision, choice = select(b, Op.REQUEST_ITEM)
    spec = b.choices.resolve(decision, choice)
    report = b.choose(decision, choice, "refused")
    assert any(e.actor == "npc" and e.intent.social == Social.REFUSE for e in report.events)
    assert b.authority.head().target("pill", Rel.AT) == "npc"
    assert b.choice_history.blocked(b.beliefs("player"), spec.parsed.candidate)
    assert all(b.choices.resolve(b.choices.current()["decision_id"], c["id"]).parsed.candidate.op != Op.REQUEST_ITEM
               for c in b.choices.current()["choices"])
    a.index.client.close()
    b.index.client.close()


def test_deferred_request_wakes_only_after_relevant_inventory_change():
    sc = request_scene(pill_at="player")
    s = session(sc)
    first = s.turn("请求钟灵给我解药", "defer")
    assert first.parsed.candidate.op == Op.REQUEST_ITEM
    assert any(e.actor == "npc" and e.intent.social == Social.EXPLAIN for e in first.events)
    assert any(o.kind == "request_item" and o.state == "deferred" for o in s.beliefs("npc").obligations)
    for _ in range(3):
        s.turn("环顾四周")
    assert any(o.kind == "request_item" and o.state == "deferred" for o in s.beliefs("npc").obligations)
    # NPC 真正拿到物品后才重新考虑原请求；不能从全知库存补给。
    s.turn("把解药交给钟灵")
    assert any(o.kind == "request_item" and o.state == "accepted" for o in s.beliefs("npc").obligations)
    assert s.authority.head().target("pill", Rel.AT) == "npc"
    s.turn("等待")
    assert s.authority.head().target("pill", Rel.AT) == "player"
    assert not any(o.kind == "request_item" for o in s.beliefs("npc").obligations)
    s.index.client.close()


def test_generic_agree_cannot_accept_an_unrelated_request():
    s = session(request_scene())
    d, c = select(s, Op.REQUEST_ITEM)
    s.choose(d, c, "pending")
    me = s.beliefs("player")
    ob = next(o for o in me.obligations if o.kind == "requested_item")
    me = replace(me, obligations=(replace(ob, state="pending"),))
    unrelated = Percept(me.last_tick, Modality.SPEECH,
                        PerceivedEvent("tell", "hall", "npc", "player", outcome=Outcome.SUCCESS, social=Social.AGREE))
    assert me.revise(unrelated)[0].obligations[0].state == "pending"
    assert codec.mind_from(codec.mind_to(me)) == me
    s.index.client.close()


def test_item_request_lead_works_with_a_narration_model_and_for_failed_requests():
    from tianlong.language.lead import lead_line
    voice = ScriptedLLM(lambda *_: "钟灵点了点头，表示愿意给出解药。")
    sc = request_scene()
    s = GameSession(sc, llm=voice, policies={"hidden": Idle()}, pipeline=False)
    d, c = select(s, Op.REQUEST_ITEM)
    delivered = []
    report = s.choose(d, c, "lead-request", delivered.append)
    assert delivered[0].startswith("你对钟灵道：")
    assert "请把解药交给我" in delivered[0]
    assert report.narration == "".join(delivered)
    assert s.authority.head().target("pill", Rel.AT) == "npc"
    assert s.authority.head().attr("player", "poisoned") is True
    failed = Kernel().step(sc.state, [Intent("far", "player", Op.REQUEST_ITEM, "hidden", "pill",
                                            based_on=sc.state.version, beneficiary="player")])
    percepts = [o.percept for o in failed.observations if o.observer == "player"]
    assert failed.events[0].outcome != Outcome.SUCCESS
    line = lead_line(percepts, s.beliefs("player").entities, "player")
    assert line.startswith("你想请求") and "解药" in line
    s.index.client.close()


def test_whisper_request_does_not_disclose_item_or_beneficiary_to_bystander():
    sc = request_scene(beneficiary="hidden")
    it = Intent("whisper", "player", Op.REQUEST_ITEM, "npc", "pill", Manner.CAREFUL,
                based_on=sc.state.version, beneficiary="hidden")
    r = Kernel().step(sc.state, [it])
    heard = [o.percept for o in r.observations if o.observer == "npc" and o.percept.modality == Modality.SPEECH]
    bystander = [o.percept for o in r.observations if o.observer == "hidden" and o.percept.event]
    assert heard[0].event.beneficiary == "hidden" and heard[0].event.request_ref
    assert all(p.event.obj is None and p.event.beneficiary is None and p.event.request_ref is None for p in bystander)
    assert r.state.target("pill", Rel.AT) == "npc"


def test_request_beneficiary_is_encoded_and_persistent_obligation_remains_visible():
    pytest.importorskip("numpy")
    from tianlong.learning.featurize import encode_action, featurize
    from tianlong.learning.schema import (
        EV_REQUEST_STATE,
        FEATURES_VERSION,
        REQUEST_STATES,
        SCHEMA,
        StaleModel,
        check_schema,
    )
    s = session(request_scene())
    d, c = select(s, Op.REQUEST_ITEM)
    s.choose(d, c, "request")
    me = s.beliefs("player")
    view = belief_view(replace(me, episodes=()), me.last_tick+20)
    assert any(e.rel == "FOR" and e.dst == "player" for e in view.edges)
    tensors = featurize(view)
    code = encode_action(tensors, "player", Candidate(Op.REQUEST_ITEM, "npc", "pill", beneficiary="player"))
    assert code.beneficiary == tensors.index_of("player")
    node = next(n for n in view.nodes if n.id.startswith("request:"))
    assert tensors.x[tensors.index_of(node.id), EV_REQUEST_STATE.start + REQUEST_STATES.index("accepted")] == 1
    assert FEATURES_VERSION == "features-v3"
    with pytest.raises(StaleModel):
        check_schema({"schema": "prior-request-semantics"}, "old-model")
    check_schema({"schema": SCHEMA}, "current-model")
    s.index.client.close()


def test_request_beneficiary_survives_pyg_batch_offsets_and_model_forward():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.data")
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.samples import agent_queries, shared_graph_batch, to_data

    me = scene_beliefs(request_scene(beneficiary="hidden"))
    cands = (Candidate(Op.REQUEST_ITEM, "npc", "pill", beneficiary="player"),
             Candidate(Op.REQUEST_ITEM, "npc", "pill", beneficiary="hidden"), Candidate(Op.WAIT))
    samples = agent_queries(me, me.last_tick, "player", cands)
    batch = pyg.Batch.from_data_list([to_data(s) for s in samples])
    shared = shared_graph_batch(samples)
    n = samples[0].graph.num_nodes
    assert batch.act_beneficiary[1] == samples[1].graph.index_of("hidden") + n
    assert batch.act_has_beneficiary.tolist() == [True, True, False]
    for key in batch.keys():  # noqa: SIM118 -- PyG 直接迭代得到 (key, tensor)，不同于 dict。
        if isinstance(batch[key], torch.Tensor):
            assert torch.equal(batch[key], shared[key]), key
    model = DynamicsModel(hidden=16, layers=1).eval()
    with torch.no_grad():
        out = model(shared)
    assert out.success.shape == (3,) and torch.isfinite(out.success).all()
    assert torch.isfinite(out.attr).all()


def test_request_crop_keeps_beneficiary_or_removes_whole_candidate_and_policy_runs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray.rllib")
    from tianlong.agents.predictors import HeuristicPredictor
    from tianlong.learning.featurize import featurize
    from tianlong.learning.rl.module import GraphPolicyNet
    from tianlong.learning.rl.observation import CAND_INT, ObsSpec, build_observation, observation_space

    sc = request_scene(beneficiary="hidden")
    me = scene_beliefs(sc)
    cands = (Candidate(Op.WAIT), Candidate(Op.REQUEST_ITEM, "npc", "pill", beneficiary="hidden"))
    preds = HeuristicPredictor().predict(me, me.last_tick, cands)
    tight = build_observation(me, me.last_tick, sc.profiles["player"], cands, preds,
                              ObsSpec(max_nodes=3, max_edges=12, max_cands=2))
    assert tight.candidates == (cands[0],), "四个必要实体放不下时不能留下悬空受益人"
    spec = ObsSpec(max_nodes=8, max_edges=24, max_cands=2)
    kept = build_observation(me, me.last_tick, sc.profiles["player"], cands, preds, spec)
    assert kept.candidates == cands
    bene = int(kept.obs["cand"][1, CAND_INT.index("beneficiary")])
    graph = featurize(belief_view(me, me.last_tick))
    assert torch.equal(torch.as_tensor(kept.obs["x"][bene]), torch.as_tensor(graph.x[graph.index_of("hidden")]))
    assert observation_space(spec).contains(kept.obs)
    net = GraphPolicyNet(hidden=16, layers=1).eval()
    with torch.no_grad():
        logits, value = net({key: torch.as_tensor(val).unsqueeze(0) for key, val in kept.obs.items()})
    assert logits.shape == (1, 2) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


@pytest.mark.parametrize("text", ["请求钟灵给我解药", "请求钟灵把解药交给我，以便救助我"])
def test_explicit_request_has_a_rule_fast_path_without_a_model(text):
    me = scene_beliefs(request_scene())
    def forbidden(*_):
        raise AssertionError("明确物品请求应走规则快路径")
    llm = ScriptedLLM(forbidden)
    p = Interpreter(llm).interpret(text, me)
    assert p.candidate.op == Op.REQUEST_ITEM and p.candidate.beneficiary == "player"
    assert p.kind == MoveKind.SAY and p.source == "rules"
    assert not llm.prompts


def test_request_negation_and_unknown_participants_do_not_execute():
    me = scene_beliefs(request_scene())
    for text in ("不要请求钟灵给我解药", "如果钟灵同意，我再请求她给我解药", "请求陌生人给我解药",
                 "请求钟灵把解药交给我，以便救助陌生人"):
        assert Interpreter(None).interpret(text, me).candidate is None, text


def test_old_records_decode_defaults_but_old_kernel_save_needs_explicit_migration(tmp_path):
    plain = codec.intent_to(Intent("legacy", "player", Op.WAIT))
    plain.pop("beneficiary")
    plain.pop("request_ref")
    assert codec.intent_from(plain).beneficiary is None
    mind = codec.mind_to(replace(scene_beliefs(request_scene()), yielded={"npc": 42}))
    assert codec.mind_from(mind).yielded["npc"] == 42
    mind.pop("yielded")
    assert not codec.mind_from(mind).yielded
    sc = request_scene()
    store = SQLiteWorldStore(tmp_path / "old.sqlite3")
    from tianlong.runtime.authority import WorldAuthority
    WorldAuthority.found(store, sc, versions={**current_versions(), "kernel": "kernel-v2"})
    with pytest.raises(IncompatibleSave):
        session(sc, store)
    migrated = GameSession(sc, store=store, allow_migration=True, pipeline=False)
    assert migrated.migrated_from["kernel"] == "kernel-v2"
    migrated.index.client.close()
