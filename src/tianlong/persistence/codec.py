"""
[INPUT]: 依赖 core 的全部值对象，cognition 的 Belief / Episode / Obligation / Said / SocialCue，persistence/store 的 TurnEnvelope
[OUTPUT]: 对外提供 core 值对象 ⇄ JSON 兼容 dict 的显式编解码函数（intent / change / event / percept / fact / sketch / belief / episode /
          obligation（待答与待回话，命题可缺）/ said（闲话无命题、带言语行为）/ cue 社交线索 / attitudes 态度 / envelope 请求进度 / world 全世界 / mind 完整认知 / memory 经历）；
          请求含冻结选项绑定与展示文字；旧记录缺新键时取缺省（无选项绑定、言语行为为无、态度为空）
[POS]: persistence 的序列化边界；逐字段手写而非反射或 pickle——数据库里的内容不能决定构造哪个类，这是安全边界也是版本边界
       物品请求的受益人、回应编号与义务状态显式往返；yielded 随完整认知保存，不随重启丢失；旧记录缺新键取缺省。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tianlong.cognition import Belief, BeliefStore, Episode, Obligation, Said
from tianlong.cognition.agenda import SocialCue
from tianlong.core import (
    AddRelation,
    Change,
    Entity,
    EntitySketch,
    Event,
    Fact,
    Intent,
    Kind,
    Manner,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    Relation,
    RemoveRelation,
    SetAttr,
    Social,
    WorldState,
)
from tianlong.core.memories import MemoryRecord
from tianlong.persistence.store import ChoiceUse, TurnEnvelope

J = dict[str, Any]


# ============================================================
#  命题与事实
# ============================================================


def prop_to(p: Proposition) -> J:
    return {"s": p.subject, "p": p.predicate, "v": p.value}


def prop_from(d: J) -> Proposition:
    return Proposition(d["s"], d["p"], d["v"])


def fact_to(f: Fact | None) -> J | None:
    return None if f is None else {"prop": prop_to(f.prop), "holds": f.holds}


def fact_from(d: J | None) -> Fact | None:
    return None if d is None else Fact(prop_from(d["prop"]), bool(d["holds"]))


# ============================================================
#  意图、变化、事件
# ============================================================


def _social(v: str | None) -> Social | None:
    return Social(v) if v else None       # 旧记录没有这个键：None


def intent_to(it: Intent) -> J:
    return {"id": it.id, "actor": it.actor, "op": it.op.value, "target": it.target, "obj": it.obj,
            "manner": it.manner.value, "topic": fact_to(it.topic), "based_on": it.based_on, "utterance": it.utterance,
            "social": it.social.value if it.social else None, "beneficiary": it.beneficiary, "request_ref": it.request_ref}


def intent_from(d: J) -> Intent:
    return Intent(d["id"], d["actor"], Op(d["op"]), d.get("target"), d.get("obj"), Manner(d["manner"]),
                  fact_from(d.get("topic")), int(d["based_on"]), d.get("utterance"), _social(d.get("social")),
                  d.get("beneficiary"), d.get("request_ref"))


def change_to(c: Change) -> J:
    if isinstance(c, SetAttr):
        return {"t": "set", "e": c.entity, "k": c.key, "old": c.old, "new": c.new}
    tag = "add" if isinstance(c, AddRelation) else "remove"
    return {"t": tag, "src": c.rel.src, "rel": c.rel.type.value, "dst": c.rel.dst}


def change_from(d: J) -> Change:
    if d["t"] == "set":
        return SetAttr(d["e"], d["k"], d["old"], d["new"])
    rel = Relation(d["src"], Rel(d["rel"]), d["dst"])
    return AddRelation(rel) if d["t"] == "add" else RemoveRelation(rel)


def event_to(e: Event) -> J:
    return {"id": e.id, "tick": e.tick, "intent": intent_to(e.intent), "place": e.place,
            "outcome": e.outcome.value, "reason": e.reason, "changes": [change_to(c) for c in e.changes]}


def event_from(d: J) -> Event:
    return Event(d["id"], int(d["tick"]), intent_from(d["intent"]), d.get("place"), Outcome(d["outcome"]),
                 d.get("reason"), tuple(change_from(c) for c in d["changes"]))


# ============================================================
#  感知
# ============================================================


def sketch_to(s: EntitySketch) -> J:
    return {"id": s.id, "kind": s.kind.value, "name": s.name, "attrs": [list(a) for a in s.attrs], "seen": s.seen}


def sketch_from(d: J) -> EntitySketch:
    return EntitySketch(d["id"], Kind(d["kind"]), d["name"], tuple((k, v) for k, v in d["attrs"]), bool(d.get("seen", True)))


def pevent_to(v: PerceivedEvent | None) -> J | None:
    if v is None:
        return None
    return {"kind": v.kind, "place": v.place, "actor": v.actor, "target": v.target, "obj": v.obj,
            "outcome": v.outcome.value if v.outcome else None, "topic": fact_to(v.topic),
            "reason": v.reason, "utterance": v.utterance, "social": v.social.value if v.social else None,
            "beneficiary": v.beneficiary, "request_ref": v.request_ref}


def pevent_from(d: J | None) -> PerceivedEvent | None:
    if d is None:
        return None
    return PerceivedEvent(d["kind"], d["place"], d.get("actor"), d.get("target"), d.get("obj"),
                          Outcome(d["outcome"]) if d.get("outcome") else None, fact_from(d.get("topic")),
                          d.get("reason"), d.get("utterance"), _social(d.get("social")),
                          d.get("beneficiary"), d.get("request_ref"))


def percept_to(p: Percept) -> J:
    return {"tick": p.tick, "modality": p.modality.value, "event": pevent_to(p.event),
            "facts": [fact_to(f) for f in p.facts], "scopes": list(p.scopes),
            "sketches": [sketch_to(s) for s in p.sketches], "informant": p.informant}


def percept_from(d: J) -> Percept:
    return Percept(int(d["tick"]), Modality(d["modality"]), pevent_from(d.get("event")),
                   tuple(fact_from(f) for f in d["facts"]), tuple(d["scopes"]),  # type: ignore[misc]
                   tuple(sketch_from(s) for s in d["sketches"]), d.get("informant"))


# ============================================================
#  认知
# ============================================================


def belief_to(b: Belief) -> J:
    return {"holds": b.holds, "confidence": b.confidence, "modality": b.modality.value,
            "learned_at": b.learned_at, "informant": b.informant}


def belief_from(prop: Proposition, d: J) -> Belief:
    return Belief(prop, bool(d["holds"]), float(d["confidence"]), Modality(d["modality"]),
                  int(d["learned_at"]), d.get("informant"))


def episode_to(e: Episode) -> J:
    return {"tick": e.tick, "modality": e.modality.value, "event": pevent_to(e.event), "informant": e.informant}


def episode_from(d: J) -> Episode:
    ev = pevent_from(d["event"])
    assert ev is not None
    return Episode(int(d["tick"]), Modality(d["modality"]), ev, d.get("informant"))


def obligation_to(o: Obligation) -> J:
    return {"kind": o.kind, "counterpart": o.counterpart, "topic": fact_to(o.topic), "since": o.since,
            "social": o.social.value if o.social else None, "item": o.item, "beneficiary": o.beneficiary,
            "request_ref": o.request_ref, "state": o.state}


def obligation_from(d: J) -> Obligation:
    return Obligation(d["kind"], d["counterpart"], fact_from(d.get("topic")), int(d["since"]), _social(d.get("social")),
                      d.get("item"), d.get("beneficiary"), d.get("request_ref"), d.get("state", "pending"))


def said_to(s: Said) -> J:
    return {"listener": s.listener, "fact": fact_to(s.fact), "tick": s.tick, "social": s.social.value if s.social else None}


def said_from(d: J) -> Said:
    return Said(d["listener"], fact_from(d.get("fact")), int(d["tick"]), _social(d.get("social")))


def cue_to(c: SocialCue) -> J:
    return {"frm": c.frm, "op": c.op.value, "social": c.social.value if c.social else None, "tick": c.tick,
            "utterance": c.utterance, "to": c.to}


def cue_from(d: J) -> SocialCue:
    return SocialCue(d["frm"], Op(d["op"]), _social(d.get("social")), int(d["tick"]), d.get("utterance"), d.get("to"))


def attitudes_to(a: Mapping[str, int]) -> J:
    return {k: int(v) for k, v in sorted(a.items())}


def attitudes_from(d: J | None) -> dict[str, int]:
    return {str(k): int(v) for k, v in (d or {}).items()}      # 旧存档没有态度：空


# ============================================================
#  请求进度：叙述文字不进这份 JSON，由 record_render 单独补写
# ============================================================


def envelope_to(e: TurnEnvelope) -> J:
    return {"request_id": e.request_id, "payload_hash": e.payload_hash, "intent": intent_to(e.intent),
            "planned_ticks": e.planned_ticks, "start_version": e.start_version, "start_clock": e.start_clock,
            "versions": list(e.versions), "ticks": list(e.ticks), "percepts": [percept_to(p) for p in e.percepts],
            "fresh": list(e.fresh), "done": e.done, "source": e.source,
            "followups": [intent_to(i) for i in e.followups], "reaction": e.reaction,
            "choice": {"decision_id": e.choice.decision_id, "choice_id": e.choice.choice_id} if e.choice else None,
            "command": e.command}


def envelope_from(d: J, narration: str | None = None) -> TurnEnvelope:
    return TurnEnvelope(d["request_id"], d["payload_hash"], intent_from(d["intent"]), int(d["planned_ticks"]),
                        int(d["start_version"]), int(d["start_clock"]), tuple(int(v) for v in d["versions"]),
                        tuple(int(t) for t in d["ticks"]), tuple(percept_from(p) for p in d["percepts"]),
                        tuple(d["fresh"]), bool(d["done"]), d.get("source", "rules"), narration,
                        tuple(intent_from(i) for i in d.get("followups", ())), bool(d.get("reaction", False)),
                        ChoiceUse(d["choice"]["decision_id"], d["choice"]["choice_id"]) if d.get("choice") else None,
                        d.get("command", ""))


def world_to(s: WorldState) -> J:
    return {"seed": s.seed, "version": s.version, "clock": s.clock,
            "entities": [{"id": e.id, "kind": e.kind.value, "name": e.name, "attrs": list(e.attrs)}
                         for e in s.entities.values()],
            "relations": [list(r.sort_key()) for r in s.sorted_relations()]}


def world_from(d: J) -> WorldState:
    return WorldState.build(d["seed"], d["clock"],
                            [Entity(e["id"], Kind(e["kind"]), e["name"], tuple(tuple(a) for a in e["attrs"]))
                             for e in d["entities"]],
                            [Relation(s, Rel(r), t) for s, r, t in d["relations"]], d["version"])


def mind_to(s: BeliefStore) -> J:
    return {"owner": s.owner, "entities": [sketch_to(e) for e in s.entities.values()],
            "beliefs": [{"prop": prop_to(b.prop), **belief_to(b)} for b in s.sorted_beliefs()],
            "episodes": [episode_to(e) for e in s.episodes], "trust": dict(s.trust), "last_tick": s.last_tick,
            "surveyed": dict(s.surveyed), "searched": dict(s.searched),
            "obligations": [obligation_to(o) for o in s.obligations], "said": [said_to(x) for x in s.said],
            "cues": [cue_to(c) for c in s.cues], "attitudes": dict(s.attitudes),
            "company": dict(s.company), "allies": list(s.allies), "yielded": dict(s.yielded)}


def mind_from(d: J) -> BeliefStore:
    sketches = [sketch_from(e) for e in d["entities"]]
    beliefs = [belief_from(prop_from(b["prop"]), b) for b in d["beliefs"]]
    return BeliefStore(d["owner"], {s.id: s for s in sketches}, {b.prop: b for b in beliefs},
                       tuple(episode_from(e) for e in d["episodes"]), d["trust"], d["last_tick"],
                       d["surveyed"], d["searched"], tuple(obligation_from(o) for o in d["obligations"]),
                       tuple(said_from(s) for s in d["said"]), tuple(cue_from(c) for c in d["cues"]),
                       d["attitudes"], d["company"], tuple(d["allies"]), d.get("yielded", {}))


def memory_to(m: MemoryRecord) -> J:
    return {"id": m.id, "world_id": m.world_id, "branch_id": m.branch_id, "owner": m.owner,
            "kind": m.kind, "text": m.text, "occurred_at": m.occurred_at, "known_at": m.known_at,
            "source": m.source, "subjects": list(m.subjects), "informant": m.informant, "verdict": m.verdict}


def memory_from(d: J) -> MemoryRecord:
    return MemoryRecord(d["id"], d["world_id"], d["branch_id"], d["owner"], d["kind"], d["text"],
                        d["occurred_at"], d["known_at"], d["source"], tuple(d["subjects"]),
                        d.get("informant"), d.get("verdict"))
