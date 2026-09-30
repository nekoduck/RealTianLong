"""
[INPUT]: cognition/Candidate、language/Parsed 与 persistence/codec 的显式事实编码
[OUTPUT]: ChoiceSpec、冻结 Parsed 的 JSON 编解码与稳定动作语义键
[POS]: 玩家选项的数据边界；展示文字不参与动作解析，服务器保存完整路线、方式、话题与多步计划。
       冻结动作显式保存请求受益人与回应编号；旧菜单缺字段时取 None。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tianlong.cognition.candidates import Candidate
from tianlong.core import Manner, Op, Social, digest, make_id
from tianlong.language.parser import MoveKind, Parsed
from tianlong.persistence.codec import fact_from, fact_to


def candidate_to(c: Candidate) -> dict[str, Any]:
    return {"op": c.op.value, "target": c.target, "obj": c.obj, "manner": c.manner.value,
            "topic": fact_to(c.topic), "social": c.social.value if c.social else None,
            "beneficiary": c.beneficiary, "request_ref": c.request_ref}


def candidate_from(d: dict[str, Any]) -> Candidate:
    return Candidate(Op(d["op"]), d.get("target"), d.get("obj"), Manner(d["manner"]),
                     fact_from(d.get("topic")), Social(d["social"]) if d.get("social") else None,
                     d.get("beneficiary"), d.get("request_ref"))


def parsed_to(p: Parsed) -> dict[str, Any]:
    assert p.candidate is not None
    return {"candidate": candidate_to(p.candidate), "utterance": p.utterance, "kind": p.kind.value,
            "repeat": p.repeat, "until": p.until, "followups": [candidate_to(c) for c in p.followups]}


def parsed_from(d: dict[str, Any]) -> Parsed:
    return Parsed(candidate_from(d["candidate"]), d.get("utterance"), source="choice",
                  repeat=int(d.get("repeat", 1)), until=d.get("until"), kind=MoveKind(d["kind"]),
                  followups=tuple(candidate_from(c) for c in d.get("followups", ())))


def semantic_key(p: Parsed) -> str:
    return digest("choice-action-v1", parsed_to(p))


@dataclass(frozen=True, slots=True)
class ChoiceSpec:
    id: str
    label: str
    parsed: Parsed
    focus_id: str | None
    evidence_refs: tuple[str, ...]
    semantic_key: str

    @classmethod
    def of(cls, label: str, parsed: Parsed, focus_id: str | None = None,
           evidence_refs: tuple[str, ...] = ()) -> ChoiceSpec:
        parsed = replace(parsed, source="choice", command=None)
        key = semantic_key(parsed)
        return cls(make_id("choice", key), label, parsed, focus_id, evidence_refs, key)

    def to_data(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "parsed": parsed_to(self.parsed),
                "focus_id": self.focus_id, "evidence_refs": list(self.evidence_refs),
                "semantic_key": self.semantic_key}

    @classmethod
    def from_data(cls, d: dict[str, Any]) -> ChoiceSpec:
        return cls(d["id"], d["label"], parsed_from(d["parsed"]), d.get("focus_id"),
                   tuple(d.get("evidence_refs", ())), d["semantic_key"])
