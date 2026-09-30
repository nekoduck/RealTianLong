"""
[INPUT]: WorldStore 的冻结决策快照、GameSession 共用执行链路、choice_builder 与 choice_model
[OUTPUT]: ChoiceService.current()/resolve()，给浏览器仅公开 ID 和文案
[POS]: 决策发布只写派生菜单；消费由请求首 tick 与世界同事务完成。幂等重放先于菜单过期检查。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tianlong.core import make_id
from tianlong.persistence.store import ChoiceConflict
from tianlong.runtime.choice_builder import build_choices
from tianlong.runtime.choice_model import ChoiceSpec

if TYPE_CHECKING:
    from tianlong.runtime.session import GameSession

CHOICE_SCHEMA = 1


class ChoiceService:
    def __init__(self, session: GameSession) -> None:
        self.session = session

    def current(self) -> dict[str, Any]:
        s = self.session
        if s.ending is not None:
            return {"decision_id": None, "choices": []}
        version = s.authority.head().version
        snapshot = s.store.decision(s.ref)
        if snapshot is None or snapshot.get("version") != version:
            choices = build_choices(s.beliefs(s.player), goals=s.scenario.profiles[s.player].goals,
                                    history=s.choice_history)
            snapshot = s.store.publish_decision(s.ref, {
                "schema": CHOICE_SCHEMA, "id": make_id("decision", str(s.ref), version), "version": version,
                "choices": [c.to_data() for c in choices], "consumed_request": None,
            }, version)
        if snapshot.get("schema") != CHOICE_SCHEMA:
            raise ChoiceConflict("这份选项存档需要更新，请重新读取游戏")
        return {"decision_id": snapshot["id"],
                "choices": [{"id": c["id"], "label": c["label"]} for c in snapshot["choices"]]}

    def resolve(self, decision_id: str, choice_id: str) -> ChoiceSpec:
        s = self.session
        snapshot = s.store.decision(s.ref)
        if (snapshot is None or snapshot.get("schema") != CHOICE_SCHEMA or snapshot["id"] != decision_id
                or snapshot["version"] != s.authority.head().version or snapshot.get("consumed_request")):
            raise ChoiceConflict("局势已经变化，请刷新后重新选择")
        for choice in snapshot["choices"]:
            if choice["id"] == choice_id:
                return ChoiceSpec.from_data(choice)
        raise ChoiceConflict("这项选择不属于当前局势，请刷新后重新选择")
