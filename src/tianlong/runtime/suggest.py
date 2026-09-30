"""
[INPUT]: 玩家 BeliefStore 与 choice_builder 的结构化选择
[OUTPUT]: suggestions() 仅为旧调用方提供选项文案
[POS]: 兼容入口；菜单由 ChoiceService 冻结并以 ID 执行，无固定分类槽位或“研读过就删除”的规则。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.cognition.beliefs import BeliefStore
from tianlong.runtime.choice_builder import build_choices


def suggestions(me: BeliefStore, limit: int = 3) -> tuple[str, ...]:
    return tuple(c.label for c in build_choices(me, limit))
