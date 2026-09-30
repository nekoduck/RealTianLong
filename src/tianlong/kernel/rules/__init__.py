"""
[INPUT]: 依赖 kernel/rules 下各行动规则模块
[OUTPUT]: 对外提供 ActionRule 与 default_rules()（Op → 规则实例的注册表）
[POS]: kernel/rules 包入口；注册表是 kernel 的扩展点——自定义世界可传入自己的规则表
       注册 RequestItemRule；新增操作必须覆盖在默认规则表。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Op
from tianlong.kernel.rules.base import ActionRule
from tianlong.kernel.rules.combat import AttackRule
from tianlong.kernel.rules.cultivation import StudyRule, UseRule
from tianlong.kernel.rules.handling import GiveRule, PutRule, TakeRule
from tianlong.kernel.rules.locks import LockRule, UnlockRule
from tianlong.kernel.rules.movement import MoveRule, WaitRule
from tianlong.kernel.rules.senses import InspectRule
from tianlong.kernel.rules.speech import AskRule, RequestItemRule, TellRule


def default_rules() -> dict[Op, ActionRule]:
    rules: list[ActionRule] = [
        MoveRule(), WaitRule(), TakeRule(), PutRule(), GiveRule(),
        UnlockRule(), LockRule(), InspectRule(), TellRule(), AskRule(),
        AttackRule(), StudyRule(), UseRule(), RequestItemRule(),
    ]
    table = {r.op: r for r in rules}
    missing = set(Op) - set(table)
    assert not missing, f"行动缺少规则: {missing}"
    return table


__all__ = ["ActionRule", "default_rules"]
