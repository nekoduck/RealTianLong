"""
[INPUT]: 汇总 scenarios 各模块
[OUTPUT]: 对外提供 Scenario、Ending、Variant、Beat、Card、build_warehouse、build_wuliang、build_wuliang_commoner、SCENARIOS 注册表
[POS]: scenarios 包入口；场景是纯内容层，只依赖 core 与 kernel 的感知构造器
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from collections.abc import Callable

from tianlong.scenarios.base import Beat, Card, Ending, Scenario, Variant
from tianlong.scenarios.tianlong import build_wuliang, build_wuliang_commoner
from tianlong.scenarios.warehouse import build_warehouse

# 场景注册表：命令行 --world 按名取用。"wuliang" 是普通人版（玩家是挑茶伙计阿顺），"wuliang-duanyu" 是段誉作主角的旧版
SCENARIOS: dict[str, Callable[[int], Scenario]] = {
    "warehouse": build_warehouse, "wuliang": build_wuliang_commoner, "wuliang-duanyu": build_wuliang}

__all__ = ["SCENARIOS", "Beat", "Card", "Ending", "Scenario", "Variant", "build_warehouse", "build_wuliang",
           "build_wuliang_commoner"]
