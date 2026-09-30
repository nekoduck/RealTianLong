"""
[INPUT]: 依赖标准库 dataclasses / enum
[OUTPUT]: 对外提供 GoalKind / Goal / Profile
[POS]: core 的角色设定卡（含时间闸门与盟友，以及主持层用的腔调 voice、谈资 knows、公开来历 intro、话多 chatty、脾气 temper）；
       agents 的脚本策略据此行动（回话、搭话、先礼后兵），learning 的奖励据此计算，主持人之声据 voice 写台词——
       目标是角色条件化的，不存在统一的“剧情精彩度”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GoalKind(StrEnum):
    PROTECT = "protect"  # 让物品留在 home 或回到所有者手中
    ACQUIRE = "acquire"  # 把物品弄到自己身上
    DELIVER = "deliver"  # 把物品交到 recipient 手中
    GUARD = "guard"      # 守住地点 home：不许盟友之外的人在此逗留
    HOSTILE = "hostile"  # 与 person 为敌：找到他、制住他
    ESCAPE = "escape"    # 悄悄到达 home；途中撞见外人便灭口
    DEFEND = "defend"    # 护着 person：谁对他动手，就对谁动手


@dataclass(frozen=True, slots=True)
class Goal:
    kind: GoalKind
    item: str | None = None       # PROTECT / ACQUIRE / DELIVER 的物品
    home: str | None = None       # PROTECT 的原处；GUARD / ESCAPE 的地点
    recipient: str | None = None  # DELIVER 的收件人
    person: str | None = None     # HOSTILE / DEFEND 的对象
    weight: float = 1.0
    not_before: int | None = None  # 时间闸门：此刻之前不追求（“入夜才动身”）
    until: str = "subdued"         # HOSTILE 的了结条件：对方受伤（wounded）即解气，或须制住（subdued）

    def active(self, now: int) -> bool:
        return self.not_before is None or now >= self.not_before


@dataclass(frozen=True, slots=True)
class Profile:
    agent: str
    role: str
    persona: str
    goals: tuple[Goal, ...] = ()
    is_player: bool = False
    trust: tuple[tuple[str, float], ...] = ()   # 对他人说法的信任度（未列出者取默认值）
    allies: tuple[str, ...] = ()                # 自己人：守地、灭口时不会对他们动手
    voice: str = ""                             # 说话的腔调与待人的样子（公开的一面）：主持人之声据此写他的台词
    knows: str = ""                             # 谈资：此人知道、且肯对人讲的掌故与背景（不含秘密），按“；”分条
    intro: str = ""                             # 公开的来历：认识他的人都知道的身份（不含秘密），别人被问到“他是什么来头”时据此回答
    chatty: float = 0.0                         # 主动搭话的倾向 [0, 1]：话多的人在玩家身边会找话说
    temper: float = 0.0                         # 脾气 [-1, 1]：越高越受不得激（被辱即翻脸），越低越能忍

    def interests(self) -> tuple[str, ...]:
        """目标涉及的物品与人物：言语话题、预测器“预期获知”的关注点都以此为准。"""
        found = {x for g in self.goals for x in (g.item, g.recipient, g.person) if x}
        return tuple(sorted(found))
