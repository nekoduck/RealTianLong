"""
[INPUT]: 依赖 core 的 WorldState / Percept / Profile
[OUTPUT]: 对外提供 Scenario（初始世界 + 角色设定 + 初始认知 + 文风/外观描写/别称（common_words 只供解析）+ 逐级提示 guide（guide_at 按所在处定起点）
          + 结局 endings + 秘密词表 secrets）、Ending
[POS]: scenarios 的容器类型；初始认知以“过去的感知”给出，于是信念从第一刻起就只有一个来源——感知，没有“直接注入信念”的后门
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tianlong.core import Percept, WorldState
from tianlong.core.profiles import Profile


@dataclass(frozen=True, slots=True)
class Ending:
    """本幕结局：玩家（据世界真相）身处 place 即落幕。key 是稳定的结局编号，title 是给玩家看的名字，
    epilogue 是终章的前提与基调（交给叙述者，真相揭晓部分由事件日志生成）。"""

    key: str
    title: str
    place: str
    epilogue: str = ""


@dataclass(frozen=True, slots=True)
class Scenario:
    world_id: str
    state: WorldState
    profiles: Mapping[str, Profile]
    priors: Mapping[str, tuple[Percept, ...]]
    setting: str = ""                                    # 世界前提：开场讲给玩家，也交给叙述者
    lore: Mapping[str, str] = field(default_factory=dict)   # 实体外观描写（键 "id" 或 "id@night"），只在被看见时用于叙述
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # 实体别称，供玩家输入解析
    hints: str = ""                                                     # 给玩家的指令示例
    style: str = ""                                                     # 文风要求：只交给 LLM 叙述者
    guide: tuple[str, ...] = ()        # 逐级提示（/hint、“我该做什么”）：由浅入深，只点方向不给步骤；玩家目标见其 Profile.goals
    endings: tuple[Ending, ...] = ()   # 本幕的结局：玩家抵达某地即落幕，终章据事件日志收束并揭晓真相
    secrets: tuple[str, ...] = ()      # 剧情秘密的说法（正则片段，如“私奔”）：只交给叙述闸门，玩家没听说过就不许写进正文
    guide_at: Mapping[str, int] = field(default_factory=dict)   # 玩家身在某地时提示至少从第几条说起（走过的路不再提）
    common_words: frozenset[str] = frozenset()  # 别称里同时是普通名词的（“石壁”）：只供解析，闸门不据此拒绝

    @property
    def gate_aliases(self) -> dict[str, tuple[str, ...]]:
        """交给叙述闸门的别称：去掉同时是普通名词的（“石壁”在石洞里只是石壁），解析玩家输入仍用全部别称。"""
        return {k: tuple(a for a in v if a not in self.common_words) for k, v in self.aliases.items()}

    @property
    def player(self) -> str | None:
        return next((p.agent for p in self.profiles.values() if p.is_player), None)

    @property
    def npcs(self) -> tuple[str, ...]:
        return tuple(sorted(a for a, p in self.profiles.items() if not p.is_player))
