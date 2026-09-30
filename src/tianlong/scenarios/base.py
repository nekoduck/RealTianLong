"""
[INPUT]: 依赖 core 的 WorldState / Percept / Profile / Drive / Op / Social
[OUTPUT]: 对外提供 Scenario（初始世界 + 角色设定 + 初始认知 + 文风/外观描写/别称（common_words 只供解析）+ 逐级提示 guide（guide_at 按所在处定起点）
          + 结局 endings + 秘密词表 secrets + 各角色的驱力 drives + 外貌称呼 epithets / 开场相识 introduced / 时钟事实 moments /
          看点识别器 beats / 写法卡 cards / 纪事角色 chronicle / 细节卡组 details / 叩首几个 tick kowtow_ticks / 看不见天的地方 enclosed）、Ending（抵达某地或时钟到点即落幕，带变体）、
          Variant、Beat（看点识别器）、Card（写法卡：只有修辞，自带识别器 cues）
[POS]: scenarios 的容器类型；初始认知以“过去的感知”给出，于是信念从第一刻起就只有一个来源——感知，没有“直接注入信念”的后门。
       新字段一律默认空（kowtow_ticks=1）：仓库、程序化世界与旧版无量山逐字节不变
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tianlong.core import Drive, Op, Percept, Social, WorldState
from tianlong.core.profiles import Profile


@dataclass(frozen=True, slots=True)
class Variant:
    """结局的变体：requires 里任一条（据世界真相）成立就在标题后缀上 label。
    requires 取 "skill:技能"（玩家已会）| "with:人"（此人与玩家同在一处）| "holds:物"（在玩家身上）| "at:地点"（玩家身在此处）。"""

    label: str
    requires: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ending:
    """本幕结局：玩家（据世界真相）身处 place，或时钟已到 at_clock，即落幕。key 是稳定的结局编号，title 是给玩家看的名字，
    epilogue 是终章的前提与基调（交给叙述者，真相揭晓或纪事由事件日志生成），variants 按落幕那一刻的真相给标题加后缀。"""

    key: str
    title: str
    place: str | None = None
    epilogue: str = ""
    at_clock: int | None = None
    variants: tuple[Variant, ...] = ()


@dataclass(frozen=True, slots=True)
class Beat:
    """看点识别器（不是触发器）：只匹配已结算、且玩家已感知到的事件或景观。字段为空即不限。
    op/actors/target/obj/social/reason 对感知到的事件；door：经这道门的 MOVE，或感知的事实揭示了这道门；
    place：事件发生处（景观：玩家所在处）；clock_from：时钟下限（整数，或 moments 的键）；
    lore：景观——玩家在 place 看见 lore 键所指的实体（"yubi@moon" 指 yubi），没有事件；
    status：感知里这一下让某人有了这个状态（点穴得手：subdued）；words：感知到的原话或姿态里须有这几个字（“举着火把”），空 = 不限；
    gloss 是玩家口吻的一句看点，stage 是认出它时生效的写法卡（Scenario.cards 的键，只有修辞），allowed 是额外许可词
    （只许点名，不许状态）；once：一局只算一次（等待只为它停一次）。同一个 key 可以有几条识别器（“叫阵或动手”）。"""

    key: str
    op: Op | None = None
    actors: tuple[str, ...] = ()
    target: tuple[str, ...] = ()
    obj: str | None = None
    door: str | None = None
    place: tuple[str, ...] = ()
    clock_from: int | str | None = None
    social: Social | None = None
    reason: str | None = None
    success: bool = True
    gloss: str = ""
    stage: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()
    lore: str = ""
    once: bool = False
    status: str | None = None
    words: str = ""


@dataclass(frozen=True, slots=True)
class Card:
    """写法卡：交给叙述者的一段修辞（貂扑出时是“灰白影子一闪”），不加任何事实——文字须在它生效的场面里过得了叙述闸门。
    生效于：某个看点认出时（Beat.stage 点了它），或它自己的识别器 cues 认出时（与看点同一套匹配，但不计 B1、不停等待）。"""

    text: str
    cues: tuple[Beat, ...] = ()


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
    endings: tuple[Ending, ...] = ()   # 本幕的结局：玩家抵达某地（或时钟到点）即落幕，终章据事件日志收束
    secrets: tuple[str, ...] = ()      # 剧情秘密的说法（正则片段，如“私奔”）：只交给叙述闸门，玩家没听说过就不许写进正文
    guide_at: Mapping[str, int] = field(default_factory=dict)   # 玩家身在某地时提示至少从第几条说起（走过的路不再提）
    common_words: frozenset[str] = frozenset()  # 别称里同时是普通名词的（“石壁”）：只供解析，闸门不据此拒绝
    drives: Mapping[str, tuple[Drive, ...]] = field(default_factory=dict)   # 角色 → 驱力（性情，按表序）：会话把它套在策略外面；
                                                                            # 没有驱力的角色（及仓库、程序化世界、旧版）一切照旧
    epithets: Mapping[str, str] = field(default_factory=dict)   # 外貌称呼：没人道出姓名之前怎样叫此人（“青衫少女”）
    introduced: Mapping[str, frozenset[str]] = field(default_factory=dict)   # 开场谁已认得谁（知其名）
    moments: Mapping[str, int] = field(default_factory=dict)    # 场景的时钟事实（"moon" 月出、"dawn" 天亮）：等待与看点据此换算
    beats: tuple[Beat, ...] = ()                                # 看点识别器（B1 与等待的停点）
    cards: Mapping[str, Card] = field(default_factory=dict)     # 写法卡目录（只有修辞；叙述者的系统提示里列全，本回合只点编号）
    chronicle: tuple[str, ...] = ()                             # 终章纪事取材的角色；非空时纪事代替旧的真相揭晓
    details: Mapping[str, tuple[str, ...]] = field(default_factory=dict)   # 地点或陈设的细节卡组：每查看一次给下一条没给过的
    kowtow_ticks: int = 1                                       # 对着可拜的陈设叩首占几个 tick（>1 时展开为多 tick 叩首再细看）
    enclosed: frozenset[str] = frozenset()                      # 看不见天的地方（石洞、石室、隧道）：天色只报时辰，不提日月

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
