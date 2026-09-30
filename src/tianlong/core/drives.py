"""
[INPUT]: 依赖 core/schema 的 Op / Social，core/events 的 Outcome
[OUTPUT]: 对外提供 Level（URGENT / IDLE / VETO）、驱力条件（At / Between / Here / Status / Saw / Heard / Holds / HeldBy / Knows /
          Fond / Menaced / Alone / Arrived / Searched / Fired / Lost / Not / AnyOf，合称 Cond）、驱力行动（Go / Flee / Follow /
          Pursue / Ask / Cross / Inspect / Take / Study / Use / Give / Pose / Say / Hold，合称 Act）、Drive、SELF / PLAYER / TARGET
[POS]: core 的驱力词表：角色性情的类型化写法（与 Goal 同属角色设定），纯数据、冻结、可哈希，不带语义——
       条件怎样在角色自己的认知上求值、行动怎样落到候选集上，归 agents/drives；驱力只产出意图，事实仍只由内核裁定。
       who 取 "self"（自己）| "player"（主角）| "target"（只在 VETO 里：被否决那一步的对象）| 实体 ID。
       技能也是状态：Status("self", "evasion") 即“自己已会凌波微步”。
       VETO 的时间窗一律以驱力自己的标记计（Fired），不看会被挤掉的线索与经历
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tianlong.core.events import Outcome
from tianlong.core.schema import Op, Social

SELF, PLAYER, TARGET = "self", "player", "target"


class Level(StrEnum):
    URGENT = "urgent"   # 先于脚本：按表序取第一条成立、不在冷却、实现得了的
    IDLE = "idle"       # 脚本闲着（等待或闲谈）时才试
    VETO = "veto"       # 对最终选择把关：选了 veto 所指的行动且条件成立，就改做 do（缺省原地不动）


# ============================================================
#  条件：只在该角色自己的认知（Situation）与自己的驱力标记上求值
# ============================================================


@dataclass(frozen=True, slots=True)
class At:
    place: str                       # 自己以为身在此处


@dataclass(frozen=True, slots=True)
class Between:
    start: int                       # 时钟窗口 [start, end)；end=None 即此后一直成立。start 落进两次决策之间就唤醒
    end: int | None = None


@dataclass(frozen=True, slots=True)
class Here:
    who: str                         # 以为此人就在眼前


@dataclass(frozen=True, slots=True)
class Status:
    who: str = SELF
    status: str = "wounded"          # 以为此人处于该状态（wounded / poisoned / subdued / 技能名）


@dataclass(frozen=True, slots=True)
class Saw:
    """近 within 个 tick 的经历里有别人的这样一次行动（亲眼看见、亲耳听见或身受）；字段为 None 即不限。
    MOVE 的 obj 是所走的门。"""

    op: Op | None = None
    actor: str | None = None
    target: str | None = None
    obj: str | None = None
    socials: frozenset[Social] = frozenset()
    outcome: Outcome | None = None
    within: int = 3


@dataclass(frozen=True, slots=True)
class Heard:
    """近 within 个 tick 听见 speaker 开口（任何带原话或言语行为的行动，不只是线索）；socials 为空即不限，to 限定冲着谁。"""

    speaker: str
    socials: frozenset[Social] = frozenset()
    to: str | None = None
    within: int = 3


@dataclass(frozen=True, slots=True)
class Holds:
    item: str                        # 以为在自己身上


@dataclass(frozen=True, slots=True)
class HeldBy:
    item: str
    who: str | None = None           # None：在眼前的别人身上


@dataclass(frozen=True, slots=True)
class Knows:
    eid: str                         # 认识（听说过也算）


@dataclass(frozen=True, slots=True)
class Fond:
    who: str = PLAYER
    at_least: int = 1                # 对此人的态度 ≥ at_least


@dataclass(frozen=True, slots=True)
class Menaced:
    at_most: int = -2                # 眼前有态度 ≤ at_most、以为既没中毒也没被制的人


@dataclass(frozen=True, slots=True)
class Alone:
    allowed: tuple[str, ...] = ()    # 眼前除了 allowed 没有别人


@dataclass(frozen=True, slots=True)
class Arrived:
    """allowed 之外有人新来：company 记下的“自何时起在身边”早于此刻、且不出 within 个 tick。
    同一 tick 走进来的人（since == now）不算：他那一 tick 听不见正常音量的话，推迟一 tick 才拦。"""

    allowed: tuple[str, ...] = ()
    within: int = 1


@dataclass(frozen=True, slots=True)
class Searched:
    eid: str                         # 亲手仔细翻查过此处


@dataclass(frozen=True, slots=True)
class Fired:
    key: str                         # 自己的某条驱力兑现过 ≥ times 次（within 个 tick 内；None 即不限时）
    within: int | None = None
    times: int = 1


@dataclass(frozen=True, slots=True)
class Lost:
    who: str                         # 此人不在眼前，追人（避开认为单向的门）已无路可走：到他那里要经单向门，或无处可找、无人可问


@dataclass(frozen=True, slots=True)
class Not:
    cond: Cond


@dataclass(frozen=True, slots=True)
class AnyOf:
    conds: tuple[Cond, ...]


Cond = (At | Between | Here | Status | Saw | Heard | Holds | HeldBy | Knows | Fond | Menaced | Alone | Arrived
        | Searched | Fired | Lost | Not | AnyOf)


# ============================================================
#  行动：只落到候选集（或候选之外的一句话、一个带字的姿态）上；实现不了就退回脚本
# ============================================================


@dataclass(frozen=True, slots=True)
class Go:
    place: str
    avoid_oneway: bool = True        # 路线不经认为单向的门（跳下去就回不来）
    careful: bool = False


@dataclass(frozen=True, slots=True)
class Flee:
    prefer: tuple[str, ...] = ()     # 往哪里逃的先后；其余按“没去过的”在前，平手由种子定


@dataclass(frozen=True, slots=True)
class Follow:
    who: str                         # 跟上此人（他不在眼前才有意义）


@dataclass(frozen=True, slots=True)
class Pursue:
    who: str                         # 找到他、制住他（先礼后兵照旧）
    avoid_oneway: bool = True


@dataclass(frozen=True, slots=True)
class Ask:
    about: str                       # 不知此人（物）下落时向眼前的人打听；知道就不问
    who: str | None = None           # None：眼前近来没问过的人


@dataclass(frozen=True, slots=True)
class Cross:
    door: str


@dataclass(frozen=True, slots=True)
class Inspect:
    eid: str


@dataclass(frozen=True, slots=True)
class Take:
    item: str


@dataclass(frozen=True, slots=True)
class Study:
    item: str


@dataclass(frozen=True, slots=True)
class Use:
    item: str
    on: str = SELF


@dataclass(frozen=True, slots=True)
class Give:
    item: str
    to: str


@dataclass(frozen=True, slots=True)
class Pose:
    text: str                        # 看得见的姿态：原地等待带上这句字，在场的人当 tick 就看得见
    social: Social | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("姿态必须带字：不带字的等待谁也看不见")


@dataclass(frozen=True, slots=True)
class Say:
    to: str
    social: Social
    careful: bool = False            # 压低声音


@dataclass(frozen=True, slots=True)
class Hold:
    """原地不动（不带字）。"""


Act = Go | Flee | Follow | Pursue | Ask | Cross | Inspect | Take | Study | Use | Give | Pose | Say | Hold


# ============================================================
#  驱力
# ============================================================


@dataclass(frozen=True, slots=True)
class Drive:
    """key 在角色内唯一；when 全部成立才触发；do 依次试，第一条实现得了的为准（VETO 可空：改为原地不动）。
    line 是一次性的修辞（成为意图的原话）；lines 按兑现次数轮换（起点由种子定）；两者都没有时姿态的字就是原话。
    once：兑现一次后不再触发；cooldown：兑现后多少个 tick 内不再触发——“兑现”指其意图结算成功（驱力标记）。
    veto：VETO 否决的行动；gloss：给人看的一句说明（对照组圣经用）。"""

    key: str
    level: Level
    when: tuple[Cond, ...] = ()
    do: tuple[Act, ...] = ()
    line: str = ""
    lines: tuple[str, ...] = ()
    once: bool = False
    cooldown: int = 0
    veto: Op | None = None
    gloss: str = ""

    def __post_init__(self) -> None:
        for name in ("when", "do"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, (value,))       # 单条也可以直接写
        if (self.level == Level.VETO) != (self.veto is not None):
            raise ValueError(f"驱力 {self.key}：VETO 必须且只有 VETO 指明否决的行动")
        if self.level != Level.VETO and not self.do:
            raise ValueError(f"驱力 {self.key}：没有行动")
        if self.line and self.lines:
            raise ValueError(f"驱力 {self.key}：line 与 lines 只取其一")
