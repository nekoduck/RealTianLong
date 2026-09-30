"""
[INPUT]: 依赖 core/schema 的 Op / Manner / Kind / Social，core/changes 的 Change，core/propositions 的 Fact，core/entities 的 Scalar
[OUTPUT]: 对外提供 Intent / Outcome / Event / PerceivedEvent / Modality / EntitySketch（含亲见/只闻其名）/ Percept / Observation、
          结算原因封闭词表 RULE_REASONS / ADMISSION_REASONS / REASONS 与 reason_key()
[POS]: core 的因果链数据：意图 → 事件（真相，含变化）→ 观察（服务端溯源记录）→ 感知（角色可见的片面内容）；
       原话/姿态（utterance）与言语行为（social）只是修辞：随感知传给在场的人，从不产生事实；
       MOVE 的 obj 是所走的路线（门），目的地与路线一起构成行动，内核不替角色挑路；
       原因词表是 kernel（产出）与 learning（编码）之间的契约
       物品请求与回应随 beneficiary/request_ref 传递；request_ref 是语义编号，不是世界事件 ID。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tianlong.core.changes import Change
from tianlong.core.entities import Scalar
from tianlong.core.propositions import Fact
from tianlong.core.schema import Kind, Manner, Op, Social

# ============================================================
#  意图：角色想做什么（结构化，尚未发生）
#  based_on 记录决策依据的世界版本——过时意图会被权威写入器拒绝
# ============================================================


@dataclass(frozen=True, slots=True)
class Intent:
    id: str
    actor: str
    op: Op
    target: str | None = None
    obj: str | None = None          # 工具或被操作的物件（开锁的钥匙、递交的物品）；MOVE 时是所走的路线（门）
    manner: Manner = Manner.NORMAL
    topic: Fact | None = None       # 语义内容：tell 的命题（可以是谎言）、ask 的问题
    based_on: int = 0
    utterance: str | None = None    # 言语的表层文字，或 WAIT 时看得见的姿态；只是修辞，事实内容以 topic 为准
    social: Social | None = None    # 言语/姿态的社交含义（赔罪、威胁……）：修辞层，只影响旁人的态度与回应
    beneficiary: str | None = None # REQUEST_ITEM 的受益人；物品仍请求交给 actor，由持有者随后施用
    request_ref: str | None = None # TELL/GIVE 对哪条已听见请求的回应；独立语义编号，非世界事件溯源 ID


# ============================================================
#  事件：实际发生了什么（真相）。只存在于世界日志，不直接交给角色
# ============================================================


class Outcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"    # 尝试了但没成功（门锁着、东西不在）——失败本身也是事实
    REJECTED = "rejected"  # 根本没进入世界：语法非法、过时版本


# ============================================================
#  结算原因的封闭词表：规则只能使用登记过的原因（kernel/resolution 在构造时校验），
#  learning 据此把事件节点的“原因”编码成定长特征——新增原因必须先在这里登记，词表指纹随之改变
# ============================================================

RULE_REASONS: tuple[str, ...] = (
    # 尚未触及世界的尝试
    "out_of_reach", "not_found", "not_holding", "already_there", "already_held", "self_target", "subdued",
    "nothing_to_learn", "already_learned", "route_not_here", "route_mismatch", "not_adjacent",
    # 触及世界之后的失败
    "held_by_other", "door_locked", "one_way", "wrong_key", "already_locked", "already_unlocked", "no_effect",
    "parried", "evaded",
    # 成功但有进境之分
    "mastered", "progress",
)
ADMISSION_REASONS: tuple[str, ...] = ("unknown_actor", "duplicate_actor", "stale", "unknown_op", "syntax")
REASONS: tuple[str, ...] = RULE_REASONS + ADMISSION_REASONS


def reason_key(reason: str | None) -> str | None:
    """语法拒绝带着细节（"syntax: ..."），归一到词表里的 "syntax"。"""
    if reason is None:
        return None
    return "syntax" if reason.startswith("syntax") else reason


@dataclass(frozen=True, slots=True)
class Event:
    id: str
    tick: int
    intent: Intent
    place: str | None
    outcome: Outcome
    reason: str | None = None
    changes: tuple[Change, ...] = ()

    @property
    def op(self) -> Op:
        return self.intent.op

    @property
    def actor(self) -> str:
        return self.intent.actor


# ============================================================
#  感知：某个角色实际获得的、片面的信息
#  - PerceivedEvent 的字段可以缺失：听到响动的人不知道是谁、做了什么
#  - Percept 不携带来源事件 ID——角色不能顺着 ID 摸到真相
#  - Observation 才携带 source_event，仅供服务端溯源与调试
# ============================================================


class Modality(StrEnum):
    SELF = "self"      # 自己行动的结果
    SIGHT = "sight"    # 亲眼看到别人的行动
    SOUND = "sound"    # 只听到响动
    SPEECH = "speech"  # 听到别人说的话（事实是“说法”，可信度取决于说话者）
    SCENE = "scene"    # 环顾四周：看到的在场事物（附带负证据的范围）


@dataclass(frozen=True, slots=True)
class PerceivedEvent:
    kind: str                       # Op 值，或 "noise"
    place: str
    actor: str | None = None
    target: str | None = None
    obj: str | None = None
    outcome: Outcome | None = None
    topic: Fact | None = None
    reason: str | None = None       # 失败原因（门锁着、没找到……）：看得见失败的人也看得见原因
    utterance: str | None = None    # 听得见的人才有：说话者的原话；看得见的人才有：姿态
    social: Social | None = None    # 与原话同进退：耳语时旁人既听不到原话，也不知道是赔罪还是威胁
    beneficiary: str | None = None
    request_ref: str | None = None


@dataclass(frozen=True, slots=True)
class EntitySketch:
    """角色得以认识某实体时获得的样子：种类、名字，以及——只有亲眼见过才有的——外观属性。

    seen=False 表示只闻其名（听人提起、隔墙听见、门那头的地点）：attrs 恒为空，外观是“未知”而不是“没有”。
    """

    id: str
    kind: Kind
    name: str
    attrs: tuple[tuple[str, Scalar], ...] = ()
    seen: bool = True


@dataclass(frozen=True, slots=True)
class Percept:
    tick: int
    modality: Modality
    event: PerceivedEvent | None = None
    facts: tuple[Fact, ...] = ()
    scopes: tuple[str, ...] = ()           # 本次完整看清其直接内容的容纳者（负证据的适用范围）
    sketches: tuple[EntitySketch, ...] = ()
    informant: str | None = None           # SPEECH 的说话者：事实只是他的说法


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    observer: str
    source_event: str | None
    percept: Percept
