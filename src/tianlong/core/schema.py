"""
[INPUT]: 依赖标准库 enum / dataclasses
[OUTPUT]: 对外提供 Kind / Rel / RelSpec / RELATIONS / ATTR_PREFIX / Op / Manner / Social（言语行为封闭词表）与 HOSTILE_SOCIAL /
          FRIENDLY_SOCIAL / OpSignature（命题“接受”与“必须”分开：言语的命题可选）/ OP_SIGNATURES / is_functional
[POS]: core 的领域词汇表；kernel 据此裁定物理，cognition 据此生成候选与修正信念，learning 据此构造特征维度。
       属性的类型与获知途径在 core/attributes（唯一真相源）
       REQUEST_ITEM 的物品与受益人有类型签名，请求不代表交付或疗效。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ============================================================
#  实体种类
#  规则：有独立生命周期、参与者和因果关系的对象才成为实体；
#        简单数值（锁状态、警觉度）作为属性挂在实体上
# ============================================================


class Kind(StrEnum):
    PERSON = "person"    # 行动者：玩家与 NPC
    PLACE = "place"      # 地点：可以“身处其中”的空间
    SURFACE = "surface"  # 家具/台面：固定在地点里、可以放东西
    ITEM = "item"        # 可携带物件
    DOOR = "door"        # 连接两个地点，可上锁


# ============================================================
#  关系类型
#  “位于(AT)”与“拥有(OWNS)”刻意正交：拿走钥匙改变位置，不改变所有权
# ============================================================


class Rel(StrEnum):
    AT = "AT"              # 物理位置（函数型：每个实体恰有一个位置）
    OWNS = "OWNS"          # 所有权：人 → 物
    MATCHES = "MATCHES"    # 钥匙 → 门
    CONNECTS = "CONNECTS"  # 门 → 地点（每扇门恰好两条）


@dataclass(frozen=True, slots=True)
class RelSpec:
    functional: bool                 # 同一 src 至多一个 dst（决定信念修正时的“互斥槽位”）
    src_kinds: frozenset[Kind]
    dst_kinds: frozenset[Kind]


RELATIONS: dict[Rel, RelSpec] = {
    Rel.AT: RelSpec(
        functional=True,
        src_kinds=frozenset({Kind.PERSON, Kind.ITEM, Kind.SURFACE}),
        dst_kinds=frozenset({Kind.PLACE, Kind.SURFACE, Kind.PERSON}),
    ),
    Rel.OWNS: RelSpec(False, frozenset({Kind.PERSON}), frozenset({Kind.ITEM})),
    Rel.MATCHES: RelSpec(False, frozenset({Kind.ITEM}), frozenset({Kind.DOOR})),
    Rel.CONNECTS: RelSpec(False, frozenset({Kind.DOOR}), frozenset({Kind.PLACE})),
}

# 属性命题的谓词前缀：命题 (door, "attr.locked", True)
ATTR_PREFIX = "attr."

def is_functional(predicate: str) -> bool:
    """谓词是否构成互斥槽位：属性天然互斥；关系看 RelSpec。"""
    if predicate.startswith(ATTR_PREFIX):
        return True
    return RELATIONS[Rel(predicate)].functional


# ============================================================
#  行动词汇：操作类型 + 目标 + 工具/对象 + 方式 + 语义内容
#  OpSignature 只描述“语法/类型”层面的合法性——
#  它可以被任何角色在只知道实体种类的前提下检查，不泄露世界真相
# ============================================================


class Op(StrEnum):
    MOVE = "move"        # 经由指定的门（路线）移动到相邻地点
    TAKE = "take"        # 拿起物件
    PUT = "put"          # 把手中物件放到地点/台面（careful 方式会藏起来）
    GIVE = "give"        # 把手中物件交给某人
    UNLOCK = "unlock"    # 用钥匙开锁
    LOCK = "lock"        # 用钥匙上锁
    INSPECT = "inspect"  # 仔细查看地点/台面（发现藏匿物）或搜身
    TELL = "tell"        # 告诉某人一个命题（可以是谎言）
    ASK = "ask"          # 向某人询问一个命题
    WAIT = "wait"        # 什么也不做
    ATTACK = "attack"    # 动手：先伤、再制住（点穴，限时自解）
    STUDY = "study"      # 研读手中的秘籍，累积进度后习得技能
    USE = "use"          # 把手中物品用在某人（含自己）身上：解药解毒
    REQUEST_ITEM = "request_item"  # 请求听者把物品交给说话者，以帮助明确的受益人；请求本身不转移物品


class Manner(StrEnum):
    NORMAL = "normal"
    CAREFUL = "careful"  # 更安静；put 时藏匿物件
    ROUGH = "rough"      # 更响、更快


class Social(StrEnum):
    """言语与姿态的社交含义（言语行为）：修辞层，不是事实——它只影响听者/看者心里的态度与回应，
    从不改变谁在哪、谁拿着什么。封闭词表：解释器、NPC 策略、叙述者共用同一套说法。"""
    GREET = "greet"            # 招呼、见礼
    THANK = "thank"
    APOLOGIZE = "apologize"    # 赔罪、认错
    PLEAD = "plead"            # 求情、求救
    PRAISE = "praise"          # 称赞、恭维
    THREATEN = "threaten"      # 威胁、亮兵刃
    TAUNT = "taunt"            # 挑衅、讥讽
    INSULT = "insult"          # 辱骂
    REFUSE = "refuse"          # 拒绝、回绝
    AGREE = "agree"            # 答应、附和
    JOKE = "joke"              # 说笑、打趣
    COMFORT = "comfort"        # 安慰、劝解
    EXPLAIN = "explain"        # 解释、讲道理、讲掌故
    CHALLENGE = "challenge"    # 叫阵、邀斗
    COMMAND = "command"        # 命令、呵斥
    FAREWELL = "farewell"      # 告辞
    REMARK = "remark"          # 随口一说、自言自语
    SUBMIT = "submit"          # 服软、磕头、示弱


HOSTILE_SOCIAL = frozenset({Social.THREATEN, Social.TAUNT, Social.INSULT, Social.CHALLENGE, Social.COMMAND})
FRIENDLY_SOCIAL = frozenset({Social.GREET, Social.THANK, Social.APOLOGIZE, Social.PLEAD, Social.PRAISE,
                             Social.COMFORT, Social.SUBMIT, Social.AGREE, Social.JOKE})


@dataclass(frozen=True, slots=True)
class OpSignature:
    target: frozenset[Kind] | None = None  # None 表示不接受目标
    obj: frozenset[Kind] | None = None
    topic: bool = False                    # 是否接受语义内容（命题）
    needs_topic: bool = False              # 是否必须带命题（言语的命题可选：没有命题就是只有原话与言语行为的闲话）
    beneficiary: frozenset[Kind] | None = None


def _k(*kinds: Kind) -> frozenset[Kind]:
    return frozenset(kinds)


OP_SIGNATURES: dict[Op, OpSignature] = {
    Op.MOVE: OpSignature(target=_k(Kind.PLACE), obj=_k(Kind.DOOR)),    # 目的地 + 路线：知道地名不等于知道路
    Op.TAKE: OpSignature(target=_k(Kind.ITEM)),
    Op.PUT: OpSignature(target=_k(Kind.PLACE, Kind.SURFACE), obj=_k(Kind.ITEM)),
    Op.GIVE: OpSignature(target=_k(Kind.PERSON), obj=_k(Kind.ITEM)),
    Op.UNLOCK: OpSignature(target=_k(Kind.DOOR), obj=_k(Kind.ITEM)),
    Op.LOCK: OpSignature(target=_k(Kind.DOOR), obj=_k(Kind.ITEM)),
    Op.INSPECT: OpSignature(target=_k(Kind.PLACE, Kind.SURFACE, Kind.PERSON)),
    Op.TELL: OpSignature(target=_k(Kind.PERSON), topic=True),   # 有命题：传递“说法”；没有：只有原话与言语行为
    Op.ASK: OpSignature(target=_k(Kind.PERSON), topic=True),
    Op.WAIT: OpSignature(),
    Op.ATTACK: OpSignature(target=_k(Kind.PERSON)),
    Op.STUDY: OpSignature(target=_k(Kind.ITEM)),
    Op.USE: OpSignature(target=_k(Kind.PERSON), obj=_k(Kind.ITEM)),
    Op.REQUEST_ITEM: OpSignature(target=_k(Kind.PERSON), obj=_k(Kind.ITEM), beneficiary=_k(Kind.PERSON)),
}
