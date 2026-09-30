"""
[INPUT]: 依赖 core 的 ATTRIBUTES / ATTR_SPECS / AttrType / DYNAMIC_ATTRS / ATTRS_VERSION / REASONS / Kind / Op / Manner / Modality /
         Outcome / Rel / ATTR_PREFIX / digest，cognition/view 的 VIEW_RELS / EVENT_RELS / EVENT_KIND
[OUTPUT]: 对外提供 FEATURES_VERSION / SCHEMA（特征 + 行动 + 目标 + 预测目标的规格指纹）、StaleModel / check_schema()、
          节点列布局（AttrBlock / ATTR_BLOCKS / F_NODE 及各事件列切片、个人探索记录列 SURVEY）、REL_VOCAB / F_EDGE、TOPIC_PREDICATES、
          行动编码字段 ACTION_FIELDS、预测目标规格（DYN_BOOL / DYN_NUM / HOLDER_EXTRA / TARGETS）
[POS]: learning 的类型化特征规格（Schema v2）——一张表说清“每个属性占哪几列、怎样表示已知/未知/不适用、
       行动与言语命题怎样编码、动态模型声明预测哪些变化”。featurize/samples/model/rl 都只从这里取维度；
       规格指纹写进每个检查点：属性、词表、行动或目标的任何增删都让旧模型在加载时被明确拒绝，而不是在张量形状上报错。
       数值以 value/scale 进入（不再压成布尔），类别以独热进入；known 列三态：1 已知、0 未知、-1 不适用
       features-v3 增加受益人行动指针、FOR 边和持久请求状态，旧检查点明确拒绝。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.cognition.view import EVENT_KIND, EVENT_RELS, VIEW_RELS
from tianlong.core import (
    ATTR_PREFIX,
    ATTR_SPECS,
    ATTRIBUTES,
    ATTRS_VERSION,
    DYNAMIC_ATTRS,
    REASONS,
    AttrType,
    Kind,
    Manner,
    Modality,
    Op,
    Outcome,
    Rel,
    digest,
)

FEATURES_VERSION = "features-v3"

# ============================================================
#  词表
# ============================================================

NODE_KINDS = (*(k.value for k in Kind), EVENT_KIND)
EVENT_OPS = (*(o.value for o in Op), "noise")
MODALITIES = tuple(m.value for m in Modality)
OUTCOMES = tuple(o.value for o in Outcome)
# 言语命题可以谈论的谓词：世界关系 + 每个属性
TOPIC_PREDICATES = (*(r.value for r in Rel), *(ATTR_PREFIX + a.key for a in ATTRIBUTES))
OPS = tuple(o.value for o in Op)
MANNERS = tuple(m.value for m in Manner)
N_OPS, N_MANNERS, N_TOPICS = len(OPS), len(MANNERS), len(TOPIC_PREDICATES)
OP_INDEX = {o: i for i, o in enumerate(Op)}
MANNER_INDEX = {m: i for i, m in enumerate(Manner)}
TOPIC_INDEX = {p: i for i, p in enumerate(TOPIC_PREDICATES)}

# 关系词表：(关系, 否定, 反向)。认知关系有正负极性；事件关系只有正。每条边都配一条反向边，让信息双向流动
REL_VOCAB: tuple[tuple[str, bool, bool], ...] = tuple(
    (r, neg, rev)
    for r in (*VIEW_RELS, *EVENT_RELS)
    for neg in ((False, True) if r in VIEW_RELS else (False,))
    for rev in (False, True)
)
REL_INDEX = {k: i for i, k in enumerate(REL_VOCAB)}
F_EDGE = len(REL_VOCAB) + 3          # + 可信度、时效、传闻
AGE_SCALE = 120.0                    # 两小时以上的时效视为同等陈旧

# ============================================================
#  节点列布局
#  [种类独热 | 自身 | 每个属性一块（值列 + known 列）| 事件：操作、渠道、结果、原因、命题谓词、命题极性、是否提问]
# ============================================================


@dataclass(frozen=True, slots=True)
class AttrBlock:
    key: str
    type: AttrType
    start: int            # 值列起点
    width: int            # 值列宽度：BOOL/NUM 为 1，CAT 为类别数（全零 = none）
    known: int            # known 列：1 已知、0 未知、-1 不适用
    scale: float
    categories: tuple[str, ...]


def _layout() -> tuple[tuple[AttrBlock, ...], int]:
    col = len(NODE_KINDS) + 1
    blocks = []
    for a in ATTRIBUTES:
        width = len(a.categories) if a.type == AttrType.CAT else 1
        blocks.append(AttrBlock(a.key, a.type, col, width, col + width, a.scale, a.categories))
        col += width + 1
    return tuple(blocks), col


KIND_COLS = slice(0, len(NODE_KINDS))
SELF_COL = len(NODE_KINDS)
ATTR_BLOCKS, _ATTR_END = _layout()
ATTR_BLOCK = {b.key: b for b in ATTR_BLOCKS}


def _span(start: int, width: int) -> slice:
    return slice(start, start + width)


EV_OP = _span(_ATTR_END, len(EVENT_OPS))
EV_MODALITY = _span(EV_OP.stop, len(MODALITIES))
EV_OUTCOME = _span(EV_MODALITY.stop, len(OUTCOMES))
EV_REASON = _span(EV_OUTCOME.stop, len(REASONS))
EV_TOPIC = _span(EV_REASON.stop, len(TOPIC_PREDICATES))
EV_TOPIC_HOLDS = EV_TOPIC.stop            # +1 肯定 / -1 否定 / 0 无命题
EV_TOPIC_QUERY = EV_TOPIC.stop + 1        # 1 = 提问（宾语未知）
REQUEST_STATES = ("pending", "accepted", "deferred", "refused")
EV_REQUEST_STATE = _span(EV_TOPIC.stop + 2, len(REQUEST_STATES))
# 个人探索记录（角色视角）：[看清过, 看清的新鲜度, 翻查过, 翻查的新鲜度]；新鲜度 = 1 - 时效/AGE_SCALE（下限 0.05，以别于从没看过）
SURVEY = slice(EV_REQUEST_STATE.stop, EV_REQUEST_STATE.stop + 4)
F_NODE = SURVEY.stop

# ============================================================
#  行动编码：言语命题不再被丢弃——“说钥匙在桌上”与“说钥匙在他身上”、肯定与否定，是不同的行动
# ============================================================

ACTION_FIELDS = ("op", "manner", "target", "obj", "actor", "topic_pred", "topic_subj", "topic_val",
                 "topic_holds", "topic_query", "beneficiary")

# ============================================================
#  预测目标（StateDelta）：声明覆盖范围，指标按此分项
#  - 位置：可定位节点的下一容纳者指针 + 三个“空”类：UNKNOWN（不知道在哪）、GONE（确知不在原处、去向不明）、
#          NEW（确知在一个此前还不认识的容纳者那里——指针指不到，但绝不是“不知道”）
#  - 动态属性：布尔三态（否/未知/是），数值（值 + 是否已知）
#  - 发现：下一刻认识了新实体（角色视角）
#  - 观察增益：行动之后信念发生的、并非本行动直接效果的变化条数（角色视角；“预期有效新观察数”）
# ============================================================

DYN_BOOL = tuple(k for k in DYNAMIC_ATTRS if ATTR_SPECS[k].type == AttrType.BOOL)
DYN_NUM = tuple(k for k in DYNAMIC_ATTRS if ATTR_SPECS[k].type == AttrType.NUM)
HOLDER_EXTRA = ("unknown", "gone", "new")
OBS_GAIN_CAP = 10
TARGETS = ("success", "holder", *(f"attr:{k}" for k in DYN_BOOL), *(f"num:{k}" for k in DYN_NUM), "discover",
           "obs_gain")

SCHEMA = digest(
    FEATURES_VERSION, ATTRS_VERSION, NODE_KINDS, EVENT_OPS, MODALITIES, OUTCOMES, REASONS, TOPIC_PREDICATES,
    REL_VOCAB, OPS, MANNERS, ACTION_FIELDS, REQUEST_STATES, TARGETS, HOLDER_EXTRA, OBS_GAIN_CAP,
    tuple((b.key, b.start, b.width, b.known, b.scale) for b in ATTR_BLOCKS), (SURVEY.start, SURVEY.stop),
)

VIEWS = ("env", "agent")


class StaleModel(ValueError):
    """检查点的特征规格或视角与当前代码不一致。"""


def check_schema(ckpt: dict, path: object, view: str | None = None) -> None:
    """规格指纹不同 = 属性/词表/行动编码/预测目标有增减：请重训。视角不同 = 装错了模型（角色入口装了全知模型）。"""
    if ckpt.get("schema") != SCHEMA:
        raise StaleModel(f"{path} 是用另一套特征规格训练的（属性、行动或预测目标有增减），请按 README“训练与结果”重训")
    if view is not None and ckpt.get("view") != view:
        raise StaleModel(f"{path} 是 {ckpt.get('view')!r} 视角的模型，这里需要 {view!r} 视角：不能拿全知模型充当角色的预测")
