"""
[INPUT]: 依赖 core 的 Social
[OUTPUT]: 对外提供 VoiceLine（一句要替 NPC 说出口的话：结构来自内核与说话者的认知，措辞交给主持人之声）、
          SceneBrief（一回合叙述所需的上下文：要说出口的话、最近几回合的正文、玩家本回合的原话/姿态、眼下的钩子，
          以及前后照应——玩家自己的身体状况、本回合的意外与变化、身在何处身边有谁、此地叫得出名字的东西、
          是否这一幕的最后一段、玩家的话有没有人接）、
          TextSink（流式叙述的交付回调）
[POS]: language 的主持层契约：会话（runtime）只用已落库的数据拼出 SceneBrief，叙述者（narrator）据此写出本回合的正文。
       台词的事实内容只能是 claim（说话者相信的命题）；其余一切——腔调、客套、叫阵、讥讽——都是修辞。
       may_name 是台词闸门的依据：说话者没听说过的人与物，不许出现在他嘴里
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tianlong.core import Social

TextSink = Callable[[str], None]     # 流式叙述：每通过闸门一句就交付一句（CLI 逐字打印，Web 端推送）


@dataclass(frozen=True, slots=True)
class VoiceLine:
    speaker: str                    # 说话者 ID
    speaker_name: str
    listener_name: str | None       # 对谁说（玩家视角的名字；对玩家说时是“你”）
    op: str                         # tell / ask
    social: Social | None           # 言语行为：答话的腔调、叫阵、讥讽、道谢……
    claim: str | None               # 命题的说法（说话者视角的一句陈述）；None = 闲话，没有事实内容
    template: str | None            # 内核里记下的原话（模板措辞）：可以改写，意思不能变
    voice: str = ""                 # 说话者的腔调与待人的样子
    knows: str = ""                 # 说话者的谈资（被问到掌故时的依据）
    may_name: frozenset[str] = frozenset()   # 台词里允许点名的名字：说话者认识的实体名与别称
    answering: str | None = None    # 对方刚才对他说的话（回话的由头）
    lately: str = ""                # 说话者近来亲历的事（出人意料地出现时才给：好让他自己说出怎么来的）
    about: str = ""                 # 玩家问到的人，说话者所知的公开来历（“龚光杰：东宗左子穆门下……”）：答“他是什么来头”的依据


@dataclass(frozen=True, slots=True)
class SceneBrief:
    lines: tuple[VoiceLine, ...] = ()        # 本回合玩家听见的 NPC 言语（按发生先后）
    recent: tuple[str, ...] = ()             # 最近几回合的正文（旧 → 新）：接续上下文、避免重复
    player_line: str | None = None           # 玩家本回合说出口的话，或做出的姿态（原样；叙述者不得替他改口或加戏）
    stakes: str | None = None                # 眼下的处境与去向（写到哪里停在钩子上）；None = 由叙述者自行收在此刻
    # ---- 前后照应（全部取自玩家自己的认知：本回合之前以为的 vs 此刻看见的） ----
    condition: str | None = None             # 玩家自己的身体状况（“你受了伤”）：叙述别写得像没事人一样
    notes: tuple[str, ...] = ()              # 本回合的意外与变化（“你原以为钟灵被点了穴道、还困在剑湖宫大殿”“钟灵没有跟来”）
    present: tuple[str, ...] = ()            # 玩家以为此刻身在何处、身边有谁（首项是地点）：无事发生时也有东西可写
    statuses: frozenset[str] = frozenset()   # 上面这些话里出现的状态（wounded/poisoned/subdued）：说到它们不算状态升级
    afflicted: tuple[tuple[str, str], ...] = ()   # (本名, 状态)：玩家自己身上确有的、以为身边的人身上有的状态，状态词落在他们身上不算错
    closing: bool = False                    # 这一幕的最后一段：收在余韵上，不再抛出选择
    nearby: tuple[str, ...] = ()             # 此地叫得出名字的东西、门与门那头的地点：可以点名（“回头是剑湖宫”），不进提示词
    unanswered: str | None = None            # 玩家冲着谁说了话、本回合他却没接话（“马五德”）：别替他编答案，写出他没顾上回答的样子
    hurt: tuple[str, ...] = ()               # 身边的人玩家以为的伤毒被制（“钟灵受了伤”）：写他们的举动别像没事人
    hooks: tuple[str, ...] = ()              # 玩家干等、眼前却有可做的事（行动建议的原话）：可把他的目光引过去，不替他决定
