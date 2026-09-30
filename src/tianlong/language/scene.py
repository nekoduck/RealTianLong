"""
[INPUT]: 依赖 core 的 Social
[OUTPUT]: 对外提供 VoiceLine（一句要替 NPC 说出口的话：结构来自内核与说话者的认知，措辞交给主持人之声）、
          SceneBrief（一回合叙述所需的上下文：要说出口的话、最近几回合的正文、玩家本回合的原话/姿态、眼下的钩子，
          以及前后照应——玩家自己的身体状况、本回合的意外与变化、身在何处身边有谁、此地叫得出名字的东西、
          是否这一幕的最后一段、玩家的话有没有人接；天色与开口者最近说过的原话；本回合的看点、写法卡、眼前的景象、细节与许可词；
          开口者在正文里的别称）、
          Sky / MOON_NONE / MOON_RISING / MOON_UP（此刻的天色：文字、是否入夜、月亮在哪），TextSink（流式叙述的交付回调）
[POS]: language 的主持层契约：会话（runtime）只用已落库的数据拼出 SceneBrief，叙述者（narrator）据此写出本回合的正文。
       台词的事实内容只能是 claim（说话者相信的命题）；其余一切——腔调、客套、叫阵、讥讽——都是修辞。
       may_name 是台词闸门的依据：说话者没听说过的人与物，不许出现在他嘴里（名字经相识账本过滤：玩家还不认识的人，
       他的本名也不许出现，除非是那人初次见面自报姓名 intro）；said 标出录入的原话（话即事实：照录，一致性闸门 fidelity 把关）；sky 是天色审计（audit.check_sky）的依据，
       None = 场景没给天色（旧版），天色一项整个不查；said_before 只交给提示词（别让他把同一句话再说一遍）；
       astir 是动作能力审计（audit.check_affordance）的豁免：本回合先动了手、后被制住的人；
       看点四样（runtime/staging 给，旧版全空、提示词逐字不变）：focus 是玩家口吻的一句看点（只交给提示词），cards 是写法卡的编号
       （卡文在系统提示的目录里，只有修辞），spectacle（景观）与 details（细节）是外观原文、算出处，allowed 是额外许可词（只许点名）；
       people 是开口者的别称 → 他的称呼（台词账本 quotes.said_by 据此认得“钟姑娘道：”）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tianlong.core import Social

TextSink = Callable[[str], None]     # 流式叙述：每通过闸门一句就交付一句（CLI 逐字打印，Web 端推送）

# 月亮在哪：月出之前 / 本回合正是月出 / 已在天上
MOON_NONE, MOON_RISING, MOON_UP = "none", "rising", "up"


@dataclass(frozen=True, slots=True)
class Sky:
    text: str                       # 天色的文字（原样进提示词）：“酉时将尽，天色已暗，月亮还没出来”
    night: bool                     # 已经入夜：不许再有日头
    moon: str = MOON_NONE           # MOON_NONE / MOON_RISING / MOON_UP：只有 rising 才许写月亮升起


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
    act: str = ""                   # 不是说话、是动手时顺口喝的一声（“向钟灵出手”）：可说可不说，只说这一下的事；漏写不补、没有模板
                                    # （said=True 时是说这句话时的动作——“往后院去”：原话是真说了的，照录、漏写照补）
    said: bool = False              # 录入的原话（驱力台词：随意图落库、在场的人都听见了）：照录，可截取连续一段，只添神态动作，不改字
    intro: str = ""                 # 玩家还不认识他、他初次冲玩家开口：他的本名，可以自报姓名（“我叫……”）；空 = 不提


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
    sky: Sky | None = None                   # 此刻的天色；None = 不给天色、天色不查
    said_before: tuple[tuple[str, tuple[str, ...]], ...] = ()   # (说话者本名, 他最近说过的原话，旧→新)：本回合要开口的人
    astir: tuple[str, ...] = ()              # 本回合自己动过手脚的人（本名；出手、走动、拿放，成败不论，因被制落空的不算）：
                                             # 被制也许是这之后的事，审计不拿“被制”管他的动作（“你挥拳打去……随即被点了穴道”）
    focus: str | None = None                 # 本回合的看点（玩家口吻）：围绕它写，其余一笔带过
    cards: tuple[str, ...] = ()              # 写法卡的编号（卡文在系统提示的目录里）：只加修辞，不加事实
    spectacle: tuple[str, ...] = ()          # 眼前的景象（景观 lore 原文，月下玉璧）：算出处
    details: tuple[str, ...] = ()            # 这回合查看时多看出的一处细节（细节卡组原文）：算出处
    allowed: frozenset[str] = frozenset()    # 额外许可词（“仙人”“小貂”）：只许点名，不许状态
    people: tuple[tuple[str, str], ...] = () # (别称, 称呼)：本回合开口的人在正文里的别称（“钟姑娘”→“钟灵”），台词账本据此归属
