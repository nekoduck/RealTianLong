"""
[INPUT]: 依赖 core 的 Op / Social，scenarios/base 的 Beat，scenarios/tianlong/wuliang 的 LOVERS，scenarios/tianlong/drives_c 的 HUNT
[OUTPUT]: 对外提供 BEATS_C（普通人版的 11 个看点识别器，plan 附录 A）与 BEAT_KEYS（看点的次序）
[POS]: scenarios/tianlong 普通人版的看点：识别器，不是触发器——只匹配已结算、且玩家已感知到的事件或景观（runtime/staging 求值），
       供 B1 计数与“等待在看点处停下”。同一个看点可以有几条识别器（叫阵或动手；私语是说话或带字的姿态）。
       写法卡与细节卡组（DETAILS）留给 M3：这里的 gloss 只是玩家口吻的一句看点
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Op, Social
from tianlong.scenarios.base import Beat
from tianlong.scenarios.tianlong.drives_c import HUNT
from tianlong.scenarios.tianlong.wuliang import LOVERS

_GG, _DY, _ZL = ("gongguangjie",), ("duanyu",), ("zhongling",)
_SCROLLS = ("scroll_lb", "scroll_bm")

BEATS_C: tuple[Beat, ...] = (
    Beat("challenge", Op.TELL, _GG, _DY, social=Social.CHALLENGE, gloss="龚光杰冲着段公子叫阵"),
    Beat("challenge", Op.ATTACK, _GG, _DY, success=False, gloss="龚光杰对段公子动手"),
    Beat("beating", Op.ATTACK, _GG, _DY, gloss="段公子挨了龚光杰一掌"),
    Beat("marten", Op.ATTACK, _ZL, _GG, gloss="一道灰影扑上龚光杰，他中了貂毒"),
    Beat("bargain", Op.USE, _ZL, _GG, obj="antidote", gloss="梁上的少女拿解药换段公子平安"),
    Beat("whisper", Op.TELL, LOVERS, LOVERS, gloss="后院里有人低声私语"),
    Beat("whisper", Op.WAIT, LOVERS, place=("houyuan",), social=Social.REMARK, gloss="后院里有人低声私语"),
    Beat("chase", None, _GG, place=("houyuan", "houshan", "yading"), clock_from=HUNT, gloss="火把与叫骂声追了过来"),
    Beat("cliff", Op.MOVE, _DY, door="d_cliff", gloss="段公子攀着藤萝跳下断崖"),
    Beat("moon", place=("jianhu",), clock_from="moon", lore="yubi@moon", once=True, gloss="月光照上玉璧，壁上似有仙人舞剑"),
    Beat("crack", Op.INSPECT, door="d_cave", place=("jianhu",), gloss="玉璧旁露出一道石缝"),
    Beat("kowtow", Op.WAIT, place=("langhuan",), social=Social.SUBMIT, gloss="有人对着玉像磕头"),
    Beat("scroll", Op.TAKE, target=_SCROLLS, gloss="蒲团里的帛卷被人取了出来"),
    Beat("scroll", Op.STUDY, target=_SCROLLS, reason="mastered", gloss="帛卷上的步法被人参透了"),
)
BEAT_KEYS: tuple[str, ...] = tuple(dict.fromkeys(b.key for b in BEATS_C))
