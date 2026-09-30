"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的同处查询，kernel/perception 的 Witnessing
[OUTPUT]: 对外提供 TellRule、AskRule
[POS]: kernel/rules 的言语行动；说话不改变物理世界，只改变听者获得的“说法”——命题可以是谎言，真假由听者自行权衡
       RequestItemRule 只传播请求，不改库存与伤毒，也不替 NPC 接受请求。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator

from tianlong.core import Fact, Intent, Manner, Op, Outcome, Percept, WorldState
from tianlong.kernel import space
from tianlong.kernel.perception import Witnessing
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


class _SpeechRule(ActionRule):
    loudness_base = 0.3
    initiative = 0.5
    usable_when_subdued = True    # 点了穴道，嘴还能说

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        listener = it.target
        assert listener is not None
        if listener == it.actor:
            return fail("self_target")
        if space.place_of(s, listener) != space.place_of(s, it.actor):
            return fail("out_of_reach")
        return succeed()

    def conveyed(self, it: Intent) -> tuple[Fact, ...]:
        """听者因此获得的“说法”。"""
        return ()

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        yield w.actor_percept()
        it = w.event.intent
        if w.resolution.outcome != Outcome.SUCCESS:
            return
        whisper = it.manner == Manner.CAREFUL
        place = w.event.place or ""
        seers = w.witnesses((place,))
        for person in seers:
            if person == it.target or not whisper:
                yield person, w.speech(self.conveyed(it))
            else:
                # 耳语：旁人只看见两人在交谈，听不到内容
                yield person, w.sight(w.public_view(with_topic=False), (), place)
        yield from w.sounds(place, w.loudness, seers)


class TellRule(_SpeechRule):
    op = Op.TELL

    def conveyed(self, it: Intent) -> tuple[Fact, ...]:
        return (it.topic,) if it.topic is not None else ()


class AskRule(_SpeechRule):
    op = Op.ASK
    # 提问不传递事实；问题本身随 PerceivedEvent.topic 抵达对方


class RequestItemRule(_SpeechRule):
    """请求只传递物品与受益人；绝不检查/代替听者的给予决策，也不产生位置或疗效变化。"""
    op = Op.REQUEST_ITEM
