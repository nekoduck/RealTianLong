"""
[INPUT]: 依赖 language/narrator 的 grams（三字片段）
[OUTPUT]: 对外提供 topics()（谈资拆成一条条）、fresh()（还没说过的谈资，至多 FRESH_KEEP 条）、told()（一段正文说到了哪几条）
[POS]: runtime 的谈资账本，gm.build_brief 与会话共用。纯模型主持人与早先的我们都犯同一个毛病：NPC 每开口就把设定背一遍
       （“东西二宗五年一比剑”说了四回、“我家住万劫谷”说了三回）。谈资按“；”拆成条目，会话记下每个 NPC 哪几条已经
       出现在正文里（三字片段过半即算说过；谁说的都算——满堂的人都听见了；交给了叙述者、说话者也开了口的，人称归一后
       两字片段三成出现就算——换了说法也认得出），此后只把还没说过的交给叙述者；都说完了就不再给，
       让他只回应眼前的事。账本是派生数据，随会话运行态落库
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection

from tianlong.language.narrator import grams

SEP = "；"
FRESH_KEEP = 2          # 一次至多交给叙述者这么多条没说过的谈资
TOLD = 0.5              # 一条谈资的三字片段过半出现在正文里：算说过了
TOLD_OFFERED = 0.3      # 交给了叙述者、说话者也开了口：两字片段（人称归一）三成出现就算说过——换了说法也认得出
_PERSON = str.maketrans({c: "某" for c in "你我他她它"})


def topics(knows: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in knows.split(SEP) if t.strip())


def fresh(knows: str, told: Collection[int] = ()) -> str:
    """还没说过的谈资（保持设定里的先后），至多 FRESH_KEEP 条。"""
    return SEP.join([t for i, t in enumerate(topics(knows)) if i not in told][:FRESH_KEEP])


def told(knows: str, text: str, offered: str = "") -> frozenset[int]:
    """这段正文说到了哪几条谈资。offered 是这回合交给叙述者、说话者也确实开了口的那几条：
    模型常把“最听她的话”说成“最听我的话啦”，三字片段认不出，按人称归一后的两字片段认。"""
    seen, loose = grams(text), grams(text.translate(_PERSON), 2)
    given = set(topics(offered))
    out = set()
    for i, t in enumerate(topics(knows)):
        mine = grams(t)
        pairs = grams(t.translate(_PERSON), 2)
        if (mine and len(mine & seen) >= TOLD * len(mine)) or (
                t in given and pairs and len(pairs & loose) >= TOLD_OFFERED * len(pairs)):
            out.add(i)
    return frozenset(out)
