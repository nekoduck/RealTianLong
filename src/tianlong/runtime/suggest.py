"""
[INPUT]: 依赖 cognition/beliefs 的 BeliefStore，cognition/candidates 的 candidates()，core 的 Kind / Op / Manner / HOSTILE_SOCIAL
[OUTPUT]: 对外提供 suggestions(me, limit=3, friends=()) -> tuple[str, ...]：此刻“可以这样做”的几句输入
[POS]: runtime 的行动建议：玩家面对空白输入框不知从何下手时，给两三句现成的话，点一下就能照做。
       只从玩家自己的认知与候选集生成（他以为在场的人、以为在这的东西、以为相邻的地方），从不参考真相，所以不剧透；
       措辞刻意落在解释器快路径的句式上（去 P、查看 S、拿起 S 上的 I、研读 I、环顾四周），点了不必等模型；
       行礼与赔罪交给解释器（规则认得，有模型时一次快模型调用）。每族至多一条，按“同伴挨了打（向动手的人替他求情、拉着他逃）或刚走开（跟上他）
       > 有人冲我来（狠话或动手：赔罪、脱身）> 没看清的 > 没试过的 > 没去过的”排序，研读过的不再提，
       同一局面给同样的建议（确定性）。被 web.py 在开场与每回合收尾时附上
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection

from tianlong.cognition.beliefs import BeliefStore
from tianlong.cognition.candidates import candidates
from tianlong.core import HOSTILE_SOCIAL, Kind, Manner, Op

RECENT = 3        # 多少个 tick 内冲我来的话算“眼前的事”（与会话的反应 tick 同量级）

# 族的次序：同优先级时谁先入选
_FAMILIES = ("friend", "people", "flight", "look", "study", "take", "move", "wait")


def suggestions(me: BeliefStore, limit: int = 3, friends: Collection[str] = ()) -> tuple[str, ...]:
    """按玩家此刻的认知给出至多 limit 句可以直接照做的输入；每族至多一句，优先级相同按族序与字面排序。
    friends：玩家的同伴（自己人与要护着的人）——同伴挨了打先替他求情，同伴刚走开就跟上。"""
    player, now = me.owner, me.last_tick
    here = me.location_of(player)
    if here is None:
        return ("环顾四周",)

    def name(eid: str | None) -> str | None:
        sk = me.sketch(eid) if eid else None
        return sk.name if sk and sk.name else None

    pool: list[tuple[int, str, str]] = []          # (优先级, 族, 句子)：优先级小者先
    cands = [c for c in candidates(me) if c.manner == Manner.NORMAL]
    recent = [e.event for e in me.episodes if e.tick >= now - RECENT]
    tried = {(e.event.kind, e.event.target) for e in me.episodes if e.event.actor == player}

    # ---- 眼前的人：冲我来的（狠话或动手）先赔罪；没打过交道的先行个礼 ----
    threatened = False
    for p in sorted({c.target for c in cands if c.op == Op.ATTACK and c.target}):
        who = name(p)
        if who is None:
            continue
        cues = [c for c in me.cues_from(p, since=max(0, now - RECENT)) if c.to == player]
        hit = any(e.kind == Op.ATTACK.value and e.actor == p and e.target == player for e in recent)
        greeted = any(s.listener == p for s in me.said)
        if hit or any(c.social in HOSTILE_SOCIAL for c in cues):
            threatened = True
            pool.append((0, "people", f"向{who}赔罪"))
        elif not greeted and not cues:
            pool.append((2, "people", f"向{who}行礼"))

    # ---- 看：此处没看清过就环顾；此处的陈设没亲手翻过就细看 ----
    if here not in me.surveyed:
        pool.append((1, "look", "环顾四周"))
    for c in cands:
        if c.op == Op.INSPECT and c.target != here and name(c.target) and me.sketch(c.target).kind == Kind.SURFACE \
                and c.target not in me.searched:
            pool.append((3, "look", f"查看{name(c.target)}"))

    # ---- 手里的秘籍可以研读；以为在这的东西可以拿起 ----
    for c in cands:
        if c.op == Op.STUDY and name(c.target) and (Op.STUDY.value, c.target) not in tried:
            pool.append((2, "study", f"研读{name(c.target)}"))
        elif c.op == Op.TAKE and name(c.target):
            loc = me.location_of(c.target)
            on = name(loc) if loc and me.sketch(loc) and me.sketch(loc).kind == Kind.SURFACE else None
            pool.append((4, "take", f"拿起{on}上的{name(c.target)}" if on else f"拿起{name(c.target)}"))

    # ---- 走：有人冲我来就是脱身；否则没看清过的地方在前 ----
    for c in cands:
        if c.op == Op.MOVE and name(c.target):
            if threatened:
                pool.append((1, "move", f"逃去{name(c.target)}"))
            else:
                pool.append((3 if c.target not in me.surveyed else 5, "move", f"去{name(c.target)}"))

    # ---- 同伴：挨了打就替他求情（或拉着他逃），刚从这里走开就跟上 ----
    for f in sorted(friends):
        who = name(f)
        if who is None or f == player:
            continue
        if me.location_of(f) == here:
            foe = next((e.actor for e in reversed(recent) if e.kind == Op.ATTACK.value and e.target == f
                        and e.actor not in (None, player) and me.location_of(e.actor) == here and name(e.actor)), None)
            if foe is not None:
                pool.append((0, "friend", f"向{name(foe)}替{who}求情"))
                way = next((c for c in cands if c.op == Op.MOVE and name(c.target)), None)
                if way is not None:
                    pool.append((1, "flight", f"拉着{who}逃去{name(way.target)}"))
        elif any(e.kind == Op.MOVE.value and e.actor == f and e.place == here for e in recent) and me.location_of(f):
            pool.append((0, "friend", f"跟上{who}"))

    pool.append((9, "wait", "等一会儿"))

    order = {f: i for i, f in enumerate(_FAMILIES)}
    picked: list[str] = []
    used: set[str] = set()
    for _, family, text in sorted(set(pool), key=lambda x: (x[0], order[x[1]], x[2])):
        if family in used or text in picked:
            continue
        picked.append(text)
        used.add(family)
        if len(picked) >= limit:
            break
    return tuple(picked)
