"""
[INPUT]: 依赖 cognition 的 BeliefStore，core 的 derive_seed，language/quotes 的 introduces / said_by（自报姓名、正文里确凿归属的引语），
         language/scene 的 SceneBrief，scenarios 的 Scenario（外貌称呼 epithets、开场相识 introduced、别称、先验）；
         runtime/authority 的 Settlement（只作类型）
[OUTPUT]: 对外提供 Acquaintance（相识账本：谁叫得出谁的名字，另记玩家抢先打出真名的人 early；JSON 往返不变）、initial()（开场的账本）、
          forms()（一个人的本名与带名的别称）、veiled()（某人还叫不出名字的人 → 这些名字）、learn_heard()（从录入的原话里学）、
          learn_delivered()（玩家从交付的正文里学，并记下抢先打出的真名）、masked()（展示用的认知副本：没引介的人换成外貌称呼）、
          lookup()（解析用的别称：外貌称呼与真名都认得）、gate_aliases()（叙述闸门的别称：没引介的人的名字只用于拒绝）、
          may_name()（NPC 台词里可点名的名字）、early_line()（玩家抢先叫出真名时附的一句，只一次）
[POS]: runtime 的称呼与相识（plan §3.9），只由会话调用、从不进入 NPC 的 Situation：NPC 的决策永远读它自己的认知，不经过这里。
       名字要有人道出才算知道——开场唱过名的、同门同帮、先验里彼此打过交道的认得；此后任何人从听见（看见）的录入原话里学
       （任何操作上的原话：说话、走动、出手……；里面出现某人的本名或带名的别称，感知到这句话的人就认得了他——自报姓名同理；
       自己说的不算），在 annotate 里推进、随世界同一事务落库；玩家另从交付的正文里学：归属清楚的引语里说话者自报姓名、
       交付的“来历”回答点了名、以此人为主语的命题，在叙述之后推进、随下一次提交落库。
       展示给玩家的一切（解释器、叙述、行动建议、场外问答、终章）都用 masked 的副本：还叫不出名字的人显示为外貌称呼（梁上的青衫少女），
       他的本名与带名的别称（钟姑娘、灵儿）交给叙述闸门只用于拒绝——出现即违规；副本永不落库。玩家抢先打出真名：解释从宽照常解析，
       人在眼前就只一次地附一句不带引号的反应（没开口的人不替他编台词），记进 early 后不再提示。
       没有外貌称呼的场景（旧版、仓库）里账本照记，展示、闸门与可点名逐字不变
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from tianlong.cognition import BeliefStore
from tianlong.core import derive_seed
from tianlong.language.quotes import introduces, said_by
from tianlong.language.scene import SceneBrief
from tianlong.scenarios import Scenario

if TYPE_CHECKING:
    from tianlong.runtime.authority import Settlement

# 玩家抢先叫出真名时附的一句：不带引号——那人没开口，不替他编台词（话即事实）；措辞按 (那人, 原话) 派生的种子轮换
_EARLY = ("{who}一愣，显是没料到你叫得出这个名字。", "{who}怔了一怔，像是奇怪你怎会知道这个名字。",
          "{who}眨了眨眼，似乎纳闷你从哪里听来这个名字。")


# ============================================================
#  账本：谁叫得出谁的名字
# ============================================================


@dataclass(frozen=True)
class Acquaintance:
    known: Mapping[str, frozenset[str]] = field(default_factory=dict)   # 角色 → 他叫得出名字的人
    early: frozenset[str] = frozenset()                                 # 玩家抢先打出真名的人（那一句只附一次）

    def __post_init__(self) -> None:
        object.__setattr__(self, "known", {a: frozenset(v) for a, v in sorted(self.known.items()) if v})
        object.__setattr__(self, "early", frozenset(self.early))

    def of(self, agent: str) -> frozenset[str]:
        return self.known.get(agent, frozenset())

    def learn(self, learned: Mapping[str, Iterable[str]]) -> Acquaintance:
        known = {a: set(v) for a, v in self.known.items()}
        for a, eids in learned.items():
            known.setdefault(a, set()).update(e for e in eids if e != a)
        return replace(self, known=known)

    def to_state(self) -> dict:
        return {"known": {a: sorted(v) for a, v in self.known.items()}, **({"early": sorted(self.early)} if self.early else {})}

    @classmethod
    def from_state(cls, state: Mapping) -> Acquaintance:
        return cls({str(a): frozenset(map(str, v)) for a, v in (state.get("known") or {}).items()},
                   frozenset(map(str, state.get("early", ()))))


def initial(scenario: Scenario) -> Acquaintance:
    """开场的账本：场景给的开场相识（唱过名的、同门同帮、同行），加上先验里与他彼此打过交道的人（见过礼、说过话）。"""
    known = {a: set(v) for a, v in scenario.introduced.items()}
    for agent, percepts in scenario.priors.items():
        for p in percepts:
            ev = p.event
            if ev is not None and agent in (ev.actor, ev.target) and ev.actor and ev.target:
                known.setdefault(agent, set()).update({ev.actor, ev.target} & set(scenario.profiles))
    return Acquaintance().learn(known)


def forms(scenario: Scenario, eid: str) -> tuple[str, ...]:
    """一个人的本名与带名的别称（与本名共用一个字：钟姑娘、灵儿、段公子）；“书呆子”“青衫少女”不带名，叫不出名字的人也能用。"""
    name = scenario.state.entity(eid).name
    return (name, *(a for a in scenario.aliases.get(eid, ()) if set(a) & set(name)))


def veiled(acq: Acquaintance, agent: str, scenario: Scenario) -> dict[str, tuple[str, ...]]:
    """agent 还叫不出名字的人（只有场景给了外貌称呼的才会被遮住）→ 他们的本名与带名的别称。"""
    return {eid: forms(scenario, eid) for eid in sorted(scenario.epithets) if eid != agent and eid not in acq.of(agent)}


def _named(text: str, scenario: Scenario) -> set[str]:
    return {eid for eid in scenario.profiles if any(f in text for f in forms(scenario, eid))}


# ============================================================
#  学名字：听见的原话（任何人），交付的正文（玩家）
# ============================================================


def learn_heard(acq: Acquaintance, settlement: Settlement, scenario: Scenario) -> Acquaintance:
    """感知到一句录入的原话（任何操作上的：说话、走动、出手……）的人，认得了话里点了名的人——自报姓名同理；自己说的不算。"""
    learned: dict[str, set[str]] = {}
    for o in settlement.observations:
        ev = o.percept.event
        if ev is not None and ev.utterance and ev.actor != o.observer:
            learned.setdefault(o.observer, set()).update(_named(ev.utterance, scenario))
    return acq.learn(learned) if learned else acq


def learn_delivered(acq: Acquaintance, player: str, text: str, brief: SceneBrief | None, command: str,
                    scenario: Scenario) -> tuple[Acquaintance, tuple[str, ...]]:
    """玩家从交付的正文里学：归属清楚的引语里说话者自报姓名（我叫/在下/本姑娘 + 名或姓）、交付的“来历”回答点了名、
    以此人为主语的命题说法；并记下他抢先打出真名的人（named_early，人在眼前才算——那一句只附一次）。"""
    hidden = veiled(acq, player, scenario)
    early = named_early(acq, player, command, brief, scenario)
    if not hidden or brief is None:
        return (replace(acq, early=acq.early | set(early)) if early else acq), early
    ids = {vl.speaker_name: vl.speaker for vl in brief.lines}
    learned = {ids[n] for n, words in said_by(text, brief) if ids[n] in hidden and introduces(words, hidden[ids[n]])}
    for vl in brief.lines:
        for eid, fs in hidden.items():
            told = any(f in text for f in fs)
            if told and (any(f in vl.about for f in fs) or any((vl.claim or "").startswith(f) for f in fs)):
                learned.add(eid)
    acq = acq.learn({player: learned}) if learned else acq
    return (replace(acq, early=acq.early | set(early)) if early else acq), early


def named_early(acq: Acquaintance, player: str, command: str, brief: SceneBrief | None,
                scenario: Scenario) -> tuple[str, ...]:
    """玩家打出了一个还没人道出的真名（解释照常从宽），那人就在眼前、之前没提示过：只提示这一回。"""
    here = set(brief.present) if brief is not None else set()      # 玩家以为身边有谁（展示用的称呼）
    return tuple(eid for eid, fs in veiled(acq, player, scenario).items()
                 if eid not in acq.early and scenario.epithets[eid] in here and any(f in command for f in fs))


def early_line(eids: Collection[str], command: str, scenario: Scenario) -> str:
    """抢先叫出真名的那一句反应（至多一人，另起一行）；过得了叙述闸门：不带引号、只点外貌称呼。"""
    if not eids:
        return ""
    eid = sorted(eids)[0]
    pick = _EARLY[derive_seed("named_early", eid, command) % len(_EARLY)]
    return "\n" + pick.format(who=scenario.epithets[eid])


# ============================================================
#  展示与闸门：玩家看见的称呼、叙述闸门的别称、NPC 台词里可点名的名字
# ============================================================


def masked(me: BeliefStore, acq: Acquaintance, scenario: Scenario) -> BeliefStore:
    """展示用副本：还叫不出名字的人，草图名换成外貌称呼。只给解释器、叙述、行动建议、场外问答与终章看，永不落库。"""
    hide = veiled(acq, me.owner, scenario)
    if not hide:
        return me
    ents = {eid: replace(sk, name=scenario.epithets[eid]) if eid in hide else sk for eid, sk in me.entities.items()}
    return replace(me, entities=ents)


def lookup(scenario: Scenario) -> dict[str, tuple[str, ...]]:
    """解析玩家输入用的别称：外貌称呼与真名都认得——玩家抢先打出真名也照常解析到那个人（解释从宽）。"""
    out = {k: tuple(v) for k, v in scenario.aliases.items()}
    for eid, epithet in scenario.epithets.items():
        out[eid] = tuple(dict.fromkeys((*out.get(eid, ()), epithet, scenario.state.entity(eid).name)))
    return out


def gate_aliases(acq: Acquaintance, player: str, scenario: Scenario, said: str = "") -> dict[str, tuple[str, ...]]:
    """叙述闸门的别称：玩家还叫不出名字的人，带名的别称从他名下拿走、连同本名挂到一个不会出场的键上——
    闸门只拿它们来拒绝（hidden）；玩家本回合自己打出的那几个字照旧算他的（复述不算泄露）。"""
    base = scenario.gate_aliases
    hide = veiled(acq, player, scenario)
    if not hide:
        return base
    out = dict(base)
    for eid, fs in hide.items():
        mine = {f for f in fs if said and f in said}
        out[eid] = (*(a for a in base.get(eid, ()) if a not in fs), *sorted(mine))
        out[f"{eid}@name"] = tuple(f for f in fs if f not in mine)
    return out


def may_name(agent: str, mind: BeliefStore, acq: Acquaintance, scenario: Scenario) -> frozenset[str]:
    """NPC 台词里可点名的名字：他认识的实体的名与别称，但还叫不出名字的人只能用不带名的称呼（他自己的名字他当然知道）。"""
    hide = veiled(acq, agent, scenario)
    return frozenset(n for eid, sk in mind.entities.items() for n in (sk.name, *scenario.aliases.get(eid, ()))
                     if n and n not in hide.get(eid, ()))
