"""
[INPUT]: 依赖 runtime/gm 的 reveal / closing_prompt / leaks / CLOSING_SYSTEM，language/llm 的 LLMUnavailable，scenarios 的 Scenario / Ending，
         cognition 的 BeliefStore，core 的 Rel / Op / Outcome / Event / WorldState / AddRelation / SetAttr / clock_label / true_value；
         宿主 GameSession 的 scenario / player / llm / ending / authority / store / ref / beliefs() / _recent
[OUTPUT]: 对外提供 ended()（据世界真相：玩家身处结局地点，或时钟已到）、title_for()（结局标题加变体后缀）、chronicle()（终章纪事
          “那一夜你没看见的事”）、names_for()（纪事用的称呼）、EndingMixin（GameSession 的落幕：_ended / _reach_ending / epilogue() / _closing）
[POS]: runtime/session 的落幕：玩家（据世界真相）身处结局地点、或时钟到了结局的时刻即落幕，之后的回合不再推进；读档时落幕与否同样由世界真相推出。
       终章 = 场景给全的结局标题（按落幕那一刻的真相加变体：身负奇功、与段公子同行、怀揣帛卷、身在何处）+ 收束（模型只取玩家亲历与最近正文、
       从不看真相，点了他不认识的名字即不用）+ 场景给了纪事角色（Scenario.chronicle）就是江湖传闻口吻的纪事，否则照旧是明确标作“真相”的揭晓。
       纪事只写真相里发生过、玩家当时不在场的事：单向门的穿越、学成、以药救人（讨价还价）、动手（灭口）与求情（饶命）、
       撂下狠话的放弃（驱力表里 give_up 的原话）；一律用 names_for 的称呼（开场认得的叫名字，其余用外貌称呼）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tianlong.cognition import BeliefStore
from tianlong.core import AddRelation, Event, Op, Outcome, Pose, Rel, Social, WorldState, clock_label
from tianlong.language.llm import LLMUnavailable
from tianlong.runtime import gm
from tianlong.scenarios import Ending, Scenario

CHRONICLE_HEAD = "—— 那一夜你没看见的事 ——"
MAX_RUMORS = 12
_OPENERS = ("听说", "后来江湖上传说", "据说", "有人说")


def ended(scenario: Scenario, head: WorldState, player: str) -> Ending | None:
    """据世界真相：玩家身处某个结局的地点；否则时钟已到某个结局的时刻。"""
    place = head.target(player, Rel.AT)
    by_place = next((e for e in scenario.endings if e.place is not None and e.place == place), None)
    return by_place or next((e for e in scenario.endings if e.at_clock is not None and head.clock >= e.at_clock), None)


def _holds(req: str, head: WorldState, player: str) -> bool:
    kind, _, what = req.partition(":")
    if kind == "skill":
        return bool(head.attr(player, what, False))
    if kind == "with":
        return head.target(what, Rel.AT) == head.target(player, Rel.AT)
    if kind == "holds":
        return head.target(what, Rel.AT) == player
    if kind == "at":
        return head.target(player, Rel.AT) == what
    raise ValueError(f"未知的结局变体条件：{req}")


def title_for(ending: Ending, head: WorldState, player: str) -> str:
    """结局标题：按落幕那一刻的真相，把成立的变体接在后面（“第一幕终 · 澜沧江畔 · 身负奇功、与段公子同行”）。"""
    labels = [v.label for v in ending.variants if any(_holds(r, head, player) for r in v.requires)]
    return ending.title + (" · " + "、".join(labels) if labels else "")


def names_for(scenario: Scenario, player: str, head: WorldState) -> dict[str, str]:
    """纪事用的称呼：开场就认得的人叫名字，其余有外貌称呼的用外貌称呼；地点物件照名字。"""
    known = scenario.introduced.get(player, frozenset())
    return {eid: (scenario.epithets[eid] if eid in scenario.epithets and eid not in known else e.name)
            for eid, e in head.entities.items()}


# ============================================================
#  纪事：江湖传闻口吻，只写真相里发生过、玩家当时不在场的事
# ============================================================


def chronicle(scenario: Scenario, head: WorldState, events: Sequence[Event], names: Mapping[str, str]) -> str:
    cast = set(scenario.chronicle)
    player = str(scenario.player)
    boasts = {d.line or next((x.text for x in d.do if isinstance(x, Pose)), "")
              for a in cast for d in scenario.drives.get(a, ()) if d.key.startswith("give_up")} - {""}
    place = scenario.state.target(player, Rel.AT)
    rows: list[str] = []
    for e in events:
        seen = e.place == place or (e.op == Op.MOVE and e.intent.target == place)
        if e.actor in cast and not seen and e.outcome == Outcome.SUCCESS:
            text = _rumor(e, head, names, boasts)
            if text:
                rows.append(f"· {_OPENERS[len(rows) % len(_OPENERS)]}{clock_label(e.tick)[-5:]}前后，{text}")
        for c in e.changes:
            if isinstance(c, AddRelation) and c.rel.src == player and c.rel.type == Rel.AT:
                place = c.rel.dst
    if not rows:
        return ""
    more = len(rows) - MAX_RUMORS
    return "\n".join([CHRONICLE_HEAD, *rows[:MAX_RUMORS]] + ([f"……另有 {more} 件，传得更没边了"] if more > 0 else []))


def _rumor(e: Event, head: WorldState, names: Mapping[str, str], boasts: set[str]) -> str | None:
    it = e.intent
    who = names.get(e.actor, e.actor)

    def n(eid: str | None) -> str:
        return names.get(eid or "", eid or "某处")

    if e.op == Op.MOVE and it.obj and head.attr(it.obj, "oneway", None) is not None:
        return f"{who}经{n(it.obj)}去了{n(it.target)}——那是条只去不回的路。"
    if e.op == Op.STUDY and e.reason == "mastered":
        return f"{who}在{n(e.place)}参透了{n(it.target)}上的功夫。"
    if e.op == Op.USE and it.target not in (None, e.actor) and it.obj and head.attr(it.obj, "cures", None):
        return f"{who}在{n(e.place)}拿{n(it.obj)}救了{n(it.target)}，换下了一场干戈。"
    if e.op == Op.ATTACK and it.target:
        return f"{who}在{n(e.place)}对{n(it.target)}下了手。"
    if e.op == Op.TELL and it.social == Social.PLEAD and it.target:
        return f"{who}在{n(e.place)}向{n(it.target)}求了情。"
    if it.utterance and it.utterance in boasts:
        return f"{who}追到{n(e.place)}，撂下一句狠话，掉头回去了。"
    return None


class EndingMixin:
    """落幕：玩家（据世界真相）身处结局地点或时钟到点。宿主（GameSession）给出场景、玩家、声音模型、权威写入器与最近正文。"""

    def _ended(self) -> Ending | None:
        return ended(self.scenario, self.authority.head(), self.player)

    def _reach_ending(self) -> Ending | None:
        """推进过的回合之后检查一次：落幕即定，之后的回合不再推进。"""
        self.ending = self.ending or self._ended()
        return self.ending

    def epilogue(self) -> str:
        """终章：先是一段收束（有模型时据 Ending.epilogue 与玩家亲历写成、过名字闸门；否则只有标题），
        再是纪事（场景给了纪事角色）或明确标作“真相”的揭晓——都由世界状态、事件日志与玩家认知确定地生成。"""
        me = self.beliefs(self.player)
        st = self.authority.head()
        head = f"【{title_for(self.ending, st, self.player)}】" if self.ending else "【尚未落幕】"
        events = self.store.events(self.ref)
        if self.scenario.chronicle:
            truth = chronicle(self.scenario, st, events, names_for(self.scenario, self.player, st))
        else:
            truth = gm.reveal(self.scenario, st, me, events)
        return "\n\n".join(x for x in (head, self._closing(me), truth) if x)

    def _closing(self, me: BeliefStore) -> str:
        """终章收束只取玩家亲历（他自己的经历记录与最近的正文），从不给模型看真相；点了他不认识的名字即不用。"""
        if self.llm is None or self.ending is None:
            return ""
        lived = sorted(self.store.recent_memories(self.ref, self.player, 0), key=lambda m: (m.known_at, m.id))
        prompt = gm.closing_prompt(self.ending.epilogue or self.ending.title, [m.text for m in lived[-40:]],
                                   self._recent)
        system = gm.CLOSING_SYSTEM + (f"\n世界：{self.scenario.setting}" if self.scenario.setting else "") + (
            f"\n文风：{self.scenario.style}" if self.scenario.style else "")
        try:
            text = self.llm.generate(prompt, system=system, temperature=0.7).strip()
        except LLMUnavailable:
            return ""
        return "" if not text or gm.leaks(text, me, self.scenario) else text
