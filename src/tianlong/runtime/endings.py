"""
[INPUT]: 依赖 runtime/gm 的 reveal / closing_prompt / leaks / CLOSING_SYSTEM，language/llm 的 LLMUnavailable，scenarios 的 Ending，
         cognition 的 BeliefStore，core 的 Rel；宿主 GameSession 的 scenario / player / llm / ending / authority / store / ref / beliefs() / _recent
[OUTPUT]: 对外提供 EndingMixin（GameSession 的落幕：_ended 由世界真相推出、_reach_ending 落幕即定、epilogue() 终章、_closing 收束）
[POS]: runtime/session 的落幕：玩家（据世界真相）身处结局地点即落幕，之后的回合不再推进；读档时落幕与否同样由世界真相推出。
       终章 = 场景给全的结局标题 + 收束（模型只取玩家亲历与最近正文、从不看真相，点了他不认识的名字即不用）+ 明确标作“真相”的揭晓
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.cognition import BeliefStore
from tianlong.core import Rel
from tianlong.language.llm import LLMUnavailable
from tianlong.runtime import gm
from tianlong.scenarios import Ending


class EndingMixin:
    """落幕：玩家（据世界真相）身处结局地点。宿主（GameSession）给出场景、玩家、声音模型、权威写入器与最近正文。"""

    def _ended(self) -> Ending | None:
        place = self.authority.head().target(self.player, Rel.AT)
        return next((e for e in self.scenario.endings if e.place == place), None)

    def _reach_ending(self) -> Ending | None:
        """推进过的回合之后检查一次：落幕即定，之后的回合不再推进。"""
        self.ending = self.ending or self._ended()
        return self.ending

    def epilogue(self) -> str:
        """终章：先是一段收束（有模型时据 Ending.epilogue 与玩家亲历写成、过名字闸门；否则只有标题），
        再是明确标作“真相”的揭晓——你以为的 vs 实际的、你没看见的事——由世界状态与玩家认知确定地生成。"""
        me = self.beliefs(self.player)
        head = f"【{self.ending.title}】" if self.ending else "【尚未落幕】"     # 标题由场景给全（“第一幕终 · 澜沧江畔”）
        truth = gm.reveal(self.scenario, self.authority.head(), me, self.store.events(self.ref))
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
