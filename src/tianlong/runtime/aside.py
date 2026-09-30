"""
[INPUT]: 依赖 runtime/gm 的主持层纯函数（is_ooc / leaked / META_HELP / self_view / goal_text / asks_direction / aside_prompt / gated_stream /
         ASIDE_SYSTEM / ASIDE_INNER / ASIDE_TOKENS），language/command 的 clarify，language/parser 的 MoveKind / Parsed，
         language/render 的 Rendered / RenderStatus / Violation，persistence 的 RequestConflict，cognition 的 BeliefStore，core 的 WorldState / clock_label / digest；
         宿主 GameSession 的 scenario / player / llm / ending / beliefs() / belief_lines() / _hint / _recent / _asides 与 runtime/session 的 TurnReport（调用时取，免得成环）
[OUTPUT]: 对外提供 AsideMixin（GameSession 的不推进回合：_aside 分派、_replay_aside 重试原样返回、_clarify 追问、_meta 元指令、
          _next_hint / _floor 逐级提示、_gm_aside 场外问答、_aside_pieces 读流）、ENDED / PARDON / ASIDE_KEEP
[POS]: runtime/session 的不推进时间的回合：场外问答、元指令、追问、落幕之后。只读玩家自己的认知、目标、逐级提示与最近正文，
       不落库；带 request_id 的只记在本进程里（最近 ASIDE_KEEP 个），重试原样返回——提示不多翻、模型不再问，异内容抛 RequestConflict。
       场外回答边生成边逐句过名字闸门，模型写的追问同样过闸门（玩家自己说出的名字不算），拦下即换成不带名字的追问
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

from tianlong.cognition import BeliefStore
from tianlong.core import WorldState, clock_label, digest
from tianlong.language.command import clarify
from tianlong.language.parser import MoveKind, Parsed
from tianlong.language.render import Rendered, RenderStatus, Violation
from tianlong.persistence import RequestConflict
from tianlong.runtime import gm

if TYPE_CHECKING:
    from tianlong.runtime.session import TurnReport, _Sink, _Stopwatch

ENDED = "第一幕已终。可以输入 /recap 回顾，或重新开始一局。"
PARDON = "没太听明白，能换个说法吗？"   # 模型写的追问点了玩家不该知道的名字时，换成这句
ASIDE_KEEP = 64      # 带 request_id 的不推进回合在本进程里记住最近这么多个（重试原样返回；它们不是世界事实，不落库）


class AsideMixin:
    """不推进时间的回合：场外问答、元指令、追问、落幕之后。宿主（GameSession）给出场景、玩家、声音模型、落幕与会话运行态。"""

    def _aside(self, parsed: Parsed, text: str, head: WorldState, me: BeliefStore, clock: _Stopwatch, sink: _Sink,
               request_id: str | None) -> TurnReport:
        """不推进时间、不落库：后台预算的决策作废（它从不写任何东西）。带 request_id 的记在本进程里，重试原样返回。"""
        from tianlong.runtime.session import TurnReport  # 报告类型属于会话：模块顶层导入会成环
        if self.ending is not None and parsed.kind not in (MoveKind.ASK_GM, MoveKind.META):
            render = Rendered(ENDED, RenderStatus.TEMPLATE)
        elif parsed.kind == MoveKind.ASK_GM:
            render = self._gm_aside(parsed.question or "", me, sink, gm.is_ooc(text))     # 边生成边交付
        elif parsed.kind == MoveKind.META:
            render = Rendered(self._meta(parsed.question or ""), RenderStatus.TEMPLATE)
        else:
            render = self._clarify(parsed, text, me)
        clock.lap("aside")
        if parsed.kind != MoveKind.ASK_GM:
            sink(render.text)
        report = TurnReport(clock_label(head.clock), parsed, render.text, advanced=False, timings=clock.laps,
                            render=render, request_id=request_id, kind=parsed.kind, ending=self.ending,
                            first_text_ms=sink.first_ms)
        if request_id is not None:
            self._asides[request_id] = (digest("turn", text), report)
            while len(self._asides) > ASIDE_KEEP:
                del self._asides[next(iter(self._asides))]
        return report

    def _replay_aside(self, request_id: str, payload: str, clock: _Stopwatch, sink: _Sink) -> TurnReport:
        """重试一次不推进的回合：同 ID 同内容原样返回（提示不再往下翻、场外问答不再问一遍模型），异内容抛 RequestConflict。"""
        bound, report = self._asides[request_id]
        if bound != payload:
            raise RequestConflict(f"请求 {request_id} 已绑定另一份内容：拒绝执行")
        sink(report.narration)
        return replace(report, timings=clock.laps, replayed=True, first_text_ms=sink.first_ms)

    def _clarify(self, parsed: Parsed, text: str, me: BeliefStore) -> Rendered:
        """追问与场内说法。模型写的（未必是模板：它熟读原著，可能点出玩家不该知道的名字）过名字闸门——玩家自己说出的名字不算；
        拦下即换成不带名字的追问。"""
        reply = parsed.clarification or "……"
        found = gm.leaked(reply, me, self.scenario, said=text) if parsed.source == "llm" else []
        if not found:
            return Rendered(reply, RenderStatus.TEMPLATE)
        cmd = parsed.command
        plain = clarify(cmd) if cmd is not None and not cmd.immediate else PARDON
        return Rendered(plain, RenderStatus.GATED_FALLBACK, tuple(Violation("entity", n) for n in found))

    def _meta(self, name: str) -> str:
        if name == "hint":
            return self._next_hint()
        if name == "recap":
            return "\n\n".join(self._recent) if self._recent else "（故事才刚开始。）"
        if name == "beliefs":
            return "\n".join(self.belief_lines()) or "（你一无所知。）"
        return gm.META_HELP

    def _next_hint(self) -> str:
        """逐级提示：每次给下一条没给过的（给完了就重复最后一条）；进度随下一次提交落库。"""
        guide = self.scenario.guide
        if not guide:
            return "（这一幕没有提示。）"
        self._hint = max(self._hint, self._floor())
        line = guide[min(self._hint, len(guide) - 1)]
        self._hint = min(self._hint + 1, len(guide))
        return f"提示：{line}"

    def _floor(self) -> int:
        """提示至少从第几条说起：走过的路不再提（身在崖底还说“龚光杰来者不善”，是主持人没跟上故事）。"""
        here = self.beliefs(self.player).location_of(self.player) or ""
        return min(self.scenario.guide_at.get(here, 0), max(len(self.scenario.guide) - 1, 0))

    def _gm_aside(self, question: str, me: BeliefStore, sink: _Sink, ooc: bool = True) -> Rendered:
        """场外问答：只用玩家自己的认知、他的目标、逐级提示（到下一条为止，不剧透更深的）与最近几段正文。
        模型的回答边生成边交付，逐句过名字闸门：点了玩家不认识的名字那句不交付、流也不再读；
        一句都没交出（首句即违规、空答、模型不可用）就交模板 = 处境摘要 + 下一条提示。
        ooc：玩家明说了“GM：”才是跳出故事，答话标“（场外）”；故事里的自问（“我身上还有什么？”）用故事里的口吻答，
        不标场外；提示只在问到“怎么办、往哪走”时才给（问身上有什么，不顺手塞一条攻略）。"""
        view = gm.self_view(me)
        guide, floor = self.scenario.guide, self._floor()
        level = max(self._hint, floor)
        nxt = guide[min(level, len(guide) - 1)] if guide else None
        lead = "（场外）" if ooc else ""
        guided = ooc or gm.asks_direction(question)
        plain = lead + "；".join(view[:3]) + "。" + (f"\n提示：{nxt}" if nxt and guided else "")
        if self.llm is None:
            sink(plain)
            return Rendered(plain, RenderStatus.TEMPLATE)
        prof = self.scenario.profiles[self.player]
        goals = [t for t in (gm.goal_text(g, me) for g in prof.goals) if t]
        prompt = gm.aside_prompt(question, view, prof.persona, goals if ooc else (),       # 元目标只在场外说
                                 guide[floor:level + 1] if guided else (), self._recent)
        system = gm.ASIDE_SYSTEM if ooc else gm.ASIDE_INNER
        text, found, failed = gm.gated_stream(self._aside_pieces(prompt, system), lambda t: gm.leaked(t, me, self.scenario),
                                              sink, lead=lead)
        violations = tuple(Violation("entity", n) for n in found)
        if failed:
            status = RenderStatus.LLM_UNAVAILABLE
        elif text:
            status = RenderStatus.LLM            # 中途拦下一句：已交付的照旧算数，拦下的记在 violations
        else:
            status, violations = RenderStatus.GATED_FALLBACK, violations or (Violation("empty", ""),)
        if not text:
            sink(plain)
        return Rendered(text or plain, status, violations)

    def _aside_pieces(self, prompt: str, system: str = gm.ASIDE_SYSTEM) -> Iterator[str]:
        """有流式接口就边生成边交付，没有就一次拿全文（同样逐句过闸门）；两者都限两三句的长度。"""
        stream = getattr(self.llm, "stream", None)
        if callable(stream):
            yield from stream(prompt, system=system, max_tokens=gm.ASIDE_TOKENS)
        else:
            yield self.llm.generate(prompt, system=system, temperature=0.4, max_tokens=gm.ASIDE_TOKENS)
