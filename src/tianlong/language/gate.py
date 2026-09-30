"""
[INPUT]: 依赖 language/render 的 RenderPlan / Violation / check / restated_hearsay，language/deeds 的 check_deeds，
         language/quotes 的 check_quotes，language/scene 的 SceneBrief，language/voice_prompt 的 CLOCK_ANY
[OUTPUT]: 对外提供 violations()（叙述闸门对一句的全部判定，纯函数）
[POS]: language 的逐句判定入口：narrator._Gate._found 只调用它，闸门精度的语料测试（tests/test_gate_precision）调用同一个函数，
       测的就是线上那条路。一句话 = 已交付的正文 before + 本句 piece；依次跑叙述闸门 check()（传闻归属收尾整段查，这里不查）、
       逐句传闻 restated_hearsay()、人事闸门 check_deeds()、台词闸门 check_quotes()（只查本句里的引语）与钟点数字。
       有台词时，引语里的名字与状态词交给台词闸门按说话者查（quoted：要说台词的人可点名的名字全集）；收幕段落（brief.closing）
       里玩家见过的陈设可以回忆
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.language.deeds import check_deeds
from tianlong.language.quotes import check_quotes
from tianlong.language.render import RenderPlan, Violation, check, restated_hearsay
from tianlong.language.scene import SceneBrief
from tianlong.language.voice_prompt import CLOCK_ANY


def violations(piece: str, text: str, before: str, plan: RenderPlan, brief: SceneBrief, known: frozenset[str],
               command: str) -> list[Violation]:
    """text = before + piece：闸门看到此为止的全部正文，台词闸门只查本句（since=len(before)）。"""
    # 有台词时，引语里的名字与状态词交给台词闸门按说话者查（他认识的、他的说法谈资与近来经历里有的才许）
    quoted = frozenset(n for vl in brief.lines for n in (*vl.may_name, vl.speaker_name)) if brief.lines else None
    found = [v for v in check(text, plan, known, quoted, recall=brief.closing)
             if v.kind != "hearsay"]                                          # 传闻有没有归属：收尾整段查
    found += restated_hearsay(piece, text, plan)                               # 这一句替传闻作保：当场丢
    found += check_deeds(text, plan, known)
    found += check_quotes(text, brief, plan, known, command=command, since=len(before))
    found += [Violation("clock", m.group(0)) for m in CLOCK_ANY.finditer(piece)]
    return found
