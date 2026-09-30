"""
[INPUT]: 依赖 tianlong.language 的 narrator（Narrator / HOLD_AFTER）/ render / scene / llm（ScriptedLLM / LLMUnavailable），
         tianlong.runtime.authority（真实内核产出的玩家感知），tianlong.scenarios 的 build_wuliang
[OUTPUT]: 叙述者 M1 验收：压句——模型漏掉一件必讲的事时，交付文本的最后一句等于模型的最后一句（钩子），补句紧挨在它前面，
          钩子之后不再追加原始事实行；什么也没漏时压句看不出来（文字与模型逐字相同）；压住钩子之后丢满两句或断了流，
          补句同样插在钩子前面，不追加“看见……”“你看到：……”；
          悬空的反应句——起因句被丢，紧随的“钟灵的笑声一下子断了”一并略过，起因句没被丢时它照常交付
[POS]: tests 的叙述收尾与悬空句（language/narrator 的 _Gate）；证伪“补上的事实行落在钩子后面”“起因没了，反应还在”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools

from tianlong.core import Intent, Manner, Op
from tianlong.language.llm import LLMUnavailable, ScriptedLLM
from tianlong.language.narrator import HOLD_AFTER, Narrator
from tianlong.language.render import RenderStatus
from tianlong.language.scene import SceneBrief
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_wuliang

SC = build_wuliang()
KNOWN = frozenset(e.name for e in SC.state.entities.values()) | {a for al in SC.aliases.values() for a in al}
_ids = itertools.count()

# 只写殿里光景的一段正文（满 HOLD_AFTER 字），最后一句是钩子
BODY = ["大殿里的喧笑声忽地一静。",
        "满堂宾客交头接耳，有人端起茶碗，又慢慢放下，杯盖磕在碗沿上，叮的一声轻响。",
        "檐外的日光斜斜照进来，照得青砖地上一片明晃晃的，几只麻雀在檐下吵个不休，谁也没去理会。",
        "东首几位客人低声议论着什么，声音压得极低，听不真切，只偶尔有一两个字飘过来，又被风吹散了。"]
HOOK = "满堂目光都落在你身上，等你答话。"


def _settle(*intents):
    auth = WorldAuthority.found(InMemoryWorldStore(), SC)
    v = auth.head().version
    s = auth.settle([Intent(f"nr{next(_ids)}", *x, None, v) for x in intents])
    return [o.percept for o in s.observations_of("duanyu")], auth.store.beliefs(auth.ref, "duanyu").entities


def _run(view, text: str, brief: SceneBrief | None = None):
    percepts, names = view
    got: list[str] = []
    voice = Narrator(ScriptedLLM(lambda p, s, sc: text), SC.setting, SC.lore, SC.style, SC.gate_aliases, lead_after=None)
    r = voice.narrate_scene("duanyu", percepts, names, brief=brief or SceneBrief(), show_scene=True, known=KNOWN,
                            on_text=got.append)
    assert r.text == "".join(got), "Rendered.text 恒等于交付给玩家的全部文字"
    return r, got


def test_no_tail_after_hook():
    """龚光杰冲你出手（必讲），模型通篇只写殿里的光景：补句插在钩子前面，钩子仍是最后一句；补句是人话，不是“看见……”。"""
    assert len("".join(BODY)) >= HOLD_AFTER
    view = _settle(("duanyu", Op.WAIT, None, None, Manner.NORMAL),
                   ("gongguangjie", Op.ATTACK, "duanyu", None, Manner.NORMAL))
    r, got = _run(view, "".join(BODY) + HOOK)
    assert r.status == RenderStatus.GATED_FALLBACK and "omitted" in {v.kind for v in r.violations}
    assert got[-1] == HOOK, "交付的最后一句就是模型的最后一句"
    patch = got[-2]
    assert "龚光杰" in patch and not patch.startswith(("\n", "看见")), "补句紧挨在钩子前面，读来是一句人话"
    assert r.text == "".join(BODY) + patch + HOOK
    quiet = _settle(("duanyu", Op.WAIT, None, None, Manner.NORMAL))
    r, got = _run(quiet, "".join(BODY) + HOOK)
    assert r.status == RenderStatus.LLM and got == [*BODY, HOOK], "什么也没漏：压句看不出来"


class _Broken:
    """写到钩子就断了流的模型。"""

    def stream(self, prompt, *, system=None, max_tokens=None):
        yield "".join(BODY) + HOOK
        raise LLMUnavailable("断流")


def test_hook_last_on_fallback():
    """压住钩子之后丢满两句、或写到钩子就断了流：补句同样是人话、插在钩子前面，钩子之后不再追加“看见……”“你看到：……”。"""
    view = _settle(("duanyu", Op.WAIT, None, None, Manner.NORMAL),
                   ("gongguangjie", Op.ATTACK, "duanyu", None, Manner.NORMAL))
    r, got = _run(view, "".join(BODY) + HOOK + "琅嬛福地的深处隐隐传来水声。神仙姐姐的玉像在暗处微微一笑。")
    assert r.status == RenderStatus.GATED_FALLBACK and r.dropped == 2, r.violations
    assert got[-1] == HOOK and "龚光杰" in got[-2] and "看见" not in got[-2] and "你看到" not in r.text, got
    percepts, names = view
    voice = Narrator(_Broken(), SC.setting, SC.lore, SC.style, SC.gate_aliases, lead_after=None)
    r = voice.narrate_scene("duanyu", percepts, names, brief=SceneBrief(), show_scene=True, known=KNOWN)
    assert r.status == RenderStatus.LLM_UNAVAILABLE and r.text.endswith(HOOK), r.text
    assert "看见" not in r.text and "你看到" not in r.text and "龚光杰" in r.text[len("".join(BODY)):]


def test_dangling_reaction():
    """起因句（点了不在眼前的琅嬛福地）被丢：紧随的“钟灵的笑声一下子断了。”没了着落，一并略过；起因句没被丢时照常交付。"""
    view = _settle(("duanyu", Op.WAIT, None, None, Manner.NORMAL))
    brief = SceneBrief(present=("剑湖宫大殿",))
    cause, react, calm = "琅嬛福地的深处隐隐传来水声。", "钟灵的笑声一下子断了。", "殿中一时静了下来。"
    r, _ = _run(view, cause + react + calm, brief)
    assert r.text == calm and r.dropped == 1, r.text
    fine = "横梁上传来一阵清脆的笑声。"
    r, _ = _run(view, fine + react + calm, brief)
    assert r.text == fine + react + calm and r.dropped == 0, r.violations
