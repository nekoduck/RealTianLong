"""
[INPUT]: 依赖 language/gate 的 violations（与 narrator._Gate._found 同一条路），tests/gate_fixtures 的 corpus / seeded（冻结的闸门输入），
         tests/test_replay_run6 的 RUN6_DROPS 所钉住的那五句（这里按原句重新列出）
[OUTPUT]: M1 闸门精度验收：run6 的五句误杀过闸门为空；误杀语料里每条 FP 都放行、每条 TP 仍被拦；
          植入的硬事实错误按类别被拦对（违规种类与类别对得上，KINDS）的比例 ≥95%（报出实际比例）、阴性对照全部放行
          （M3 起含硬事实审计：易手 possession、被制者的肢体动作 affordance、天色 sky）；
          语料与植入条目的形状自检
[POS]: tests 的闸门精度规格。只依赖核心（不 import langgraph 等可选包），核心零依赖 CI 同样跑。
       每一条放宽（比喻、回忆、否定的去向、陈设件数、拟声、物件上的字、眼神、照着出处写的景、声音的主人、复述意图、
       当作以为、破折号、省略的主语、通道）都在 tests/data/gate_seeded.json 里配着“相似但应拦”的植入条目
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import Counter

import pytest

from tianlong.language.gate import violations

from .gate_fixtures import Case, corpus, seeded

RECALL_FLOOR = 0.95
# 植入条目的类别 → 算“拦对了”的违规种类：被别的理由碰巧拦下不算召回
KINDS: dict[str, frozenset[str]] = {
    "status": frozenset({"status", "quote_status"}), "entity": frozenset({"entity", "quote_entity"}),
    "possession": frozenset({"outcome", "possession"}), "teleport": frozenset({"teleport"}), "duplicate": frozenset({"duplicate"}),
    "secret": frozenset({"secret"}), "clock": frozenset({"clock"}), "voice": frozenset({"voice"}),
    "puppet": frozenset({"puppet"}), "topology": frozenset({"topology"}),
    "affordance": frozenset({"affordance"}), "sky": frozenset({"sky"}),
}

# run6 在 M0 代码上被叙述闸门丢掉的五句（test_replay_run6 的基线，全是误杀）
RUN6_FALSE_POSITIVES = (
    "你打定主意要在这湖边捱到天黑，便抱膝坐着，看湖面上的光一寸寸挪动。",
    "那面玉璧上，忽然隐隐现出一个人影，衣袂飘飘，手中似握着长剑，竟像有仙人在壁上翩然舞剑。",
    "不知哪根藤萝上的露水，“嗒”的一声，滴在了你手背上。",
    "蒲团面子上绣着两行细字，年深日久，丝线已有些褪色，凑近了才辨认得出：“既入此室，叩首千遍，自有所得。”"
    "字下那道朽裂的口子不大，两卷帛书就嵌在里头，一卷“北冥神功”，一卷“凌波微步”，都还好端端地躺在原处，"
    "帛面上落着薄薄一层灰。",
    "剑湖宫里的喝彩与冷笑，崖下的月色与玉像，此刻都远了，只剩这一江流水，浩浩荡荡，不舍昼夜地向南奔去。",
)


def _found(c: Case) -> list:
    return violations(c.piece, c.text, c.before, c.plan, c.brief, c.known, c.command)


def test_run6_false_positives():
    """M1 出口条件的五句（舞剑 / 长剑、绣字、嗒、打定主意捱到天黑、回忆玉像）：过闸门没有一条违规。"""
    cases = {c.piece.strip(): c for c in corpus() if c.id.startswith("run6/")}
    for sentence in RUN6_FALSE_POSITIVES:
        assert sentence in cases, sentence
        assert _found(cases[sentence]) == [], sentence


@pytest.mark.parametrize("case", corpus(), ids=lambda c: c.id)
def test_corpus(case: Case):
    """误杀语料：FP 放行，TP 至少报一条违规。"""
    found = _found(case)
    if case.verdict == "FP":
        assert found == [], (case.category, case.piece, found)
    else:
        assert found, (case.category, case.piece)


def test_seeded_recall():
    """植入的硬事实错误（TP）按它的类别被拦下（违规种类对得上）的比例 ≥95%，阴性对照（OK）全部放行。"""
    bad = [c for c in seeded() if c.verdict == "TP"]
    good = [c for c in seeded() if c.verdict == "OK"]
    missed = [(c.id, c.piece, _found(c)) for c in bad if not {v.kind for v in _found(c)} & KINDS[c.category]]
    rate = 1 - len(missed) / len(bad)
    print(f"植入召回 {len(bad) - len(missed)}/{len(bad)} = {rate:.1%}；阴性对照 {len(good)} 条")
    assert rate >= RECALL_FLOOR, missed
    assert [(c.id, c.piece, _found(c)) for c in good if _found(c)] == []


def test_fixture_shape():
    """语料 49 条、id 唯一、标签只有 FP/TP；植入 ≥60 条错误、≥15 条阴性对照，且覆盖设计点名的每一类
    （M3 的易手、动作能力、天色各有植入，审计的植入 ≥15 条、阴性对照 ≥8 条）。"""
    cases, seeds = corpus(), seeded()
    assert len(cases) == 49 and len({c.id for c in cases}) == 49
    assert {c.verdict for c in cases} == {"FP", "TP"}
    tally = Counter(c.verdict for c in seeds)
    assert tally["TP"] >= 60 and tally["OK"] >= 15 and set(tally) == {"TP", "OK"}
    assert len({c.id for c in seeds}) == len(seeds)
    kinds = {c.category for c in seeds if c.verdict == "TP"}
    assert {"status", "teleport", "possession", "duplicate", "entity", "secret", "clock", "voice", "puppet",
            "affordance", "sky"} <= kinds
    assert kinds <= set(KINDS)
    audit = Counter(c.verdict for c in seeds if c.reason.startswith("M3 审计"))
    assert audit["TP"] >= 15 and audit["OK"] >= 8, "M3 硬事实审计：植入 ≥15 条、阴性对照 ≥8 条"
