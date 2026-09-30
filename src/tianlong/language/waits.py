"""
[INPUT]: 依赖标准库 re
[OUTPUT]: 对外提供 wait_length(t) -> (分钟数, until)：until ∈ {None, "night", "moon", "dawn"}
[POS]: language 的等待时长（从 parser.py 迁出）：“等一炷香”= 30 分钟，只认紧挨“分”的十进制数字（“²”“①”不是，退回按说法估）；
       “等到天黑/入夜/晚上/夜里”等到入夜，“等到月亮出来/月上”等到月出，“等到天亮/天明/等下去”等到天亮——
       时刻交给会话按场景的时钟事实（Scenario.moments）换算，上限仍是会话的 MAX_WAIT；场景没有月出这个时刻时，月亮照旧按入夜算
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re

# 等待时长（分钟）
_DURATIONS: tuple[tuple[str, int], ...] = (
    ("一个时辰", 120), ("半个时辰", 60), ("一炷香", 30), ("一盏茶", 15), ("一会", 10), ("片刻", 5),
)
# 等到某个时刻：先认天亮与月出，再认入夜（“等到月亮出来”不是入夜，“天黑了等到天亮”是天亮）
_UNTIL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dawn", ("天亮", "天明", "破晓", "等下去", "鸡叫")),
    ("moon", ("月亮出来", "月亮升起", "月出", "月上", "等到月亮", "等月亮")),
    ("night", ("天黑", "入夜", "晚上", "夜里", "月亮")),
)


def wait_length(t: str) -> tuple[int, str | None]:
    for key, words in _UNTIL:
        if any(w in t for w in words):
            return 1, key
    hit = re.search(r"(\d+)\s*分", t)
    if hit is not None:
        return max(1, int(hit.group(1))), None
    return next((m for word, m in _DURATIONS if word in t), 1), None
