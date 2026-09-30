"""
[INPUT]: 依赖 tianlong.core 的 derive_seed，tianlong.language.llm 的 ScriptedLLM，同目录 bench_rival 的脚本模型（scripted_respond / scripted_gm）
[OUTPUT]: FAST_FIRST / VOICE_P50 / VOICE_P95 / CPS / FAST_CPS / TOK_REF / PREFILL_TPS，first_token(kind, i, tok)，SimLLM，sim_llms(scale)
[POS]: scripts/bench_gm --latency-sim 的延迟模拟（plan §8.4 与 §5）：代理测不了真实延迟，L1/L2 用 ScriptedLLM 按公开分布模拟——
       快模型首 token 0.4–0.6 s 均匀、每秒吐 500 字 JSON（假设）；叙述模型首 token 对数正态（p50 2.2 s、p95 5.3 s），每秒 60 字；
       公开分布对应 TOK_REF 量级的提示词，超出部分按同一模型的预填充速度加时（对照组每回合重发整份对话，首 token 随 TOK 增长就来自这里，
       本引擎的提示词长度恒定）。第 i 次调用的抽样由 derive_seed(种类, i) 派生：同一局重跑，延迟逐项相同。
       scale 同比缩放一切等待（含会话的先声时限，经 lead_scale 交给 bench_gm._session）：测试缩到很小也不改变快慢的次序。
       报告必须标注“模拟”：这是分布的推演，不是实测
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math
import random
from typing import Any

from tianlong.core import derive_seed
from tianlong.language.llm import ScriptedLLM

FAST_FIRST = (0.4, 0.6)            # 快模型（解释器）首 token，秒，均匀分布
VOICE_P50, VOICE_P95 = 2.2, 5.3    # 叙述模型首 token，秒，对数正态
CPS = 60                           # 叙述模型吐字：每秒 60 字
FAST_CPS = 500                     # 快模型吐 JSON：每秒 500 字（假设：小模型、短输出，报告里写明）
TOK_REF = 3000                     # 公开分布对应的提示词规模（token）
PREFILL_TPS = 5000                 # 超出 TOK_REF 的部分每秒预填充这么多 token（同一模型；假设，报告里写明）
_SIGMA = math.log(VOICE_P95 / VOICE_P50) / 1.6449


def first_token(kind: str, i: int, tok: float) -> float:
    """第 i 次调用的首 token 秒数：kind="fast" 取快模型；其余（voice 本引擎叙述、rival 对照组）取叙述模型并按 TOK 加预填充。"""
    rng = random.Random(derive_seed("latency-sim", kind, i))
    if kind == "fast":
        return rng.uniform(*FAST_FIRST)
    return VOICE_P50 * math.exp(_SIGMA * rng.gauss(0.0, 1.0)) + max(0.0, tok - TOK_REF) / PREFILL_TPS


class SimLLM(ScriptedLLM):
    """按公开分布抽样首 token 的脚本模型：回复照旧由 respond 写，等待 = 抽到的秒数 × scale；drawn 记下每次抽到的（未缩放）秒数。"""

    def __init__(self, respond: Any, model: str, kind: str, scale: float = 1.0) -> None:
        cps = FAST_CPS if kind == "fast" else CPS
        super().__init__(respond, model, 0.0, cps / scale if scale > 0 else 0.0)
        self.kind, self.scale, self.lead_scale = kind, scale, scale
        self.drawn: list[float] = []

    def _arm(self, prompt: str, system: str | None) -> float:
        first = first_token(self.kind, len(self.drawn), (len(system or "") + len(prompt)) / 1.6)
        self.drawn.append(round(first, 3))
        return first * self.scale

    def generate(self, prompt: str, **kw: Any) -> str:
        self.first_delay = self._arm(prompt, kw.get("system"))
        return super().generate(prompt, **kw)

    def stream(self, prompt: str, **kw: Any):
        delay, inner = self._arm(prompt, kw.get("system")), super().stream(prompt, **kw)

        def run():
            self.first_delay = delay               # 生成器开跑前才读等待：先设好这一次抽到的
            yield from inner
        return run()


def sim_llms(scale: float = 1.0) -> dict[str, Any]:
    """与 bench_rival.scripted_llms 同样的四个角色：叙述、解释、对照组按分布抽样，评审不计时。"""
    from bench_rival import scripted_gm, scripted_judge, scripted_respond
    return {"voice": SimLLM(scripted_respond, "sim-voice", "voice", scale),
            "fast": SimLLM(scripted_respond, "sim-fast", "fast", scale),
            "baseline": SimLLM(scripted_gm, "sim-voice", "rival", scale),
            "judge": ScriptedLLM(scripted_judge, "scripted-judge")}
