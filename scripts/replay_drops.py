"""
[INPUT]: 依赖同目录 bench_online 的 run_key / load_probes / load_answers / load_interp / run_dir / PROBE_FILES，
         标准库 logging（tianlong.language.narrator 的日志）
[OUTPUT]: 命令行 python scripts/replay_drops.py DIR [KEY] [--seed N] [--probes FILE] [--answers FILE] [--interp FILE] [--inputs FILE]，
          也是可导入的库：gate_log()（收下叙述闸门日志的上下文）、load_inputs()（整局游玩的输入文件）、
          replay()（重放一个会话并收下闸门日志）、dropped()、DROP / SKIP / PATCH
[POS]: scripts 的闸门误杀排查：拿录下的代理答案在当前代码上重放一个会话（与 bench_online dump 同一条路），
       把叙述者记下的每一句“丢弃”（没过闸门）、“略过”（复述先声、悬空的“她说完……”）与“补上”（漏讲补模板行）逐条列出。
       误杀语料与 M1 的基线都从这里来：tests/test_replay_run6 钉住 run6 在当前代码上丢掉的句子。
       --probes 选探针文件（它的 variant 决定场景，缺省普通人版；答案缺省按 bench_online 的运行目录找）；
       --inputs 让整局游玩的输入取自钉住的文件而不是评测探针（探针改版不动重放）：M1 的出口条件按
       python scripts/replay_drops.py tests/data/run6 --probes scripts/bench_probes_duanyu.json
       --answers tests/data/run6/playthrough.answers.json --inputs tests/data/playthrough_duanyu.json
       跑（旧版世界），与 test_replay_run6 重放的是同一份输入
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:     # 按文件路径当库加载时（测试），也找得到同目录的 bench_online
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_online  # noqa: E402

NARRATOR = "tianlong.language.narrator"
DROP = "叙述句未通过闸门，丢弃"          # 闸门丢句：args = (句子, 违规明细)
SKIP = "略过"                            # 复述先声、台词被丢后悬空的那句
PATCH = "补上"                           # 漏讲的必讲之事补模板行、没通过闸门补事实清单


class _Keep(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if any(k in record.msg for k in (DROP, SKIP, PATCH)):
            self.records.append(record)


@contextmanager
def gate_log() -> Iterator[list[logging.LogRecord]]:
    """期间叙述者记下的丢弃 / 略过 / 补上，按先后收进交出的列表；退出时日志级别原样还回去。"""
    log, keep = logging.getLogger(NARRATOR), _Keep()
    level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(keep)
    try:
        yield keep.records
    finally:
        log.removeHandler(keep)
        log.setLevel(level)


def load_inputs(path: Path) -> list[str]:
    """整局游玩的输入：JSON 字符串数组（如 tests/data/playthrough_duanyu.json），取代探针文件里的 playthrough。"""
    data = json.loads(Path(path).read_text("utf-8"))
    if not (isinstance(data, list) and data and all(isinstance(x, str) and x.strip() for x in data)):
        raise ValueError(f"{path}：整局游玩的输入须是非空的字符串数组")
    return data


def replay(given: Sequence[str], interp: Mapping[str, str], key: str = "playthrough", seed: int = 7,
           probes: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[logging.LogRecord]]:
    """在当前代码上重放一个会话（叙述按调用序号取 given，解释器按玩家原文取 interp）→ (会话记录, 闸门日志)。"""
    with gate_log() as records:
        rec = bench_online.run_key(key, given, interp, probes or bench_online.load_probes(), seed)
    return rec, list(records)


def dropped(records: Sequence[logging.LogRecord]) -> list[str]:
    """闸门丢掉的句子本身（不含违规明细）。"""
    return [str(r.args[0]) for r in records if r.msg.startswith(DROP) and r.args]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="重放一局录下的代理答案，列出叙述闸门丢掉、略过与补上的每一句")
    ap.add_argument("dir", type=Path, help="一轮评测的目录（bench_online 的布局：<运行目录>/online/KEY.answers.json + interp_answers.json）")
    ap.add_argument("key", nargs="?", default="playthrough")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probes", type=Path, default=None, help="探针文件（variant 决定场景；缺省 scripts/bench_probes.json）")
    ap.add_argument("--answers", type=Path, default=None, help="叙述答案文件（缺省 <运行目录>/online/KEY.answers.json）")
    ap.add_argument("--interp", type=Path, default=None, help="解释器答案文件（缺省 DIR/interp_answers.json）")
    ap.add_argument("--inputs", type=Path, default=None,
                    help="整局游玩的输入（JSON 字符串数组，取代探针文件里的 playthrough）")
    args = ap.parse_args(argv)
    probes = bench_online.load_probes(path=args.probes or bench_online.PROBE_FILES["wuliang"])
    rd = bench_online.run_dir(args.dir, bench_online.bench_gm.world_of(probes), args.seed)
    given = bench_online.load_answers(args.answers or rd / "online" / f"{args.key.replace(':', '__')}.answers.json")
    interp = bench_online.load_interp(args.interp or args.dir / "interp_answers.json")
    if args.inputs is not None:
        try:
            probes = dict(probes, playthrough=load_inputs(args.inputs))
        except (OSError, ValueError) as e:              # 读不到、不是 JSON、不是字符串数组：照 argparse 的口径报错退出
            ap.error(str(e))
    rec, records = replay(given, interp, args.key, args.seed, probes)
    for r in records:
        print("NARR:", r.getMessage()[:400], flush=True)
    print(f"{args.key}: {rec['calls']} 次叙述调用（答案 {len(given)} 条），闸门丢句 {len(dropped(records))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
