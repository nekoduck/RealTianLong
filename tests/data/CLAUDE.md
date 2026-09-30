# tests/data/
> L2 | 父级: /tests/CLAUDE.md

测试用的固定数据：只被测试读取，不被引擎引用。都是“钉住”的东西——改动它们等于改动规格，必须随同一次提交说明理由。

成员清单
goldens.json: 金标准指纹（tests/test_goldens 读取与写出）——仓库 50 tick、程序化世界种子 0–9（jianghu=1）各 200 tick、旧版无量山 30 回合（模板模式）的世界指纹、事件日志摘要、版本与时钟；内核、ScriptedPolicy、候选规则或程序化世界的行为一变就对不上。重写：PYTHONPATH=src python -m tests.test_goldens --write
playthrough_duanyu.json: 旧版（段誉作玩家）评测整局游玩的 37 句输入，照录 scripts/bench_probes.json 此时的 playthrough——test_goldens 取前 30 句、test_replay_run6 按它重放 run6（第 35 句抵达结局）；与探针文件脱钩，探针改版不动金标准与基线
run6/: 第六轮在线代理评测的录制答案（tests/test_replay_run6 按原样重放）——playthrough.answers.json 是 37 次叙述调用（开场 + 35 回合 + 终章收束）的代理答案（按调用序号），interp_answers.json 是解释器的代理答案（玩家原文 → JSON，同一句话同一份）；与 scripts/bench_online 的 DIR 布局相容（--answers 指过来即可）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
