# tests/data/
> L2 | 父级: /tests/CLAUDE.md

测试用的固定数据：只被测试读取，不被引擎引用。都是“钉住”的东西——改动它们等于改动规格，必须随同一次提交说明理由。

成员清单
goldens.json: 金标准指纹（tests/test_goldens 读取与写出）——仓库 50 tick、仓库·报告 30 tick（NPC 之间带命题的一问一答）、程序化世界种子 0–9（jianghu=1）各 200 tick、旧版无量山 30 回合（模板模式），每一局记下世界指纹、事件日志、全部感知、每个角色结束时的整份认知与经历记录（不含文字）、版本与时钟，不经会话的几局另记每一次 NPC 决策（候选集、预测、Choice.index 与 tag）。钉住的是这几局实际走过的路：内核、认知、ScriptedPolicy、候选规则、启发式预测或程序化世界的改动，只要让这几局里有人换了候选、换了选择、多听见或少听见一句、多知道或少知道一件事，就对不上；这几局从没走到的分支（例如耳语、火爆者升级动手的概率）与没让任何一次掷骰翻面的参数微调（例如交手的临场幅度、拿取的响动）照样对得上——金标准守不住它们。措辞不在其列：台词只记有无、经历记录不记文字。重写：PYTHONPATH=src python -m tests.test_goldens --write
playthrough_duanyu.json: 旧版（段誉作玩家）评测整局游玩的 37 句输入，照录 scripts/bench_probes.json 此时的 playthrough——test_goldens 取前 30 句、test_replay_run6 与 scripts/replay_drops --inputs（M1 出口条件的命令）按它重放 run6（第 35 句抵达结局）；与探针文件脱钩，探针改版不动金标准与基线
run6/: 第六轮在线代理评测的录制答案（tests/test_replay_run6 按原样重放）——playthrough.answers.json 是 37 次叙述调用（开场 + 35 回合 + 终章收束）的代理答案（按调用序号），interp_answers.json 是解释器的代理答案（玩家原文 → JSON，同一句话同一份）；与 scripts/bench_online 的 DIR 布局相容（--answers 指过来即可）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
