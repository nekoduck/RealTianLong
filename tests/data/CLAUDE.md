# tests/data/
> L2 | 父级: /tests/CLAUDE.md

测试用的固定数据：只被测试读取，不被引擎引用。都是“钉住”的东西——改动它们等于改动规格，必须随同一次提交说明理由。

成员清单
goldens.json: 金标准指纹（tests/test_goldens 读取与写出）——仓库 50 tick、仓库·报告 30 tick（NPC 之间带命题的一问一答）、程序化世界种子 0–9（jianghu=1）各 200 tick、旧版无量山 30 回合（模板模式），每一局记下世界指纹、事件日志、全部感知、每个角色结束时的整份认知与经历记录（不含文字）、版本与时钟，不经会话的几局另记每一次 NPC 决策（候选集、预测、Choice.index 与 tag）。钉住的是这几局实际走过的路：内核、认知、ScriptedPolicy、候选规则、启发式预测或程序化世界的改动，只要让这几局里有人换了候选、换了选择、多听见或少听见一句、多知道或少知道一件事，就对不上；这几局从没走到的分支（例如耳语、火爆者升级动手的概率）与没让任何一次掷骰翻面的参数微调（例如交手的临场幅度、拿取的响动）照样对得上——金标准守不住它们。措辞不在其列：台词只记有无、经历记录不记文字。重写：PYTHONPATH=src python -m tests.test_goldens --write
playthrough_duanyu.json: 旧版（段誉作玩家）评测整局游玩的 37 句输入，照录 scripts/bench_probes.json 此时的 playthrough——test_goldens 取前 30 句、test_replay_run6 与 scripts/replay_drops --inputs（M1 出口条件的命令）按它重放 run6（第 35 句抵达结局）；与探针文件脱钩，探针改版不动金标准与基线
gate_corpus.json: M1 闸门误杀语料（tests/test_gate_precision 读取，经 tests/gate_fixtures 解码）——run3–run6 在 M0 代码上重放时叙述闸门丢掉的全部 49 句，每条冻结闸门的全部输入（本句 piece、此前已交付的正文 before、名字全集 known、玩家原话 command、RenderPlan 含补抓的 seen / scenery、SceneBrief；同一次叙述调用的 plan / brief 去重放在 plans / briefs 表里按下标引用，去掉了提示词与答案）与四名标注者的标签（verdict：FP 误杀 28 条 / TP 真拦 21 条，category，reason）；复核改过的条目在 reason 里写明“复核：……”（30、32 两条的 before 删去了新通道规则会丢掉的“回廊通向后院”那一句）
gate_seeded.json: M1 植入条目——以语料条目为底（base），换上本句与此前正文：TP 是应被拦下的硬事实错误（状态升级、瞬移与不在场者出现、易手、复制、未引介的名字、秘密、钟点、NPC 多出台词、替玩家开口或拿主意、门那头没见过），id 带口子名的专测每一处放宽被滥用（比喻里夹带状态或后一小句易手、回忆里定位或写动作、编的“刻字”、两字真话装拟声、真开口的“目光一沉问”、“你打定主意”接没说过的决定、陈设旁边的真复制、“别以为我不知道”的反话、破折号前已点名的人……），category 是期望的违规类别；OK 是应当放行的阴性对照（玩家自己的原话、合法比喻、合法回忆……）。M3 追加硬事实审计的植入 21 条与阴性对照 14 条（reason 以“M3 审计”开头）：易手（钟灵从你怀里抽过帛卷、只查看了蒲团却抽出帛卷、没有 GIVE 的递给）、被制者的肢体动作（拍手、起身、转身便走）、天色（月出之前的明月与月色、月已在天上再探出/爬上、入夜的夕阳与阳光）；条目可另给 plan / brief 对象按字段覆盖底（天色 sky = {text, night, moon}，先动后被制的人 astir）
run6/: 第六轮在线代理评测的录制答案（tests/test_replay_run6 按原样重放）——playthrough.answers.json 是 37 次叙述调用（开场 + 35 回合 + 终章收束）的代理答案（按调用序号），interp_answers.json 是解释器的代理答案（玩家原文 → JSON，同一句话同一份）；与 scripts/bench_online 的 DIR 布局相容（--answers 指过来即可）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
