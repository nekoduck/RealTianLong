# scripts/
> L2 | 父级: /CLAUDE.md

本地开发基础设施与演示工具。不含引擎逻辑：只准备环境，或把引擎跑一遍给人看。

成员清单
neo4j_local.sh: 下载并启动 Neo4j Community（仅监听 127.0.0.1，限定堆与页缓存），要求预设 NEO4J_PASSWORD；供契约测试与 `--store neo4j` 持久化游玩
make_colab_notebook.py: notebooks/train_colab.ipynb 的唯一源头（单元格以 Python 字符串维护，可审阅可测试）；--commit 填入固定提交生成交给 Colab 的那一份，仓库里的版本不填（tests/test_notebook.py 核对一致）
play_demo.py: 按固定指令序列录制一局（默认无量山原著路线），写出 docs/demo/{world}.md：每回合玩家所见的叙述 / 内核裁定的全部事件 / NPC 各自理由，结尾对照玩家以为与真相；叙述经 CachedLLM 缓存，重跑不再花费调用
bench_gm.py: 主持层评测（设计 §7 与“评测细则”），命令行 --out DIR [--llm auto|scripted|none] [--baseline] [--judge] [--turns N] [--limit K]，也是可导入的库；把 GameSession 当黑盒逐回合跑（会话新字段一律 getattr，缺了就退化），每条探针从全新会话出发、先走 setup、异常逐条记下不中断；按 TurnReport 与世界真相算 L1/L2（挂钟 + on_text 首字）、R4、C1（kernel.violations）、C2/R1（声称之物在真相里成立且归到玩家名下）、G1、N1、F2（4-gram）、P1（词法），写 bench.json 与带门槛 PASS/FAIL 的 report.md；auto 用真模型且缓存关闭、无密钥退回 scripted，scripted 的报告注明不作验收依据；本引擎走到结局时把终章（收束 + 真相揭晓）一并记下、交给整局盲评——玩家落幕时读得到它
bench_rival.py: bench_gm 的模型侧（单独成文件只为不超 800 行，不依赖 bench_gm）：PureLLMGM 纯模型主持人对照组（由场景生成的全知世界圣经——玩家那一行的目标用玩家自己的口吻（与主持层场外问答同一张 runtime/gm.PLAYER_GOALS），不借 NPC 的行事语义——+ Jenova 公开主持规矩 + 每回合全量对话、流式计时）、独立评审（premise_prompt / pairwise_prompt，严格 JSON，parse_verdict 容忍坏输出、拿不准判无效，盲评顺序由种子固定打乱）、--llm scripted 的离线脚本模型（解释器 JSON 兼容新旧 schema、照事实清单写叙述、顺着玩家的主持人，模拟首字 0.6/0.3 s 与 250 字/秒）
bench_online.py: 在线代理评测（设计 §8.4，没有模型密钥时由只看这一次调用的隔离代理替模型作答），python scripts/bench_online.py {keys|step|answer|assemble|dump} DIR [KEY] [--seed N] [--open-actions K]：step 用已有答案从头重放一个会话（同样的答案 → 同样的轨迹）、停在第一个没答的叙述调用上打印其 system 与 prompt 并立刻退出，answer 以 stdin 追加答案，assemble 在全部 DONE 后拼出本引擎记录、按已有代理回复回放对照组、写盲评面板/评审提示词/报告，dump 写下一个会话的每次叙述调用；解释器按玩家原文取 DIR/interp_answers.json；也是可导入的库（run_key 给定答案重放一个会话），重放时关掉迟到的先声（代理作答瞬时，它本不上场）
replay_drops.py: 闸门误杀排查，python scripts/replay_drops.py DIR [KEY] [--answers FILE] [--interp FILE] [--inputs FILE]：经 bench_online 在当前代码上重放录下的代理答案，逐条列出叙述者丢弃、略过与补上的句子并给出丢句数；--inputs 以 JSON 字符串数组取代探针文件里的整局游玩（M1 出口条件：DIR 取 tests/data/run6、--answers 取其 playthrough.answers.json、--inputs 取 tests/data/playthrough_duanyu.json，与测试重放同一份输入，探针改版不动它）；load_inputs() / replay() / dropped() 供 tests/test_replay_run6 钉住 M1 的误杀基线
sim_beats.py: 普通人版调参台（plan §7 M2），python scripts/sim_beats.py [--seeds 1-20] [--players ...] [--json FILE] [--check] [--jobs N]：不接模型、模板模式，种子 × 五种脚本化玩家（被动等待、跟随段誉、撒谎者、抢帛卷的、夜里溜走的——只凭玩家自己的认知与时钟行事）逐回合跑真实会话，报告 B1（按场景识别器只算玩家感知到的看点）、段誉到达琅嬛/澜沧江的种子数、到达的结局、讨价还价是否成立、钟灵是否被制、各驱力兑现次数、E1（空转推进回合比例）；--check 按 GATES（跟随型 ≥16/20 目击 ≥8/11、跟随型 E1 ≤ 10%、被动玩家下段誉 ≥12/20 到琅嬛且 ≥8/20 到澜沧江、讨价还价 ≥16/20）判定；也是可导入的库（play / summarize / table / PLAYERS），tests/test_commoner 按文件加载；结果表入库为 docs/results/sim_beats.md
bench_probes.json: bench_gm 的中文探针：整局游玩（大殿到澜沧江畔，说话/姿态/行动/等待/提问混杂）、60 句花样输入（R4，带参考类别 expect）、15 句瞎编前提（C2）、10 句不可能的企图（R1）；每条可带 setup 铺垫与 require 前提，claim 为真相可检的声称（持有/学成/身处/交到/被制/受伤），assert 为词法旁证关键词，note 为评审的事实依据

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
