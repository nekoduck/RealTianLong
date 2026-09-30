# scripts/
> L2 | 父级: /CLAUDE.md

本地开发基础设施与演示工具。不含引擎逻辑：只准备环境，或把引擎跑一遍给人看。

成员清单
neo4j_local.sh: 下载并启动 Neo4j Community（仅监听 127.0.0.1，限定堆与页缓存），要求预设 NEO4J_PASSWORD；供契约测试与 `--store neo4j` 持久化游玩
serve_public.py: --allow-migration 显式传给网页服务； 免费 HTTPS 外网试玩入口（本机需在线）；启动仅监听本机的网页服务与官方 cloudflared Quick Tunnel，回合用 JSON，入口与进程号写 .cache/public/deployment.json；关闭时一并停止两个子进程，端口被占用即拒绝启动，不代理其他服务
make_colab_notebook.py: notebooks/train_colab.ipynb 的唯一源头（单元格以 Python 字符串维护，可审阅可测试）；--commit 填入固定提交生成交给 Colab 的那一份，仓库里的版本不填（tests/test_notebook.py 核对一致）
play_demo.py: 按固定指令序列录制一局（默认无量山原著路线），写出 docs/demo/{world}.md：每回合玩家所见的叙述 / 内核裁定的全部事件 / NPC 各自理由，结尾对照玩家以为与真相；叙述经 CachedLLM 缓存，重跑不再花费调用
bench_gm.py: 主持层评测（设计 §7 与“评测细则”），命令行 --out DIR [--llm auto|scripted|none] [--baseline] [--judge] [--turns N] [--limit K]，也是可导入的库；把 GameSession 当黑盒逐回合跑（会话新字段一律 getattr，缺了就退化），每条探针从全新会话出发、先走 setup、异常逐条记下不中断；按 TurnReport 与世界真相算 L1/L2（挂钟 + on_text 首字）、R4、C1（kernel.violations）、C2/R1（声称之物在真相里成立且归到玩家名下）、G1、N1、F2（4-gram）、P1（词法），写 bench.json 与带门槛 PASS/FAIL 的 report.md；auto 用真模型且缓存关闭、无密钥退回 scripted，scripted 的报告注明不作验收依据
bench_rival.py: bench_gm 的模型侧（单独成文件只为不超 800 行，不依赖 bench_gm）：PureLLMGM 纯模型主持人对照组（由场景生成的全知世界圣经 + Jenova 公开主持规矩 + 每回合全量对话、流式计时）、独立评审（premise_prompt / pairwise_prompt，严格 JSON，parse_verdict 容忍坏输出、拿不准判无效，盲评顺序由种子固定打乱）、--llm scripted 的离线脚本模型（解释器 JSON 兼容新旧 schema、照事实清单写叙述、顺着玩家的主持人，模拟首字 0.6/0.3 s 与 250 字/秒）
bench_probes.json: bench_gm 的中文探针：整局游玩（大殿到澜沧江畔，说话/姿态/行动/等待/提问混杂）、60 句花样输入（R4，带参考类别 expect）、15 句瞎编前提（C2）、10 句不可能的企图（R1）；每条可带 setup 铺垫与 require 前提，claim 为真相可检的声称（持有/学成/身处/交到/被制/受伤），assert 为词法旁证关键词，note 为评审的事实依据

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
