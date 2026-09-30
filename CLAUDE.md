# RealTianLong - 图世界文字游戏引擎：规则内核裁定事实，角色经由各自的认知图理解世界
Python 3.11 + 标准库内核 + Neo4j（事实持久化）+ Qdrant（经历检索）+ LangGraph（智能体编排）+ PyTorch Geometric（关系动态模型）+ RLlib（角色策略）+ Gemini（开放语义与叙述）

<directory>
src/tianlong/ - 引擎本体（10 子包: core, kernel, cognition, persistence, memory, language, agents, learning, scenarios, runtime）
src/tianlong/core/ - 领域语言：实体/关系/命题/事件/变化，只依赖标准库
src/tianlong/kernel/ - 世界规则内核：唯一裁定事实处，纯函数结算（1 子目录: rules）
src/tianlong/cognition/ - 角色心智：信念存储与修正、认知图投影、候选行动
src/tianlong/persistence/ - 事实持久化：WorldStore 协议与实现，唯一写路径 commit
src/tianlong/memory/ - 可回忆经历：记忆写入策略、嵌入、向量索引、回忆
src/tianlong/language/ - 开放语义：模板文本、LLM 接入、对白、玩家输入解析与叙述
src/tianlong/agents/ - 智能体：LangGraph 单角色决策图、多角色扇出编排、脚本策略、后果预测、调度
src/tianlong/learning/ - 学习：PyG 关系动态模型（环境/角色两入口）、RLlib 模仿学习 + PPO 角色策略（1 子目录: rl）
src/tianlong/scenarios/ - 内容：初始世界 + 角色设定 + 以感知形式给出的初始认知（1 子目录: tianlong 天龙八部，第一幕无量山：普通人版与段誉旧版）
src/tianlong/runtime/ - 装配：权威写入器、游戏会话（主持层回合循环）、命令行与网页前端
tests/ - 验收即规格：每条设计边界对应可证伪断言（1 子目录: data 金标准指纹、录制的代理答案与闸门语料）
scripts/ - 本地基础设施与演示脚本（无 Docker 环境下的 Neo4j、录制一局“玩家所见 vs 世界真相”、Colab 笔记本生成器、主持层评测“本引擎 vs 纯模型主持人”、在线代理评测与闸门误杀重放、普通人版看点调参台）
docs/ - 入库的产出物与设计（3 子目录: demo 演示录像, results 训练报告与结果表, design 手写施工图）
notebooks/ - Colab GPU 训练笔记本（生成物：只调用同一套训练 CLI，固定提交、失败即停、产物写进 Drive）
.github/ - CI（1 子目录: workflows——核心零依赖套件 + 学习层与 Neo4j 全量套件）
</directory>

<config>
pyproject.toml - 包元数据；核心零依赖，graph/memory/agents/learn/rl 为可选 extras；pytest 标记 neo4j/learn/rl/slow（slow 默认不跑）
.env.example - 运行期环境变量模板（GEMINI_API_KEY、NEO4J_*、QDRANT_URL）；真实 .env 被 gitignore，密钥永不入库
.gitignore - 排除虚拟环境、缓存、密钥、日志与训练产物（artifacts/）
constraints.txt - 学习层依赖的锁定版本（torch-geometric / ray / gymnasium）：CI 与 Colab 笔记本同用一份，版本随提交走
README.md - 面向人的入口：架构、快速开始、验收用例、训练结果、边界与测试对照
</config>

<constitution>
依赖方向: core ← kernel, cognition ← persistence, memory, language ← agents, learning ← runtime；cognition 永不依赖 kernel
事实只由 kernel 裁定；GNN 预测、LLM 输出、智能体共识都只是意图，必须经 WorldAuthority 结算后才成为事实
先隔离信息、再做消息传递：learning 只接受 GraphView，角色入口在类型上拿不到 WorldState
确定性: 一切 ID 与随机种子由 blake2b 从语义输入派生；相同存档 + 相同意图 → 相同指纹
</constitution>

法则: 极简·稳定·导航·版本精确
