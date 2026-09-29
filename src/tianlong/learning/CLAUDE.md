# learning/
> L2 | 父级: /CLAUDE.md

GNN 学习"接下来可能发生什么"（阶段 B），RL 学习"应该选择什么"（阶段 C，见 rl/CLAUDE.md）。两者的决策输入都只来自角色认知：featurize 只接受 GraphView，角色入口在类型上拿不到 WorldState。

设计决策：编码器用 TransformerConv(edge_dim) 而非 RGCN/HGT——关系类型（含极性与方向）与可信度/时效/传闻都在边特征里，RGCN 类卷积会丢掉这些认知语义；节点种类作为特征，小图上与异构图表达力等价、且能与 RL 的定长观测互通。环境动态与角色视角两类样本分开构造、分开训练：前者学真相如何演化（输入含规则用到的全部机制变量——表示丢信息与真实随机性必须分开），后者学"做了这件事之后我会相信什么"（含"仍然不知道"与"确知不在原处"）。标签只来自内核实际执行的结果。维度与语义只在 schema.py 定义一次。

成员清单
schema.py: 类型化特征规格（Schema v2）唯一一处：节点列布局（种类 | 自身 | 每个属性一块值列 + known 三态列（1 已知 / 0 未知 / -1 不适用），数值以 value/scale、类别以独热进入 | 事件的操作/渠道/结果/原因/命题谓词/极性/提问）、关系词表（认知关系含极性与方向、ONEWAY_TO、事件关系含命题主语宾语）、行动编码字段（含言语命题）、预测目标声明 TARGETS（位置指针 + UNKNOWN/GONE、动态布尔三态、动态数值、发现、有效新观察数）；SCHEMA 指纹写进每个检查点，check_schema() 对规格不符或视角不符（全知模型冒充角色预测）以 StaleModel 明确拒绝
featurize.py: 按 schema 把 GraphView 编码为 GraphTensors（numpy 中立格式），每条边配反向边；bool_tri()/num_known() 从节点列读回动态属性（标签与输入同一定义）；encode_action() 把候选编码为 操作/方式/目标/对象（MOVE 为路线门）/行动者 + 言语命题（谓词、主语、宾语、极性、是否提问）
samples.py: 监督信号（StateDelta）：env_sample() 真实状态（含全部机制变量）→ 真实下一状态，agent_sample() 个人认知 → 结算后的认知，agent_query() 推理输入；“不知道”拆成仍不知道（UNKNOWN）/确知不在原处去向不明（GONE）/确知在一个此前不认识的容纳者那里（NEW）/认识新实体（discover）；动态属性按类型给标签（布尔三态、数值 + 是否已知：修习进度、学成技能、被吸走的内力）；有效新观察数由 datagen 给出；DynData 让所有指针字段随批偏移、空类用独立掩码；推理热路径 agent_queries() + shared_graph_batch()：一次决策的候选共用一张认知图，只构图一次、按块复制成批，与逐样本拼接张量逐项相同
task.py: arg_type()（配置字段展平成 CLI 参数的解析器，布尔严格按 true/false）；TaskConfig 任务分布契约（江湖化比例、规模与人数（越界即拒）、修习覆盖旋钮、“先探查”任务旋钮 hide_goal_items、守地与潜逃角色 roles、启用目标族——同时约束取样与奖励注册、时限）——GNN 数据、模仿示范、PPO 的每个 env runner、评测与检查点读同一份；registry() 只注册启用目标族，fingerprint() 进 manifest
datagen.py: 数据工厂，按 TaskConfig 取样的程序化世界里“脚本策略 + 分层随机探索”行动、内核执行，每 tick 只一人行动（孤立行动效果模型）；own_effect_slots()/observation_gain() 把“信念变化中扣除本行动直接效果（含拿到手的东西随之而来的手感、走过的门没锁）”定义为有效新观察数；记录样本所属世界供按世界切分；collect(workers) 按世界分给子进程、按世界序号拼回，与顺序收集逐项相同
model.py: RelationalEncoder（残差 TransformerConv 栈）、ActionEncoder（操作/方式/命题谓词嵌入 + 五个引用的节点表示 + 命题极性，策略网络同构）与 DynamicsModel（行动条件化 + 成败头 + 带 UNKNOWN/GONE 两空类与惯性项的位置指针头 + 动态布尔三态头 + 数值残差头与已知性惯性 + 发现头 + 有效新观察数头）；loss_terms() 每个目标一项，角色专有目标只在角色样本上计；关系编码在行动条件化之前、与行动无关，forward(encoded=) 让“一张图 × 多个候选”只编码一次
train.py: 训练与验收 CLI（python -m tianlong.learning.train --view env|agent，任务字段展平为参数，--resume true 按轮断点续训且与不中断逐位相同、换配置的断点被拒绝，不进 run_id），按世界切分；每轮在校准世界上算损失、取最低的一轮（测试世界从不参与选择），报告 model_selection 与逐轮历史；split 同时报世界数与样本数；指标按 TARGETS 逐项：位置召回只叫 holder_*（不冒充“全部事实”）、布尔属性逐属性、数值属性 MAE 对照“不变”、成败按操作的 Brier 对照**训练集**常数（测试集常数只作诊断）、发现与有效新观察数；世界三分 训练/校准/测试，温度只在校准世界拟合、测试世界报原始与校准后两种 Brier；coverage 报告每类机制在数据里出现几次；报告与检查点带 manifest、schema 指纹、视角与温度；--workers 并行生成数据（0 = 全部核）；device=auto 有 GPU 即用
provenance.py: 溯源 manifest：提交号、工作区是否干净与未提交改动的哈希（拿不到就写 None）、compat_signature()（全部语义版本的唯一定义：特征规格、属性、目标、奖励、候选规则、规则内核、观测布局——部署包比对的也是它）、任务指纹、完整配置、种子、依赖版本；run_id 由提交 + 工作区状态 + 配置 + 种子派生；EXACT_RESOURCE_KEYS（workers：只改执行方式、不改结果）不进 run_id、不挡续训
results.py: 结果出口 CLI（python -m tianlong.learning.results 报告... [--pair A B]... [--out]）：从机器可读报告生成带 run_id/提交号/任务指纹/种子的 Markdown 表；跨种子汇总只合并同一提交（含改动哈希）且除种子外配置相同的运行、同种子只取最新；--pair 用报告里的逐世界记录把两次运行（如训练期有/无预测、有/无记忆）在同一批留出世界上配对比较（rl/stats，不需要 torch）；README 的数字只从这里来
predictor.py: GNNPredictor 以 OutcomePredictor 协议接入 LangGraph 决策图（成败头按检查点里校准世界拟合的温度缩放；进展与风险取模型预测的下一刻认知为假想分支——与启发式共用 agents 的 BranchValuer，只省功不改值——加上自己下一刻受伤中毒被制的概率；孤立行动效果模型）；加载时核对规格与视角（只收 agent 模型）；成功率来自成败头，预期获知来自有效新观察数头（确定地走到已知处不算获知）；一次决策只构图、只编码一次（与逐候选路径逐项相同，环境步长 170→60 ms）
bundle.py: 部署边界（python -m tianlong.learning.bundle 目录 --predictor --policy --reports）：bundle.json 记下**训练时**的全部语义版本（取自检查点 manifest；打包时预测器与策略须同版本且与当前代码一致）、相对路径与 sha256；load_bundle() 逐项核对（格式、语义版本、哈希、策略声明的配套预测器是否在包里），任何不一致列出并拒绝（StaleModel，不做隐式迁移）；策略用哪个预测器训练（启发式或某个 GNN 文件的哈希），上线就用哪个
parallel.py: 并行原语 ordered_map()（spawn 进程池、按输入顺序拼回）、resolve_workers()（0 = 全部核）与 single_thread()（torch 的 CPU 运算随线程数差几个 ulp：子进程与顺序路径一律单线程）；世界与回合彼此独立，并行与顺序逐项相同——并行度是资源旋钮不是实验配置
profile.py: 剖析 CLI（python -m tianlong.learning.profile --worlds --steps --device --out）：世界生成、候选、启发式/GNN 预测、认知视图与编码、观测构造、策略前向、内核结算、认知修正逐段计时并写明设备——先量再放大；假想分支微基准在同一批直接效果上并列 branch_value_definition（逐候选完整假想）与 branch_value_shared（BranchValuer）
rl/: 强化学习（见 rl/CLAUDE.md）
__init__.py: 包入口（torch / torch_geometric / ray 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
