# learning/rl/
> L2 | 父级: /src/tianlong/learning/CLAUDE.md

角色策略的强化学习（RLlib 新 API 栈）。强化学习发生在训练阶段；游玩时运行训练好的网络（LearnedPolicy），不在玩家输入后临时重训。
训练环境与线上共用同一个内核与同一套认知修正，观测只来自各自的认知图；奖励 = 目标状态跃迁的任务奖励（经 core/goals 同一套语义，ACHIEVE 首次达成、MAINTAIN 按窗口判破坏与恢复，开局即满足者不给）+ 命名的势函数塑形（γΦ′−Φ）+ 分项成本（步长、失败、搜身落空）。

动作空间是"指针式"候选打分：候选对象由认知图提供（cognition.candidates），策略只学在候选集中选哪一个；掩码只屏蔽候选集之外的空位，从不按真相屏蔽行动。
冻结的世界模型（启发式或 GNN）为每个候选提供 (成功率, 有效新观察, 目标进展, 风险, 不确定性) 特征——"在每批策略采样期间固定其版本"。

成员清单
observation.py: 请求受益人进入候选编码和裁剪引用； OBS_VERSION（观测布局与语义的指纹，形状相同而含义变了也能拒绝旧策略）；ABLATIONS / ablate()（消融 = 某类输入整列置零：预测列、记忆列——训练、评测、上线同一个定义）；gymnasium 只在 observation_space() 里按需导入；ObsSpec 与定长观测契约 build_observation()：节点（叠加角色标记：自身/目标物/目标地/收件人/目标人物/盟友，未激活目标不标）、边、目标槽位（种类/权重/是否激活/距激活时间/了结条件/一次性或持续/信念读者给出的达成状态与进展 + 指向实体的指针）、完整候选编码（含言语命题）、候选预测、掩码；裁剪契约：自身、目标所指、保留候选的全部引用优先入图，放不下引用的候选整个剔除并记入 CropReport——绝不悬空；RL 环境与游戏内 LearnedPolicy 共用；另有逐节点的长期记忆摘要列（被提起几次、说法被证实/证伪几次、最近一次相关经历距今多久）
env.py: TianlongEnv（MultiAgentEnv，config["ablate"] 为训练期消融——消融了预测就不调用预测器；config 原样保留供子进程重建，world_seed 为当前一局的世界种子）；map_episodes() 把逐局函数分给 spawn 子进程、按顺序拼回并把 coverage（含值为 0 的键）并回，世界按 TaskConfig 取样（江湖化一路贯通到 reset，coverage 报告实际抽到的场景与目标），启用目标族之外的目标在 reset 时明确报错；同一 tick 同时出招、统一结算；先取候选全集再按行动族配额截断，动作编号对应裁剪后保留的候选，crops 暴露裁剪报告（配额截掉的也算）；GoalTracker 跟踪目标真实状态，每个角色的 MemoryView 由与线上同一个 records_for 增量维护，infos 携带分项奖励；expert_choices()/expert_actions() 给出脚本示范（含结构化标签）；last_events 只供评测数行为
rewards.py: RewardWeights / RewardBreakdown（任务、塑形、步长、失败、搜身落空五项）/ GoalRecord（激活时刻、初态、首次达成、首次破坏、持有 tick）/ GoalTracker（未注册目标构造即报错）/ step_reward()；γ 与 PPO 一致
module.py: 候选实体指针含受益人，按新观测布局打分； GraphPolicyNet（纯 torch：定长观测还原为稀疏批图 → RelationalEncoder + 角色标记 + 记忆摘要 → 目标槽位编码（特征 + 所指实体表示，掩码平均）→ 逐候选打分（操作/方式/命题谓词 + 目标/对象/命题主语/命题宾语表示 + 极性 + 预测）+ 价值头）与 CandidateScoringModule（TorchRLModule + ValueFunctionAPI 外壳）
imitation.py: 模仿学习：Demo（观测、动作、示范者标签、世界）、collect_demos()（示范与留出示范用各自的种子流，均 ≥ DEMO_SEED_FLOOR，与评测种子不相交；workers 并行、按局序拼回）、bc_weights()（固定逐样本权重使等待总份额恰为声明值，[0,1] 之外拒绝、单类时份额归存在的一类）、weighted_batch_loss()（除以名义批大小：epoch 贡献与分批方式无关，全等待批不被分母抵消）、behavior_clone()（逐轮报告等待损失份额与全等待批数）、holdout_metrics()（留出世界的等待/行动精确率召回率与按示范标签的准确率）
evaluation.py: 跑局与事件口径：行为计数只读事件与事前认知（与奖励权重无关）——搜身落空/无证据；动手得手/落空/被拒，缘由分目标所驱（寻仇、守地/潜逃时对非盟友、为护/取/送的物品对自己认定的持有者）/还手护人/无端；误指控；repeat_kind() 把重复确定性失败算无效循环、招架闪避后再出手算随机重掷；目标分开局即满足/新达成/持续守住并报用时与可达性；策略都可 pickle：RandomPolicy（无状态：每步随机数由种子、世界种子与局内步数派生，复用对象、换执行顺序都不变）、永远等待、脚本、NetPolicy（可测试期消融）；evaluate(workers) 各局分给子进程，与顺序评测逐项相同；统计推断转自 stats
stats.py: 统计口径（纯 numpy）：METRICS 分子分母定义、EpisodeLog（一局一个世界，可 JSON 往返）、cluster_ci 以世界为单位重采样、compare 在同一批世界上配对并按预先声明的容许差判等效（世界少于 MIN_WORLDS 不下结论）；算不出来的一律 None，报告是严格 JSON
train.py: 流水线 CLI（python -m tianlong.learning.rl.train）：模仿学习（示范世界与留出世界分开）→ PPO（所有角色共享参数、各自观测，env_config 携带同一份 TaskConfig）→ 同一批留出世界上评测 随机/永远等待/脚本/模仿/PPO/测试期置零预测 并两两配对比较。RLConfig 含展平的 TaskConfig 字段、γ、评测种子、等效容许差与训练期消融 --ablate-predictions / --ablate-memory（ablations() 进 env_config、检查点与报告；tag() 给出 policy_ppo[_noPred][_noMem]_s{种子}；训练期已无预测时不再做测试期置零）；--resume true 续用模仿学习权重与 PPO 断点（配置不符即拒绝，只许改轮数/评测规模/并行度；不进 run_id）；报告是严格 JSON、带 manifest 与逐世界记录（供 results --pair 跨运行配对），检查点带预测器哈希；--env-runners/--gpus/--workers（评测与示范的并行进程，结果不变、不进 run_id）供 Colab 放大
policy.py: LearnedPolicy 以 Policy 协议接入 LangGraph 决策图，替换 ScriptedPolicy 而不改图；加载时核对规格指纹并沿用训练时的 ObsSpec 与训练期消融（没见过的输入上线时照样置零），选中的观测下标映射回决策图的候选下标
__init__.py: 包入口（ray[rllib] 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
