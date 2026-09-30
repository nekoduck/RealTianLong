# core/
> L2 | 父级: /CLAUDE.md

领域语言层。只依赖标准库，是整座依赖图的最底层：kernel 用它裁定事实，cognition 用它表达信念，persistence 用它存储，learning 用它构造标签。所有值对象不可变、可哈希、可 pickle，映射字段一律是 FrozenMap（连内容也改不动），查询一律返回排序元组——回放确定性与快照安全在这一层就被保证。

成员清单
schema.py: 领域词汇表，Kind/Rel/Op/Manner 枚举（含动手/研读/施用）+ Social 言语行为封闭词表（赔罪/威胁/叫阵……修辞层，HOSTILE_SOCIAL/FRIENDLY_SOCIAL 分组）+ RelSpec（函数型关系决定信念互斥槽位）+ OpSignature（行动语法，MOVE = 目的地 + 路线门；命题分“接受”与“必须”，言语的命题可选）+ ATTR_PREFIX
attributes.py: 类型化属性规格——每个属性的类型（布尔/数值/类别）、适用种类、获知途径（外观/状态/交互/手感/内省/只有世界知道）、是否被行动改变、数值尺度；派生 OBSERVABLE/STATUS/TACTILE/INTROSPECTIVE/PRIVATE 集合与 DYNAMIC_ATTRS；true_value() 从存储值 + 时钟派生规则真正使用的值（点穴余时、兵刃缺省锋利），kernel 与环境视图都经由它读值；ATTRS_VERSION 规格指纹
goals.py: 目标语义单一定义——七类目标的达成（三态，不知道 = None）/势能（只作塑形）/一次性或持续（ACHIEVE/MAINTAIN），GoalReader 协议让同一套语义由世界读者（奖励、评测）与信念读者（角色观测）分别求值；GoalRegistry 对未注册或缺字段的目标明确报错 UnsupportedGoal，绝不静默套用递送兜底
entities.py: Entity/Relation 值对象，属性存为有序元组以保证可哈希与 repr 稳定
changes.py: 世界变化语言 AddRelation/RemoveRelation/SetAttr，带前置条件（删除要求存在、改值要求旧值匹配），relocate() 表达 AT 的一删一增
world.py: WorldState 不可变实际世界图（实体表与内部邻接索引为 FrozenMap），apply() 只改事实、stamp() 才推进版本，fingerprint() 是回放验收判据
frozen.py: FrozenMap 只读快照映射，dict 子类封死全部就地修改（赋值/删除/update/pop/popitem/clear/setdefault/|=），__reduce__ 重建使 pickle/deepcopy 可用，JSON 可序列化、与 dict 判等、拷贝走 C 实现——frozen dataclass 冻不住的“字段里那个 dict”由它封死
propositions.py: Proposition（命题内容，不含"谁相信"）与 Fact（带极性），slot 定义函数型谓词下的互斥槽位
events.py: 因果链数据 Intent → Event（真相）→ Observation（服务端溯源）→ Percept（角色可见的片面内容，刻意不带来源 ID）；结算原因封闭词表 RULE_REASONS/ADMISSION_REASONS（kernel 产出、learning 编码的契约，reason_key() 把语法拒绝归一）；MOVE 的 obj 是所走的路线（门）；EntitySketch.seen 区分亲眼所见（有外观）与只闻其名（外观未知）；言语原话/姿态 utterance 与言语行为 social 随意图与感知传递，只是修辞，从不产生事实
grammar.py: signature_error() 行动语法检查（言语不带命题也合法），kernel 用真实种类、cognition 用已知种类调用同一把尺子；只查"能不能这样说"，不查"能不能做成"
profiles.py: Goal/Profile 角色设定卡，目标角色条件化（守护/获取/递送/守地/寻仇/灭口/护人）+ 时间闸门 not_before + 寻仇了结条件 until + 盟友 + 信任度 + 主持层用的腔调 voice / 谈资 knows（按“；”分条）/ 公开来历 intro / 话多 chatty / 脾气 temper；interests() 汇总关注的人与物
memories.py: MemoryRecord 经历权威记录，known_at 用于检索时的时间过滤，informant/verdict 结构化记下“谁的说法后来被亲眼证实/证伪”，向量索引只是它的派生
ids.py: blake2b 确定性派生 digest/derive_seed/make_id，拒绝 hash() 与 uuid4 以保证可回放
clock.py: 游戏时间为整数分钟，1 tick = 1 分钟，clock_label() 渲染"第1日 08:10"，is_night()/minutes_until_night() 定义昼夜（戌时入夜、卯时破晓）
__init__.py: 再导出全部公共类型

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
