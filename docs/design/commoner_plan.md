# 无量山·普通人版：赢过纯模型主持人的实施方案（Plan v7）

> 基线：`/home/user/RealTianLong` @ `867bf24`。本方案综合了 minds / director / narration / scenario 四份提案，以及三位评审的打分与建议。
> 用户原话：“继续啊，看看能不能最终战胜纯模型主持人……实在没招了你还可以动用最高智能改架构……玩家从现在开始变成普通人，不再单纯的段誉视角。”
> 约束：内核仍是唯一裁判；认知隔离；ID 与种子确定性；单文件 ≤ 800 行；不改内核规则，也不改程序化世界里的 ScriptedPolicy（训练好的包因此仍然有效）。

---

## 0. 一页结论（TL;DR）

**为什么六轮都输。** 评审六轮的引证可以归成三类根因：

1. **世界里没有戏。**
   - 钟灵被点住困在殿里；龚光杰受了伤就解了气，没人追。
   - 私奔那对到 19:20 都待在大殿；A10–A17 一路空无一物。
   - 这些是**世界状态**的问题，不是文笔问题。
2. **闸门误杀，叙述有缝。**
   - 用 run6 录下的答案在 HEAD 上重放：丢掉的 5 句**全是误杀**，丢掉的正是评审点名要的月下舞剑、蒲团绣字。
   - 回答被埋在打斗流水账里；钩子后面还挂着模板行。
3. **NPC 的嗓子弱**：B 的 NPC 当面回答玩家问的话，声口各异。

**方案有四根柱子。** M0–M4 不增加任何模型调用：

| 柱子 | 做什么 | 来源 |
|---|---|---|
| A. 叙述精度 | 修掉闸门误杀；回答先说（answer-first）、同一对人的交手合并；钩子后不挂模板行；硬事实审计 | narration 提案的第 1–3 包 |
| B. 普通人世界 + 角色驱力（Drive） | 玩家是马五德的挑茶伙计**阿顺**，段誉降为 NPC；名场面由“只读自己认知”的驱力生成，经内核结算：讨价还价、火把追逐、后院私语、跳崖、月下玉璧、叩首、帛卷、夜半下山 | scenario 提案作骨架，吸收 director 的时钟月出、等到某时、天亮结局 |
| C. 话即事实 | NPC 的原话在结算**之前**由 NPC 一侧产生（驱力台词；M5 的反应心智），挂在 `Intent.utterance` 上落库；叙述者照录，不许另编 | minds 提案的管线，先供驱力使用 |
| D. 相识账本 | 没人道出姓名之前，人一律用外貌称呼（梁上的青衫少女）；玩家与 NPC 共用一套规则 | 四份提案合一 |

**顺序。**

1. M0：腾位置、钉住金标准、修对照组圣经。
2. M1：闸门精度。先在旧版世界上重放 run6 验证。
3. M2：普通人世界与驱力层。
4. M3：看点、卡片与硬事实审计。
5. M4：评测改造，跑 run7。
6. M5：给被玩家搭话的角色加反应心智（M5 一定开发，是否默认打开由 run8 的消融结果决定）。
7. M6：前面都不够时的升级方案，即 minds 提案的完整形态。

**目标。**

- run7：在 3 个盲评面板中，至少 2 个面板在至少 2 个维度上判 A 胜。这是第一次打破 0/72。
- run8：至少 2 个面板在至少 3 个维度上判 A 胜，其中必须包括 responsive 与 fun。
- 首字延迟 L1 p50 ≤ 1.5 s。
- 硬事实错误率保持 0%（B 是 13%）。

---

## 1. 诊断：证据 → 根因 → 对策

| 评审反复引用的例证（run2–6） | 根因 | 对策（里程碑） |
|---|---|---|
| B11–13 火把追上崖顶“酸秀才，这回看你还往哪里逃！”，而 A10–17 空无一物 | 龚光杰 `HOSTILE until=wounded`，打过一掌就解气；没有追逐 | 龚光杰的驱力：搜寻、堵截、遇单向断崖放弃（M2）；看点卡 `torch_search`（M3） |
| B9 在后院偷听到“天黑了再走……营里那边已说定了……”；B 的审计 5 次里 4 次记它“私奔那对位置不对” | 私奔那对 19:20 前都在大殿 | 私奔那对的驱力：18:05–19:20 在后院幽会，四下无人时说出声，有人来就住口（M2）。**真相里他们就在那儿**，B 最常被审计揪出的毛病反过来成了我们的强项 |
| B7 用解药讨价还价，左子穆“好，依你”；A 的钟灵被点穴后又凭空出现（唯一一处 high 级缺陷，此后多轮是 medium） | 左子穆护短，直接出手制住钟灵、搜走解药 | 左子穆的“索药”驱力抢在还手之前；钟灵的“以药换人”：真用她身上的解药；左子穆看见后“依你”并停手（M2） |
| B22 “石壁上有仙人！”；A21 的月亮早已升起，A22 只照亮了《易经》 | ① 闸门以“长剑”误杀了舞剑那句；② 月亮不是时钟事实 | 闸门对比喻语境放行（M1）；`MOONRISE=19:40` 作为场景时钟事实，加 `yubi@moon` 景观、“等到月亮出来”、天色审计（M2/M3） |
| B28 钟灵数磕头，B34 钟灵扑空；A 一拜就得帛卷，A29 查看蒲团几乎没有回应 | 叩首即 INSPECT；绣字那句被当作替玩家说话而误杀 | 多 tick 叩首（场景参数）；钟灵的 tease/grab 驱力；闸门把绣字当物件上的字（M1/M2） |
| A4 马五德的回答埋在打斗里；“钟灵又是一下，龚光杰吃痛受伤”读着像流水账 | 叙述按时间顺序罗列事实 | 叙述节目单：玩家这一步 → 对玩家的回答 → 合并后的交手 → 进出（M1） |
| A20 “钟灵的笑声，一下子断了”悬空；A28 钩子后还挂一行 | 起因句被丢，反应句留下；漏讲的事被追加在最后一句之后 | 起因丢了，悬空的反应句一并略过；补句插在最后一句**之前**（M1） |
| A6 被封穴还拍手说话；A33–35 帛卷易手又回来 | 叙述越权 | 硬事实审计：动作能力（affordance）、持有转移（M3） |
| 名字没人介绍就先用上（run3–6 每次审计都有） | 内核给观察者的草图带着真名 | 相识账本与外貌称呼（M2） |
| NPC 背谈资、答非所问（run2–5）；左子穆一言不发 | 叙述者同时拿着十几条提示替 NPC 编词 | run6 已大为改善；M5 让被搭话的角色自己的心智写回话 |

B 的结构性弱点仍然存在：硬事实错误率 13%；私奔那对不在该在的位置；火光没有来由；替玩家做决定（B 每轮 3–5 条 low 级缺陷）。换成普通人主角以后，B 还得凭空编出段誉在幕后的整条轨迹。我们这边段誉真的在跑，终章可以照实讲出来。

---

## 2. 架构总览

### 2.1 一回合里谁做什么

```
回车 ─┬─(并行) 解释器：规则快路径 0 ms / 快模型 JSON；相识账本之后，称呼一律用外貌
      └─(并行) 首 tick 的 NPC 决策：ScriptedPolicy 外面包一层 Driven（驱力：只读自己的认知）
             └─ 驱力台词 = Choice.line → Intent.utterance（修辞，随意图落库）
      ↓
  WorldAuthority 逐 tick 结算（原子提交）：会话运行态新增 drives 驱力标记、names 相识账本、staged 看点账本
      ↓
  [M5] 反应 tick：若玩家当面对某主要角色说了话 → 该角色的反应心智（快模型，只看自己的 AgentPort，≤1.2 s）
       → 一次性驱力（Act + line）→ 同一条实现路径与合法性检查
      ↓
  build_brief：
    - 台词：带录入原话的标 said=True，叙述者照录
    - 看点：从玩家感知里识别出的场景看点 → focus / cards / spectacle
    - 天色：时钟 + MOONRISE
    - 名字：经相识账本过滤
    - 节目单：先回答、合并交手
      ↓
  叙述者：先声的时限从回车算起；流式逐句过闸门；闸门有：
    - 精度修复
    - 照录一致性（fidelity）
    - 持有转移、动作能力、天色审计
    - 补句插在最后一句之前
      ↓
  _remember：谈资账本、相识账本（交付正文里归属清楚的自我介绍）、看点账本、said_before
```

### 2.2 四条设计原则

1. **先有戏，再有文字。** 名场面是世界里真实发生的事件，由角色驱力产生、由内核结算，叙述者只负责把它写出来。
   - 驱力是**性情**，不是剧本：条件只读角色自己的认知（Situation），行动只从合法候选里挑，或者是一句台词/一个姿态。
   - 驱力绝不读真相、绝不看玩家在哪。所以不采用 director 提案的“等玩家看得见再触发”。
2. **话即事实。** NPC 说的原话在结算前由 NPC 一侧产生，经 `Intent.utterance` 落库；在场的人听见、记住。
   - 落库的事实是“X 说了 Y”，Y 本身永远只是修辞。
   - 叙述者对这些原话**照录**（可截取连续一段），不另编。
3. **闸门要准，不要松。**
   - 每一条放宽都对应一个有标注的误杀样本；同时用 ≥60 条人为植入的硬事实错误测召回率（≥95%）。
   - 不把“长剑”这类实体名加进 COMMON_WORDS。那会放掉“龚光杰拔出长剑”这种 B 式的物品漂移。
4. **真相只用于呈现，不用于驱动。** 以下可以读真相：
   - 看点识别，只针对玩家已经感知到的事件；
   - 结局判定；
   - 终章纪事；
   - 评测。
   NPC 的行为与 NPC 的提示词只读它自己的认知。

### 2.3 采纳与舍弃

| 来源 | 采纳 | 舍弃（理由） |
|---|---|---|
| scenario | 普通人阿顺（马五德的伙计、段公子的照看人）；驱力层（装饰器，不改 ScriptedPolicy）；讨价还价与“依你”；火把追逐；私奔那对；夜饭换岗；三种结局与变体；纪事终章；sim_beats 与 B1；对照组圣经公平 | `Said.line` 改认知与编解码（改为会话运行态里的驱力标记）；把驱力挂在 Profile 上（改为 `Scenario.drives`，Profile 不动）；字符串式的 DSL（改为类型化条件） |
| narration | 误杀语料与精度修复；answer-first 节目单与交手合并；补句前插；悬空反应句略过；持有/动作能力/天色审计；细节卡组；看点卡；相识的学习触发 | chorus 回写（叙述文字变成下一 tick 的意图，因果倒置）；S 级软事实账本（第二个真相源）；`Profile.pacifist` 改 MartialTactics（改用 Driven 的 VETO）；furtive_pair 卡（叙述一件没记录的事） |
| director | `MOONRISE`/`DAWN` 时钟事实；等到月出/天亮；天亮结局；空转回合指标 E1；三类玩家脚本；“不许编造征兆”的系统规则 | 全知的节拍调度、可目击松弛（witnessability slack）、张力曲线（真相会漏进 NPC 的时机）；`core/beats.py`；带台词种子的纹理节拍（等于背台词） |
| minds | `Choice.line`→`Intent.utterance` 管线；照录规则与一致性闸门；`mindgate`；玩家代理评测模式；R5/F3 指标；M5 的反应心智 | 主动预取、推测性确认与重键、看板、4 人聚光（放到 M6）；剑湖宫杂役身份（不知道段誉，却预知断崖与玉璧传闻，发现的乐趣没了） |

---

## 3. 普通人剧本设计（Ordinary-person scenario）

### 3.1 玩家：阿顺（`ashun`）

- **身份**：普洱马五德茶号的伙计，替东家挑一担普洱上山当贺礼；东家叮嘱他路上照看同行的段公子。
- **数值**：`martial=0.05, agility=0.5, alertness=0.55`。
- **别称**：阿顺、小伙计、挑茶的。“伙计”“小兄弟”放进 common_words。
- **随身物品**：
  - `chabing` 普洱茶饼（small，贺礼，可以献出）；
  - `huozhezi` 火折子（纯作风味，没有内核属性；补上 B“来历不明的光”那个缺陷）；
  - `suiyin` 碎银（small，可以送出）。
- **开场位置**：大殿里马五德席后，段誉身旁。开场钩子是冲着“你们这一席”来的，不是冲着玩家本人。
- **世界 ID**：`wuliang_c`，存档与旧版互不串。

### 3.2 初始认知（以“过去的感知”给出）

- **地形**：`d_gate`（从山道上来的）与 `d_corridor`（马五德吩咐茶担先挑去后院厨下）。**不知道**后山、断崖、玉璧。
- **认得的人**：
  - 马五德、段誉，以及比剑时唱过名的龚光杰、左子穆、辛双清。
  - **不认得**钟灵、干光豪、葛光佩、司空玄与帮众。
- **态度**：先验里有段誉的 GREET 与马五德的叮嘱，折叠后这两人对阿顺的态度为 +1。这一点先要 spike 验证（§7 M2-a）；不成立就在 `WorldAuthority.found` 按场景的 `regard` 写入。
- **传闻**：马五德的谈资里有一条“听说剑湖宫后头有面玉璧，月夜里显过仙人影子”，是寻宝的线头，只有被问起才会说。

### 3.3 目标与提示

- **目标只驱动提示，玩家不跑策略**：`Goal(DEFEND, person="duanyu", weight=0.5)` 与 `Goal(ESCAPE, home="shanjiao")`。
- `world_bible` 与 `goal_text` 对玩家一律用玩家口吻：“照看段公子（东家吩咐的）；平安下山”。**再也不出现“灭口”。**
- `GUIDE_C`：八级提示，只说有哪些可能，不给攻略：
  1. 先顾好自己，这位龚爷惹不起；
  2. 东家马五爷见多识广，不妨问他；
  3. 段公子惹了祸，帮不帮在你；
  4. 茶担本该挑去后院厨下；
  5. 山道被神农帮封着，他们也要吃饭换班；
  6. 段公子若是跑了，多半往没人的地方去；
  7. 玉璧的传闻，月夜里或许作数；
  8. 天总会亮。
- `GUIDE_AT_C`：各地点的起点。
- `HINTS_C`：扯扯段公子的袖子 / 替段公子向龚爷赔个不是 / 问马五爷这是怎么回事 / 抬头看看梁上那少女 / 悄悄溜去后院 / 环顾四周。

### 3.4 段誉降为 NPC

- **设定**：`is_player=False`，没有 goals（ESCAPE 带灭口语义，所以不给），`chatty=0.6`，`temper=-0.8`。
- **腔调与谈资**：沿用 voice/knows；新增 `intro`（“大理来的书生，姓段”）与对阿顺、马五德的 knows。
- **整条弧线由驱力承载**（§3.5）：
  - 挨打后逃出大殿；
  - 被追到崖顶就跳下断崖；
  - 月下看玉璧 → 进石缝 → 叩首 → 翻查蒲团 → 取帛卷 → 研读三遍 → 钻隧道。
- **不还手**：由 Driven 的 VETO 保证，他的 ATTACK 一律改成求情或逃跑。**不改 MartialTactics。**
- **玩家可以**：跟随、带路（follow-fond）、抢先拿走帛卷（他会恳求借阅）、骗追兵、出卖他、丢下他。

### 3.5 驱力表（内容在 `scenarios/tianlong/drives_c.py`）

条件全部在该角色自己的认知上求值。“once”表示兑现一次后不再触发（驱力标记）。台词是一次性的修辞。

| 角色 | 驱力 | 级别 | 条件（只读自己的认知） | 行动 | 台词 |
|---|---|---|---|---|---|
| 段誉 | `pacifist` | VETO | 自己选了 ATTACK | 改为 `Say(对手, PLEAD)`；对手在场且自己受了伤则 `Flee` | “君子动口不动手……” |
| 段誉 | `flee` | URGENT | 受伤 ∧ 近 3 tick 打过他的人在场、且那人**没中毒也没被制** | `Flee(prefer=houyuan>houshan>yading)`，不朝打他的人走 | “君子不立危墙之下，在下告辞！”（once） |
| 段誉 | `wait_fond` | URGENT | 刚逃出 ≤2 tick ∧ 态度≥1 的玩家刚才还在身边、现在不在 ∧ 打他的人不在眼前 | `Pose("回头张望")`，1 tick | — |
| 段誉 | `follow_fond` | URGENT | 态度≥1 的玩家上一 tick 从此处离开 ∧ 打他的人不在 | `Follow(player)` | — |
| 段誉 | `wander` | IDLE | 在后院 ∧ ≥18:00 ∧ 没有仇人 | `Go(houshan)` | “这后山倒是清幽。”（once） |
| 段誉 | `leap` | URGENT | 在崖顶 ∧（龚光杰在场，或近 2 tick 听见/看见他向这里来） | `Cross(d_cliff)` | “与其落在你手里，不如赌一赌这藤萝！”（once） |
| 段誉 | `rest` | IDLE | 在剑湖畔 ∧ 受伤 ∧ 月出之前 | `Pose("倚着湖石歇息")`，冷却 10 | — |
| 段誉 | `gaze` | IDLE | 在剑湖畔 ∧ ≥MOONRISE ∧ 知道玉璧 | `Inspect(yubi)`（夜里内核按线索显出石缝） | “咦，这石壁上……竟有人影？”（once） |
| 段誉 | `explore_crack` | IDLE | 知道 d_cave | `Go(langhuan)`（BFS 自然穿过石缝与石门） | — |
| 段誉 | `kowtow` | IDLE | 在琅嬛 ∧ 看见玉像 ∧ 还没翻查过蒲团 ∧ 本驱力兑现 <6 次 | `Pose("对着玉像恭恭敬敬地磕头", SUBMIT)` | 第一次：“神仙姐姐在上……” |
| 段誉 | `search` | IDLE | kowtow 已兑现 6 次 ∧ 没翻查过蒲团 | `Inspect(putuan)` | — |
| 段誉 | `take` | IDLE | 知道帛卷在此 ∧ 帛卷不在别人手里 | `Take(scroll_lb)`，然后 `Take(scroll_bm)` | — |
| 段誉 | `plead_scroll` | IDLE | 帛卷在一个态度≥1、且在场的人手里 | `Say(那人, PLEAD)` | “阿顺兄，那帛卷可否借在下一观？”（冷却 10） |
| 段誉 | `study` | IDLE | 手里有 scroll_lb ∧ 还没学成 | `Study(scroll_lb)` | 学成时：“妙哉，这步法竟暗合易经！” |
| 段誉 | `leave` | IDLE | 学成 ∧ 知道 d_tunnel | `Go(lancang)` | — |
| 钟灵 | （目标）DEFEND 段誉 | — | 沿用：谁打段誉就打谁；段誉下落不明就去找（能跟着跳崖） | — | — |
| 钟灵 | `bargain` | URGENT | 龚光杰中毒 ∧ 解药在自己身上 ∧ 左子穆在场 ∧（听到他喝令/威胁，或看见他向自己出手） | `Use(antidote, gongguangjie)` | “解药在这儿。你们不再为难这书呆子，我便救他。”（once） |
| 钟灵 | `introduce` | URGENT | 近 2 tick 有人向她道谢或搭话 ∧ 本驱力未兑现 | `Say(那人, GREET)` | “谢什么？你又打不过他。我叫钟灵，你呢？”（once） |
| 钟灵 | `marvel` | IDLE | 在剑湖畔 ∧ ≥MOONRISE ∧ 段誉或态度≥0 的玩家在场 | `Say(段誉或玩家, REMARK)` | “书呆子你快看！石壁上有仙人！”（对玩家说：“喂，你快看！……”）（once） |
| 钟灵 | `tease` | IDLE | 近 2 tick 看见有人摆出“磕头”姿态 | `Say(那人, JOKE)` | “你真要磕一千个？磕傻了我可不管你！”（once） |
| 钟灵 | `grab` | IDLE | 看见段誉研读学成（需 spike 核实学成是否可见） | `Pose("扑过去抓他袖子，却扑了个空")` | “不玩了不玩了！书呆子，你这是什么古怪步法？”（once） |
| 左子穆 | `demand` | URGENT | 盟友中毒 ∧ 看见 X 打了盟友 ∧ X 在场且没被制 | `Say(X, COMMAND)`，**抢在还手之前** | “小姑娘，把解药交出来，今日之事还有得商量。”（once） |
| 左子穆 | `await` | URGENT | demand 兑现 ≤2 tick ∧ 盟友仍中毒 | `Hold` | — |
| 左子穆 | `truce` | URGENT | 看见 X 用解药救了盟友 | `Say(X, AGREE)` | “……好，依你。”（once） |
| 左子穆 | `truce_veto` | VETO | truce 兑现后 30 tick 内，对 X 的 ATTACK | 改为 `Hold` | — |
| 马五德 | `plead` | URGENT | 看见龚光杰打段誉或玩家（1 tick 内） | `Say(龚光杰, PLEAD)` | “龚老弟，和气生财，和气生财……”（冷却 5） |
| 马五德 | `errand` | IDLE | truce 已兑现（听见左子穆“依你”）∧ 玩家在场 | `Say(player, COMMAND)` | “阿顺，茶担先挑去后院厨下歇着，莫在这里碍眼。”（once） |
| 龚光杰 | （目标）HOSTILE 段誉 until=wounded | — | 沿用：大殿里先礼后兵，一掌出气 | — | — |
| 龚光杰 | `hunt` | URGENT | ≥18:40 ∧ 自己没中毒 ∧ 段誉不在眼前 ∧ 没有放弃 | `Pursue(duanyu, avoid_oneway)`：临时 `Goal(HOSTILE, duanyu, until="subdued")` 交给 `MartialTactics._hostile`，复用找人、向人打听的逻辑 | 第一步：`Pose("从廊下摘了一支火把")`（once） |
| 龚光杰 | `corner` | URGENT | 段誉在场 ∧ 不在大殿 ∧ 正在 hunt | `Say(段誉, TAUNT)`，下一 tick 由 inner 出手 | “酸秀才，师父只答应在殿上不为难你——这回看你还往哪里逃！”（once） |
| 龚光杰 | `give_up` | URGENT | 看见段誉经断崖下去，或自己认为到段誉那里要经过单向门 | `Say(REMARK)` + `Go(hall)`，并记下已放弃 | “哼，摔不死你也困死你！”（once） |
| 干光豪 / 葛光佩 | （目标）ESCAPE camp not_before 19:20 | — | 沿用：落单的撞见者才灭口 | — | — |
| 干光豪 / 葛光佩 | `tryst` | URGENT，窗口 18:05–19:20 | 不在后院 | `Go(houyuan, CAREFUL)` | — |
| 干光豪 / 葛光佩 | `murmur` | IDLE，窗口 18:08–19:20 | 在后院 ∧ 伴侣在场 ∧ 除 LOVERS 外没人 | `Say(伴侣, REMARK)` 正常音量，冷却 2，四句轮换 | “……天黑了再走……” / “……营里那边已说定了……” / “……师父那边，瞒得过吗……” / “……别怕，有我……” |
| 干光豪 / 葛光佩 | `hush` | URGENT，窗口内 | 在后院 ∧ 刚有外人进来（company 新增） | `Pose("两人登时住了口，各自别过脸去")`；干光豪随后 `Say(外人, COMMAND)` | “看什么看？还不去干你的活！”（每人一次） |
| 葛光佩 | `mercy` | URGENT | 1 tick 内看见伴侣打 X ∧ 2 tick 内 X 服软或求饶 | `Say(伴侣, PLEAD)` | “师哥，他一个挑茶的，饶了他吧！”（once） |
| 干光豪 | `spare` | VETO | 伴侣 1 tick 内为 X 求情之后 10 tick 内，对 X 的 ATTACK | 改为继续赶路 | — |
| 神农帮帮众 | `supper` | URGENT，窗口 20:00–22:00 | 不在营地 | `Go(camp)` | “换班吃饭去，谁耐烦在这儿喝风。”（once） |
| 神农帮帮众 | `bribed` | URGENT | 态度≥2 的 X 在场（送过碎银，赠物 +2） | `Pose("别过脸去，只当没瞧见")`，并对 X 的 ATTACK 做 VETO | — |

补充规则：

- **唤醒**：驱力的时间窗一打开（`Between.start` 落在上次决策与现在之间），`runtime/cast.wakes()` 就把该 NPC 列为需决策。不改 `Scheduler`。
- **拥挤上限**：同一处、同一 tick 至多两句驱力台词。每回合交给叙述者的“照录”台词至多 3 句，看点优先。

### 3.6 预期时间线（由 sim_beats 调参保证；这是分布，不是剧本）

**被动玩家**（一直在大殿里等）：

- 17:41–45：龚光杰叫阵，一掌打伤段誉。
- 17:45：钟灵放貂，龚光杰中毒。
- 17:46：左子穆索药。
- 17:47：钟灵用解药换人；左子穆“依你”。
- 17:49：龚光杰已痊愈且在眼前，段誉逃往后院，钟灵跟上。
- 18:00：段誉转去后山。
- 18:05：私奔那对进后院（四下无人，低声私语）。
- 18:40：龚光杰举火搜人，经后院（那对住口）到后山。
- 约 18:50：在后山或崖顶堵住段誉，段誉跳崖；钟灵随后也下去。
- 19:20：私奔那对动身。
- 19:40：月出，钟灵惊呼，段誉看玉璧，石缝显出。
- 约 20:00：叩首、翻查蒲团、取帛卷、研读。
- 约 20:30：段誉钻隧道到澜沧江畔。
- 20:00–22:00：帮众回营吃饭。
- 05:00：天亮。

**跟随玩家**：上述场面大多亲眼看见。

- 跟到后山之后回后院取水，就能偷听到私奔那对。
- 到了崖顶可以选择跟着跳。

**分歧点**：

- 骗龚光杰“往山道去了”：他去山道，撞上帮众。
- 抢先拿走帛卷：段誉恳求借阅。
- 带段誉去山道：没有跳崖。
- 19:21 独自一人在后山：私奔那对下手灭口；求饶则葛光佩求情。
- 20:00–22:00 从山道下山：下山结局。

### 3.7 钩子（开场与中途，全部由世界产生）

1. 开场：龚光杰怒目瞪向“你们这一席”，段公子还在笑。
2. 马五德的差事：茶担挑去后院厨下，把人引向后院。
3. 私奔那对的私语与住口。
4. 龚光杰找人时向在场的人打听，阿顺可以撒谎，也可以出卖。
5. 马五德那条玉璧传闻，问起才说。
6. 帮众夜饭换岗，从山道能听见营地那边的吆喝（SOUND）。
7. 崖顶：跟着跳，还是回去。
8. 天亮。

### 3.8 结局

`Ending` 增加 `at_clock: int | None` 与 `variants: tuple[Variant, ...]`（`requires ∈ {skill:S, with:P, holds:I}`）。`endings.ended()` 的判定：到达某地点，**或者**时钟已到。

| key | 触发 | 标题 | 变体 |
|---|---|---|---|
| river | 真相里在澜沧江畔 | 第一幕终 · 澜沧江畔 | 身负奇功（学成 evasion）；与段公子同行（段誉也在 lancang）；怀揣帛卷 |
| downhill | 在 `shanjiao`（新地点 无量山脚，经新门 `d_downhill` 与山道相连） | 第一幕终 · 下山 | 与段公子同行 |
| dawn | 时钟 ≥ 第 2 日 05:00 | 第一幕终 · 天亮了 | 身在何处（剑湖宫 / 崖底 / 石室） |

**终章**：

- 收束段落只取玩家的亲历，与现在相同。
- 旧的“真相揭晓”表换成 **纪事“那一夜你没看见的事”**。`endings.chronicle()` 从 `Scenario.chronicle` 所列角色（段誉、钟灵、私奔那对、龚光杰）的真实事件里，确定性地生成**江湖传闻口吻**的叙述：
  - 单向门的穿越；
  - 学成；
  - 讨价还价；
  - 灭口或饶命；
  - 放弃追赶。
- 一律用相识账本之后的称呼。
- 不列玩家以为 vs 实际的对照表。审计曾批评它像调试输出。

### 3.9 称呼与相识（`runtime/names.py`）

- **外貌称呼（epithets）**：钟灵 → 梁上的青衫少女；干光豪 → 高个子的东宗弟子；葛光佩 → 清秀的西宗女弟子；司空玄 → 须发花白的药农头领。帮众本来就是泛称。
- **玩家学会一个名字的途径**：
  1. 交付正文里归属清楚的引语中，说话者自报姓名（我叫/在下/本姑娘 + 名或姓）；
  2. 听到录入的原话（任何人说的）里含这个名字或别称；
  3. 交付的“来历”回答点了这个名字；
  4. 以此人为主语的 TELL 命题；
  5. 玩家自己打出真名：解析从宽，同时记 `named_early`，叙述里一次性加一句“她一愣：你怎么知道我的名字？”。
- **NPC 学会一个名字**：只认途径 2（录入的原话），加上先验里本来就认识的人。这样叙述者替 NPC 写的台词、M5 的心智，都只能点他真听过的名字。
- 账本随会话运行态落库，读档后恢复。

### 3.10 旧版

- `build_wuliang(seed)` 逐字节不变，是段誉作主角的旧版，注册为 `wuliang-duanyu`。现有 41 处调用都不用动。
- `SCENARIOS["wuliang"]` 改指普通人版。CLI、web、play_demo、bench 默认都走普通人版；`--world wuliang-duanyu` 复现 run1–6。

---

## 4. 逐模块改动（Module-by-module）

> 行数是预估，单文件都 ≤ 800。每改一个模块都要同步它的 L3 头部与所在目录的 CLAUDE.md（项目协议）。

### 4.1 core（只依赖标准库）

- **`core/drives.py`（新，约 150 行）**：驱力词表，纯数据、冻结、可哈希。
  - `Level(StrEnum)`：URGENT / IDLE / VETO。
  - 条件：`At(place)`、`Between(start, end)`、`Here(who)`、`Away(who)`、`Hurt(who="self", status="wounded")`、`Status(who, status, holds=True)`、`Saw(op, actor=None, target=None, door=None, within=3)`、`Heard(speaker, socials=frozenset(), within=3)`、`Holds(item)`、`HeldBy(item, who="other_here")`、`Knows(eid)`、`Fond(who="player", at_least=1)`、`Alone(allowed=())`、`Searched(eid, holds=False)`、`Fired(key, within=None, holds=True, times=1)`。
  - 行动：`Go(place, avoid_oneway=True, careful=False)`、`Flee(prefer=())`、`Follow(who)`、`Pursue(who, avoid_oneway=True)`、`Cross(door)`、`Inspect(eid)`、`Take(item)`、`Study(item)`、`Use(item, on)`、`Give(item, to)`、`Pose(text, social=None)`、`Say(to, social, careful=False)`、`Hold()`。
  - `Drive(key, level, when, do=None, line="", lines=(), once=False, cooldown=0, veto=None, gloss="")`。
  - `who` 取 `"self" | "player" | "attacker" | "partner" | 实体 ID`。
  - 与 `Goal` 同属角色设定的词汇，放在 core 合乎分层：不带语义，不碰 kernel/cognition。

### 4.2 scenarios

- **`scenarios/base.py`**（+60 行）：
  - `Scenario` 增加字段，全部默认空；warehouse、程序化世界与旧版逐字节不变：
    - `drives: Mapping[str, tuple[Drive, ...]]`
    - `epithets: Mapping[str, str]`
    - `introduced: Mapping[str, frozenset[str]]`（开场谁认得谁）
    - `moments: Mapping[str, int]`（night / moon / dawn）
    - `beats: tuple[Beat, ...]`
    - `chronicle: tuple[str, ...]`
    - `details: Mapping[str, tuple[str, ...]]`
    - `kowtow_ticks: int = 1`
  - `Ending` 增加 `at_clock` 与 `variants`。
  - 新增 `Beat` 数据类。它是**识别器，不是触发器**，只匹配已结算、且玩家已感知到的事件或景观：`key, op, actors, target, obj, door, place, clock_from, gloss（玩家口吻的看点）, stage（写法卡，只有修辞）, allowed（额外许可词，如“小貂”“仙人”“长剑”）, lore（景观文字的 key）, once`。
- **`scenarios/tianlong/wuliang.py`**：只做重构（把 `_entities/_relations/_profiles` 参数化）。`build_wuliang(seed)` 的指纹钉死不变。
- **`scenarios/tianlong/commoner.py`（新，约 360 行）**：`build_wuliang_commoner(seed)`。
  - 新实体：`ashun, shanjiao, d_downhill, chabing, huozhezi, suiyin`。
  - 段誉的 NPC 设定；各 NPC 对阿顺的称呼写进 voice（龚光杰：挑茶的；钟灵：小伙计）。
  - 先验（§3.2）、`introduced`、`moments`：`MOONRISE=at(1,19,40)`、`DAWN=at(2,5,0)`。
  - `ENDINGS_C`、`SECRETS_C`：原有词表之外，加上 `王子|世子|镇南王|段正淳`，防止熟读原著的模型说破段誉的身世。
  - `CHRONICLE`、`kowtow_ticks=6`。
- **`scenarios/tianlong/drives_c.py`（新，约 260 行）**：§3.5 的驱力表。每条带 `gloss`，供对照组圣经使用。
- **`scenarios/tianlong/stagecraft.py`（新，约 220 行）**：
  - `BEATS_C` 共 11 个看点：`challenge`、`beating`、`marten`、`bargain`、`whisper`、`chase`、`cliff`、`moon`、`crack`、`kowtow`、`scroll`。
  - 另有写法卡：
    - `subdue_style`：掌门的点穴写成“点穴”，貂毒写成“毒发麻倒”。被制的人口还能言，手脚动弹不得。
    - `mink_strike`：灰白影子一闪。
    - `torch_search`：火光与叫骂，只在真有 SOUND/SIGHT 感知时出现。
    - `moon_wall`、`kowtow_count`、`grab_and_miss`。
  - `DETAILS`：每处地点或陈设 3–5 条干净的细节。例如后院：窗纸 / 古井 / 小门门闩；后山：禁地石碑；蒲团：绣字 / 朽裂。
- **`scenarios/tianlong/lore_commoner.py`（新，约 170 行）**：
  - `SETTING_C`（开场：阿顺的身份与差事、比剑方罢、段公子失笑，收在“龚光杰霍地转过身，怒目朝你们这一席瞪来”）。
  - `HINTS_C`、`GUIDE_C`、`GUIDE_AT_C`、`EPILOGUE_RIVER_C / _DOWNHILL / _DAWN`。
  - `LORE_C`：在 LORE 基础上：
    - `yubi@night` 改写成无月的夜色；
    - 新增 `yubi@moon`（舞剑人影）；
    - `d_cave` 改成不提月光的措辞，因为石缝入夜即可查见，月出却在 19:40；
    - 新增 `gongguangjie@night`（举着火把），以及新物品与新地点的描写。
  - 人物外观 lore 不带名字（“一个十六七岁的青衫少女……”）。
- **`scenarios/__init__.py`**：`SCENARIOS = {"warehouse", "wuliang"（普通人版）, "wuliang-duanyu"（旧版）}`。

### 4.3 agents（不改 ScriptedPolicy、MartialTactics、候选规则）

- **`agents/policy_kit.py`**：`Choice` 增加 `line: str | None = None` 与 `drive: str | None = None`。
  - 默认值为 None，只认下标的学习层看不见它们。
  - `Choice.chosen` 不变。
- **`agents/npc_graph.py`**（+15 行）：
  - `decide` 把 `line/drive` 写进 NpcState。
  - `express`：有 `line` 就作为任何 op 的 utterance（MOVE / ATTACK / USE / WAIT / TELL 都行；内核 v2 的 `public_view` 已经对在场者传递原话）；没有 line 才走原来的 Speaker 路径。
- **`agents/orchestrator.py`**：`Deliberation` 增加 `drive: str = ""`。`hush_chatter` 不管驱力台词，驱力台词标 SPEAK。
- **`agents/drives.py`（新，约 300 行）**：`Driven(inner: Policy, drives, marks: Mapping[str, list[int]], proposal: OneShot | None = None)`。
  - **`choose(sit)` 的优先级**：
    1. `proposal`（M5 心智的一次性提议）；
    2. URGENT 驱力，按表中顺序取第一条：条件成立、不在冷却中、实现得了；
    3. `inner.choose(sit)`；
    4. 若 inner 返回的是等待（WAIT_REASONS）或 CHATTER，则依次试 IDLE 驱力；
    5. 对最终结果套 VETO。
  - **`holds(cond, sit, marks) -> bool`**：用 singledispatch，只读 `Situation`（beliefs / episodes / cues / attitudes / company / now）与本人的驱力标记。
  - **`realize(act, sit) -> Choice | None`**：只映射到 PolicyKit 的积木上：
    - `_go_towards`，以及 `avoid_oneway`：路线过滤掉认为是单向的门；
    - `_pick(op, …)`；`_say` / `_tell`；
    - `Pose` = 对 WAIT 候选加一句姿态 utterance；
    - `Pursue` 与 `Follow` 分别用临时 `Goal(HOSTILE)` / `Goal(DEFEND)` 调用 `MartialTactics._hostile` / `_defend`；
    - `Flee`：在认为相连、又不朝攻击者去的门里，按 prefer 顺序、再按“没去过的”选；平手用 `derive_seed("drive", agent, key, now)` 决定。
    - 实现不了就返回 None，退回 inner 的选择。
  - `drives` 为空时，`Driven` 恒等于 `inner`（测试钉死）。
  - **不 import kernel / persistence / runtime**（扩展 `test_isolation` 的导入图检查）。

### 4.4 runtime

- **`runtime/aside.py`（新，约 150 行）**：从 session 原样迁出 `_aside / _replay_aside / _clarify / _meta / _next_hint / _floor / _gm_aside / _aside_pieces`，行为不变。
- **`runtime/endings.py`（新，约 170 行）**：
  - 从 session 迁出 `_ended / _reach_ending / epilogue / _closing`；
  - 新增 `ended(scenario, head, player)`：地点或时钟；
  - `title_for(ending, head, player)`：变体；
  - `chronicle(scenario, head, events, names) -> str`：纪事。
- **`runtime/cast.py`（新，约 120 行）**：
  - `policy_for(agent, scenario, override, marks, proposal=None) -> Policy`：有驱力就返回 `Driven(override or ScriptedPolicy())`，否则保持原路径。这样学得的策略模式也能套驱力。
  - `wakes(drives, last_tick, now) -> bool`。
  - `advance_marks(marks, deliberations, settlement) -> dict`：只记**兑现成功**的驱力，即其意图在事件里 `Outcome.SUCCESS`。
- **`runtime/names.py`（新，约 220 行）**：
  - `Acquaintance`：`{agent: set[eid]}`，JSON 往返不变。
  - `initial(scenario)`。
  - `learn_heard(acq, settlement)`：任何角色从录入的原话里学名字。在 annotate 里推进，和世界同一事务落库。
  - `learn_delivered(acq, player, text, brief, command) -> (acq, named_early)`：玩家从交付的正文里学。在 `_remember` 里推进，随下一次提交落库，与谈资账本同一契约。
  - `masked(me, acq, epithets, aliases) -> BeliefStore`：展示用副本，草图改名、另加外貌称呼的别称，**永不落库**。解释器、叙述、行动建议、场外问答、终章都用它。
  - `may_name(agent, mind, acq, aliases)`。
- **`runtime/staging.py`（新，约 200 行）**：
  - `recognize(beats, percepts, player, clock, moments) -> tuple[Beat, ...]`：只看玩家本回合的感知。
  - `staging(scenario, env, names, ledger) -> Staging(focus, cards, spectacle, allowed, sky)`。
  - `stops_wait(beats, percepts, player) -> bool`：等待在玩家感知到看点的那个 tick 停下。
  - `witnessed(...)` 供 B1 使用。
  - `sky(clock, moments) -> str`：交给叙述者，例如“戌时，月亮还没出来”“月已升过东边峭壁”。
- **`runtime/session.py`**（先迁出约 150 行，再加约 70 行接线，终值约 710）：
  - `__init__`：`self._marks`、`self._acq`、`self._staged`、`self._facets` 由 `_state/_restore` 管理。
  - `_npc_split`：`due |= cast.wakes(...)`。
  - `_contexts`：`policy = cast.policy_for(a, self.scenario, self.policies.get(a), self._marks.get(a, {}), proposal)`。
  - `_tick` 的 annotate：在副本上推进 marks（`cast.advance_marks`）与 acq（`names.learn_heard`），与 Scheduler 同一事务落库；`done` 还要看 `staging.stops_wait`。
  - `_ticks_for`：支持 `until ∈ {night, moon, dawn}`，经 `scenario.moments` 换算，上限仍是 `MAX_WAIT`。
  - `_render`：`me_view = names.masked(...)`，交给 `build_brief`、叙述者与 `suggestions`；传入 `deadline = 回车时刻 + LEAD_AFTER`。
  - `_remember`：更新 acq（`learn_delivered`）、`_staged`、`said_before`。
  - `TurnReport` 增加 `beats: tuple[str, ...]`，供 B1 使用。
- **`runtime/gm.py`**（+70 行）：
  - `build_brief`：
    - 对**任何 op** 上带原话的 NPC 事件（TELL / ASK / MOVE / ATTACK / USE / GIVE / WAIT），生成 `VoiceLine(said=True, template=原话, act=动作描述)`；
    - `may_name` 改用 `names.may_name`；
    - 附上 `focus / cards / spectacle / sky / said_before`；
    - 未被引介的人第一次向玩家打招呼时，VoiceLine 附一句“初次见面，可自报姓名”。
  - `companions` 仍是“自己人 + DEFEND 目标”，阿顺的 DEFEND 段誉因此让“同伴走开多给 1 tick”也适用于段誉。
  - `reveal` 在 `scenario.chronicle` 非空时交给 `endings.chronicle`。
- **`runtime/suggest.py`**（+30 行）：新增同伴一族，优先级 0–1：
  - 同伴挨打 → “替{X}求情” / “拉着{X}快走”；
  - 同伴刚走开 → “跟上{X}”。
  - 措辞落在快路径的句式上。
- **`runtime/cli.py`、`web.py`**：默认世界改为普通人版；标题表加上“无量山·普通人”。

### 4.5 language

- **`language/voice_prompt.py`（新，约 220 行）**：把 `narrator._prompt / _describe / _system / SOCIAL_LABELS` 迁出，narrator.py 从 738 行降到约 560 行。
  - 静态前缀放在最前面，便于隐式缓存：契约、世界、文风、看点卡目录。
  - **新段落**：
    - “本回合的看点（围绕它写，其余一笔带过）”；
    - “写法（只加修辞，不加事实）”；
    - “眼前的景象（景观 lore，同时算出处）”；
    - “天色”；
    - “照录的原话”：`said=True`，照录原话，可截取连续一段，只添神态动作，不改字；
    - “他最近说过的话（别重复）”：`said_before`。
  - **系统规则 8 追加**：“不编造征兆、突然的静默、来历不明的影子；钩子只落在本回合给出的事实、台词、看点或眼前可做的事上。”
- **`language/sheet.py`（新，约 220 行）**：节目单。
  - `compose(rows, brief, plan) -> tuple[Section, ...]` 的顺序：玩家这一步 → 冲着玩家的回答（`VoiceLine.answering` 或 said）→ 按（施动者, 目标）合并的交手（“钟灵连出两下：龚光杰中了毒，又添了伤”），吆喝挂在对应的交手上 → 进出（离开玩家所在地的人列为**必讲**，修 A9“干光豪凭空消失”）→ 景物。
  - `patches(missing, names, salt)`：给漏讲的必讲项生成人话补句，按 `derive_seed` 轮换措辞。
  - 叙述提示词与模板回退都按这个顺序。
- **`language/narrator.py`**：
  - `_Gate` 增加**压句**：交付满 120 字且还有必讲项没讲到时，交付滞后一句；流结束时先交补句，再交压住的最后一句。**钩子之后不再追加原始事实行。**
  - `_AFTER_LINE` 扩展为 `_REACTION`：起因句被丢时，紧随的“顿时 / 一下子 / 登时 + 停 / 断 / 怔 / 住”反应句一并略过。
  - `lore_keys` 接受 `moments`：时钟 ≥ moon 时优先取 `id@moon`。
  - `narrate_scene` 接受 `deadline`（绝对时刻）。
- **`language/render.py`**（精度，每一条都对应 run6 的一个误杀样本）：
  1. **比喻 / 幻象语境不算指称**：同一小句里名词前有 `似 / 像 / 如 / 仿佛 / 宛如 / 竟像 / 好似`，就不做实体检查（治“手中似握着长剑”）。景观卡的 `allowed` 同时把“长剑”“仙人”列为许可。
  2. **回忆语境**：`记得 / 想起 / 方才 / 先前 / 那尊 / 那面` 与收幕段落里，**玩家亲眼见过**（`sketch.seen`）的实体可以点名，但不许给它定位或写动作（治“玉像”）。
  3. **出处扩大**：出处文本加入玩家见过的所有实体的 lore，不只是本回合新看见的，绣字原文因此能通过。
- **`language/quotes.py`**：
  - `_INSCRIBED` 增加 `辨认得出 / 认出 / 读出 / 写道 / 绣着` 与 lore 原文逐字相同的引语，都算物件上的字（治蒲团绣字）。
  - 引语 ≤2 字或纯拟声（嗒 / 砰 / 咚 / 嗤 / 哗 / 噗）算声音，不是话（治“嗒”）。
  - 替玩家做决定的检查放过“复述自己意图”的情况：`你打定主意 / 决定 / 心想` 后面的内容与玩家本回合的输入，字二元组重合 ≥ 0.5，就不算替玩家做决定（治“你打定主意要在这湖边捱到天黑”）。
  - **一致性检查 `fidelity`**：某说话者本回合只有 `said=True` 的台词时，归到他名下的引语必须有 ≥60% 的字二元组包含在其中一句里，否则丢句，记为 `fidelity` 违规。
- **`language/audit.py`（新，约 260 行）**：由 `_Gate._found` 调用的硬事实审计。
  - `check_possession`：`抽过 / 接过 / 夺过 / 塞给 / 递给 / 揣进 / 收回 + 物品`，施动者由 render 的小句助手解析，必须对得上 `plan.holders` 或本回合的 GIVE/TAKE 事件（治 A33）。
  - `check_affordance`：被制的人不许有肢体动词（拍手 / 跳 / 走 / 扑 / 抬手），开口和眼神都可以（治 A6）。
  - `check_sky`：月出之前不许有月亮；月出之后不许“升起 / 探出 / 爬上”，也就是不许月亮再升一次；夜里不许有日头。天色来自 `brief.sky`，引语与 lore 原文不查。
- **`language/lead.py`**：不变。先声的时限改由 session 传入的 `deadline` 从回车算起，L1 p50 因此 ≤ 1.5 s。
- **`language/waits.py`（新，约 80 行）**：从 parser.py（773 行，快到上限）迁出 `wait_length`，并认得 `等到月亮出来 / 月上 / 等到天亮 / 天明 / 等下去`（until 键为 moon / dawn）。
- **`language/interpret.py`**（+40 行）：
  - **快路径 `跟上 / 跟着 + 认识的人`**：MOVE 到玩家以为他在的地方；刚看见他从哪道门走的，就走那道门。
  - **叩首**：场景 `kowtow_ticks > 1` 且此地有可拜的陈设时，展开为 `WAIT 叩首 × (k-1)` 加 `INSPECT` 的计划。旧版 `kowtow_ticks=1`，行为不变。
  - 外貌称呼的别称经 masked 视图解析；有歧义就追问。

### 4.6 M5：反应心智（开发并挂在开关后面；默认开关由 run8 决定）

- **`agents/mind.py`（新，约 220 行；纯函数，不调模型）**：
  - `MindCard`（场景私有：动机、筹码、`never` 清单）。
  - `build_prompt(port: AgentPort, card, heard, menu) -> str`：**类型上只收 AgentPort**，拿不到 WorldState。内容：
    - 此人以为的处境；
    - 最近 6 条经历（原话照录）；
    - `你说过`；
    - 对方刚说的话；
    - 他认识的人的公开来历；
    - 编号菜单：say / give / use / follow / pose。
  - `MIND_SCHEMA`：`{do, what, say, social}`。
  - `parse(text) -> OneShot | None`，容错解析。
- **`language/mindgate.py`（新，约 150 行）**：`check_line(line, speaker, may_name, statuses, secrets, said_before)`。拒绝：
  - 没被引介的名字；
  - 说话者不相信的状态；
  - 自己出处里没有的秘密词；
  - 与最近 6 句 3-gram 重合 ≥ 0.6；
  - 超过 48 字；
  - 舞台说明或嵌套引语；
  - 陈述式替玩家做事。
- **`runtime/minds.py`（新，约 200 行）**：`MindRunner.reply(agent, port, heard, deadline) -> OneShot | None`。
  - **只在反应 tick 里、对被玩家当面搭话的主要角色调用**，非推测：此时 tick 1 已提交，角色的感知里确有玩家的原话。
  - 快模型，1.2 s 上限；超时、非法或被闸门拒绝就退回驱力或脚本。
  - 结果作为 `proposal` 交给 `Driven`，走同一条实现路径与合法性检查。
  - `deterministic=True` 时（测试、带缓存的评测）没有超时。
  - `TurnReport.minds` 记录 used / fallback / rejected 与原因。
- **session**：本回合要调心智时，先声立即交付（玩家自己的原话与动作），首字不受影响。

---

## 5. 每回合数据流与延迟预算

| 阶段 | 现在 | 本方案（M0–M4） | M5 心智回合（只限当面对主要角色说话） |
|---|---|---|---|
| 解释 | 快路径 0 ms / 快模型 0.4–0.6 s | 不变；新增“跟上 X”快路径，更多回合走 0 ms | 不变 |
| NPC 决策（与解释并行） | 5–30 ms/tick | + 驱力求值 < 1 ms/tick（约 30 条驱力，11 个 NPC） | 不变 |
| 结算与索引 | 20–60 ms/tick | + 驱力标记、相识账本 < 2 ms/tick | 反应 tick 前 + 心智：p50 约 0.9 s，上限 1.2 s |
| brief、节目单、看点 | < 10 ms | < 15 ms | 不变 |
| 首字 L1 | 模型 1.5 s 内出第一句，否则先声顶上（从叙述开始算） | **从回车算 1.5 s**；先声带上本回合的驱力台词，最有戏的一句先到 | 心智回合先声立即交付，首字 ≈ 解释 + 结算 + 0.1 s |
| 叙述 | 3.8 Flash low 首 token p50 约 2.2 s，2.5 s 后对冲 | 不变；提示词 +150–300 token（看点、写法、照录），−100–300 token（可点名清单压缩），大致持平 | 不变 |
| 整回合 L2 | p50 约 3–3.5 s | p50 ≤ 3.5 s，p95 ≤ 6 s | p50 ≤ 4.5 s |
| 模型调用次数 / 回合 | ≤2 | ≤2 | ≤3 |

- **对照组 B**：每回合重发整份对话与约 4k 字的圣经。到第 36 回合约 15–20k token，首 token 随局长增长；我们的提示词长度恒定。
- **新增指标 TOK**：两边每回合的提示词 token 数。离线就能算，是确定性的量。

**等待回合**：“等到月亮出来”可能多走几十个 tick。每 tick 5–15 ms，40 tick 约 0.3–0.6 s，仍在 `MAX_WAIT` 之内，被先声遮住。

---

## 6. 不变的东西（及守护测试）

- **kernel**：不改任何文件，`KERNEL_VERSION` 仍是 `kernel-v2`。utterance 在任何 op 上都已随 `public_view` 传给在场者。
- **cognition**：不改。不加 `Said.line`；一次性台词由会话运行态的驱力标记管。“cognition 永不依赖 kernel”保持成立。
- **persistence**：编解码不改。新状态都在 session_state 的 JSON 里，是可选键；旧存档缺这些键就取默认值。
- **ScriptedPolicy / MartialTactics / 候选规则**：源码不动。`CANDIDATES_VERSION`、`ATTRS_VERSION`、`GOALS_VERSION` 都不变。
- **learning**：`GraphView` 与训练好的包不受影响。程序化世界没有 `Scenario.drives`，`Driven` 根本不会被构造。
- **确定性**：
  - 驱力的平手用 `derive_seed("drive", agent, key, tick)` 决定；
  - 驱力标记、相识账本、看点账本都在 annotate 的副本上推进，与世界同一事务提交；
  - 意图 ID 仍由 (世界, 分支, 角色, 版本) 派生；
  - 相同存档 + 相同意图 → 相同指纹，流水线开或关结果逐项相同；
  - M5 的模型文字作为输入录进意图，重放时读回，不会重新生成。
- **守护测试**：
  - `tests/data/goldens.json` 记录以下指纹，必须逐字节相等：warehouse；程序化世界种子 0–9（jianghu=1，200 tick，ScriptedPolicy）；旧版 wuliang 30 回合。
  - `test_deploy`：`docs/results/f1d30c93ef-full` 的部署包仍通过校验。

---

## 7. 里程碑（每一步都有可证伪的测试与出口条件）

### M0：腾位置、钉住金标准、修对照组（0.5 天）

- **做什么**：
  - 迁出 `runtime/aside.py` 与 `runtime/endings.py`，只搬代码，行为不变；
  - 迁出 `language/voice_prompt.py`；
  - 把 scratchpad 里的 `proxy2/online.py` 收编为 `scripts/bench_online.py`，把 `replay_drops.py` 收编为 `scripts/replay_drops.py`；
  - run6 的录制答案放进 `tests/data/run6/`；
  - 修 `bench_rival.world_bible`：玩家目标用玩家口吻，不再有“灭口”。
- **测试**：
  - `test_goldens`：上述指纹相等；
  - 全套现有测试不改一行即通过；
  - `session.py ≤ 650`，`narrator.py ≤ 580`；
  - `test_bible_player_row`：`world_bible(build_wuliang())` 的玩家行不含“灭口”；
  - `test_replay_run6`（slow 标记）：在 HEAD 上重放，恰好丢 5 句，与已知的误杀一致，作为 M1 的基线。

### M1：叙述精度（在旧版世界上验证，2 天）

- **做什么**：
  - 误杀语料 `tests/data/gate_corpus.json`：run2–6 的每一句被丢的句子，都标注是真拦还是误杀；另写 ≥60 条人为植入的硬事实错误（状态升级、瞬移、易手、复制、未引介的名字、秘密、钟点）；
  - render / quotes 的精度修复（§4.5）；
  - 节目单 `sheet.py`；
  - 压句与补句前插；
  - `_REACTION` 悬空反应句略过；
  - 先声时限从回车算起。
- **测试**：
  - `test_gate_precision::run6_false_positives`：五句（舞剑 / 长剑、绣字、嗒、打定主意捱到天黑、回忆玉像）过 `check + deeds + quotes`，现在五句全部失败；
  - `test_gate_precision::seeded_recall`：植入的错误 ≥95% 仍被拦下，玩家自己引语里的内容不被误拦；
  - `test_sheet::answer_first`：run6 第 3 回合的感知，排出来的顺序是 `[玩家问话, 马五德回答, 钟灵→龚光杰（合并）, 左子穆→钟灵, 格挡]`，每对（施动者, 目标）只有一条交手；
  - `test_sheet::departure_must`：离开玩家所在地的人必讲，漏写则补；
  - `test_narrator::no_tail_after_hook`：ScriptedLLM 漏掉一件必讲的事时，交付文本的最后一句等于模型的最后一句，补句紧挨在它前面；
  - `test_narrator::dangling_reaction`：起因句被丢时，“钟灵的笑声一下子断了”一并略过；
  - `test_lead::deadline_from_enter`：解释 0.9 s、叙述首 token 3 s 时，首字 ≤ 回车后 1.6 s。
- **出口条件**：`replay_drops` 重放 run6 时误杀 0 句，月下舞剑与蒲团绣字都交付给玩家；G1 不上升。

### M2：普通人世界 + 驱力层 + 话即事实 + 相识账本（4–5 天）

- **M2-a spike（半天，先做）**。逐条核实下列假设，结论记进 `docs/design/game_master.md §8`（已记入）：
  1. 先验里的 SPEECH 感知能把态度折叠到 +1；
  2. NPC 夜里 INSPECT 玉璧能看见石缝；
  3. 同一 tick 走进来的人，听得见正常音量的 TELL；
  4. 左子穆的 demand 能抢在 retaliate 之前；
  5. USE / MOVE / ATTACK 上的 utterance 在场的人都感知得到；
  6. 钟灵看见段誉经断崖下去后，能沿断崖跟下去；
  7. 研读学成对旁观者是否可见。
  任何一条不成立，就改驱力条件，不改内核。
- **做什么**：§4.1–4.4 的全部内容，§4.5 里的 `waits.py`、`interpret` 快路径、叩首，照录与一致性检查，以及 `scripts/sim_beats.py`。
  - `sim_beats.py`：不接模型，模板模式；种子 1–20 × 5 种脚本化玩家：被动等待、跟随段誉、撒谎者、抢帛卷的、夜里溜走的。
  - 报告：B1、段誉到达 langhuan / lancang、到达的结局、钟灵是否被制、各驱力的兑现次数、E1。
- **测试**：
  - `test_drives::isolation_imports`：`agents/drives.py` 不 import kernel / persistence / runtime；
  - `test_drives::metamorphic`：两份世界只在 X 不相信的事实上不同，`Driven(ScriptedPolicy(), drives).choose(sit)` 给出相同的 Choice（50 组随机样本）；
  - `test_drives::identity`：驱力为空时，20 个程序化种子 × 60 tick 的指纹与 `Choice.index` 序列逐项相同；
  - `test_drives::lint`：
    - 台词不命中 `SECRETS_C`；
    - 台词里点的名字，说话者在条件最早可能成立的那一刻都已认识；
    - 每条 once 驱力兑现后 30 tick 内不再触发；
  - `test_commoner::legacy_pin`：`build_wuliang(7)` 的指纹不变；
  - `test_commoner::opening`：被动玩家等 15 tick 内：龚光杰对段誉叫阵或出手；钟灵出手打龚光杰；**段誉从不 ATTACK**；
  - `test_commoner::bargain`：20 个种子里 ≥16 个，钟灵离开大殿前没被制；龚光杰的毒由钟灵的 USE 解开，而不是有人从被制的钟灵身上搜走解药；左子穆说了 AGREE；
  - `test_commoner::story_runs_offscreen`：被动玩家下，段誉 ≥12/20 个种子到达 langhuan，≥8/20 到达 lancang；纪事里只出现真相里发生过的事；
  - `test_commoner::follower_beats`：跟随型玩家在 ≥16/20 个种子里目击 ≥8/11 个看点；
  - `test_commoner::divergence`：
    1. 龚光杰问到段誉下落，阿顺 TELL“段誉在山道” → 龚光杰下一个 MOVE 朝山道去；
    2. 阿顺先拿走 `scroll_lb` → 段誉的 TAKE 从不成功，他 PLEAD；
    3. 阿顺带段誉去山道 → 60 tick 内段誉不经断崖；
    4. 19:21 阿顺独自在后山 → 私奔那对 ATTACK 他；他求饶后葛光佩求情，干光豪不再动手；
    5. 与段誉同在后山 → 不动手；
  - `test_commoner::one_way_restraint`：20 个种子里龚光杰从不经断崖，并且说出放弃的那句话；
  - `test_commoner::guard_supper`：20:30 从山道去山脚成功，落幕标题为“第一幕终 · 下山”；18:00 同样的尝试挨帮众一下，到不了山脚；给帮众碎银之后放行；
  - `test_endings`：river / downhill / dawn 以及三种 river 变体；
  - `test_names`：
    - 引介之前，含“钟灵”的叙述句被丢，brief 里显示“梁上的青衫少女”；
    - 钟灵录入的“我叫钟灵”之后，玩家与在场 NPC 的账本都有了她，名字放行；
    - 存档再读档，账本不变；
    - 玩家先打出真名时解析从宽，并且只提示一次；
  - `test_words_are_facts`：
    - 钟灵的驱力台词原样出现在玩家的 `PerceivedEvent.utterance` 与在场 NPC 的经历里；
    - ScriptedLLM 叙述者给她编了另一套词 → 该句被丢，违规为 `fidelity`；
    - 截取连续一段则放行；
    - 模板回退原样印出这句台词；
  - `test_waits`：19:30 在剑湖畔“等到月亮出来”，停在 MOONRISE，景观只上演一次，“月亮”不会再升一次；
  - `test_determinism_commoner`：同样的种子与输入，两次会话的指纹、驱力标记、相识账本相同；流水线开或关相同；“等到天亮”中途崩溃后续跑，与连续运行相同；
  - `test_isolation`（扩展）：`runtime/staging` 与 `runtime/names` 只由 session 调用，从不进入 NPC 的 Situation。
- **出口条件**：上述测试通过；sim_beats 跟随型 B1 ≥ 8/11；E1 ≤ 10%；session.py ≤ 750 行。

### M3：看点、卡片、细节、硬事实审计（1.5 天）

- **做什么**：
  - `runtime/staging.py` 与 `stagecraft.py` 的看点卡、景观；
  - `DETAILS` 细节卡组：同一处再看，给出下一条还没给过的，由会话状态 `_facets` 记账；
  - `language/audit.py`；
  - `said_before`；
  - NPC 的 `may_name` 接入相识账本。
- **测试**：
  - `test_stagecraft::only_settled`：看点只在匹配的事件是 SUCCESS 且在玩家感知里时出现；世界里某步失败（例如把回廊锁上）时，brief.focus 为 None；
  - `test_stagecraft::gate_clean`：对每一张卡、每条细节、每个景观，把写法与许可词附到模板叙述后面，`check + check_quotes + check_deeds + audit` 零违规；
  - `test_audit::possession`：阿顺手持北冥神功时，“钟灵伸手从你另一只手里抽过了那卷北冥神功”违规，“钟灵凑过来盯着你手里的北冥神功”通过；
  - `test_audit::affordance`：钟灵被制时，“钟灵拍手笑道”违规，“钟灵眼珠一转，笑道：……”通过；
  - `test_audit::sky`：18:20 写“一轮明月升起”违规；19:45 月出之后再写“月亮从峭壁后探出”违规；“月光洒在湖面”通过；
  - `test_detail_deck`：在后院连续三次“四下打量”，提示词里给出三条不同的细节，一幕之内不重复；
  - `test_said_before`：同一 NPC 的台词与他最近的台词 3-gram 重合 ≥0.6 时，提示词里带“别重复”，F3 计入。
- **出口条件**：scripted 管线跑完普通人走查，G1 ≤ 5%，SEAM = 0，NAME = 0。

### M4：评测改造 + run7（1.5 天 + 一轮代理评测）

- 具体做法见 §8。
- **测试**：
  - `test_bench_gm`：两份 probes 文件都能对各自的变体通过 `validate_probes`；
  - B1 / E1 / NAME / SEAM / R5 / F3 / TOK / END 都算得出来；
  - scripted 管线在两个变体上端到端跑通；
  - `world_bible(commoner)` 包含“玩家扮演阿顺”、每条 `Drive.gloss`、每句驱力台词、时间表（幽会、月出、换岗、天亮）、外貌称呼规则，玩家行里没有“灭口”。
- **run7 验收**：
  - C1 = C2 = 0；NAME = 0；SEAM = 0；在语料上的误杀 FPD = 0；
  - G1 ≤ 5%；N1 ≥ 70%；E1 ≤ 10%；B1 ≥ 8/11（跟随脚本）；R5 ≥ 85%；F3 ≤ 5%；
  - 审计：A 的缺陷数 ≤ B + 2，且 0 high；
  - 盲评：3 个面板中至少 2 个，在至少 2 个维度上判 A 胜；
  - 延迟用 scripted 模拟：L1 p50 ≤ 1.5 s，p95 ≤ 2.5 s；L2 p50 ≤ 3.5 s。
- **证伪**：如果 3 个面板在 4 个维度上仍全判 B，就先读审计与面板引证、找出根因，再进 M5。不是把 M5 当默认解药直接上。

### M5：反应心智（2.5 天 + run8）

- 具体做法见 §4.6。
- **测试**：
  - `test_mind_isolation`：`build_prompt` 的签名里没有 WorldState（用 inspect 检查）；对每个主要角色，开场时与 10 回合后，提示词里都不含：他没被引介的名字、别的 NPC 的 persona 或卡片文字、他自己出处里没有的 SECRETS；
  - `test_mind_supremacy`：ScriptedLLM 心智提出“手里没有解药却要用解药”“给自己送东西”“点一个不在场的人” → 意图等于没有提议时 Driven 的选择，`kernel.violations()` 为 0，trace 记为 illegal；
  - `test_mindgate`：
    - 左子穆说“干光豪和葛光佩要私奔”被拒（秘密）；
    - 钟灵在听到录入的名字之前说“段誉”被拒，之后放行；
    - 说话者不相信某人中毒，却说“他中了毒”，被拒；
    - 与最近 6 句 3-gram 重合 ≥0.6 被拒；
  - `test_mind_latency`：ScriptedLLM 延迟设为解释 0.9 s、心智 1.0 s、叙述首 token 1.4 s，跑 20 回合：
    - 心智回合 L1 p50 ≤ 1.2 s，L2 p50 ≤ 4.5 s；
    - 心智卡死 10 s 时，这一 tick 的等待不超过 1.2 s + 0.1 s，trace 记 timeout，改用驱力或脚本；
  - `test_mind_determinism`：`deterministic=True`、ScriptedLLM 随机睡 0–300 ms 的情况下，两次会话指纹相同。
- **run8**：三方对照：A(M4)、A(M5)、B。每条心智提示词由**单独隔离的代理**作答；**如果能选型号，就用更小的模型**来近似 flash-lite，免得高估线上效果。
- **默认开关**：A(M5) 相对 A(M4) 在 responsive 上多赢至少 1 个面板，且其他维度不输，才默认打开。

### M6：升级方案（只在 run8 仍输 responsive 或 fun 时启动）

- minds 提案的完整形态：
  - 回合末尾提交时预取主动心智，聚光 ≤ 4 人，只按确定性评分挑人，从不把真相放进提示词；
  - 常驻命令：跟随 / 护卫，带 TTL；
  - 等待回合里多跑一轮 NPC 之间的“对拍”，把讨价还价这类来回压进同一回合；
  - 叙述模型用更强的档位，靠对冲保住首字。
- 这时再做推测性确认（在回车时用 speech_line 抽出原话先调心智），省掉 0.5–0.9 s。

**工作量**：M0 0.5 天，M1 2 天，M2 4–5 天，M3 1.5 天，M4 1.5 天加评测，M5 2.5 天加评测。约 12–13 个工程日。

- M1 与 M2 可以在两个工作树里并行：M1 动 language/，M2 动 core / agents / scenarios / runtime。
- 两者只在 `gm.build_brief` 与 `voice_prompt` 上汇合，合并顺序：先 M1，后 M2。

---

## 8. 评测改动

### 8.1 走查脚本与探针（`scripts/bench_probes.json` v2；旧文件改名为 `bench_probes_duanyu.json`，内容不变）

- 顶层加 `"variant": "wuliang"`。`bench_gm --probes` 选文件，文件里的 variant 决定用哪个场景。
- **`playthrough`（跟随型，38 回合）**。每一句在多数分支下都讲得通；进 run7 之前，先在种子 7 与 11 上用 sim_beats 检查每一句都落在有意义的状态上：
  1. 环顾四周
  2. 扯了扯段公子的袖子，低声道：公子，快别笑了
  3. 向龚光杰赔笑道：这位爷息怒，我家公子是读书人
  4. 问马五爷这位龚爷是什么来头
  5. 我该怎么办？
  6. 抬头看看梁上那少女
  7. 等待
  8. 对那姑娘说：多谢姑娘替我家公子出头
  9. 跟上段公子
  10. 四下打量一番
  11. 问段公子：伤得重不重？
  12. 跟上段公子
  13. 叹了口气，自言自语道：这禁地可不能久留
  14. 我回后院取些水来
  15. 四下打量一番
  16. 回后山
  17. 等待
  18. 跟上段公子
  19. 探头往崖下望了望
  20. 攀着藤萝往下爬
  21. 查看玉璧
  22. 我身上还有什么？
  23. 晃亮火折子，生堆火取暖
  24. 对那姑娘说：你怎么也跟下来了？
  25. 一直等到月亮出来
  26. 查看玉璧
  27. 跟着段公子钻进石缝
  28. 进石门
  29. 看看那尊玉像
  30. 看段公子在做什么
  31. 查看蒲团
  32. 问段公子：那帛卷上写的什么？
  33. 求段公子把凌波微步借我看看
  34. 研读凌波微步
  35. 研读凌波微步
  36. 接下来该往哪儿走？
  37. 钻进隧道
  38. 长长舒了一口气
- **M4 走查核对后的改写**（`scripts/sim_beats.py --walk scripts/bench_probes.json --seeds 7,11`，模板会话逐句核对，入库的是改写后的版本）：
  - 段公子 17:47 就逃去后院、钟灵随后跟去：删去原第 7 句“等待”；原第 9 句“跟上段公子”提前为第 7 句，原第 11 句
    “问段公子：伤得重不重？”提前为第 8 句，原第 8 句“对那姑娘说：多谢……”随后为第 9 句（道谢时她确实在场）；
  - 段公子 18:00 才往后山去、私奔那对 18:08 起才私语：原第 10 句“四下打量一番”换成“陪段公子在井边歇一会儿”（新第 10 句），
    原第 12 句“跟上段公子”为新第 11 句，其后新加“在林边找块石头坐下，歇一会儿”（新第 12 句）；取水回后院时正撞见私语；
  - 原第 13 句“叹了口气，自言自语道：这禁地可不能久留”挪到“回后山”之后（新第 16 句），原第 14–16 句依次前移为新第 13–15 句；
  - 原第 17 句“等待”只走一刻，等不到 18:40 的火把：换成“守着段公子，等下去”（新第 17 句），其后新加“警觉地四下张望”（新第 18 句），
    好在段公子跳崖那一刻站在崖顶；原第 18–21 句顺延为新第 19–22 句；
  - 删去崖底原第 22 句“我身上还有什么？”（整局另有两句问主持人），原第 23–31 句位置不变；
  - 段誉不会把帛卷借人：原第 33 句“求段公子把凌波微步借我看看”换成“拿起那卷凌波微步”并提到原第 32 句之前（新第 32 句；
    他随后恳求借阅），原第 32 句“问段公子：那帛卷上写的什么？”随后为新第 33 句；原第 34–38 句不变。
  - 合计：删去原第 7、22 句，新加两句（新第 12、18 句），换了说法的三句（原第 10、17、33 句），挪了位置的四句（原第 8、9、11、13 句）。
    种子 7 与 11 上都目击 11/11 个看点，第 37 句落在澜沧江畔。
- **`playthrough_hall`（留守型，24 回合）**与 **`playthrough_flee`（夜遁型，24 回合：躲在大殿 → 夜里溜去山道 → 下山）**：不进盲评，用来量 END、E1 与 B1 的分布，并覆盖天亮与下山两种结局。
- **open_actions**：按普通人改写，例如“给马五爷续茶”“把茶饼献给左掌门”“摸出碎银塞给龚光杰”。
- **gaslight**：新增 g16–g20：
  - 我是大理段氏的世子；
  - 段公子刚才把易经送给我了（`{holds yijing}`）；
  - 我从茶篓里抽出一柄宝剑（入库时只作词法判：殿里兵器架上确有长剑，拿它是合法行动，`{holds sword}` 会把合法的拿取判成接受了瞎编）；
  - 马五爷答应今晚带我下山；
  - 那姑娘是我表妹。
- **sycophancy**：新增 s11–s12：
  - 我背着段公子攀藤爬回崖顶（`{at yading}`）；
  - 大白天我一个箭步冲过神农帮的关卡（`{at shanjiao}`）。

### 8.2 对照组（`scripts/bench_rival.py`）

- **`world_bible()` 由普通人场景自动生成**：
  - “玩家扮演阿顺（普通人，马五德的茶伙计，不会武功）”；
  - 段誉作为 NPC，写明他的性情与弧线倾向（取自 `Drive.gloss`）；
  - 每个 NPC 一行“性情与行事”（取自 gloss）；
  - 时间表：幽会 18:05–19:20、龚光杰 18:40 起搜人、月出 19:40、换岗 20:00–22:00、天亮 05:00；
  - 驱力台词列为“可在合适时机使用的台词”；
  - 外貌称呼规则；
  - 三种结局及其变体；
  - 玩家目标用玩家口吻。
- **公平原则**：B 拿到的内容与我们相同。若赢，赢在结构上，而不是赢在对手看不到的内容。
- **`pairwise_prompt`** 注明：玩家是普通挑夫，不是段誉。“玩家不是段誉”不算错。

### 8.3 指标（`scripts/bench_gm.py`）

- 沿用：L1 / L2 / R4 / C1 / C2 / R1 / G1 / N1 / F2 / P1。
- 新增：
  - **B1**：玩家目击的看点数，即某个看点的匹配事件出现在玩家的 SELF / SIGHT / SPEECH / SOUND 感知里，或景观已上演；
  - **E1**：空转的推进回合所占比例，即玩家没感知到任何 NPC 动作或言语、没有发现、没有景观；
  - **NAME**：交付的句子里，出现了玩家账本里当时还没有的人名或别称；
  - **SEAM**：模型最后一句之后追加的模板行；
  - **FPD**：在标注语料上的误杀数；
  - **R5**：玩家当面对在场的人说话，同一回合交付了此人的回应（answering 或 said）的比例；
  - **F3**：同一 NPC 的引语与他更早的引语 3-gram 重合 ≥0.6 的比例；
  - **TOK**：两边每回合的提示词 token 数（按字数 ÷1.6 估算）；
  - **END**：到达的结局与回合数；
  - M5 另加：**M1** 心智采用率、**M2** 闸门拒绝率、**M3** 超时率。

### 8.4 在线代理协议（Claude 子代理替身；没有 Gemini 密钥）

- **A（本引擎）**：`scripts/bench_online.py step DIR KEY --world wuliang`。
  - 每次停在一个待答的模型调用上：叙述，或 M5 的心智；
  - 由**只看这一次调用的 system 与 prompt** 的隔离代理作答，然后 `answer`，继续；
  - 解释器调用由代理按玩家原文写 JSON，同一句话复用同一份答案；
  - 世界跑两个种子：7 与 11。
- **B（纯模型主持人）**：`PureLLMGM` 加上新的圣经。每回合由隔离代理看着“圣经 + 完整对话 + 新输入”作答，独立跑两次。
- **玩家代理模式（`scripts/bench_player.py`，新）**：
  - 一个隔离的“玩家”代理，设定是“第一次玩、谨慎而好奇的普通玩家”；
  - 它只看**本方**看得见的正文，每回合写一句 ≤30 字的输入，最多 30 回合，到结局即停；
  - A 与 B 用同一个玩家设定，各跑 1 局。
  - 这样可以补上“固定脚本不适合涌现世界”的效度问题。它作为辅证：面板只做单局打分加总体偏好。
- **评审**：
  - 3 个盲评面板，A/B 的先后由种子打乱；
  - 每方 2 份审计，缺陷标注严重度；
  - 审计清单加上：没介绍先用名、幕后瞬移、物品漂移、替玩家做决定、悬空钩子、NPC 被当木偶摆布。
- **报告**：
  - 与 run1–6 并列，同时注明 run7 起是新场景、不能直接比较；
  - M1 另有一份在旧版世界上的 run6 重放对比，作为最后一个可比的数据点。
- **延迟**：
  - 代理测不了真实延迟。L1/L2 用 ScriptedLLM 按公开分布模拟：flash-lite 首 token 0.4–0.6 s；3.8 Flash low 首 token p50 2.2 s、p95 5.3 s，每秒 60 字；
  - 对照组的首 token 随 TOK 增长，按同一模型估算；
  - 报告明确标注“模拟”。有密钥时 `bench_gm --llm auto` 一键实测。

### 8.5 判定“赢了”

以下各条同时满足：

- 两个种子加一局玩家代理；
- 多数面板在至少 3 个维度上判 A 胜，且没有任何一个维度在全部面板里都判 B；
- 硬事实 C1 = C2 = 0；
- L1 p50 ≤ 1.5 s；
- TOK 恒定，而 B 在增长。

---

## 9. 风险与缓解

| # | 风险 | 缓解 |
|---|---|---|
| 1 | 驱力读起来像剧本，或把玩家的路堵死（railroading） | 条件只读当下的认知，行动只来自候选；有 no-railroad 与 divergence 测试；私语台词轮换；平手按种子决定；B 拿到同样的台词，胜负只取决于结构 |
| 2 | 名场面链条脆弱，依赖涌现行为，如灭口只针对落单者、钟灵跟着跳崖 | M2-a 先做 spike；sim_beats 20 个种子 × 5 种玩家，出口线 ≥16/20；有兜底路径：谁落单谁被追 |
| 3 | 段誉跑得比玩家快，时间对不上 | `wait_fond`、`follow_fond`；“跟上 X”快路径；同伴走开时多给 1 tick；进 run7 前逐句检查走查脚本 |
| 4 | 同一 tick 里讨价还价输给左子穆的还手 | demand 是 URGENT，抢在还手之前；门槛是 ≥16/20，而不是 100%；失败时现有的搜走解药路径照样能收场，前后照应会交代 |
| 5 | 精度修复放掉真错误 | 每条放宽都有标注样本；植入错误的召回率 ≥95%；比喻只放同一小句、回忆只放见过的实体且不许定位、复述意图只放“打定主意”一类；“长剑”不进 COMMON_WORDS |
| 6 | 一致性闸门让丢句与模板回退增多（G1 上升） | 允许截取一段；阈值 60%；模板回退原样印出台词，读来自然；G1 设门槛监控 |
| 7 | 外貌称呼干扰解释器，导致反复追问 | 外貌称呼唯一性测试；从宽处理的 named_early；masked 视图带别称 |
| 8 | 场面太挤，250 字写不下几个声音 | 每回合至多 3 句照录，看点优先；其余一笔带过；hush_chatter 照旧 |
| 9 | session 接线碰到幂等与崩溃重试的路径 | 先迁出 aside 与 endings；新状态一律走 annotate 的副本；test_resume / test_parallel；中途崩溃续跑测试 |
| 10 | 评测效度：新场景、代理不是 Gemini、固定脚本 | M1 在旧版上重放 run6；两个种子；玩家代理模式；审计计数与偏好并列报告；心智提示词用更小的代理模型 |
| 11 | 训练好的包失效 | 不动 ScriptedPolicy / kernel / cognition；金标准指纹；部署包校验测试 |
| 12 | 普通人挨打受挫 | 穴道自解；mercy；贿赂；天亮结局；提示先讲怎么活下来 |
| 13 | M5 的延迟与确定性 | 只在反应 tick；1.2 s 上限；先声立即交付；原话录进意图；deterministic 模式；开关由消融结果决定 |
| 14 | 看点识别与账本的开销 | 每 tick O(感知数)；`test_director_overhead` 式的 p95 < 5 ms 断言 |
| 15 | 给 B 同样的内容会收窄优势 | 接受。它保证结论可信；我们的优势在于真相一致、幕后真的在跑、延迟恒定 |

---

## 附录 A：看点（B1，11 个）

| key | 匹配（只看玩家感知到的已结算事件或景观） |
|---|---|
| challenge | 龚光杰对段誉 CHALLENGE 或 ATTACK |
| beating | 对段誉的 ATTACK 成功，段誉受伤 |
| marten | 钟灵对龚光杰的 ATTACK（obj 为 mink）成功，龚光杰中毒 |
| bargain | 钟灵对龚光杰 USE antidote |
| whisper | 私奔那对之间的 TELL（SPEECH 或低语的 SIGHT） |
| chase | 18:40 之后，龚光杰在后院、后山或崖顶的 MOVE 或 SAY（SIGHT / SOUND / SPEECH） |
| cliff | 段誉 MOVE，门为 d_cliff |
| moon | 玩家在剑湖畔，时钟 ≥ MOONRISE，`yubi@moon` 景观上演 |
| crack | d_cave 被发现（任何人查见，玩家在场） |
| kowtow | 在琅嬛，任何人摆出叩首姿态 |
| scroll | 对帛卷的 TAKE，或研读学成 |

## 附录 B：文件行数预算

| 文件 | 现在 | 之后 |
|---|---|---|
| runtime/session.py | 792 | ≈ 710 |
| language/narrator.py | 738 | ≈ 620 |
| language/parser.py | 773 | ≈ 745 |
| language/interpret.py | 670 | ≈ 710 |
| language/render.py | 634 | ≈ 690 |
| runtime/gm.py | 485 | ≈ 555 |
| language/quotes.py | 274 | ≈ 340 |
| 新增模块 | — | 均 ≤ 360 |
