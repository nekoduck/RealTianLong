# 普通人版看点调参结果（sim_beats）

- 命令：`PYTHONPATH=src python scripts/sim_beats.py --seeds 1-20 --jobs 4 --check`（不接模型、模板模式；`--jobs` 与否结果逐项相同）
- 代码：提交 `da275e1`（M2 第二段）；`--check` 全部达标，退出码 0
- 玩家脚本只凭玩家自己的认知与时钟行事：passive 一直“等下去”；follower 跟着段公子（在后山陪着他时回后院取一趟水），崖顶不见人就攀下去；
  liar 天黑前溜去后院，龚光杰问起段公子就说他在山道；grabber 跟到琅嬛福地，抢先翻蒲团、拿走凌波微步自己研读；slipper 躲在大殿，夜饭换班的时辰一到就下山
- B1：玩家目击的看点数（附录 A 的 11 个识别器，只算玩家自己的感知）；E1：空转的推进回合所占比例（这一回合没感知到任何 NPC 的动作或言语、没有新认识的东西、没有看点）

## 出口条件

| 条件 | 门槛 | 结果 |
|---|---|---|
| 跟随型目击 ≥8/11 个看点的种子数 | ≥16/20 | 20/20（均值 10.55） |
| 跟随型 E1 | ≤10% | 0.0% |
| 被动玩家下段誉到琅嬛福地 | ≥12/20 | 20/20 |
| 被动玩家下段誉到澜沧江畔 | ≥8/20 | 20/20 |
| 讨价还价成立（钟灵离开大殿前没被制、解药由她用在龚光杰身上、左子穆说“依你”） | ≥16/20 | 20/20 |

## 结果表

| 玩家 | 种子 | B1 均值 | B1≥8 | 到琅嬛 | 到澜沧江 | 讨价还价 | 钟灵被制 | E1 | 结局 |
|---|---|---|---|---|---|---|---|---|---|
| passive | 20 | 4.0 | 0 | 20 | 20 | 20 | 0 | 0.0% | dawn 20 |
| follower | 20 | 10.55 | 20 | 20 | 20 | 20 | 0 | 0.0% | river 20 |
| liar | 20 | 5.2 | 0 | 20 | 20 | 20 | 0 | 14.1% | dawn 20 |
| grabber | 20 | 10.55 | 20 | 20 | 20 | 20 | 0 | 0.0% | river 20 |
| slipper | 20 | 4.0 | 0 | 20 | 20 | 20 | 0 | 16.8% | downhill 20 |

| 看点 | passive | follower | liar | grabber | slipper |
|---|---|---|---|---|---|
| challenge | 20 | 20 | 20 | 20 | 20 |
| beating | 20 | 20 | 20 | 20 | 20 |
| marten | 20 | 20 | 20 | 20 | 20 |
| bargain | 20 | 19 | 20 | 19 | 20 |
| whisper | 0 | 20 | 4 | 20 | 0 |
| chase | 0 | 20 | 20 | 20 | 0 |
| cliff | 0 | 12 | 0 | 12 | 0 |
| moon | 0 | 20 | 0 | 20 | 0 |
| crack | 0 | 20 | 0 | 20 | 0 |
| kowtow | 0 | 20 | 0 | 20 | 0 |
| scroll | 0 | 20 | 0 | 20 | 0 |

| 驱力兑现（次） | passive | follower | liar | grabber | slipper |
|---|---|---|---|---|---|
| duanyu.bolt | 20 | 40 | 20 | 40 | 20 |
| duanyu.explore_crack | 40 | 40 | 40 | 40 | 40 |
| duanyu.exult | 20 | 20 | 20 | 0 | 20 |
| duanyu.flee | 20 | 20 | 20 | 20 | 20 |
| duanyu.follow_fond | 0 | 20 | 0 | 40 | 0 |
| duanyu.gaze | 20 | 20 | 20 | 20 | 20 |
| duanyu.kowtow | 100 | 100 | 100 | 100 | 100 |
| duanyu.kowtow_first | 20 | 20 | 20 | 20 | 20 |
| duanyu.leap | 20 | 20 | 20 | 20 | 20 |
| duanyu.leave | 20 | 20 | 20 | 0 | 20 |
| duanyu.pacifist | 20 | 20 | 20 | 20 | 20 |
| duanyu.pacifist_more | 30 | 30 | 30 | 30 | 30 |
| duanyu.rest | 80 | 80 | 60 | 80 | 80 |
| duanyu.search | 20 | 20 | 20 | 0 | 20 |
| duanyu.study | 60 | 60 | 60 | 0 | 60 |
| duanyu.take | 40 | 40 | 40 | 0 | 40 |
| duanyu.wait_fond | 40 | 40 | 40 | 40 | 40 |
| duanyu.wander | 20 | 20 | 20 | 20 | 20 |
| ganguanghao.hush | 20 | 40 | 20 | 40 | 20 |
| ganguanghao.murmur | 680 | 380 | 336 | 380 | 680 |
| ganguanghao.shoo | 0 | 20 | 20 | 20 | 0 |
| ganguanghao.tryst | 20 | 20 | 20 | 20 | 20 |
| geguangpei.hush | 20 | 40 | 20 | 40 | 20 |
| geguangpei.murmur | 680 | 360 | 332 | 360 | 680 |
| geguangpei.tryst | 20 | 20 | 20 | 20 | 20 |
| gongguangjie.corner | 20 | 20 | 20 | 20 | 20 |
| gongguangjie.defer | 25 | 25 | 25 | 25 | 25 |
| gongguangjie.give_up | 20 | 20 | 20 | 20 | 20 |
| gongguangjie.go_back | 40 | 40 | 40 | 40 | 40 |
| gongguangjie.hunt | 100 | 60 | 216 | 60 | 100 |
| gongguangjie.no_leap | 20 | 20 | 20 | 20 | 20 |
| gongguangjie.numb | 35 | 35 | 35 | 35 | 35 |
| gongguangjie.torch | 20 | 20 | 20 | 20 | 20 |
| mawude.errand | 20 | 14 | 20 | 14 | 20 |
| mawude.plead | 20 | 20 | 20 | 20 | 20 |
| shennong.eat | 160 | 0 | 160 | 0 | 20 |
| shennong.relock | 20 | 0 | 20 | 0 | 0 |
| shennong.supper | 20 | 0 | 20 | 0 | 20 |
| shennong.unbar | 20 | 0 | 20 | 0 | 20 |
| zhongling.bargain | 20 | 20 | 20 | 20 | 20 |
| zhongling.grab | 20 | 20 | 20 | 0 | 20 |
| zhongling.hold_bite | 15 | 15 | 15 | 15 | 15 |
| zhongling.hold_truce | 10 | 10 | 10 | 10 | 10 |
| zhongling.marvel | 20 | 20 | 20 | 20 | 20 |
| zhongling.tease | 20 | 20 | 20 | 20 | 20 |
| zuozimu.await | 15 | 15 | 15 | 15 | 15 |
| zuozimu.demand | 20 | 20 | 20 | 20 | 20 |
| zuozimu.truce | 20 | 20 | 20 | 20 | 20 |
| zuozimu.warn | 5 | 5 | 5 | 5 | 5 |
| zuozimu.warn_hold | 6 | 6 | 6 | 6 | 6 |

## 调参记录

| 轮次 | 改动（只动驱力条件、时间窗、先验、内容与脚本化玩家） | 结果 |
|---|---|---|
| 0 | 初版驱力表（种子 1–4） | 被动玩家整条弧线跑通；跟随型卡在“跟上段公子”（解释器没认出）、跟上后与段誉来回换位（follow_fond 把玩家走到他刚离开之处也当成“从这里走开”） |
| 1 | 解释器“跟上 + 认识的人”快路径；Saw 加 here，follow_fond 只认从自己此处走开；私语改为两人一递一句（葛光佩只接话头）；龚光杰追到崖顶不见人即作罢（give_up），找遍另有 give_up_lost；跟随型脚本回后院取水 | 种子 1–20 开场：讨价还价 15/20（第一口被格开时左子穆当场还手、中毒的龚光杰连着还手，钟灵离殿前被制） |
| 2 | 左子穆 warn：弟子没中毒、对方没打他本人时，出言警告而不亲自动手；龚光杰 numb：中了貂毒动不了手 | 讨价还价 20/20；全量 20 种子：跟随型 B1≥8 20/20、E1 0%，被动到琅嬛/澜沧江 20/20；slipper 下山 16/20（脚本“等下去”一觉等过了换班） |
| 3 | slipper 脚本改为掐着换班的时辰短等 | slipper 下山 20/20；各门槛全部达标（即上表） |

## 已知局限

- 不同种子之间差异很小：种子只改内核的掷骰（先手抖动、出手胜负），驱力与脚本策略本身是确定的，所以多数统计是 20/20 或 0/20。
- 跟随型看到“跳崖”只有 12/20：玩家与段誉同一 tick 走动时，先手值决定玩家到崖顶时他是否已经跳下。
- E1 的“感知到 NPC 的言语”包括闲谈；长等待被要紧的事或看点打断，所以空转回合很少。
