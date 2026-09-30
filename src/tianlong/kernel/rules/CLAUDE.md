# kernel/rules/
> L2 | 父级: /src/tianlong/kernel/CLAUDE.md

行动规则插件。kernel 只认识 ActionRule 接口；新增行动 = 在 core/schema 加 Op 与 OpSignature + 在这里加一个子类并注册，kernel 与 perception 不改（OCP）。规则面对真实世界状态，产出 Resolution；再由 fragments() 说明结果在物理世界里留下的可感知片段，默认 perceive() 按片段投影（覆写 fragments()/perceive() 组合 Witnessing 积木）。

成员清单
base.py: ActionRule 抽象基类（resolve/loudness/fragments/perceive + usable_when_subdued + public_reasons 旁人看得出的失败原因）+ 方式系数表（careful 更轻更慢，rough 更响更快）
movement.py: MoveRule = 目的地 + 路线（Intent.obj 是门），只检查这条路是否真在身边、真通往那里、方向与锁，不替角色挑路（路不在这里/不通往那里即获知纠正）；推不开即获知"门锁着"、逆着单向通道即获知"只能往下"；成功才在目的地留下"抵达"片段，门边失败只在这一侧留下动作与响动；WaitRule 不带姿态时不产生感知，带姿态（坐下喝茶、拔剑、磕头）时在场的人亲眼看见，不附带任何事实
handling.py: TakeRule/PutRule/GiveRule 只改 AT 从不改 OWNS；careful 放置即藏匿（hidden），正常放置或拿起时解除；可以搜走被制住者身上的东西；"东西在别人手里"旁人看得见
locks.py: UnlockRule/LockRule 共享前置条件（手持、门在身边、钥匙匹配），钥匙不配即获知"不配"
senses.py: InspectRule 仔细查看地点/台面发现藏匿物与暗门（night_only 只在夜里、clue 指明线索物如玉璧）、搜身发现藏在身上的小物件，并给出含藏匿物的"完整看清"范围
combat.py: AttackRule 武斗物理（被挡开/闪开旁人看得见），身手 + 带种子随机裁定；得手依次致伤、点穴（限时自解）、带毒兵刃致毒；evasion 闪避、absorb 吸走徒手攻击者内力——技能以效果命名，书名只是内容
cultivation.py: StudyRule 研读秘籍逐次累积（进度私密、旁人看不出学没学成）、UseRule 施用物品（cures 对症才有效）
speech.py: RequestItemRule 只传递请求，耳语不向旁观者披露物品与受益人；TellRule/AskRule 言语不改物理世界，只给听者"说法"（可为谎言）；命题可选：不带命题的闲话只传原话与言语行为；careful 即耳语，旁人只见交谈、不闻内容也不知是赔罪还是威胁；穴道被制仍可开口
__init__.py: default_rules() 注册表，启动时校验每个 Op 都有规则

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
