"""
[INPUT]: 依赖 core 的 at / Op / Social / Outcome，core/drives 的驱力词表，scenarios/tianlong/wuliang 的 ELOPE / LOVERS
[OUTPUT]: 对外提供 DRIVES_C（角色 → 驱力表）与时刻 TRYST / MURMUR / WANDER / HUNT / MOONRISE / SUPPER / SUPPER_END / DAWN
[POS]: scenarios/tianlong 普通人版的角色性情（plan §3.5，凡与之冲突处按 game_master §8 的对策）：条件只读该角色自己的认知，
       行动只从合法候选里挑，或是一句台词、一个带字的姿态；兑现与否由内核裁定。每条带 gloss——玩家口吻的一句说明，供对照组圣经。
       §8 的改动：钟灵 hold_bite / hold_truce 与龚光杰 defer 两条 VETO（交易不再被互相还手搅黄）；私语 murmur 改为带字的姿态
       （走进来的人当 tick 看得见），住口 hush 推迟一 tick（Arrived）；学成对旁人不可见，段誉学成后一 tick 摆 exult 姿态，
       钟灵的 grab 挂在它上面；搜人先打听（Ask），追人绕开认为单向的门，找遍或看见他跳崖即放弃（give_up 在 t+1 抢先，
       另有 no_leap 在崖顶否决一切去路、改回大殿）；逃跑看眼前态度 ≤ −2、没中毒没被制的人（Menaced），刚挨了一下先愣一 tick；
       石缝要先 Go(shidong) 再 Go(langhuan)。VETO 的时间窗一律以驱力自己的标记计（Fired）。
       关卡：栅门上锁、钥匙在帮众身上——夜饭换班前先开锁（unbar），收了碎银的放行（bribed_open）并不再对他动手（wink），
       换班回来再锁上（relock）。一次性台词写在 line 上；姿态本身就是看得见的字（带引语的姿态把话写在字里）。
       马五德差阿顺去后院用 EXPLAIN 不用 COMMAND（§8 第 1 条：喝令会让阿顺对好心的东家 −1）；段誉回头等人要阿顺刚才还在身边（Near）；
       龚光杰堵住段誉时不出手（给跳崖留窗口），闲着就举火把逼近（loom），不与旁人寒暄。
       姿态是叙述看得见的那一行，引号外只用中性称呼（段公子），NPC 口吻里的昵称（书呆子、酸秀才）只留在引号里的原话
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Op, Outcome, Social, at
from tianlong.core.drives import (
    PLAYER,
    SELF,
    TARGET,
    Alone,
    AnyOf,
    Arrived,
    Ask,
    At,
    Between,
    Cross,
    Drive,
    Fired,
    Flee,
    Follow,
    Fond,
    Go,
    Heard,
    HeldBy,
    Here,
    Hold,
    Holds,
    Inspect,
    Knows,
    Level,
    Lock,
    Lost,
    Menaced,
    Near,
    Not,
    Pose,
    Pursue,
    Saw,
    Say,
    Searched,
    Status,
    Study,
    Take,
    Unlock,
    Use,
)
from tianlong.scenarios.tianlong.wuliang import ELOPE, LOVERS

URGENT, IDLE, VETO = Level.URGENT, Level.IDLE, Level.VETO

# ============================================================
#  时刻（场景的时钟事实）
# ============================================================

WANDER = at(1, 18, 0)        # 段誉嫌后院逼仄，往后山去
TRYST = at(1, 18, 5)         # 私奔那对溜去后院幽会
MURMUR = at(1, 18, 8)        # 四下无人时低声私语
HUNT = at(1, 18, 40)         # 龚光杰举火搜人
MOONRISE = at(1, 19, 40)     # 月出：玉璧上显出舞剑的人影
SUPPER, SUPPER_END = at(1, 20, 0), at(1, 22, 0)   # 帮众回营吃饭换班
DAWN = at(2, 5, 0)           # 天亮

SOFT = frozenset({Social.SUBMIT, Social.APOLOGIZE, Social.PLEAD})
DEEPER = ("yading", "houshan", "houyuan")      # 往里逃：大殿 → 后院 → 后山 → 崖顶（离人越远越好）
_GG, _ZL, _ZZ, _DY = "gongguangjie", "zhongling", "zuozimu", "duanyu"

# ============================================================
#  段誉：不还手；挨打了逃，被堵在崖顶就跳；月下看玉璧、进石缝、叩首、翻蒲团、取帛卷、研读、钻隧道
# ============================================================

DUANYU = (
    Drive("pacifist", VETO, (), Say(TARGET, Social.PLEAD), line="君子动口不动手，这位兄台有话好说……", once=True,
          veto=Op.ATTACK, gloss="段公子从不还手，挨了打也只是求饶讲理"),
    Drive("pacifist_flee", VETO, (Status(SELF, "wounded"), Menaced(-2)), (Flee(DEEPER), Say(TARGET, Social.PLEAD)),
          veto=Op.ATTACK,
          gloss="段公子受了伤还被人逼着，宁可逃也不动手"),
    Drive("pacifist_more", VETO, (), Say(TARGET, Social.PLEAD), veto=Op.ATTACK, gloss="段公子再挨打也只是讲理"),
    Drive("wait_fond", URGENT, (AnyOf((Fired("flee", within=2), Fired("bolt", within=2), Fired("pacifist_flee", within=2))),
                                Not(Here(PLAYER)), Near(PLAYER, within=3),
                                Fond(PLAYER, 1), Not(Menaced(-2)), Not(Fired("wait_fond", within=5))),
          Pose("回头张望了一眼，像是在等人"), gloss="段公子逃开时你若还在他身边，他会回头等等你"),
    Drive("leap", URGENT, (At("yading"), AnyOf((Here(_GG), Saw(actor=_GG, within=2), Heard(_GG, within=2)))),
          Cross("d_cliff"), line="与其落在你手里，不如赌一赌这藤萝！", once=True,
          gloss="段公子被追到崖顶，会攀着藤萝跳下断崖"),
    Drive("flee", URGENT, (Status(SELF, "wounded"), Menaced(-2), Not(Saw(Op.ATTACK, target=SELF, within=1))),
          Flee(DEEPER), line="君子不立危墙之下，在下告辞！", once=True,
          gloss="段公子挨了打、仇人又能动手时，会往后院、后山没人的地方逃"),
    Drive("bolt", URGENT, (Status(SELF, "wounded"), Menaced(-2), Not(Saw(Op.ATTACK, target=SELF, within=1))),
          Flee(DEEPER), gloss="段公子再撞见仇人，接着逃"),
    Drive("exult", URGENT, Status(SELF, "evasion"), Pose("忽然拍腿叫道：“妙哉，这步法竟暗合易经！”"), once=True,
          gloss="段公子学成了帛卷上的步法，会喜形于色"),
    Drive("follow_fond", URGENT, (Saw(Op.MOVE, actor=PLAYER, within=1, here=True), Not(Here(PLAYER)), Fond(PLAYER, 1),
                                  Not(Menaced(-2))), Follow(PLAYER), gloss="你走开时，段公子会跟着你走"),
    Drive("wander", IDLE, (At("houyuan"), Between(WANDER), Not(Menaced(-2))), Go("houshan"), line="这后山倒是清幽。",
          once=True, gloss="天色将晚，段公子嫌后院逼仄，会往后山去走走"),
    Drive("rest", IDLE, (At("jianhu"), Status(SELF, "wounded"), Between(WANDER, MOONRISE)),
          Pose("倚着湖石歇息，揉着挨了一掌的肩头"), cooldown=10, gloss="落到崖底，段公子会倚着湖石歇息"),
    Drive("gaze", IDLE, (At("jianhu"), Between(MOONRISE), Knows("yubi")), Inspect("yubi"),
          line="咦，这石壁上……竟有人影？", once=True, gloss="月亮出来后，段公子会去细看玉璧上的人影"),
    Drive("explore_crack", IDLE, (Knows("d_cave"), Not(At("langhuan")), Not(Fired("exult"))),
          (Go("langhuan"), Go("shidong")), gloss="段公子发现了石缝，会钻进去看个究竟"),
    Drive("kowtow_first", IDLE, (At("langhuan"), Knows("statue"), Not(Searched("putuan"))),
          Pose("对着玉像跪倒，口中念念有词：“神仙姐姐在上……”", Social.SUBMIT), once=True,
          gloss="段公子见了玉像，会痴痴地跪下磕头"),
    Drive("kowtow", IDLE, (At("langhuan"), Fired("kowtow_first"), Not(Searched("putuan")), Not(Fired("kowtow", times=5))),
          Pose("对着玉像恭恭敬敬地磕头", Social.SUBMIT), gloss="段公子一个接一个地磕头"),
    Drive("search", IDLE, (At("langhuan"), Fired("kowtow", times=5), Not(Searched("putuan"))), Inspect("putuan"),
          gloss="磕完了头，段公子会翻看蒲团"),
    Drive("take", IDLE, Knows("scroll_lb"), (Take("scroll_lb"), Take("scroll_bm")), gloss="段公子会取出蒲团里的帛卷"),
    Drive("plead_scroll", IDLE, (HeldBy("scroll_lb", PLAYER), Here(PLAYER), Fond(PLAYER, 1)), Say(PLAYER, Social.PLEAD),
          line="阿顺兄，那帛卷可否借在下一观？", once=True, gloss="帛卷若在你手里，段公子会开口恳求借阅（只求这一回）"),
    Drive("study", IDLE, (Holds("scroll_lb"), Not(Status(SELF, "evasion"))), Study("scroll_lb"),
          gloss="段公子会捧着凌波微步的帛卷一遍遍地读"),
    Drive("leave", IDLE, (Fired("exult"), Knows("d_tunnel"), Not(At("lancang"))), Go("lancang", avoid_oneway=False),
          gloss="学成之后，段公子会钻进隧道出山"),
)

# ============================================================
#  钟灵：护着段誉（目标照旧）；中毒的人不再打、交易之后不再打；以药换人；自报姓名；月下惊呼；笑人磕头；扑空
# ============================================================

ZHONGLING = (
    Drive("hold_bite", VETO, (Status(TARGET, "poisoned"), Not(Saw(Op.ATTACK, actor=TARGET, within=1))),
          Pose("护在段公子身前，冷眼瞧着"), veto=Op.ATTACK, gloss="钟灵不再打已经中了毒的人，除非他又动手"),
    Drive("hold_truce", VETO, (Fired("bargain"), Not(Saw(Op.ATTACK, actor=TARGET, within=1))),
          Pose("拍了拍腰间的貂儿，没再动手"), veto=Op.ATTACK, gloss="钟灵换了人之后说话算话，除非对方又动手"),
    Drive("bargain", URGENT, (Status(_GG, "poisoned"), Holds("antidote"), Here(_ZZ), Here(_GG),
                              AnyOf((Heard(_ZZ, frozenset({Social.COMMAND, Social.THREATEN}), to=SELF, within=3),
                                     Saw(Op.ATTACK, actor=_ZZ, target=SELF, within=2)))),
          Use("antidote", _GG), line="解药在这儿。你们不再为难这书呆子，我便救他。", once=True,
          gloss="东宗掌门向钟灵要解药，她会拿解药换段公子平安"),
    Drive("introduce", URGENT, Heard(PLAYER, frozenset({Social.THANK, Social.PRAISE}), to=SELF, within=2),
          Say(PLAYER, Social.GREET), line="谢什么？你又打不过他。我叫钟灵，你呢？", once=True,
          gloss="你向她道谢，梁上的少女会自报姓名"),
    Drive("introduce_hi", URGENT, (Heard(PLAYER, to=SELF, within=2), Not(Fired("introduce"))),
          Say(PLAYER, Social.GREET), line="咦，你这小伙计倒会说话。我叫钟灵，你呢？", once=True,
          gloss="你跟她搭话，梁上的少女会自报姓名"),
    Drive("marvel", IDLE, (At("jianhu"), Between(MOONRISE), Here(_DY), Not(Fired("marvel_you"))),
          Say(_DY, Social.REMARK), line="书呆子你快看！石壁上有仙人！", once=True,
          gloss="月亮一出来，钟灵会惊呼玉璧上有仙人"),
    Drive("marvel_you", IDLE, (At("jianhu"), Between(MOONRISE), Not(Here(_DY)), Here(PLAYER), Fond(PLAYER, 0),
                               Not(Fired("marvel"))),
          Say(PLAYER, Social.REMARK), line="喂，你快看！石壁上有仙人！", once=True, gloss="月亮一出来，钟灵会拉你看玉璧"),
    Drive("tease", IDLE, (At("langhuan"), Saw(Op.WAIT, actor=_DY, socials=frozenset({Social.SUBMIT}), within=2),
                          Here(_DY), Not(Fired("tease_you"))),
          Say(_DY, Social.JOKE), line="你真要磕一千个？磕傻了我可不管你！", once=True, gloss="钟灵会取笑磕头的人"),
    Drive("tease_you", IDLE, (At("langhuan"), Saw(Op.WAIT, actor=PLAYER, socials=frozenset({Social.SUBMIT}), within=2),
                              Here(PLAYER), Not(Fired("tease"))),
          Say(PLAYER, Social.JOKE), line="你真要磕一千个？磕傻了我可不管你！", once=True, gloss="钟灵会取笑磕头的人"),
    Drive("grab", IDLE, (Saw(Op.STUDY, actor=_DY, within=10), Saw(Op.WAIT, actor=_DY, within=1), Here(_DY)),
          Pose("扑过去要抓段公子的袖子，却扑了个空：“不玩了不玩了！书呆子，你这是什么古怪步法？”"), once=True,
          gloss="段公子学成步法后，钟灵扑过去抓他却扑了个空"),
)

# ============================================================
#  左子穆：索药抢在还手之前；等一等；见解药救了弟子就“依你”；之后三十个 tick 不动手（除非挨打）
# ============================================================

ZUOZIMU = (
    Drive("warn", VETO, (Not(Status(_GG, "poisoned")), Saw(Op.ATTACK, actor=TARGET, target=_GG, within=3),
                         Not(Saw(Op.ATTACK, actor=TARGET, target=SELF, within=1))),
          Say(TARGET, Social.THREATEN), line="小丫头，再敢放肆，休怪老夫手下无情！", once=True, veto=Op.ATTACK,
          gloss="有人对他弟子出手却没得手，东宗掌门先出言警告，自重身份不亲自动手"),
    Drive("warn_hold", VETO, (Not(Status(_GG, "poisoned")), Saw(Op.ATTACK, actor=TARGET, target=_GG, within=3),
                              Not(Saw(Op.ATTACK, actor=TARGET, target=SELF, within=1))), veto=Op.ATTACK,
          gloss="东宗掌门警告过了，只冷眼盯着"),
    Drive("truce_hold", VETO, (Fired("truce", within=30), Not(Saw(Op.ATTACK, actor=TARGET, within=1))), veto=Op.ATTACK,
          gloss="东宗掌门答应了的事，一时不会反悔"),
    Drive("truce", URGENT, Saw(Op.USE, actor=_ZL, target=_GG, outcome=Outcome.SUCCESS, within=3),
          Say(_ZL, Social.AGREE), line="……好，依你。", once=True, gloss="钟灵救了他的弟子，东宗掌门会点头答应"),
    Drive("demand", URGENT, (Status(_GG, "poisoned"), Saw(Op.ATTACK, actor=_ZL, target=_GG, within=10), Here(_ZL),
                             Not(Status(_ZL, "subdued"))),
          Say(_ZL, Social.COMMAND), line="小姑娘，把解药交出来，今日之事还有得商量。", once=True,
          gloss="弟子中了貂毒，东宗掌门会先向钟灵索要解药"),
    Drive("await", URGENT, (Fired("demand", within=2), Status(_GG, "poisoned")), Hold(),
          gloss="索过解药，东宗掌门会等她回话"),
)

# ============================================================
#  龚光杰：师父开了口就不打钟灵；18:40 举火搜人（先打听），堵住就叫骂，看见跳崖或找遍了就放弃；崖顶上从不往下跳
# ============================================================

_QUIT = AnyOf((Fired("give_up"), Fired("give_up_lost")))
_HUNTING = (Fired("torch"), Not(_QUIT))          # 举了火把、还没作罢

GONGGUANGJIE = (
    Drive("numb", VETO, Status(SELF, "poisoned"), Pose("捂着伤处，脸色发青，一时提不起劲来"), veto=Op.ATTACK,
          gloss="中了貂毒，龚光杰浑身发麻，动不了手"),
    Drive("defer", VETO, Heard(_ZZ, frozenset({Social.COMMAND, Social.AGREE}), to=TARGET, within=30), veto=Op.ATTACK,
          gloss="师父对钟灵开了口，龚光杰就不再对她动手"),
    Drive("no_leap", VETO, At("yading"), Go("hall"), veto=Op.MOVE, gloss="龚光杰从不往断崖下跳"),
    Drive("give_up", URGENT, (*_HUNTING, Not(Here(_DY)), AnyOf((Saw(Op.MOVE, actor=_DY, obj="d_cliff", within=2),
                                                                At("yading")))),
          Pose("朝崖下啐了一口：“哼，摔不死你也困死你！”"), once=True,
          gloss="龚光杰追到崖顶不见人（或看见段公子跳崖），便知他下了断崖，骂一句作罢"),
    Drive("give_up_lost", URGENT, (*_HUNTING, Lost(_DY)), Pose("把火把往地上一掼：“哼，算那酸秀才命大！”"), once=True,
          gloss="找遍了也找不到，龚光杰骂一句作罢"),
    Drive("go_back", URGENT, (_QUIT, Not(At("hall"))), Go("hall"), gloss="作罢之后，龚光杰回大殿去"),
    Drive("torch", URGENT, (Between(HUNT), Not(Status(SELF, "poisoned")), Not(Here(_DY)), Not(_QUIT)),
          Pose("从廊下摘了一支火把，要去搜人"), once=True, gloss="天黑前，龚光杰会举着火把去搜段公子"),
    Drive("corner", URGENT, (*_HUNTING, Here(_DY), Not(At("hall"))), Say(_DY, Social.TAUNT),
          line="酸秀才，师父只答应在殿上不为难你——这回看你还往哪里逃！", once=True,
          gloss="龚光杰在殿外堵住段公子，先叫骂一番"),
    Drive("corner_again", URGENT, (*_HUNTING, Here(_DY), Not(At("hall")), Not(Fired("corner", within=10)),
                                   Not(Fired("corner_again", within=10))),
          Say(_DY, Social.TAUNT), gloss="再撞见段公子，龚光杰又骂一回"),
    Drive("loom", IDLE, (*_HUNTING, Here(_DY), Not(At("hall"))), Pose("举着火把逼上一步，冷笑着堵住去路"),
          gloss="龚光杰堵住段公子时只举着火把逼近，不与旁人寒暄"),
    Drive("hunt", URGENT, (*_HUNTING, Not(Status(SELF, "poisoned")), Not(Here(_DY))), (Ask(_DY), Pursue(_DY)),
          gloss="龚光杰搜人时会向遇见的人打听段公子的下落"),
)

# ============================================================
#  马五德：替人求情；讲和之后差伙计去后院
# ============================================================

MAWUDE = (
    Drive("plead", URGENT, (Here(_GG), AnyOf((Saw(Op.ATTACK, actor=_GG, target=_DY, within=1),
                                              Saw(Op.ATTACK, actor=_GG, target=PLAYER, within=1)))),
          Say(_GG, Social.PLEAD), line="龚老弟，和气生财，和气生财……", cooldown=5,
          gloss="龚光杰对段公子或你动手，马五爷会打圆场"),
    Drive("errand", IDLE, (Heard(_ZZ, frozenset({Social.AGREE}), within=15), Here(PLAYER)), Say(PLAYER, Social.EXPLAIN),
          line="阿顺，茶担先挑去后院厨下歇着，莫在这里碍眼。", once=True,
          gloss="风波平了，马五爷会差你把茶担挑去后院厨下"),
)

# ============================================================
#  私奔那对：天擦黑溜去后院幽会，四下无人时低声私语，有人进来就住口；落单的撞见者照旧灭口，求饶则师妹求情
# ============================================================


def _lovers(me: str, partner: str, murmurs: tuple[str, ...], answer: bool = False) -> tuple[Drive, ...]:
    """answer：只接着对方的话头说（对方上一 tick 刚低语过）——两人一递一句，各自隔一 tick，谁也不会同时开口。"""
    lead = (Saw(Op.WAIT, actor=partner, within=1),) if answer else ()
    return (
        Drive("hush", URGENT, (At("houyuan"), Between(TRYST, ELOPE), Arrived(allowed=LOVERS)),
              Pose("登时住了口，别过脸去"), cooldown=10, gloss="后院里一有人进来，那一对便住了口"),
        Drive("tryst", URGENT, (Between(TRYST, ELOPE), Not(At("houyuan"))), Go("houyuan", careful=True),
              gloss="天擦黑，东宗的高个子弟子与西宗的女弟子先后溜去后院"),
        Drive("murmur", IDLE, (At("houyuan"), Between(MURMUR, ELOPE), Here(partner), Alone(allowed=LOVERS), *lead),
              Pose("低声私语", Social.REMARK), lines=murmurs, cooldown=2,
              gloss="四下无人时，那一对在后院低声私语"),
    )


GANGUANGHAO = (
    Drive("spare", VETO, Heard("geguangpei", frozenset({Social.PLEAD}), to=SELF, within=10), Go("camp", careful=True),
          veto=Op.ATTACK, gloss="师妹求了情，干光豪便不再动手"),
    *_lovers("ganguanghao", "geguangpei", ("凑在她耳边低声道：“……天黑了再走……”", "压低了嗓子道：“……营里那边已说定了……”",
                                           "握着她的手低声道：“……别怕，有我……”")),
    Drive("shoo", URGENT, (Fired("hush", within=2), At("houyuan"), Here(PLAYER)), Say(PLAYER, Social.COMMAND),
          line="看什么看？还不去干你的活！", once=True, gloss="撞见的若是挑夫，高个子弟子会喝他走开"),
)
GEGUANGPEI = (
    Drive("spare", VETO, Fired("mercy", within=10), Go("camp", careful=True), veto=Op.ATTACK,
          gloss="替人求了情，她自己也不再动手"),
    Drive("mercy", URGENT, (Heard(PLAYER, SOFT, within=2), Here("ganguanghao"), Here(PLAYER),
                            AnyOf((Saw(Op.ATTACK, actor="ganguanghao", target=PLAYER, within=3), Status(PLAYER, "wounded")))),
          Say("ganguanghao", Social.PLEAD), line="师哥，他一个挑茶的，饶了他吧！", once=True,
          gloss="被撞见的挑夫若求饶，西宗女弟子会替他求情"),
    *_lovers("geguangpei", "ganguanghao", ("低着头细声道：“……师父那边，瞒得过吗……”",
                                           "攥着衣角细声道：“……天黑了，真走得脱么……”"), answer=True),
)

# ============================================================
#  神农帮帮众：锁着关卡守山道；夜饭换班先开锁再回营；收了碎银的放行、不再动手；换班回来再锁上
# ============================================================

SHENNONG_C = (
    Drive("wink", VETO, Fond(TARGET, 2), Pose("别过脸去，只当没瞧见"), veto=Op.ATTACK, gloss="收了碎银，帮众就不再拦你"),
    Drive("eat", VETO, (Between(SUPPER, SUPPER_END), At("camp")), veto=Op.MOVE, gloss="夜饭没吃完，帮众不回山道"),
    Drive("unbar", URGENT, (Between(SUPPER, SUPPER_END), At("shandao")), Unlock("d_downhill"),
          gloss="换班吃饭前，帮众会顺手打开关卡的锁"),
    Drive("supper", URGENT, (Between(SUPPER, SUPPER_END), Not(At("camp"))), Go("camp"),
          line="换班吃饭去，谁耐烦在这儿喝风。", gloss="戌时过后，把守山道的帮众回营吃饭，一去一个时辰"),
    Drive("bribed_open", URGENT, (Here(PLAYER), Fond(PLAYER, 2), At("shandao")), Unlock("d_downhill"),
          gloss="塞了碎银，帮众会替你开关卡"),
    Drive("bribed", URGENT, (Here(PLAYER), Fond(PLAYER, 2)), Pose("掂了掂碎银，别过脸去，只当没瞧见"), once=True,
          gloss="塞了碎银，帮众只当没瞧见"),
    Drive("relock", URGENT, (At("shandao"), Not(Between(SUPPER, SUPPER_END)),
                             AnyOf((Not(Here(PLAYER)), Not(Fond(PLAYER, 2))))), Lock("d_downhill"),
          gloss="换班回来，帮众会把关卡重新锁上"),
)

DRIVES_C: dict[str, tuple[Drive, ...]] = {
    "duanyu": DUANYU, "zhongling": ZHONGLING, "zuozimu": ZUOZIMU, "gongguangjie": GONGGUANGJIE, "mawude": MAWUDE,
    "ganguanghao": GANGUANGHAO, "geguangpei": GEGUANGPEI, "shennong": SHENNONG_C,
}
