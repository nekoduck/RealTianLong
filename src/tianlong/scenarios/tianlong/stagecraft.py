"""
[INPUT]: 依赖 core 的 Op / Social，scenarios/base 的 Beat / Card，scenarios/tianlong/wuliang 的 LOVERS，scenarios/tianlong/drives_c 的 HUNT
[OUTPUT]: 对外提供 BEATS_C（普通人版的 11 个看点识别器，plan 附录 A）、BEAT_KEYS（看点的次序）、CARDS_C（写法卡目录）、
          DETAILS_C（地点与陈设的细节卡组）
[POS]: scenarios/tianlong 普通人版的看点、写法与细节：识别器，不是触发器——只匹配已结算、且玩家已感知到的事件或景观（runtime/staging 求值），
       供 B1 计数、“等待在看点处停下”与叙述的看点。同一个看点可以有几条识别器（叫阵或动手；私语是后院里的闲话或带字的姿态——替人求饶的高声不算）；
       识别器按剧情推进排列，同一回合认出几个时取靠后的那个作本回合的看点（gloss 是玩家口吻的一句）。
       写法卡只有修辞、不加事实（文字须在它生效的场面里过得了叙述闸门，tests/test_stagecraft 逐张验）：
       貂的写法随貂咬中生效（毒发麻倒也在这里——只有貂毒），点穴的写法由它自己的识别器认点穴得手，火光与叫骂随追逐看点生效（同一组识别器：举火把逼近、堵住叫骂、向人打听，
       只路过不算——作罢后走回大殿也是路过，看点与写法卡不会一个说有火把一个说没有），
       扑空只认钟灵在琅嬛的那个姿态。卡文是一句能照抄进正文的话（不用冒号：“叫骂：……”会被当成有人开口）。许可词只许点名：月下舞剑的人影说成“仙人”，貂说成“小貂”；
       许可词不得是世界里实体的名字或别称（“长剑”是兵器架上的真剑：许了它，闸门就不再查那柄剑）。
       细节只写看得见的静物：不写人、不写会变的状态（门开没开）、不写线索之外的事，不点别处的名字，不写日月
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Op, Social
from tianlong.scenarios.base import Beat, Card
from tianlong.scenarios.tianlong.drives_c import HUNT
from tianlong.scenarios.tianlong.wuliang import LOVERS

_GG, _DY, _ZL = ("gongguangjie",), ("duanyu",), ("zhongling",)
_SCROLLS = ("scroll_lb", "scroll_bm")
_SEARCHED = ("houyuan", "houshan", "yading")
# 搜人时玩家真感知得到的那几下（都只在举了火把、还没作罢时才有）：举火把逼近、堵住叫骂、向人打听；
# 只路过不算——作罢之后走回大殿也是这样路过
_HUNTING = ((Op.WAIT, None, "举着火把"), (Op.TELL, Social.TAUNT, ""), (Op.ASK, None, ""))

BEATS_C: tuple[Beat, ...] = (
    Beat("challenge", Op.TELL, _GG, _DY, social=Social.CHALLENGE, gloss="龚光杰冲着段公子叫阵"),
    Beat("challenge", Op.ATTACK, _GG, _DY, success=False, gloss="龚光杰对段公子动手"),
    Beat("beating", Op.ATTACK, _GG, _DY, gloss="段公子挨了龚光杰一掌"),
    Beat("marten", Op.ATTACK, _ZL, _GG, gloss="一道灰影扑上龚光杰，他中了貂毒", stage=("mink_strike",), allowed=("小貂",)),
    Beat("bargain", Op.USE, _ZL, _GG, obj="antidote", gloss="梁上的少女拿解药跟龚光杰讲和"),
    Beat("whisper", Op.TELL, LOVERS, LOVERS, place=("houyuan",), social=Social.REMARK, gloss="后院里有人低声私语"),
    Beat("whisper", Op.WAIT, LOVERS, place=("houyuan",), social=Social.REMARK, gloss="后院里有人低声私语"),
    *(Beat("chase", op, _GG, place=_SEARCHED, clock_from=HUNT, social=social, words=words, gloss="火把与叫骂声追了过来",
           stage=("torch_search",)) for op, social, words in _HUNTING),
    Beat("cliff", Op.MOVE, _DY, door="d_cliff", gloss="段公子攀着藤萝跳下断崖"),
    Beat("moon", place=("jianhu",), clock_from="moon", lore="yubi@moon", once=True, gloss="月光照上玉璧，壁上似有仙人舞剑",
         stage=("moon_wall",), allowed=("仙人",)),
    Beat("crack", Op.INSPECT, door="d_cave", place=("jianhu",), gloss="玉璧旁露出一道石缝"),
    Beat("kowtow", Op.WAIT, place=("langhuan",), social=Social.SUBMIT, gloss="有人对着玉像磕头", stage=("kowtow_count",)),
    Beat("scroll", Op.TAKE, target=_SCROLLS, gloss="蒲团里的帛卷被人取了出来"),
    Beat("scroll", Op.STUDY, target=_SCROLLS, reason="mastered", gloss="帛卷上的步法被人参透了"),
)
BEAT_KEYS: tuple[str, ...] = tuple(dict.fromkeys(b.key for b in BEATS_C))

# ============================================================
#  写法卡：只有修辞；叙述者的系统提示里列全（静态前缀），本回合只点编号
# ============================================================

CARDS_C: dict[str, Card] = {
    "subdue_style": Card(
        "点穴就写点穴，一指点去，被点中的人登时手脚动弹不得，嘴里却还说得出话，眼珠也还转得动。",
        cues=(Beat("subdue_style", Op.ATTACK, status="subdued"),)),
    "mink_strike": Card("貂扑出时只见一道灰白的影子一闪，快得看不清；貂毒写成毒发麻倒，被咬的地方又麻又胀，手脚渐渐不听使唤。"),
    "torch_search": Card("搜人的动静写成火光与叫骂，火把的光一晃一晃，粗声的叫嚷时远时近。"),
    "moon_wall": Card("月下的玉璧写成光与影，壁上的人影似动非动，衣袂飘飘，像有仙人在壁上舞剑，看得人出了神。"),
    "kowtow_count": Card("磕头写成一下一下地数，额头碰着地面，咚的一声，又是一声。"),
    "grab_and_miss": Card(
        "扑空写成一抓落空，指尖刚要碰到衣袖，眼前一花，人已滑开了半步。",
        cues=(Beat("grab_and_miss", Op.WAIT, _ZL, place=("langhuan",)),)),
}

# ============================================================
#  细节卡组：同一处每查看一次，给下一条还没给过的（一幕之内不重复）；只写看得见的静物
# ============================================================

DETAILS_C: dict[str, tuple[str, ...]] = {
    "hall": ("殿柱上的红漆剥落了好几处，露出底下发黑的木头。",
             "比剑的场子铺着青石板，接缝里积着些细沙。",
             "宾客席的案几上摆着茶碗，碗里的茶早凉了。",
             "檐角挂着的铜铃被穿堂风吹得轻轻作响。"),
    "houyuan": ("厢房的窗纸破了几个小洞，被风吹得一鼓一鼓。",
                "古井的井栏上磨出了一道道绳痕，井口幽幽地透着凉气。",
                "那扇小门的门闩是根老枣木，磨得油光发亮。",
                "墙根下码着一排柴禾，上头落了层薄灰。"),
    "houshan": ("林边立着一块石碑，刻着禁地二字，字口里长满了青苔。",
                "松针落了厚厚一层，踩上去软绵绵的，没什么声响。",
                "几棵老松的树皮上有刀砍过的旧痕，早已长得模糊了。"),
    "yading": ("崖边的岩石被山风磨得光秃秃的，只在岩隙里钻出几丛野草。",
               "崖壁上垂下的藤萝有老有新，粗的有手腕那么粗。",
               "往下望去，云雾一层层翻上来，看不见底。"),
    "jianhu": ("湖水清得能看见水底的卵石，几尾小鱼在水里悬着。",
               "峭壁脚下长着一圈湿漉漉的青苔。",
               "湖边散落着些被水冲圆了的白石子。"),
    "shidong": ("石阶上凿着防滑的横纹，边角都磨圆了。",
                "洞顶不时滴下水珠，落在石阶上嗒的一声。",
                "洞壁上挂着一层细密的水珠，摸上去冰凉。"),
    "langhuan": ("石几上的灰积得有铜钱厚。",
                 "四壁凿得平平整整，墙角摆着几只空了的石瓮。",
                 "石凳的腿上雕着缠枝花纹，刀工细得出奇。"),
    "shandao": ("石阶两旁的松树斜斜地探向深谷。",
                "拐弯处有块歇脚的大石，被人坐得溜光。",
                "路边的野草长得齐膝高，草叶上挂着露水。"),
    "putuan": ("蒲团边上的丝线已经褪了色，那两行细字却还绣得清清楚楚。",
               "蒲团的草编朽了，一碰便簌簌地掉下碎屑，边上还裂开一道口子。",
               "两个蒲团都压出了浅浅的凹痕，像是有人在上头跪过许多回。"),
    "yubi": ("玉璧光滑如镜，凑近了能看见自己模糊的影子。",
             "玉璧浸在湖水里的那一截，留着一圈淡淡的水痕。",
             "玉璧上有几道极细的天然纹路，弯弯曲曲伸进石里。"),
    "statue": ("玉像的衣褶雕得层层叠叠，细看竟像要飘起来似的。",
               "玉像的一双眼睛是用黑宝石嵌的，望过去仿佛也在望着你。",
               "玉像脚下的石座上刻着云纹，积着薄薄一层灰。"),
}
