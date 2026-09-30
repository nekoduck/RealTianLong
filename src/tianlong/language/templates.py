"""
[INPUT]: 依赖 core 的 EntitySketch / Kind / Fact / PerceivedEvent / Percept / Modality / Op / Outcome / Rel / Social
[OUTPUT]: 对外提供 Names 类型、render_fact()、render_event()、render_experience()、render_percept()、consequences()（事件的看得见的后果；familiar——此前已知下落的——只说“仍在”）、
          REASONS / SUCCESS_NOTES / ATTR_WORDS、
          SOCIAL_VERBS（没有原话的言语按言语行为写成动作：“向钟灵打了个招呼”）
[POS]: language 的确定性文本层（无 LLM）；memory 用它生成经历文本，narrator 在无模型时用它兜底——同一套措辞，两处复用。
       所见清单按所在处归拢成人话（也是交给声音模型的事实清单：更短、更像话，每条事实照旧都在）。
       旁人来到眼前写出他从哪道认得的门过来（“钟灵经断崖来到剑湖畔”：人是怎么来的一目了然），带服软的查看写成伏地磕头
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Container, Mapping

from tianlong.core import (
    EntitySketch,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Rel,
    Social,
)
from tianlong.core.schema import ATTR_PREFIX

# 名称表即观察者自己的实体草图：名字 + 种类（决定“在桌上/在身上/在港口”的措辞）
Names = Mapping[str, EntitySketch]

REASONS: dict[str, str] = {
    "door_locked": "门锁着",
    "not_found": "那里没有找到",
    "held_by_other": "东西在别人手里",
    "wrong_key": "钥匙对不上",
    "out_of_reach": "够不着",
    "not_holding": "手里并没有那样东西",
    "already_there": "已经在那里了",
    "already_held": "已经拿在手里了",
    "already_locked": "本来就锁着",
    "already_unlocked": "本来就没锁",
    "not_adjacent": "从这里过不去",
    "route_not_here": "这里并没有那条路",
    "route_mismatch": "那条路并不通往那里",
    "self_target": "不能对自己这么做",
    "stale": "时机已经过去了",
    "duplicate_actor": "同一时刻只能做一件事",
    "subdued": "穴道被制，动弹不得",
    "evaded": "被对方轻飘飘地闪了开去",
    "parried": "被对方化解了",          # 不说怎么化解的：弱者靠貂、靠运气避过强者的一掌，交给叙述者写得合情理
    "one_way": "陡峭异常，爬不回去",
    "nothing_to_learn": "看不出什么门道",
    "already_learned": "早已学会了",
    "no_effect": "毫无效验",
}

# 成功时的附注（研读的进境）
SUCCESS_NOTES: dict[str, str] = {"progress": "，若有所悟，却还未能融会贯通", "mastered": "，豁然贯通"}

# 属性命题的措辞：(为真, 为假)
ATTR_WORDS: dict[str, tuple[str, str]] = {
    "locked": ("锁着", "没锁"),
    "hidden": ("被藏了起来", "没有被藏起来"),
    "wounded": ("受了伤", "没有受伤"),
    "poisoned": ("中了毒", "没有中毒"),
    "subdued": ("被点了穴道，动弹不得", "行动如常"),
    "evasion": ("身法大进，步法已然纯熟", "不会闪避的步法"),
    "absorb": ("体内多了一门吸纳他人内力的奇功", "不会吸纳内力的功夫"),
}
# 事件之后值得一提的后果（状态、技能、暗道）
_NOTABLE = frozenset({"wounded", "poisoned", "subdued", "evasion", "absorb"})


def _n(names: Names, eid: str | None, viewer: str | None = None, me: str = "我") -> str:
    if eid is None:
        return "某处"
    if eid == viewer:
        return me
    sk = names.get(eid)
    return sk.name if sk else eid


def _where(names: Names, eid: str, viewer: str | None, me: str) -> str:
    """位置宾语的措辞：人 → X身上，台面 → X上，地点 → X。"""
    sk = names.get(eid)
    suffix = {Kind.PERSON: "身上", Kind.SURFACE: "上"}.get(sk.kind, "") if sk else ""
    return _n(names, eid, viewer, me) + suffix


# ============================================================
#  命题
# ============================================================


def render_fact(f: Fact, names: Names, viewer: str | None = None, me: str = "我") -> str:
    p = f.prop
    subj = _n(names, p.subject, viewer, me)
    neg = not f.holds
    if p.predicate.startswith(ATTR_PREFIX):
        key = p.attr_key
        if key in ATTR_WORDS:
            return f"{subj}{ATTR_WORDS[key][1 if neg else 0]}"
        if key == "oneway":
            return f"{subj}只能通往{_n(names, str(p.value), viewer, me)}"
        return f"{subj}的{key}{'不是' if neg else '是'}{p.value}"
    rel = Rel(p.predicate)
    if p.value is None:
        return f"{subj}在哪里？" if rel == Rel.AT else f"{subj}的{rel.value}是什么？"
    if rel == Rel.AT:
        return f"{subj}{'不在' if neg else '在'}{_where(names, str(p.value), viewer, me)}"
    obj = _n(names, str(p.value), viewer, me)
    phrase = {
        Rel.OWNS: ("拥有", "并不拥有"),
        Rel.MATCHES: ("能打开", "打不开"),
        Rel.CONNECTS: ("通往", "不通往"),
    }[rel]
    return f"{subj}{phrase[1] if neg else phrase[0]}{obj}"


# ============================================================
#  事件
# ============================================================


# 没有原话的言语：按言语行为写成一个动作（“向钟灵打了个招呼”），耳语（旁人连言语行为也不知道）才是“低声说了些什么”
SOCIAL_VERBS: dict[Social, str] = {
    Social.GREET: "向{t}打了个招呼", Social.THANK: "向{t}道谢", Social.APOLOGIZE: "向{t}赔不是", Social.PLEAD: "向{t}求情",
    Social.PRAISE: "称赞{t}", Social.THREATEN: "出言威胁{t}", Social.TAUNT: "出言讥讽{t}", Social.INSULT: "骂了{t}几句",
    Social.REFUSE: "回绝了{t}", Social.AGREE: "应承了{t}", Social.JOKE: "跟{t}说笑", Social.COMFORT: "宽慰{t}",
    Social.EXPLAIN: "向{t}解释了几句", Social.CHALLENGE: "向{t}叫阵", Social.COMMAND: "喝令{t}", Social.FAREWELL: "向{t}告辞",
    Social.REMARK: "对{t}说了几句", Social.SUBMIT: "向{t}服软",
}


def _speech(v: PerceivedEvent, t: str, topic: str) -> str:
    if v.topic or v.utterance:
        return f"对{t}说：“{v.utterance or topic}”" if v.kind == Op.TELL.value else f"问{t}：“{v.utterance or topic}”"
    if v.social is not None:
        return SOCIAL_VERBS[v.social].format(t=t) if v.kind == Op.TELL.value else f"向{t}问了几句"
    return f"对{t}低声说了些什么" if v.kind == Op.TELL.value else f"向{t}低声问了些什么"


def _verb(v: PerceivedEvent, names: Names, viewer: str | None, me: str) -> str:
    t, o = _n(names, v.target, viewer, me), _n(names, v.obj, viewer, me)
    topic = render_fact(v.topic, names, viewer, me) if v.topic else "一些话"
    op = Op(v.kind)
    table = {
        Op.MOVE: (f"来到{t}" if v.actor == viewer and v.outcome == Outcome.SUCCESS    # 自己只写到达，步态留给原话
                  # 旁人来到眼前：看得见他从哪道门（认得的）过来——“钟灵经断崖来到剑湖畔”，人是怎么来的一目了然
                  else (f"经{o}来到{t}" if v.obj in names else f"来到{t}")
                  if v.outcome == Outcome.SUCCESS and v.target and v.place == v.target
                  else f"走向{t}" if v.target else f"走向{o}" if v.obj else "想要离开"),  # 旁人只看见他走向哪道门
        Op.TAKE: f"拿起{t}",
        Op.PUT: f"把{o}放在{t}",
        Op.GIVE: f"把{o}交给{t}",
        Op.UNLOCK: f"用{o}开{t}的锁",
        Op.LOCK: f"用{o}锁上{t}",
        Op.INSPECT: f"伏地磕头，顺势细看{t}" if v.social == Social.SUBMIT else f"仔细查看{t}",   # 原著路线的叩拜：磕头照写
        Op.TELL: _speech(v, t, topic),
        Op.ASK: _speech(v, t, topic),
        Op.WAIT: v.utterance or "静静等待",          # 带姿态的等待：姿态是不带主语的动作短语（“坐下来喝了口茶”）
        Op.ATTACK: f"猛地向{t}出手" if v.kind == Op.ATTACK.value and v.target else "出手",
        Op.STUDY: f"埋头研读{t}",
        Op.USE: f"服下{o}" if v.target == v.actor else f"把{o}用在{t}身上",
    }
    return table[op]


def render_event(v: PerceivedEvent, names: Names, viewer: str | None = None, me: str = "我") -> str:
    if v.kind == "noise":
        return f"{_n(names, v.place, viewer, me)}那边传来一阵响动"
    text = f"{_n(names, v.actor, viewer, me)}{_verb(v, names, viewer, me)}"
    if v.outcome == Outcome.SUCCESS and v.reason in SUCCESS_NOTES:
        text += SUCCESS_NOTES[v.reason]
    if v.outcome == Outcome.FAILURE:
        text += f"，但没有成功（{REASONS.get(v.reason, v.reason)}）" if v.reason else "，但没有成功"
    elif v.outcome == Outcome.REJECTED:
        text = f"{_n(names, v.actor, viewer, me)}想要{_verb(v, names, viewer, me)}，但这行不通"
    return text


def render_percept(p: Percept, names: Names, viewer: str, me: str = "我", familiar: Container[str] = ()) -> str:
    """角色视角的一句话（me 是观察者的自称：记忆里是“我”，对玩家叙述时是“你”）。SCENE 渲染为所见清单。
    familiar：观察者此前已知下落的东西——再翻出来不算“发现”。"""
    if p.modality == Modality.SCENE:
        return _scene(p, names, viewer, me)
    if p.event is None:
        return "；".join(render_fact(f, names, viewer, me) for f in p.facts)
    text = render_experience(p.modality, p.event, names, viewer, me)
    after = consequences(p, names, viewer, me, familiar)
    return text + ("——" + "，".join(after) if after else "")


def _scene(p: Percept, names: Names, viewer: str, me: str) -> str:
    """所见清单按所在处归拢：“干光豪、葛光佩都在剑湖宫大殿；长剑在兵器架上”，而不是一人一句“某某在剑湖宫大殿”。
    自己在哪、身上带着什么不必每回念叨；归拢只改措辞，每条“在”的事实照旧都在。"""
    groups: dict[str, list[str]] = {}
    here = next((str(f.prop.value) for f in p.facts
                 if f.holds and f.prop.predicate == Rel.AT.value and f.prop.subject == viewer), None)
    ends: dict[str, list[str]] = {}
    for f in p.facts:
        prop = f.prop
        if f.holds and prop.predicate == Rel.AT.value and viewer not in (prop.subject, prop.value):
            groups.setdefault(str(prop.value), []).append(_n(names, prop.subject, viewer, me))
        elif f.holds and prop.predicate == Rel.CONNECTS.value:
            ends.setdefault(prop.subject, []).append(str(prop.value))
    # 看得见的通道与它连着的地方（“回廊连着剑湖宫后院”）：四下看看，路也是看得见的东西；不说能不能走（断崖只能下，试过才知道）
    ways = [f"{_n(names, d, viewer, me)}连着{_n(names, o, viewer, me)}" for d, places in ends.items()
            if here in places for o in places if o != here]
    rows = [f"{'、'.join(who)}{'都' if len(who) > 1 else ''}在{_where(names, where, viewer, me)}"
            for where, who in groups.items()]
    if not rows and not ways:
        return "四下空无一物"
    return "；".join(rows + ways)


def consequences(p: Percept, names: Names, viewer: str, me: str, familiar: Container[str] = ()) -> list[str]:
    """事件带来的看得见的后果：谁受伤中毒被制、学成了什么、发现了什么暗道与藏匿之物（familiar 里的——此前已知下落的——不再算发现，只说“仍在”）。"""
    out: list[str] = []
    hidden = {f.prop.subject for f in p.facts if f.prop.is_attr and f.prop.attr_key == "hidden" and f.holds}
    found: dict[str, list[str]] = {}          # 藏匿之物按藏处归拢：“发现蒲团里藏着北冥神功帛卷、凌波微步帛卷”
    still: dict[str, list[str]] = {}          # 此前已知下落的：不算发现，只说“仍在”
    for f in p.facts:
        prop = f.prop
        if prop.is_attr and prop.attr_key in _NOTABLE and f.holds:
            out.append(render_fact(f, names, viewer, me))
        elif not prop.is_attr and f.holds and prop.subject in hidden:
            if prop.predicate == Rel.CONNECTS.value and prop.value != (p.event.place if p.event else None):
                out.append(f"发现一处暗道：{_n(names, prop.subject, viewer, me)}通往{_n(names, str(prop.value), viewer, me)}")
            elif prop.predicate == Rel.AT.value:
                (still if prop.subject in familiar else found).setdefault(str(prop.value), []).append(
                    _n(names, prop.subject, viewer, me))
    for where, things in found.items():
        sk = names.get(where)
        spot = _where(names, where, viewer, me) + ("里" if sk is not None and sk.kind == Kind.PLACE else "")
        out.append(f"发现{spot}藏着{'、'.join(things)}")
    for where, things in still.items():
        out.append(f"{'、'.join(things)}仍在{_where(names, where, viewer, me)}")
    return out


def render_experience(modality: Modality, event: PerceivedEvent, names: Names, viewer: str, me: str = "我") -> str:
    """以某种感官经历一个事件：“听到……”“看见……”。"""
    prefix = {Modality.SOUND: "听到", Modality.SIGHT: "看见", Modality.SPEECH: "听见"}.get(modality, "")
    return prefix + render_event(event, names, viewer, me)

