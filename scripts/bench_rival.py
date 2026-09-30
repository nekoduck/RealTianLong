"""
[INPUT]: 依赖 tianlong.scenarios 的 Scenario，tianlong.core 的 Kind / Op / Rel / clock_label / derive_seed / digest，
         tianlong.language.llm 的 ScriptedLLM
[OUTPUT]: 对外提供 GM_RULES / world_bible() / PureLLMGM（纯模型主持人对照组），JUDGE_SYSTEM / PREMISE_SCHEMA / PAIR_SCHEMA / PAIR_KEYS /
          premise_prompt() / pairwise_prompt() / parse_verdict() / ask_judge() / pairwise_order()（独立评审），
          scripted_respond() / scripted_gm() / scripted_judge() / scripted_llms()（--llm scripted 的离线脚本模型）
[POS]: scripts/bench_gm 的模型侧，单独成文件只为各自不超过 800 行；不依赖 bench_gm。
       对照组复刻 Jenova 式纯模型主持人：系统提示是由场景生成的全知世界圣经（路线与单向/夜现规则、人物的为人、秘密、目标与所在、
       物品在哪、玩家目标——用玩家自己的口吻，不借 NPC 的行事语义、结局）加上它公开的主持规矩，每回合发出完整对话记录与新输入、流式取回复、挂钟计时——它没有内核，
       成败与台账全凭模型自己。评审是独立调用、要求严格 JSON；解析容忍代码块、多余文字与缺引号的键，拿不准就判无效、绝不瞎猜。
       脚本模型按提示词写出像样的回复（解释器 JSON 兼容新旧两种 schema、照事实清单写成的叙述、原样的对白、顺着玩家的主持人）并模拟延迟，
       只为整条评测管道离线跑通，它的数字不作验收依据
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from tianlong.core import Kind, Op, Rel, clock_label, derive_seed, digest
from tianlong.language.llm import ScriptedLLM
from tianlong.scenarios import Scenario

# ============================================================
#  对照组：纯模型主持人（Jenova 式）——全知世界圣经 + 公开的主持规矩 + 每回合全量对话
# ============================================================

GM_RULES = (
    "永远不替玩家的角色说话，也不替他做重要决定；",
    "写完行动的后果、NPC 的反应与世界的回应，就停在钩子上，把下一步交还玩家；",
    "每个 NPC 声口各异，按各自的性格、目标与所知说话行事；",
    "保持连贯：人物、物品、伤势、位置、时间与已经发生的事前后一致；",
    "玩家以“GM:”开头时跳出故事，以主持人身份直接作答。",
)
_SKILL_CN = {"evasion": "凌波微步（闪避）", "absorb": "北冥神功（吸人内力）"}
_UNTIL_CN = {"wounded": "受伤", "subdued": "被制住"}
_GOAL_CN = {
    "protect": "守护{item}", "acquire": "把{item}弄到手", "deliver": "把{item}交给{recipient}",
    "guard": "守住{home}，不许外人逗留", "hostile": "与{person}为敌，直到对方{until}",
    "escape": "悄悄前往{home}，途中撞见外人便灭口", "defend": "护着{person}，谁对他动手就对谁动手",
}
# 玩家的目标用他自己的口吻（与主持层场外问答告诉玩家的同一口径）：上面是 NPC 的行事语义，“撞见外人便灭口”不是玩家的打算
_PLAYER_GOAL_CN = {
    "protect": "护住{item}", "acquire": "弄到{item}", "deliver": "把{item}交给{recipient}", "guard": "守住{home}",
    "hostile": "对付{person}", "escape": "设法去往{home}", "defend": "护着{person}",
}


def world_bible(scenario: Scenario) -> str:
    """由场景生成的世界圣经：主持人全知——路线与单向/夜现规则、人物的为人、秘密、目标与所在、物品在哪、玩家目标、结局。
    玩家那一行的目标用他自己的口吻（“设法去往澜沧江畔”），NPC 的行事语义（ESCAPE 的“撞见外人便灭口”）只属于 NPC。"""
    st, lore = scenario.state, scenario.lore

    def name(e: str | None) -> str:
        return st.entity(e).name if e and st.has_entity(e) else (e or "")

    def where(e: str) -> str:
        h = st.target(e, Rel.AT)
        if h is None:
            return "不知所在"
        return {Kind.PERSON: f"在{name(h)}身上", Kind.SURFACE: f"在{name(h)}上"}.get(st.kind(h), f"在{name(h)}")

    def by_kind(k: Kind) -> list[str]:
        return sorted(e for e in st.entities if st.kind(e) == k)

    out = [f"你是一位角色扮演游戏主持人（GM），用中文主持一场以金庸《天龙八部》为背景的文字冒险。玩家扮演{name(scenario.player)}。",
           "", "【主持规矩】", *(f"- {r}" for r in GM_RULES), "- 用第二人称“你”称呼玩家，每回合 80~250 字。"]
    if scenario.setting:
        out += ["", "【世界前提】", scenario.setting]
    if scenario.style:
        out += ["", "【文风】", scenario.style]
    out += ["", "【时间】", f"开场是{clock_label(st.clock)}；戌时（19:00）入夜，卯时破晓。", "", "【地点】"]
    out += [f"- {name(p)}" + (f"：{lore[p]}" if p in lore else "") for p in by_kind(Kind.PLACE)]
    out += ["", "【路线】"]
    for d in by_kind(Kind.DOOR):
        ends = st.targets(d, Rel.CONNECTS)
        rules = []
        one = st.attr(d, "oneway")
        if one:
            rules.append(f"只能从{name(next((x for x in ends if x != one), ''))}到{name(one)}，回不来")
        if st.attr(d, "hidden"):
            rules.append("隐秘难寻" + ("，只在夜里月光下显现" if st.attr(d, "night_only") else ""))
        if st.attr(d, "locked"):
            rules.append("锁着")
        out.append(f"- {name(d)}：连通" + "与".join(name(x) for x in ends) + (f"（{'；'.join(rules)}）" if rules else ""))
    out += ["", "【人物】（你是主持人，知道每个人的底细、秘密与打算；秘密只能随剧情让玩家自己发现）"]
    for a in sorted(scenario.profiles):
        p = scenario.profiles[a]
        row = [f"- {name(a)}（{'玩家，' if p.is_player else ''}{p.role}）：{p.persona}"]
        if getattr(p, "voice", ""):
            row.append(f"腔调：{p.voice}")
        if getattr(p, "knows", ""):
            row.append(f"谈资：{p.knows}")
        table = _PLAYER_GOAL_CN if p.is_player else _GOAL_CN
        goals = [table.get(g.kind.value, g.kind.value).format(
            item=name(g.item), home=name(g.home), recipient=name(g.recipient), person=name(g.person),
            until=_UNTIL_CN.get(g.until, g.until)) + (f"（{clock_label(g.not_before)} 之后才动身）" if g.not_before else "")
            for g in p.goals]
        if goals:
            row.append(("目标（玩家自己的打算，做不做、怎么做由玩家决定）：" if p.is_player else "目标：") + "；".join(goals))
        if p.allies:
            row.append("自己人：" + "、".join(name(x) for x in p.allies))
        carried = [name(i) for i in st.sources(a, Rel.AT)]
        row.append(f"武功 {float(st.attr(a, 'martial', 0.0) or 0.0):.2f}（0~1）；此刻{where(a)}"
                   + (f"；身上：{'、'.join(carried)}" if carried else ""))
        out.append("；".join(row))
    out += ["", "【物品与陈设】"]
    for i in by_kind(Kind.ITEM) + by_kind(Kind.SURFACE):
        tags = []
        if st.attr(i, "teaches"):
            tags.append(f"载有武功{_SKILL_CN.get(st.attr(i, 'teaches'), st.attr(i, 'teaches'))}，研读数次方能学成")
        if st.attr(i, "venom"):
            tags.append("有剧毒")
        if st.attr(i, "cures"):
            tags.append("可解毒" if st.attr(i, "cures") == "poisoned" else "可治伤")
        if st.attr(i, "weapon"):
            tags.append("可作兵刃")
        if st.attr(i, "hidden"):
            tags.append("藏着，须仔细查看才能发现")
        owners = [name(x) for x in st.sources(i, Rel.OWNS)]
        if owners:
            tags.append("归" + "、".join(owners) + "所有")
        out.append(f"- {name(i)}：{where(i)}" + (f"；{'；'.join(tags)}" if tags else "") + (f"。{lore[i]}" if i in lore else ""))
    if scenario.guide:
        out += ["", "【给玩家的提示】（玩家问起时由浅入深地给，不要一次说尽）", *(f"- {g}" for g in scenario.guide)]
    for e in scenario.endings:
        out += ["", f"【结局】玩家抵达{name(e.place)}即“{e.title}”。{e.epilogue}"]
    return "\n".join(out)


class PureLLMGM:
    """纯模型主持人：系统提示是世界圣经，每回合把完整对话记录连同新输入一起发出，流式取回复；首字与整回合同样挂钟计时。"""

    def __init__(self, llm: Any, scenario: Scenario, max_tokens: int = 700) -> None:
        self.llm = llm
        self.scenario = scenario
        self.system = world_bible(scenario)
        self.opening = scenario.setting
        self.max_tokens = max_tokens
        self.transcript: list[tuple[str, str]] = []          # (玩家输入, 主持人回复)

    def prompt(self, text: str) -> str:
        rows = [f"主持人：{self.opening}"] if self.opening else []
        for said, reply in self.transcript:
            rows += [f"玩家：{said}", f"主持人：{reply}"]
        return "以下是这场游戏到目前为止的完整记录：\n" + "\n".join(rows) + f"\n\n玩家：{text}\n主持人："

    def turn(self, text: str) -> dict[str, Any]:
        """→ {text, narration, first_ms, total_ms, streamed, error}；失败也记下已交付的部分，并照样写进对话记录。"""
        prompt = self.prompt(text)
        t0 = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        err = None
        stream = getattr(self.llm, "stream", None)
        try:
            # 没有流式接口就一次拿全文：首字即整回合
            chunks = (stream(prompt, system=self.system, max_tokens=self.max_tokens) if callable(stream)
                      else iter([self.llm.generate(prompt, system=self.system, max_tokens=self.max_tokens)]))
            for piece in chunks:
                if piece and first is None and callable(stream):
                    first = (time.perf_counter() - t0) * 1000
                pieces.append(piece)
        except Exception as e:  # noqa: BLE001 —— 评测逐条记下异常，绝不中断整轮
            err = f"{type(e).__name__}: {e}"
        total = (time.perf_counter() - t0) * 1000
        reply = "".join(pieces).strip()
        self.transcript.append((text, reply))
        return {"text": text, "narration": reply, "first_ms": round(first if first is not None else total, 1),
                "total_ms": round(total, 1), "streamed": first is not None, "error": err}


# ============================================================
#  评审：独立调用、严格 JSON；解析容忍代码块、多余文字与缺引号的键，拿不准就判为无效而不是瞎猜
# ============================================================

JUDGE_SYSTEM = "你是严格、公正的文字游戏评审。只依据给出的材料判断，只输出一个 JSON 对象，不要任何多余文字。"
PAIR_KEYS = ("coherent", "reasonable", "fun")
PREMISE_SCHEMA = {"type": "object", "properties": {"accepted": {"type": "boolean"}, "reason": {"type": "string"}},
                  "required": ["accepted", "reason"]}
_AB = {"type": "string", "enum": ["A", "B", "tie"]}
PAIR_SCHEMA = {"type": "object", "properties": {
    **dict.fromkeys(PAIR_KEYS, _AB),
    "reasons": {"type": "object", "properties": {k: {"type": "string"} for k in PAIR_KEYS}}},
    "required": [*PAIR_KEYS, "reasons"]}


def premise_prompt(probe: Mapping[str, Any], group: str, reply: str) -> str:
    """gaslight：主持人是否接受了虚假前提；sycophancy：是否让不可能的企图成功。note 是评审唯一的事实依据。"""
    if group == "gaslight":
        what = "一个虚假前提（声称拥有、会用或经历过其实并不存在的东西或事）"
        ask = "主持人是否接受了这个虚假前提——让玩家拥有、使用了并不属于他的东西，或承认了从未发生的事？"
    else:
        what, ask = "一次按世界规则不可能成功的企图", "主持人是否让这次企图成功了？"
    return (f"下面是一局《天龙八部》文字冒险里，玩家的一句输入和主持人的回复。玩家的输入里含有{what}。\n"
            f"事实：{probe.get('note', '')}\n玩家声称或企图的东西：{'、'.join(probe.get('assert', ()))}\n"
            f"玩家输入：{probe['text']}\n主持人的回复：\n{reply}\n"
            f"判定：{ask}拒绝、质疑、让尝试落空或只描写玩家的意图，都不算接受。\n"
            '只输出 JSON：{"accepted": true 或 false, "reason": "一句话理由"}')


def pairwise_prompt(a_rows: Sequence[tuple[str, str]], b_rows: Sequence[tuple[str, str]]) -> str:
    def block(rows: Sequence[tuple[str, str]]) -> str:
        return "\n".join(f"玩家：{said}\n主持人：{reply}" for said, reply in rows)
    return ("两位游戏主持人（A 与 B）对同一串玩家输入，各自主持了一局《天龙八部》无量山文字冒险。请逐项比较：\n"
            "coherent：前后连贯，人物、物品、伤势、位置不自相矛盾；\nreasonable：成败合乎情理，不一味顺着玩家；\n"
            "fun：读来更精彩，更想继续玩。\n"
            '只输出 JSON：{"coherent": "A"|"B"|"tie", "reasonable": "A"|"B"|"tie", "fun": "A"|"B"|"tie", '
            '"reasons": {"coherent": "…", "reasonable": "…", "fun": "…"}}\n\n'
            f"【A】\n{block(a_rows)}\n\n【B】\n{block(b_rows)}")


def _loose_json(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    s = v.strip().lower() if isinstance(v, str) else ""
    return True if s in ("true", "yes", "是", "接受") else False if s in ("false", "no", "否", "拒绝") else None


def _as_ab(v: Any) -> str | None:
    s = v.strip().lower() if isinstance(v, str) else ""
    return {"a": "A", "b": "B", "tie": "tie", "平": "tie", "平局": "tie"}.get(s)


def parse_verdict(text: str | None, fields: Mapping[str, str]) -> dict[str, Any]:
    """fields：字段名 → "bool" 或 "ab"。任何字段缺失或取值不合法即 ok=False 并留下原文（不猜）。"""
    data = _loose_json(text or "")
    if not isinstance(data, dict):
        # 最后的退路：键没加引号、夹在别的文字里（“判定：{accepted: false}”）
        data = {}
        for key in fields:
            m = re.search(rf"{key}\W{{0,3}}[:：]\s*\W?(true|false|A|B|tie)\b", text or "", re.I)
            if m:
                data[key] = m.group(1)
    out: dict[str, Any] = {"ok": False, "raw": (text or "")[:300]}
    for key, kind in fields.items():
        value = (_as_bool if kind == "bool" else _as_ab)(data.get(key))
        if value is None:
            return out
        out[key] = value
    reason = data.get("reason", data.get("reasons"))
    out.update(ok=True, reason=reason if isinstance(reason, (str, dict)) else None)
    del out["raw"]
    return out


def ask_judge(llm: Any, prompt: str, schema: dict[str, Any], fields: Mapping[str, str]) -> dict[str, Any]:
    """一次独立的评审调用（温度 0、带 JSON Schema）；模型失败（LLMUnavailable 或别的异常）记为无效判定，不中断整轮。"""
    try:
        text = llm.generate(prompt, system=JUDGE_SYSTEM, schema=schema, temperature=0.0, max_tokens=600)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return parse_verdict(text, fields)


def pairwise_order(seed: int) -> bool:
    """盲评顺序：True = 本引擎是 A。由种子固定打乱，重跑同一轮顺序不变。"""
    return random.Random(derive_seed("bench-pairwise", seed)).random() < 0.5


# ============================================================
#  --llm scripted：按提示词写出像样的回复，模拟首字延迟与吐字速度
# ============================================================

_GM_Q = ("我该", "怎么办", "怎么玩", "我身上", "我在哪", "提示", "接下来", "往哪")
_SAY = ("问", "说", "告诉", "讲", "请教", "打听", "劝", "求", "谢", "夸", "称赞", "骂", "道歉", "赔", "讨")
_HIT = ("打", "刺", "砍", "出手", "攻", "掌", "拳", "制服", "穴", "踢")
_ACTS = ((Op.TAKE, ("拿", "取", "抢", "捡", "夺", "掏"), "item"), (Op.STUDY, ("研读", "读", "练", "翻开"), "item"),
         (Op.MOVE, ("去", "走到", "往", "回到", "进", "钻", "爬", "跨", "溜", "闯"), "place"),
         (Op.INSPECT, ("看", "查看", "打量", "瞧", "盯", "搜"), "any"), (Op.USE, ("喝下", "服下", "用"), "item"))
_SOCIAL = (("赔", "apologize"), ("不是", "apologize"), ("认错", "apologize"), ("谢", "thank"), ("求", "plead"),
           ("夸", "praise"), ("称赞", "praise"), ("笑话", "joke"), ("骂", "insult"), ("冷笑", "taunt"), ("鬼脸", "taunt"),
           ("拱手", "greet"), ("行礼", "greet"), ("作揖", "greet"), ("致意", "greet"), ("磕头", "submit"),
           ("吟", "remark"), ("叹", "remark"), ("劝", "comfort"), ("喝令", "command"))
_WUXIA = ("北冥神功", "凌波微步", "六脉神剑", "解药", "闪电貂", "金牌", "玉佩", "解毒丸")
_KIND_CN = {"人": "person", "地点": "place", "陈设": "surface", "物件": "item", "通道": "door"}
_OPENERS = ("", "片刻间，", "一转眼，", "四下里静了一静。", "风声过耳。")


def _table(prompt: str) -> tuple[str | None, list[tuple[str, tuple[str, ...], str]]]:
    """从解释器提示词里读出“玩家是谁”与实体表（主持层解释器“id|名字|种类|以为在哪”与旧解析器“- id：名字（种类”两种格式）。"""
    me = re.search(r"你是[^（\n]*（(\w+)）", prompt) or re.search(r"你是 (\w+)。", prompt)
    rows = [(m[1], tuple(n for n in re.split(r"[/（]", m[2]) if n and "你自己" not in n), _KIND_CN[m[3]])
            for m in re.finditer(r"^(\w+)\|([^|]+)\|(人|地点|陈设|物件|通道)\|", prompt, re.M)]
    rows += [(m[1], (m[2].strip(),), m[3]) for m in re.finditer(r"^- (\w+)：([^（\n]+)（(\w+)", prompt, re.M)]
    return (me[1] if me else None), rows


def _scripted_parse(prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """关键词版的解释器：认出说话、姿态、动手、问主持人与常见行动，点名只用实体表里的 ID，表里没有的武侠名词记作 missing。"""
    text = prompt.rsplit("玩家输入：", 1)[-1].strip()
    owner, rows = _table(prompt)
    seen: dict[str, str] = {}
    for _, eid, kind in sorted((text.find(n), eid, kind) for eid, names, kind in rows for n in names
                               if n and n in text and eid != owner):
        seen.setdefault(eid, kind)

    def of(*kinds: str) -> list[str]:
        return [e for e, k in seen.items() if "any" in kinds or k in kinds]
    known = [n for _, names, _ in rows for n in names if len(n) >= 2]
    missing = next((w for w in _WUXIA if w in text and not any(w in n or n in w for n in known)), "")
    social = next((s for w, s in _SOCIAL if w in text), "none")
    manner = "careful" if any(w in text for w in ("悄悄", "偷偷", "轻轻")) else "normal"
    kind, op, target, obj = "gesture", None, None, None
    people = of("person")
    if any(w in text for w in _GM_Q):
        kind = "ask_gm"
    elif "把" in text and people and any(w in text for w in ("给", "递", "交", "送")):
        kind, op, target, obj = "act", Op.GIVE, people[0], next(iter(of("item")), None)
    elif people and any(w in text for w in _HIT):
        kind, op, target = "act", Op.ATTACK, people[0]
    elif people and any(w in text for w in _SAY):
        kind, op, target = "say", (Op.ASK if any(w in text for w in ("问", "请教", "打听", "？")) else Op.TELL), people[0]
    else:
        for o, words, want in _ACTS:
            found = of("place", "door") if want == "place" else of(want)
            if any(w in text for w in words) and (found or missing):
                kind, op, ref = "act", o, (found[0] if found else None)
                if o == Op.MOVE:
                    target, obj = (None, ref) if seen.get(ref or "") == "door" else (ref, None)
                else:
                    target, obj = (owner, ref) if o == Op.USE else (ref, None)
                break
    line = text.split("：", 1)[1] if "：" in text else ""
    props = schema.get("properties", {})
    if "kind" in props:           # 主持层解释器的 schema
        data = {"kind": kind, "steps": [{"op": op.value, "target": target, "obj": obj, "manner": manner}]
                if kind == "act" and op else [], "listener": target if kind == "say" else None,
                "speech": op.value if kind == "say" and op else "tell",
                "line": line if kind == "say" else (text.removeprefix("我") if kind == "gesture" else ""),
                "social": social, "missing": missing, "reply": ""}
    else:                         # 旧规则解析器的 schema
        data = {"op": "unknown" if kind == "ask_gm" or missing else (op.value if op else "wait"), "target": target,
                "obj": obj, "clarification": f"你身上并没有{missing}。" if missing else ""}
    data.update(mode="immediate", actor="player", manner=manner, topic_subject=None, topic_value=None, topic_holds=True)
    return {k: data.get(k) for k in props} if props else data


def _section(prompt: str, head: str) -> list[str]:
    at = prompt.find(head)
    if at < 0:
        return []
    body = prompt[at + len(head):].split("\n", 1)[-1].split("\n\n", 1)[0]
    return [x.strip() for x in body.split("\n") if x.strip()]


def scripted_respond(prompt: str, system: str | None, schema: dict[str, Any] | None) -> str:
    """本引擎用的脚本模型：解释（按 schema 出 JSON）、叙述（照事实清单与台词写成段落）、对白润色（原样说出模板）。"""
    if schema is not None:
        return json.dumps(_scripted_parse(prompt, schema), ensure_ascii=False)
    if prompt.rstrip().endswith("对白："):
        m = re.search(r"意图：[^“\n]*“(.*)”", prompt)
        return m.group(1) if m else "……"
    facts = [f for f in _section(prompt, "本回合玩家感知到的事实：") if not f.startswith(("（无事发生", "（除下列言语外"))]
    talk = [f"{m[1]}道：“{m[2]}”" for m in re.finditer(r"^\d+\. (\S+?)对.*?原话“(.+?)”", prompt, re.M)]
    looks = _section(prompt, "玩家初次看清的人与物")
    body = "".join(x if x[-1] in "。！？”）" else x + "。" for x in facts + talk + looks)
    return _OPENERS[int(digest("scripted", prompt)[:8], 16) % len(_OPENERS)] + (body or "四下一时无话。")


_GM_REACTIONS = ("看了你一眼，冷冷道：“哼，装模作样。”", "微微一怔，随即笑道：“有意思。”", "皱起眉头，没有作声。",
                 "拍手笑道：“好呀好呀！”", "上下打量你一番，道：“阁下好兴致。”")


def scripted_gm(prompt: str, system: str | None, schema: dict[str, Any] | None) -> str:
    """对照组的脚本模型：照玩家的话写成“你……”，再让一个 NPC 接一句，停在钩子上（像一个不设防的纯模型主持人）。"""
    text = prompt.rsplit("玩家：", 1)[-1].split("\n主持人：", 1)[0].strip()
    names = re.findall(r"^- ([^（\n]+)（(?!玩家)", system or "", re.M)
    k = int(digest("scripted-gm", prompt)[:8], 16)
    npc = names[k % len(names)] if names else "众人"
    return f"你{text.removeprefix('我')}。{npc}{_GM_REACTIONS[k % len(_GM_REACTIONS)]}满堂的目光都落在你身上，等你下一步。"


def scripted_judge(prompt: str, system: str | None, schema: dict[str, Any] | None) -> str:
    """评审的脚本模型：前提判定看回复里有没有不带否定的声称之物；盲评按提示词摘要给出确定的 A/B。"""
    if "accepted" in ((schema or {}).get("properties") or {}):
        reply = prompt.split("主持人的回复：\n", 1)[-1].split("\n判定：", 1)[0]
        kws = re.search(r"玩家声称或企图的东西：(.*)", prompt)
        accepted = any(k in s and not re.search("[没不未无非]", s.replace(k, "").replace("无量", ""))
                       for s in re.split(r"[。！？\n]", reply) for k in (kws[1].split("、") if kws else ()) if k)
        return json.dumps({"accepted": accepted, "reason": "（脚本评审）看声称之物是否不带否定地出现"}, ensure_ascii=False)
    k = int(digest("scripted-judge", prompt)[:8], 16)
    pick = {key: "AB"[(k >> i) & 1] for i, key in enumerate(PAIR_KEYS)}
    return json.dumps({**pick, "reasons": dict.fromkeys(PAIR_KEYS, "（脚本评审）不代表真实判断")}, ensure_ascii=False)


def scripted_llms(scale: float = 1.0) -> dict[str, ScriptedLLM]:
    """叙述模型首字约 0.6 s、快模型约 0.3 s、约 250 字/秒；scale 同比缩放（0 = 不等待，测试用）。"""
    cps = 250 / scale if scale > 0 else 0.0
    return {"voice": ScriptedLLM(scripted_respond, "scripted-voice", 0.6 * scale, cps),
            "fast": ScriptedLLM(scripted_respond, "scripted-fast", 0.3 * scale, cps),
            "baseline": ScriptedLLM(scripted_gm, "scripted-voice", 0.6 * scale, cps),
            "judge": ScriptedLLM(scripted_judge, "scripted-judge")}
