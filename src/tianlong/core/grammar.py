"""
[INPUT]: 依赖 core/schema 的 Op / Kind / OP_SIGNATURES，core/propositions 的 Fact
[OUTPUT]: 对外提供 signature_error()：检查一个行动的“语法/类型”是否合法（命题分“接受”与“必须”：言语不带命题也合法）
[POS]: core 的行动语法；kernel 用真实实体种类调用它拒绝畸形意图，cognition 用角色已知的实体种类调用它生成候选——同一把尺子，两种视角
       检查 REQUEST_ITEM 必须有已知种类的受益人；只有 TELL/GIVE 接受请求回应编号。
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable

from tianlong.core.propositions import Fact
from tianlong.core.schema import OP_SIGNATURES, Kind, Op

# ============================================================
#  这里只检查“能不能这样说”，不检查“能不能做成”：
#  “用钥匙开门”语法合法；钥匙配不配这扇门是 kernel 的事实裁定，
#  不能因为角色不知道的真相而提前屏蔽行动（那本身就是信息泄露）
# ============================================================

KindOf = Callable[[str], Kind | None]


def signature_error(
    op: Op, kind_of: KindOf, target: str | None, obj: str | None, topic: Fact | None,
    beneficiary: str | None = None, request_ref: str | None = None,
) -> str | None:
    sig = OP_SIGNATURES[op]
    for role, value, allowed in (("target", target, sig.target), ("obj", obj, sig.obj),
                                 ("beneficiary", beneficiary, sig.beneficiary)):
        if allowed is None:
            if value is not None:
                return f"{op} 不接受 {role}"
            continue
        if value is None:
            return f"{op} 缺少 {role}"
        kind = kind_of(value)
        if kind is None:
            return f"{role} 指向未知实体 {value}"
        if kind not in allowed:
            return f"{op} 的 {role} 不能是 {kind}"
    if sig.needs_topic and topic is None:
        return f"{op} 缺少语义内容"
    if not sig.topic and topic is not None:
        return f"{op} 不接受语义内容"
    if request_ref is not None and op not in (Op.TELL, Op.GIVE):
        return f"{op} 不接受请求回应编号"
    return None
