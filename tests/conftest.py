"""
[INPUT]: 依赖 tianlong.scenarios / persistence / runtime.authority / kernel
[OUTPUT]: 对外提供 pytest fixtures：warehouse、authority、intent 工厂、fold 辅助
[POS]: tests 的共享夹具；所有测试从同一个标准场景出发，保证用例之间可比较；
       关掉 MKL/OpenMP 的动态线程（负载一高线程数就变，逐位相同的断言会偶发失败）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools
import os

# MKL 默认按机器负载动态减少线程（MKL_DYNAMIC），线程数一变，CPU 上的归约就差几个 ulp：
# “断点续训与不中断逐位相同”这类断言在满载的整套测试里会偶发失败。须在 torch 首次导入之前关掉
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import pytest

from tianlong.core import Fact, Intent, Manner, Op
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_warehouse

_counter = itertools.count()


@pytest.fixture
def warehouse():
    return build_warehouse()


@pytest.fixture
def authority(warehouse):
    return WorldAuthority.found(InMemoryWorldStore(), warehouse)


def make_intent(actor: str, op: Op, target: str | None = None, obj: str | None = None, *,
                based_on: int, manner: Manner = Manner.NORMAL, topic: Fact | None = None,
                intent_id: str | None = None) -> Intent:
    return Intent(intent_id or f"it{next(_counter)}", actor, op, target, obj, manner, topic, based_on)


@pytest.fixture
def act(authority):
    """在权威写入器上让若干角色同时行动：act(("player", Op.TAKE, "key"), ...)。"""

    def _act(*specs, **kw):
        v = authority.head().version
        intents = [make_intent(*spec, based_on=v, **kw) for spec in specs]
        return authority.settle(intents)

    return _act
