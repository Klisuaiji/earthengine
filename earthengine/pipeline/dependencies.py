"""阶段依赖图 (P0-1 / 5 节)。

为所有阶段声明有向无环依赖图；指定目标阶段时默认执行其前置依赖闭包。
若请求跳过依赖的显式模式缺少字段，抛出可读错误；检测循环依赖并指出
缺哪个阶段/字段、哪个目标依赖它、如何修复。
"""

from __future__ import annotations

# 阶段 → 直接前置依赖（DAG）
# 顺序与 pipeline/stages._STAGE_ORDER 对齐
_STAGE_DEPS = {
    "planet": [],
    "tectonics": ["planet"],
    "crust": ["tectonics"],
    "ocean": ["crust", "tectonics"],
    "terrain": ["ocean", "tectonics", "crust"],
    "atmosphere": ["terrain", "planet"],
    "climate": ["atmosphere", "terrain"],
    "hydrology": ["terrain", "climate"],
    "geology": ["terrain", "climate"],
    "conditioning": ["terrain", "tectonics", "ocean", "climate"],
    "enhance": ["conditioning", "terrain", "ocean"],
    "vegetation": ["terrain", "climate"],
    "civilization": ["vegetation", "climate", "hydrology"],
    "render": ["planet", "tectonics", "crust", "ocean", "terrain",
               "atmosphere", "climate", "hydrology", "geology",
               "conditioning", "enhance", "vegetation", "civilization"],
}

# 别名 → 规范名
_ALIASES = {
    "tectonic": "tectonics", "ocean": "ocean", "terrain": "terrain",
    "climate": "climate", "hydrology": "hydrology", "crust": "crust",
    "vegetation": "vegetation", "civilization": "civilization",
}


def canonical(stage):
    """别名归一化；未知阶段返回原串（调用方负责报错）。"""
    return _ALIASES.get(stage, stage)


def dependency_closure(stages) -> list:
    """返回按拓扑顺序排列的、包含全部前置闭包且去重的阶段列表。

    ``stages``：规范阶段名列表（未归一化由调用方负责）。
    返回顺序满足：任一阶段总在其依赖之后。
    """
    seen = set()
    order = []

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for dep in _STAGE_DEPS.get(node, []):
            visit(dep)
        order.append(node)

    for s in stages:
        if s not in _STAGE_DEPS:
            raise StageDependencyError(
                f"未知阶段 '{s}'。可用阶段：{list(_STAGE_DEPS.keys())}。")
        visit(s)
    return order


def detect_cycle() -> list:
    """检测依赖图是否有环；有则返回环路径，无则返回 []。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {k: WHITE for k in _STAGE_DEPS}
    stack = []

    def dfs(u):
        color[u] = GRAY
        stack.append(u)
        for v in _STAGE_DEPS[u]:
            if color[v] == GRAY:
                return stack[stack.index(v):] + [v]
            if color[v] == WHITE:
                cyc = dfs(v)
                if cyc:
                    return cyc
        stack.pop()
        color[u] = BLACK
        return None

    for k in _STAGE_DEPS:
        if color[k] == WHITE:
            cyc = dfs(k)
            if cyc:
                return cyc
    return []


def missing_inputs(requested: list, available: set) -> dict:
    """请求阶段集里，其前置依赖闭包中缺失（未可用）的阶段 → 供报错。

    ``requested`` 规范阶段名；``available`` 已生成字段/阶段集合。
    返回 {stage: [缺失依赖...]}。
    """
    closure = dependency_closure(requested)
    missing = {}
    for s in closure:
        need = [d for d in _STAGE_DEPS.get(s, []) if d not in available]
        if need:
            missing[s] = need
    return missing


class StageDependencyError(Exception):
    """阶段依赖错误：未知阶段、缺前置、或循环依赖。"""


def describe_stage(stage) -> dict:
    """返回单阶段契约摘要（依赖 + 说明），供文档/CLI。"""
    return {
        "stage": stage,
        "requires": _STAGE_DEPS.get(stage, []),
    }
