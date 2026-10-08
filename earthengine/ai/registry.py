"""AI 后端注册表 (42)：核心只依赖抽象，Adapter 通过注册表按需启用。"""

from __future__ import annotations

from earthengine.ai.base import TerrainEnhancer, ProceduralEnhancer

_REGISTRY = {"procedural": ProceduralEnhancer}


def register(name: str, factory):
    _REGISTRY[name] = factory


def get_enhancer(name: str = "procedural", **kwargs) -> TerrainEnhancer:
    """按名字取增强器实例；未知名字回退 procedural。"""
    if name not in _REGISTRY:
        return _REGISTRY["procedural"]()
    fac = _REGISTRY[name]
    if isinstance(fac, type):
        return fac(**kwargs)
    return fac(**kwargs)


def available() -> list:
    return list(_REGISTRY.keys())
