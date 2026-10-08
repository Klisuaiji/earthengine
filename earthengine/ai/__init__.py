"""AI 模块：TerrainEnhancer 抽象 + 程序化默认实现 (24 / 25 / 26 / 42)。

核心只依赖抽象 ``TerrainEnhancer``，``TerrainDiffusionAdapter`` 只是实现之一，
可替换为未来模型。无 AI 仍可完整生成；有 GPU/AI 可增强。
"""

from earthengine.ai.base import (
    TerrainEnhancer,
    ProceduralEnhancer,
    resolve_device,
)

__all__ = ["TerrainEnhancer", "ProceduralEnhancer", "resolve_device"]
