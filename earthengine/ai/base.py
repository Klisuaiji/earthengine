"""TerrainEnhancer 抽象 + ProceduralEnhancer 默认实现 (24 / 25 / 26 / 42)。

契约 (24.3)：``ProceduralEnhancer`` 与 ``TerrainDiffusionAdapter`` 必须返回
**同 dtype、同量纲（米）、同海陆符号** 的数组，使下游 Hydrology/Vegetation/
Rendering 无分支。
"""

from __future__ import annotations

import numpy as np

from earthengine.terrain import detail as _detail
from earthengine.terrain.noise import fbm as _fbm


class TerrainEnhancer:
    """统一地形增强接口。"""

    name = "base"

    def generate(self, conditioning: dict, region, resolution: int,
                 seed: int) -> np.ndarray:
        """返回该区域的 float32 高程（米）。"""
        raise NotImplementedError


class ProceduralEnhancer(TerrainEnhancer):
    """默认实现：无 GPU 也能跑，纯 numpy 分形细化 (24.2)。"""

    name = "procedural"

    def __init__(self, detail_gain: float = 0.35):
        self.detail_gain = detail_gain

    def generate(self, conditioning: dict, region, resolution: int,
                 seed: int) -> np.ndarray:
        """把 macro_elevation + terrain_region 细化到 ``resolution``。

        若无地理 conditioning（区域级），则用本区域现有高程 + 分形纹理增强。
        """
        macro = conditioning.get("macro_elevation")
        if macro is None:
            raise ValueError("ProceduralEnhancer needs macro_elevation")
        land = conditioning.get("land_sea")
        terrain_class = conditioning.get("terrain_region")
        h, w = macro.shape
        elev = macro.astype(np.float32)
        fine = _fbm((h, w), base_scale=10, octaves=5, seed=seed + 701)
        land_f = np.ones((h, w), dtype=bool) if land is None else land.astype(bool)
        elev = elev + fine * np.where(land_f, 60.0, 70.0).astype(np.float32)
        if terrain_class is not None:
            elev = _detail.ridge_detail(elev, terrain_class, seed)
        elev[land_f] = np.maximum(elev[land_f], 3.0)
        elev[~land_f] = np.minimum(elev[~land_f], -1.0)
        return elev.astype(np.float32)


def resolve_device(device: str = "auto") -> str:
    """解析设备：auto → cuda（若有）否则 cpu。"""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
