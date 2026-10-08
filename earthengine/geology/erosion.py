"""侵蚀与风化 (23)：把地形、土壤、河流反馈回高程与河流。

v2.0 Phase 4 的简化参数化版本：降雨驱动的流水侵蚀 + 山体剥蚀，
反馈到高程（削峰填谷），供河流与植被使用。
"""

from __future__ import annotations

import numpy as np


def erosion(elev: np.ndarray, land_mask: np.ndarray, precipitation: np.ndarray,
            seed: int = 0, iterations: int = 1):
    """流水侵蚀：坡度 + 降水 → 削峰，局部沉积。

    返回侵蚀后的高程 (h,w) float32（仅陆地）。
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover
        return elev.astype(np.float32)
    e = elev.astype(np.float32).copy()
    gy, gx = np.gradient(e)
    slope = np.sqrt(gy * gy + gx * gx)
    # 侵蚀强度 ∝ 坡度 × 降水
    p = precipitation.astype(np.float32)
    pn = (p - p.min()) / (p.max() - p.min() + 1e-9)
    erode = 0.02 * slope * pn * land_mask.astype(np.float32)
    for _ in range(iterations):
        e = gaussian_filter(e, 1.0) - erode
    e[land_mask] = np.maximum(e[land_mask], 1.0)
    return e.astype(np.float32)


def rock_hardness(terrain_class, seed: int = 0) -> np.ndarray:
    """岩石硬度 [0,1]：山脉最硬（抗侵蚀），平原最软。"""
    from earthengine.pipeline.world import (
        TT_MOUNTAIN, TT_PLATEAU, TT_HILL, TT_PLAIN, TT_BASIN,
    )
    hard = np.zeros(terrain_class.shape, dtype=np.float32)
    hard[terrain_class == TT_MOUNTAIN] = 0.9
    hard[terrain_class == TT_PLATEAU] = 0.7
    hard[terrain_class == TT_HILL] = 0.5
    hard[terrain_class == TT_PLAIN] = 0.3
    hard[terrain_class == TT_BASIN] = 0.35
    return hard
