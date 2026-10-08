"""颜色表 (29)：颜色只是显示方式，不能成为模拟数据本身。"""

from __future__ import annotations

import numpy as np

from earthengine.pipeline.world import TERRAIN_RGB, TT_MOUNTAIN, TT_PLATEAU
from earthengine.biology.vegetation import BIOME_RGB
from earthengine.climate.climate import KOPPEN_RGB

# 今怀古参考调色板：青绿大陆架 + 绿→黄褐→棕→白 陆地
RELIEF_HIRES_LAND = np.array([
    [0.0, 0.588, 0.784, 0.471],
    [0.10, 0.776, 0.804, 0.494],
    [0.28, 0.902, 0.847, 0.541],
    [0.50, 0.898, 0.788, 0.549],
    [0.70, 0.804, 0.651, 0.451],
    [0.85, 0.780, 0.569, 0.573],
    [1.00, 0.973, 0.973, 0.973],
], dtype=np.float32)

# 板块调色板（连续）：区分相邻板块
PLATE_PALETTE = np.array([
    [0.20, 0.35, 0.75], [0.80, 0.35, 0.30], [0.30, 0.70, 0.45],
    [0.85, 0.70, 0.25], [0.55, 0.35, 0.75], [0.20, 0.70, 0.75],
    [0.80, 0.55, 0.15], [0.65, 0.75, 0.80], [0.40, 0.25, 0.55],
    [0.75, 0.75, 0.30], [0.60, 0.80, 0.65], [0.85, 0.45, 0.60],
], dtype=np.float32)


def biome_palette(biome_name) -> np.ndarray:
    """生物群系 (h,w,3) float32 颜色（LUT over unique names）。"""
    uniq, inverse = np.unique(biome_name, return_inverse=True)
    lut = np.array([BIOME_RGB.get(b, (120, 160, 100)) for b in uniq], dtype=np.float32)
    return lut[inverse.reshape(biome_name.shape)] / 255.0


def koppen_palette(koppen_code) -> np.ndarray:
    """Köppen (h,w,3) float32 颜色。"""
    uniq, inverse = np.unique(koppen_code, return_inverse=True)
    lut = np.array([KOPPEN_RGB.get(c, (30, 80, 150)) for c in uniq], dtype=np.float32)
    return lut[inverse.reshape(koppen_code.shape)] / 255.0


def terrain_palette(terrain_class) -> np.ndarray:
    """地形类别 (h,w,3) float32 颜色。"""
    pal = np.zeros((12, 3), dtype=np.float32)
    for k, col in TERRAIN_RGB.items():
        pal[k] = col
    return pal[terrain_class] / 255.0
