"""植被 (27)：Köppen + 冰盖 → 生物群系。

输入 temperature, precipitation, soil, elevation, slope, seasonality →
输出 desert / grassland / forest / rainforest / taiga / tundra / wetland …。
现有 ``biome()`` 已产出字符串数组，保留为 v2.0 的 ``biology/vegetation.py``。
"""

from __future__ import annotations

import numpy as np

BIOME_OF_KOPPEN = {
    "Af": "Tropical rainforest", "Am": "Tropical monsoon", "Aw": "Tropical savanna",
    "BWh": "Hot desert", "BWk": "Cold desert", "BS": "Steppe",
    "Cfa": "Temperate broadleaf", "Cfb": "Temperate forest", "Csb": "Mediterranean",
    "Dfb": "Boreal forest", "Dfc": "Taiga", "Dfa": "Humid continental",
    "ET": "Tundra", "EF": "Ice sheet", "Ocean": "Ocean",
}
BIOME_RGB = {
    "Tropical rainforest": (30, 120, 50), "Tropical monsoon": (50, 150, 60),
    "Tropical savanna": (170, 190, 90), "Hot desert": (220, 200, 130),
    "Cold desert": (200, 185, 155), "Steppe": (210, 195, 140),
    "Temperate broadleaf": (70, 160, 90), "Temperate forest": (90, 175, 110),
    "Mediterranean": (120, 180, 100), "Boreal forest": (60, 140, 150),
    "Taiga": (80, 130, 165), "Humid continental": (75, 155, 130),
    "Tundra": (190, 210, 200), "Ice sheet": (235, 240, 245), "Ocean": (30, 80, 150),
}
BIOME_INDEX = {name: i for i, name in enumerate(BIOME_RGB)}


def biome(koppen_code, ice, h=None):
    """Köppen + 冰盖 → 生物群系字符串数组 (h,w)。"""
    codes, inverse = np.unique(koppen_code, return_inverse=True)
    names = np.array([BIOME_OF_KOPPEN.get(c, "Ocean") for c in codes])
    out = names[inverse.reshape(koppen_code.shape)]
    out = np.where(ice, "Ice sheet", out)
    return out.astype(object)


def biome_index(biome_name) -> np.ndarray:
    """生物群系 → int8 码表（供 fields.npz 存档）。"""
    uniq, inverse = np.unique(biome_name, return_inverse=True)
    lut = np.array([BIOME_INDEX.get(b, 0) for b in uniq], dtype=np.int8)
    return lut[inverse.reshape(biome_name.shape)]
