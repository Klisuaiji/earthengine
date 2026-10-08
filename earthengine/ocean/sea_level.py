"""海平面 (12 / 6.1)：由 Planet 参数与海陆比例共同决定。

海洋不能由 ``LAND_FRACTION + 随机大陆 + 噪声`` 生成，必须由板块 + 地壳 +
海盆 + 海平面 + 板块历史共同决定。
"""

from __future__ import annotations

import numpy as np


def ocean_fraction_to_sea_level(coarse_elev, target_ocean_frac):
    """由目标海洋占比反推海平面高程（米）。

    对宏观高程直方图二分查找一个阈值，使 ``elev < sea_level`` 的面积占比
    ≈ ``target_ocean_frac``。返回该海平面高度。
    """
    flat = coarse_elev.ravel()
    if target_ocean_frac <= 0.0:
        return float(flat.max()) + 1.0
    if target_ocean_frac >= 1.0:
        return float(flat.min()) - 1.0
    lo, hi = float(flat.min()), float(flat.max())
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        ocean_frac = float((flat < mid).mean())
        if ocean_frac < target_ocean_frac:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def apply_sea_level(coarse_elev, sea_level_m):
    """把海平面应用到高程，返回 (land_mask, adjusted_elev)。"""
    land_mask = coarse_elev >= sea_level_m
    return land_mask, coarse_elev
