"""大尺度地形 (16-17)：宏观高程由板块构造驱动。

地球物理高程基准（米）：
    LAND_BASE  大陆台地
    MOUNTAIN_BONUS 汇聚边界山脉
    OCEAN_BASE  洋底
    TRENCH      俯冲海沟
    RIDGE       洋中脊
"""

from __future__ import annotations

import numpy as np

LAND_BASE = 3500.0
MOUNTAIN_BONUS = 1500.0
OCEAN_BASE = -6000.0
TRENCH = -8000.0
RIDGE = -4500.0


def coarse_elevation(land_mask, boundaries, shelf_px=40.0):
    """边界应力的大陆/海洋基础高程（pre-diffusion conditioning）。

    汇聚边界：陆上山脉、海上海沟；张裂边界：洋中脊；近岸：大陆架抬升。
    返回 (h,w) float32 米。
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:  # pragma: no cover
        distance_transform_edt = None
    h, w = land_mask.shape
    bt = boundaries["boundary_type"]
    cdist = boundaries["convergent_dist"]
    ddist = boundaries["divergent_dist"]
    lm = land_mask.astype(np.float32)

    elev = np.where(lm > 0.5, LAND_BASE, OCEAN_BASE)
    if distance_transform_edt is not None:
        d_land = distance_transform_edt(~land_mask).astype(np.float32)
        shelf = np.clip(1.0 - d_land / shelf_px, 0.0, 1.0)
    else:
        shelf = _manual_shelf(~land_mask)
    elev = np.where(lm <= 0.5, elev + shelf * (-500.0 - OCEAN_BASE), elev)
    conv = 2200.0 * np.exp(-cdist / 45.0)
    elev = elev + conv * (lm + (cdist < 30).astype(np.float32))
    trench_mask = (1.0 - lm) * (1.0 - shelf)
    trench = (TRENCH - OCEAN_BASE) * np.exp(-cdist / 35.0) * trench_mask
    ridge = (RIDGE - OCEAN_BASE) * np.exp(-ddist / 45.0) * (1.0 - lm)
    elev = elev + trench + ridge
    return np.clip(elev, TRENCH, LAND_BASE + MOUNTAIN_BONUS)


def _manual_shelf(sea):
    """纯 numpy 大陆架：逐层扩张近似（无 scipy 兜底）。"""
    shelf = np.zeros(sea.shape, dtype=np.float32)
    cur = sea.copy()
    for k in range(1, 41):
        nxt = _dilate(cur)
        ring = nxt & ~cur
        shelf[ring] = np.clip(1.0 - k / 40.0, 0.0, 1.0)
        cur = nxt
    return shelf


def _dilate(mask):
    """4 邻域膨胀（经度环绕）。"""
    out = mask.copy()
    out |= np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0)
    out |= np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
    return out
