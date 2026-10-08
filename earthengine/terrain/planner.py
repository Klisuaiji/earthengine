"""Terrain Region Planner (19 / 26)：给大陆“上色”。

宏观地理 → 地形分类 → terrain_region → conditioning encoder → Terrain Diffusion。
Terrain Diffusion 不决定“这里应该是山还是大陆”（EarthEngine 已决定），只负责
“这个山脉长什么样”。

v0 的 7 类 ``terrain_type_map``（A.2 已有能力 4）升级为 12 类：
深海/大陆架/沿海/平原/丘陵/高原/山脉/盆地/洋中脊/海沟/火山/裂谷。
"""

from __future__ import annotations

import numpy as np

from earthengine.pipeline.world import (
    TT_DEEP, TT_SHELF, TT_COASTAL, TT_PLAIN, TT_HILL, TT_PLATEAU,
    TT_MOUNTAIN, TT_BASIN, TT_RIDGE, TT_TRENCH, TT_VOLCANIC, TT_RIFT,
)
from earthengine.terrain.macro import (
    OCEAN_BASE, RIDGE, TRENCH,
)
from earthengine.terrain import macro
from earthengine.terrain.mountains import mountain_belt
from earthengine.terrain.plains import plains_hills, plateaus, basins
from earthengine.terrain.noise import fbm


def _d_ocean_rel(land_mask):
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(land_mask).astype(np.float32)
    except Exception:  # pragma: no cover
        from earthengine.terrain.plains import _d_ocean_rel as _f
        return _f(land_mask)


def terrain_region_map(land_mask, boundaries, seed):
    """地理规则驱动的 12 类地形区域图 + 宏观高程。

    地理规则（扩散模型无法自行得知）：
      * 山脉沿汇聚边界（Andes/Himalaya）；
      * 1-2 块内陆高原（Tibet）；
      * 沿海平原、内陆丘陵带；
      * 洋中脊 / 海沟 / 裂谷 / 火山弧由边界类型与地壳驱动；
      * 大陆架由距陆缘距离推导（非高程硬切）。

    返回 ``(geo_elev, terrain_class)``。
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:  # pragma: no cover
        distance_transform_edt = None
    h, w = land_mask.shape
    cdist = boundaries["convergent_dist"]
    ddist = boundaries["divergent_dist"]
    bt = boundaries["boundary_type"]
    lm = land_mask
    broad = fbm((h, w), base_scale=96, octaves=3, seed=seed + 302)

    # --- 海洋 ---
    if distance_transform_edt is not None:
        d_land = distance_transform_edt(~lm).astype(np.float32)
    else:
        d_land = _d_ocean_rel(~lm)
    shelf = np.clip(1.0 - d_land / 26.0, 0.0, 1.0)
    ocean = (OCEAN_BASE + shelf * (-500.0 - OCEAN_BASE)
             + (RIDGE - OCEAN_BASE) * np.exp(-ddist / 45.0) * (1.0 - lm)
             + (TRENCH - OCEAN_BASE) * np.exp(-cdist / 35.0) * (1.0 - lm) * (1.0 - shelf)
             + broad * 250.0 * (1.0 - lm))

    # --- 陆地 ---
    land_elev = plains_hills(lm, seed, h, w)
    land_elev = land_elev + plateaus(lm, seed, h, w)
    basin_mask, basin_low = basins(lm, seed, h, w)
    land_elev = land_elev + basin_low
    land_elev = land_elev + mountain_belt(lm, cdist, seed, h, w)
    land_elev = np.maximum(land_elev, 3.0)

    geo_elev = np.where(lm, land_elev, ocean).astype(np.float32)

    # --- 12 类分类 ---
    cls = np.zeros((h, w), dtype=np.int8)
    sea = ~lm
    if distance_transform_edt is not None:
        d_sea = distance_transform_edt(lm).astype(np.float32)  # dist into ocean
        cls[sea & (d_sea < 14)] = TT_SHELF
    else:
        cls[sea] = TT_DEEP
    cls[sea & (cls != TT_SHELF)] = TT_DEEP
    cls[sea & (ddist < 12) & (cls == TT_DEEP)] = TT_RIDGE      # 洋中脊
    cls[sea & (cdist < 12) & (shelf < 0.5) & (cls == TT_DEEP)] = TT_TRENCH  # 海沟
    cls[lm & (geo_elev >= 2000)] = TT_MOUNTAIN
    cls[lm & (geo_elev >= 1100) & (geo_elev < 2000)] = TT_PLATEAU
    cls[lm & (geo_elev >= 350) & (geo_elev < 1100)] = TT_HILL
    cls[lm & (geo_elev < 350)] = TT_PLAIN
    cls[basin_mask & lm & (geo_elev < 1100)] = TT_BASIN
    # 沿海带：陆地近海缘一条带
    if distance_transform_edt is not None:
        cls[lm & (d_land < 6)] = TT_COASTAL
    # 裂谷 / 火山弧：由边界类型近似标记
    cls[lm & (bt == 2)] = TT_RIFT
    cls[sea & (bt == 1) & (cls == TT_TRENCH)] = TT_TRENCH
    return geo_elev, cls


def build_conditioning(geo_elev, terrain_class, land_mask, boundaries,
                       climate=None, hydrology=None):
    """TerrainConditioning (18)：交给 AI 增强层的条件张量集合。

    契约 (24.3)：ProceduralEnhancer 与 TerrainDiffusionAdapter 消费同一份
    conditioning，保证同 dtype、同量纲（米）、同海陆符号。
    """
    from earthengine.tectonics.boundaries import CONVERGENT, DIVERGENT
    cond = {
        "land_sea": land_mask.astype(bool),
        "terrain_region": terrain_class,
        "macro_elevation": geo_elev.astype(np.float32),
        "mountain_mask": terrain_class == TT_MOUNTAIN,
        "plateau_mask": terrain_class == TT_PLATEAU,
        "plain_mask": (terrain_class == TT_PLAIN) | (terrain_class == TT_COASTAL),
        "basin_mask": terrain_class == TT_BASIN,
        "volcanic_mask": terrain_class == TT_VOLCANIC,
        "rift_mask": terrain_class == TT_RIFT,
        "trench_mask": terrain_class == TT_TRENCH,
        "ridge_mask": terrain_class == TT_RIDGE,
        "convergent_mask": boundaries["boundary_type"] == CONVERGENT,
        "divergent_mask": boundaries["boundary_type"] == DIVERGENT,
    }
    if climate is not None:
        cond["temperature"] = climate.get("temperature")
        cond["precipitation"] = climate.get("precipitation")
    return cond
