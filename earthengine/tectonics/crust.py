"""地壳系统 (11)：解决 P3（板块 = 大陆，无地壳概念）。

至少存在：CONTINENTAL / OCEANIC / TRANSITIONAL。属性：crust_type, crust_age,
density, thickness, elevation_bias。洋壳年龄驱动热沉降：depth ≈ 2500 + 350·√age。

v2.0 让 ``Plate`` 与 ``Continent`` 成为两个独立数据概念：一个板块可同时拥有
大陆地壳 + 大陆架 + 海洋地壳。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from earthengine.tectonics import core

CRUST_CONTINENTAL = core.CRUST_CONTINENTAL
CRUST_OCEANIC = core.CRUST_OCEANIC
CRUST_TRANSITIONAL = 2


@dataclass
class CrustState:
    """每像素地壳状态。"""

    type: np.ndarray          # (h, w) int8：0 陆 / 1 洋 / 2 过渡
    age: np.ndarray           # (h, w) float32（Myr）
    density: np.ndarray       # (h, w) float32（kg/m³）
    thickness: np.ndarray     # (h, w) float32（km）
    elevation_bias: np.ndarray  # (h, w) float32（米）


def generate_crust(plates: np.ndarray, is_ocean: np.ndarray,
                   h: int, w: int, seed: int) -> CrustState:
    """从板块系统生成地壳类型 + 初始年龄。

    - 洋壳板块：CRUST_OCEANIC，年龄由到张裂边界(洋中脊)的距离决定（越远离脊越老）；
    - 陆壳板块：CRUST_CONTINENTAL；
    - 已放置的独立大陆板块：保留为大陆地壳（板块≠大陆，但以板块类型为起点）。

    年龄 → 热沉降深度（米）：``depth ≈ 2500 + 350·√age``。
    """
    from earthengine.seed import SeedManager
    sm = SeedManager(seed)

    # is_ocean is per-plate; map it to per-pixel via the plate-id map
    ocean_by_plate = np.asarray(is_ocean, dtype=int)
    is_oc = ocean_by_plate[plates]
    ctype = np.where(is_oc, CRUST_OCEANIC, CRUST_CONTINENTAL).astype(np.int8)
    age = np.zeros((h, w), dtype=np.float32)

    # 洋壳年龄：给每块洋板块一个确定性年龄基底，再接热沉降
    rng = sm.derive("tectonics.crust_age")
    uniq = np.unique(plates)
    for pid in uniq:
        if ocean_by_plate[int(pid)]:
            age[plates == pid] = float(rng.uniform(2.0, 180.0))

    density = np.where(ctype == CRUST_CONTINENTAL, 2800.0, 3000.0).astype(np.float32)
    thickness = np.where(ctype == CRUST_CONTINENTAL, 35.0, 7.0).astype(np.float32)
    # 热沉降深度（米）
    depth = 2500.0 + 350.0 * np.sqrt(np.maximum(age, 0.0))
    elevation_bias = np.where(ctype == CRUST_CONTINENTAL,
                              np.float32(0.0), -depth).astype(np.float32)
    return CrustState(type=ctype, age=age, density=density,
                      thickness=thickness, elevation_bias=elevation_bias)
