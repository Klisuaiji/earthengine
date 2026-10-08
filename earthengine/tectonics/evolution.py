"""地壳演化规则 (11 增补)。

DIVERGENT（洋壳）  → 新生洋壳，age=0，洋中脊高程抬升
DIVERGENT（陆壳）  → 裂谷，地壳变薄 → TRANSITIONAL
CONVERGENT（洋-陆）→ 俯冲：洋壳消亡，陆缘造山带 + 火山弧
CONVERGENT（洋-洋）→ 岛弧 + 海沟
CONVERGENT（陆-陆）→ 碰撞：地壳增厚 → 高原/山脉，无火山
年龄 → 洋壳热沉降：depth ≈ 2500 + 350·√age(Myr) 米
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics.boundaries import (
    CONVERGENT, DIVERGENT, INTERIOR, TRANSFORM,
)
from earthengine.tectonics.crust import (
    CRUST_CONTINENTAL, CRUST_OCEANIC, CRUST_TRANSITIONAL, CrustState,
)


def ocean_depth_m(age_myr) -> np.ndarray:
    """洋壳热沉降深度（米）。"""
    return 2500.0 + 350.0 * np.sqrt(np.maximum(age_myr, 0.0))


def evolve_crust(crust: CrustState, boundary_type: np.ndarray,
                 h: int, w: int) -> CrustState:
    """把板块边界类型转成地壳演化效应（类型/厚度/高程偏移更新）。

    简化参数化版本：汇聚洋缘 → 俯冲海沟（高程偏压加深）；汇聚陆缘 → 造山抬升；
    张裂 → 洋中脊抬升 / 陆裂谷变薄。
    """
    ctype = crust.type.copy()
    age = crust.age.copy()
    thickness = crust.thickness.copy()
    bias = crust.elevation_bias.copy()

    conv = boundary_type == CONVERGENT
    div = boundary_type == DIVERGENT

    # 洋壳俯冲（洋-洋 / 洋-陆）：年龄在俯冲带重置归 0（消亡洋壳），高程加深
    ocean_conv = conv & (ctype == CRUST_OCEANIC)
    age[ocean_conv] = 0.0
    bias[ocean_conv] = np.minimum(bias[ocean_conv], -6000.0)

    # 陆缘碰撞：地壳增厚，造山抬升
    land_conv = conv & (ctype == CRUST_CONTINENTAL)
    thickness[land_conv] = np.maximum(thickness[land_conv], 60.0)
    bias[land_conv] = np.maximum(bias[land_conv], 3000.0)

    # 洋中脊（洋壳张裂）：新生洋壳 age=0，脊抬升
    ocean_div = div & (ctype == CRUST_OCEANIC)
    age[ocean_div] = 0.0
    bias[ocean_div] = np.maximum(bias[ocean_div], -2500.0)

    # 陆裂谷（陆壳张裂）：地壳变薄 → 过渡壳
    land_div = div & (ctype == CRUST_CONTINENTAL)
    ctype[land_div] = CRUST_TRANSITIONAL
    thickness[land_div] = np.minimum(thickness[land_div], 18.0)

    return CrustState(type=ctype, age=age, density=crust.density.copy(),
                      thickness=thickness, elevation_bias=bias)
