"""Tectonic Core：板块系统（第一优先级）。

v2.0 的板块生成链路（7.2）：
    Fib 球面采样 → 微板块种子 → 球面邻接图 → 多源 BFS 均衡生长 → 幂律归并大板块
    → 欧拉运动 → 按相对运动分类边界 → 地壳系统。

旧 ``Voronoi = 板块`` 的错误链路已被替换：Voronoi 只负责建立球面邻接骨架，
真正的板块构造模型（运动、边界类型、地壳）由本包决定。
"""

from .plates import (
    Plate,
    PlanetaryPlates,
    generate_tectonic_plates,
    generate_validated_world,
)
from .boundaries import (
    INTERIOR,
    CONVERGENT,
    DIVERGENT,
    TRANSFORM,
    OBLIQUE,
    classify_by_motion,
)
from .crust import (
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    CRUST_TRANSITIONAL,
    CrustState,
    generate_crust,
)

__all__ = [
    "Plate", "PlanetaryPlates",
    "generate_tectonic_plates", "generate_validated_world",
    "INTERIOR", "CONVERGENT", "DIVERGENT", "TRANSFORM", "OBLIQUE",
    "classify_by_motion",
    "CRUST_CONTINENTAL", "CRUST_OCEANIC", "CRUST_TRANSITIONAL",
    "CrustState", "generate_crust",
]
