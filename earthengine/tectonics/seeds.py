"""微板块种子采样 (7.2)。

- 由面积目标反推微板块数量 ``n_raw``（不再固定 30）。
- 球面最远点采样（避免极区聚集），两枚海洋种子留在地图中央作为太平洋类似物。

实现复用 ``earthengine.tectonics.voronoi`` 的最远点采样；本模块提供 v2.0 面向
的面积目标 → 种子数量的换算。
"""

from __future__ import annotations

import math

import numpy as np

from earthengine.tectonics import voronoi as _sv


def microplate_count(target_area_frac: float, min_micro: int = 24,
                     density: float = 0.5) -> int:
    """按目标大板块平均面积反推微板块种子数量。

    ``density``：每单位面积(球面分数)的种子密度。真实地球 ~15 板块、分布不均，
    我们用较高的微板块密度以保证归并后有丰富的小型板块可供合并。
    """
    n = int(round(density / max(target_area_frac, 1e-6)))
    return max(min_micro, n)


def sphere_seeds(seed: int, n_plates: int, h: int, w: int):
    """返回 ``(seeds, ocean_center)``：微板块种子单位向量 + 海洋中心。

    种子布局与分辨率无关（球面坐标），只依赖 seed。
    """
    return _sv._place_seeds(seed, n_plates)


def domain_warped_sphere(h: int, w: int, seed: int) -> np.ndarray:
    """域扭曲后的球面方向场 (h, w, 3)，作为多源生长的各向异性代价场。

    Voronoi 只把噪声用于输入坐标微扰，不参与边界输出。
    """
    return _sv._warp_dirs(_sv._coords_to_vecs(h, w), seed, h, w)
