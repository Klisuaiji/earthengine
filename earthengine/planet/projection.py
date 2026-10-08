"""投影 (4.1)：3D 球面 ↔ 2D 等经纬投影互转。"""

from __future__ import annotations

import numpy as np

from .sphere import LatLonGrid


def to_xyz(h: int, w: int) -> np.ndarray:
    """(H,W,3) 等经纬投影的单元质心单位向量。"""
    return LatLonGrid(h, w).xyz()


def from_xyz(xyz: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 (N,3) 单位向量就近映射回 (H,W) 平面像素索引 (N,)。"""
    return np.asarray(LatLonGrid(h, w).nearest(xyz), dtype=np.intp)


def lat_lon_deg(h: int, w: int):
    """返回纬度 (H,1)、经度 (1,W) 度数网格。"""
    g = LatLonGrid(h, w)
    return g._lat, g._lon
