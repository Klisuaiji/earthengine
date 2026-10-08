"""网格抽象 (4.5)：SphereGrid + LatLonGrid。

全引擎只依赖 ``SphereGrid`` 抽象接口，不直接接触 (H,W) 数组或像素坐标。
Phase 1 用 ``LatLonGrid``（沿用 (H,W) 数组 + 周期经度邻域 + cos(lat) 面积权重），
使既有数组算法零改动；Phase 3+ 可选 ``IcosphereGrid`` 只需替换实现。
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree as _KDTree
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - pure numpy fallback
    _HAS_SCIPY = False


class SphereGrid:
    """球面网格抽象接口。"""

    def xyz(self) -> np.ndarray:
        """单元质心单位向量 (N,3) float32。"""
        raise NotImplementedError

    def neighbors(self) -> np.ndarray:
        """邻接对 (K,2) int32。"""
        raise NotImplementedError

    def area(self) -> np.ndarray:
        """单元面积（含 cos(lat) 权重）(N,) float32。"""
        raise NotImplementedError

    def geodesic(self, i, j) -> float:
        """大圆距离。"""
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError


def _lat_deg(h: int) -> np.ndarray:
    """每行纬度（度，+90 上 → -90 下）。"""
    yy = np.arange(h, dtype=np.float32)[:, None]
    return 90.0 - 180.0 * (yy + 0.5) / h


class LatLonGrid(SphereGrid):
    """等经纬投影 (H,W) 网格。

    - 经度周期邻域（x 环绕），避免 0°/360° 接缝。
    - 面积权重 = cos(lat)。
    - 所有 (H,W) 数组算法均可直接映射到本网格。
    """

    def __init__(self, h: int, w: int):
        self._h = int(h)
        self._w = int(w)
        self._lat = _lat_deg(h)
        self._lon = np.arange(w, dtype=np.float32)[None, :] * 360.0 / w
        self._xyz = None

    def shape(self):
        return (self._h, self._w)

    # ------------------------------------------------------------------
    def xyz(self) -> np.ndarray:
        """(H,W,3) 单元质心单位向量。"""
        if self._xyz is None:
            lat, lon = self._lat, self._lon
            cl = np.cos(np.deg2rad(lat))
            sl = np.sin(np.deg2rad(lat))
            co = np.cos(np.deg2rad(lon))
            so = np.sin(np.deg2rad(lon))
            self._xyz = np.stack([cl * co, cl * so, np.broadcast_to(sl, lat.shape)], axis=-1)
        return self._xyz.astype(np.float32)

    def neighbors(self) -> np.ndarray:
        """每像素 4 邻域（经度环绕）：返回 (H*W*4, 2)。"""
        h, w = self._h, self._w
        idx = np.arange(h * w).reshape(h, w)
        nbr = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = np.clip(idx + dy * w, 0, h * w - 1)
            xx = np.roll(idx, dx, axis=1)
            nbr.append(np.stack([idx.ravel(), (yy + (xx - idx)).ravel()], axis=1))
        return np.concatenate(nbr, axis=0).astype(np.int32)

    def area(self) -> np.ndarray:
        """(H,W) 面积权重（cos(lat)，归一化到总和 1）。"""
        a = np.cos(np.deg2rad(self._lat)).astype(np.float32)
        a = np.broadcast_to(a, (self._h, self._w))
        return a.astype(np.float32) / float(a.sum())

    def geodesic(self, i, j) -> float:
        """两点间大圆距离（rad）。"""
        xyz = self.xyz().reshape(-1, 3)
        c = float(np.clip(np.dot(xyz[i], xyz[j]), -1.0, 1.0))
        return float(np.arccos(c))

    # ------------------------------------------------------------------
    # 实用方法
    # ------------------------------------------------------------------
    def wrap_x(self, arr: np.ndarray, shift: int = 1) -> np.ndarray:
        return np.roll(arr, shift, axis=1)

    def nearest(self, pts: np.ndarray, k: int = 1):
        """给定单位向量 pts (N,3)，返回每点最近的 (H,W) 像素索引。"""
        xyz = self.xyz().reshape(-1, 3)
        d = xyz @ pts.T                       # (Npix, N)
        if k == 1:
            return int(np.argmax(d, axis=0))
        return np.argsort(-d, axis=0)[:k].T
