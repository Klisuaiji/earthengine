"""多源 BFS 均衡生长 (7.2)。

把微板块种子生长成面积大致均衡的球面分区；之后按幂律归并为大板块。
实现复用 ``earthengine.tectonics.voronoi`` 的分区/均衡/清理，接口面向 v2.0。
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics import voronoi as _sv


def partition_sphere(seed: int, w: int, h: int, seeds, n_raw: int) -> np.ndarray:
    """把球面按微板块种子分割成 ``n_raw`` 个微板块分区 (h, w)。"""
    return _sv._partition_sphere(w, h, seeds, seed, n_raw)


def smooth_labels(pid: np.ndarray, radius: int = 2, iters: int = 1) -> np.ndarray:
    """平滑标签边界（多数投票）。"""
    return _sv._smooth_labels(pid, radius=radius, iters=iters)


def cleanup_raw_plates(pid: np.ndarray, n_raw: int) -> np.ndarray:
    """清理原始分区的极小碎片/飞地。"""
    return _sv._cleanup_raw_plates(pid, n_raw)


def balanced_grow(pid: np.ndarray, ocean_center, n_big: int,
                  w: int, h: int) -> np.ndarray:
    """沿邻接图做面积均衡多源 BFS，归并为 ``n_big`` 个大板块。

    ``ocean_center`` 保证两枚海洋板块留在中央纯洋区。
    """
    merged = _sv._grow_continents(pid, ocean_center, w, h, n_big)
    merged = _sv.remove_enclaves(merged, min_frac=0.0005)
    merged = _sv._smooth_labels(merged, radius=max(4, w // 128), iters=2)
    return _sv.remove_enclaves(merged, min_frac=0.002)
