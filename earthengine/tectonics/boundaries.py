"""板块边界与分类 (7.4 / 9-10)。

边界不再由“像素坐标 hash”随机决定（那是 v2.0 要消除的不可复现源，见 38 纠错），
而是由板块相对运动推导：

    v_rel = v_A(p) - v_B(p)
    n     = 边界法向（A→B 测地线切向）
    v_n   = dot(v_rel, n)          # 法向：>0 分离，<0 汇聚
    v_t   = |v_rel - v_n·n|        # 切向

    |v_n|>eps 且 |v_t|/|v_n|<0.5 → DIVERGENT(v_n>0) / CONVERGENT(v_n<0)
    |v_t|>eps                    → TRANSFORM
    否则                          → OBLIQUE

随机性只由 seed + plate_id 决定，与像素坐标、分辨率无关 → 同一 seed 在不同
分辨率下得到同一块世界。
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics import core

INTERIOR, CONVERGENT, DIVERGENT, TRANSFORM, OBLIQUE = (
    core.INTERIOR, core.CONVERGENT, core.DIVERGENT, core.TRANSFORM, 4)


def classify_by_motion(merged, centroids, omega, h, w, lamb=0.8):
    """按相对运动把边界像素分类为 汇聚/张裂/转换/斜向。

    返回 ``(boundary_type, score)``：``score`` 为每像素类型强度（用于距离场）。
    """
    return core._classify_by_motion(merged, centroids, omega, h, w, lamb)


def boundary_pixels(merged, h, w):
    """提取边界像素集合与每像素的板块对。"""
    return core._extract_boundary_pixels(merged, h, w)
