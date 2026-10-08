"""河流 (22)：由汇水面积阈值提取河网。"""

from __future__ import annotations

import numpy as np


def rivers_from_acc(flow_acc, land_mask, min_acc=None, h=None, w=None):
    """汇水面积 >= min_acc 的陆地像素为河。返回 ``(river_mask, min_acc)``。"""
    if h is None or w is None:
        h, w = flow_acc.shape
    if min_acc is None:
        min_acc = max(120, (h * w) // 8000)
    rivers = land_mask & (flow_acc >= min_acc)
    return rivers, min_acc


def ocean_outlets(flow_dir, flow_acc, land_mask, h, w):
    """河流入海口：陆地像素流向海洋的边。返回 (h,w) bool。"""
    from earthengine.hydrology.flow import _DY, _DX
    out = np.zeros((h, w), dtype=bool)
    yy, xx = np.nonzero(land_mask & (flow_dir != 8))
    for y, x in zip(yy, xx):
        k = flow_dir[y, x]
        ny = (y + _DY[k]) % h
        nx = (x + _DX[k]) % w
        if not land_mask[ny, nx]:
            out[y, x] = True
    return out
