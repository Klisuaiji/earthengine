"""断层与构造活动 (23)：由板块边界类型推导应力场。"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics.boundaries import CONVERGENT, DIVERGENT, TRANSFORM


def fault_map(boundary_type: np.ndarray, h: int, w: int) -> np.ndarray:
    """断层分布 [0,1]：转换边界最高（剪切），汇聚次之。"""
    f = np.zeros((h, w), dtype=np.float32)
    f[boundary_type == TRANSFORM] = 0.9
    f[boundary_type == CONVERGENT] = 0.7
    f[boundary_type == DIVERGENT] = 0.5
    return f
