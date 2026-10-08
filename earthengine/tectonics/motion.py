"""板块运动 (7.3)：欧拉矢量。

每个板块一个角速度矢量 ``omega: Vector3``（轴过球心，rad/Myr）。
单元 p（单位向量）处的线速度：``v(p) = omega × p``。

小板块移动更快（真实地球相关性）。
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics import core


def euler_velocity(omega: np.ndarray, p: np.ndarray) -> np.ndarray:
    """单元 p 处的线速度向量 ``omega × p``。

    ``omega`` (3,)，``p`` (..., 3) 或 (3,)。返回同形。
    """
    return np.cross(omega, p)


def relative_velocity(omega_a: np.ndarray, omega_b: np.ndarray,
                      p: np.ndarray) -> np.ndarray:
    """两板块在单元 p 处的相对运动 ``v_A - v_B``。"""
    return euler_velocity(omega_a, p) - euler_velocity(omega_b, p)


def surface_speed(omega: np.ndarray, p: np.ndarray) -> float:
    """表面线速度标量（km/Myr）。"""
    return float(np.linalg.norm(euler_velocity(omega, p)))
