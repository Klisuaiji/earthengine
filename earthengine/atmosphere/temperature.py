"""温度 (20)：纬度梯度 + 海陆温差 + 垂直递减率。

把旧 ``solar_temperature`` 的第一输入从“纬度数组”改为 ``Planet + 纬度`` (6.2)。
"""

from __future__ import annotations

import numpy as np

from earthengine.planet.parameters import Planet
from earthengine.planet.sphere import _lat_deg


def latitude_temperature(planet: Planet, h: int) -> np.ndarray:
    """纬度温度带（含行星恒星参数缩放 + 轴倾角）。"""
    lat = _lat_deg(h)
    base = 30.0 * np.cos(np.deg2rad(lat)) - 5.0
    # 恒星参数缩放：太阳常数越高，赤道越热、极地越冷
    s = planet.solar_constant_wm2 / 1361.0
    base = planet.initial_temperature_c + (base - planet.initial_temperature_c) * s
    return base.astype(np.float32)


def solar_temperature(planet: Planet, elev: np.ndarray, h: int) -> np.ndarray:
    """温度：纬度带 + 大陆性 + 垂直递减率。

    返回 (h,w) float32 ℃。
    """
    lat = _lat_deg(h)
    t0 = latitude_temperature(planet, h)
    land = (elev >= 0).astype(np.float32)
    t0 = t0 + land * (4.0 * np.cos(np.deg2rad(lat)) - 2.0 * np.abs(lat) / 90.0 * 6.0)
    temp = t0 - 0.0065 * np.maximum(elev, 0.0)         # lapse rate
    return temp.astype(np.float32)


def continental_pressure(temp: np.ndarray, elev: np.ndarray, h: int) -> np.ndarray:
    """热力气压（暖→低压）+ 高地低压。"""
    tmean = temp.mean()
    p = -0.9 * (temp - tmean)
    p = p - 0.02 * np.maximum(elev, 0.0)
    return p.astype(np.float32)
