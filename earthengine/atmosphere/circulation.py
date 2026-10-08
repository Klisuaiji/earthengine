"""大气环流 (20 / 21)：行星风带 + 科里奥利偏转 + 季风 + 地形偏转。

引入科里奥利参数 f = 2 Ω sin φ（行星自转速度参与模拟，6.2）。
"""

from __future__ import annotations

import numpy as np

from earthengine.planet.parameters import Planet
from earthengine.planet.sphere import _lat_deg


def coriolis(planet: Planet, lat_deg: np.ndarray) -> np.ndarray:
    """科里奥利参数 f = 2 Ω sin(φ)（rad/s）。"""
    return 2.0 * planet.omega_s * np.sin(np.deg2rad(lat_deg))


def wind_field(planet: Planet, elev: np.ndarray, temp: np.ndarray,
               h: int, w: int) -> tuple:
    """行星风带 + 地形偏转 + 季风。

    返回 ``(u, v)`` 风矢量场（方向意义；量级受科里奥利调制）。
    """
    lat = _lat_deg(h)
    abslat = np.abs(lat)
    # 行星风带：0-30 信风、30-60 西风、60-90 极地东风
    band = np.select(
        [abslat < 30, (abslat >= 30) & (abslat < 60), abslat >= 60],
        [-1.0, 1.0, -1.0], default=0.0)
    u = band * (1.0 - 0.3 * np.cos(np.deg2rad(lat)) ** 2)
    # 科里奥利调制：高纬度西风带更窄更锐利
    f = coriolis(planet, lat)
    u = u * np.clip(np.abs(f) / max(1e-6, abs(2.0 * planet.omega_s * 0.5)), 0.4, 1.0)
    # 地形偏转
    gy, gx = np.gradient(elev.astype(np.float32))
    v = 0.4 * (np.roll(gx, 1, axis=0) - gx)
    # 季风：低压暖陆吸海风
    land = (elev >= 0).astype(np.float32)
    gy_t, gx_t = np.gradient(temp * land)
    v = v - 0.5 * gy_t
    mag = np.sqrt(u ** 2 + v ** 2)
    mag = np.where(mag > 0, mag, 1.0)
    u = u / mag * np.clip(mag, 0.3, 1.5)
    v = v / mag * np.clip(mag, 0.3, 1.5)
    return u.astype(np.float32), v.astype(np.float32)
