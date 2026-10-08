"""太阳辐射 (20)：行星参数参与模拟 (6.2)。

太阳常数 S = L / (4π d²)。轴向倾斜调制季节温度带边界。
"""

from __future__ import annotations

import numpy as np

from earthengine.planet.parameters import Planet
from earthengine.planet.sphere import _lat_deg


def solar_radiation(planet: Planet, lat_deg: np.ndarray) -> np.ndarray:
    """每行纬度的相对日射量（含轴倾角季节平均）。"""
    tilt = np.deg2rad(planet.axial_tilt_deg)
    # 平均年日射 ∝ cos(lat)，轴倾角扩大两极季节振幅
    base = np.cos(np.deg2rad(lat_deg))
    seasonal = tilt * np.sin(np.deg2rad(np.abs(lat_deg)))
    return base + 0.3 * seasonal
