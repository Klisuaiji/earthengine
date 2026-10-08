"""降水 (20)：纬度基础带 + 迎风坡抬升 + 雨影 + 内陆干旱。"""

from __future__ import annotations

import numpy as np

from earthengine.planet.sphere import _lat_deg


def precipitation(u: np.ndarray, v: np.ndarray, elev: np.ndarray, h: int) -> np.ndarray:
    """降水 (mm/yr)。"""
    lat = _lat_deg(h)
    abslat = np.abs(lat)
    base = (2500.0 * np.exp(-((abslat / 12.0) ** 2))
            + 800.0 * np.exp(-(((abslat - 60.0) / 18.0) ** 2))
            + 300.0)
    gy, gx = np.gradient(elev.astype(np.float32))
    # 迎风坡（风上方向抬升）
    windward = -(u * gx + v * gy) / 500.0
    oro = 1.0 + 1.6 * np.clip(windward, 0.0, 1.5)
    shadow = 1.0 - 0.4 * np.clip(-windward, 0.0, 1.5)
    arid = 1.0 - 0.35 * np.clip(elev / 4000.0, 0.0, 1.0)
    arid = arid * (1.0 - 0.25 * np.clip(np.abs(lat) - 20.0, 0.0, 1.0) / 50.0)
    precip = base * oro * shadow * arid
    return precip.astype(np.float32)
