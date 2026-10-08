"""行星参数 (6.1 / 6.2)。

参数必须真正参与模拟：自转速度→科里奥利参数 f=2Ωsinφ；轴倾角→季节与温度带；
恒星距离+光度→太阳常数 S=L/(4πd²)；行星半径→网格单元实际面积；海平面→海陆比例。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Planet:
    """一个随机架空行星的物理参数。

    v2.0 之前参数散落在 ``world_gen(temps, humids, ...)`` 里、不参与模拟 (P4)。
    v2.0 把 ``Planet`` 建成一等对象，天文参数直接进入气候与地壳方程。
    """

    # --- 基本物理 ---
    radius_km: float = 6371.0            # 行星半径（地球参考）
    mass_earth: float = 1.0              # 质量（地球倍数）
    rotation_period_h: float = 24.0      # 自转周期（小时）
    axial_tilt_deg: float = 23.4         # 轴倾角（度）
    orbital_period_d: float = 365.25     # 轨道周期（天）

    # --- 恒星参数 ---
    star_distance_au: float = 1.0        # 距恒星距离（AU）
    star_luminosity: float = 1.0         # 恒星光度（太阳倍数）

    # --- 表面参数 ---
    sea_level_m: float = 0.0             # 海平面（米，相对参考）
    atmosphere: str = "habitable"        # 大气状态（habitable / thin / thick / none）
    initial_temperature_c: float = 15.0  # 初始平均温度
    ocean_fraction: float = 0.7          # 目标海洋占比（约束海陆）

    # --- 可扩展参数 ---
    atmospheric_pressure_atm: float = 1.0
    greenhouse_factor: float = 1.0
    eccentricity: float = 0.0
    albedo: float = 0.3
    solar_constant_wm2: float = field(default=1361.0, init=False)

    def __post_init__(self):
        # S = L / (4π d²)，用地球参数标定
        self.solar_constant_wm2 = (
            self.star_luminosity / (self.star_distance_au ** 2) * 1361.0)

    # ------------------------------------------------------------------
    # 导出给下游模拟的派生物理量
    # ------------------------------------------------------------------
    @property
    def omega_s(self) -> float:
        """自转角速度（rad/s）：Ω = 2π / 自转周期。"""
        return 2.0 * math.pi / (self.rotation_period_h * 3600.0)

    @property
    def gravity_m_s2(self) -> float:
        """表面重力（m/s²），质量按半径缩放近似。"""
        return 9.81 * self.mass_earth

    def coriolis(self, lat_deg) -> float:
        """科里奥利参数 f = 2 Ω sin(φ)。"""
        return 2.0 * self.omega_s * math.sin(math.radians(lat_deg))

    def cell_area_m2(self, lat_deg, dlat_deg, dlon_deg) -> float:
        """一个 (dlat × dlon) 度网格单元在纬度 lat_deg 处的实际面积（m²）。"""
        r = self.radius_km * 1000.0
        return (
            r ** 2 * math.radians(dlat_deg) * math.radians(dlon_deg)
            * math.cos(math.radians(lat_deg))
        )

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["solar_constant_wm2"] = self.solar_constant_wm2
        d["omega_s"] = self.omega_s
        return d
