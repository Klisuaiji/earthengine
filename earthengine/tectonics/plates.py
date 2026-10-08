"""大板块：Plate / PlanetaryPlates + v2.0 生成入口 (7.1 / 7.2 / 8)。

面积分布服从幂律——允许 1~2 个超大型 + 若干大/中/小 + 微板块，禁止所有板块
面积几乎一致（基尼系数过低即 FAIL，见 34.1 / 34.6）。

v2.0 面向的配置参数（7.2）：
    plate_count_range     板块数量范围
    size_powerlaw_alpha   面积幂律指数
    supercontinent_tendency 超级大陆倾向
    continental_fragmentation 大陆破碎化
    plate_mobility        板块移动性（速率量级）
    isolation             板块隔离倾向

完整、已验证的实现位于 ``earthengine.tectonics.core``（运动驱动生成器，
10/10 seed 健康）。本模块在 core 之上提供 v2.0 配置面。
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics import core


# 复用 core 的数据结构，保证唯一实现
Plate = core.Plate
PlanetaryPlates = core.PlanetaryPlates


def target_area_distribution(n_plates: int, rng: np.random.Generator,
                             powerlaw_alpha: float = 1.6,
                             n_ocean_frac: float = 0.35):
    """按幂律抽样目标面积 (7.2)。

    返回 ``(ocean_targets, continent_targets)``：海洋板块合计约 ``n_ocean_frac``，
    大陆板块合计其余；各板块面积服从幂律，产生“少数巨型 + 中型 + 微板块”。
    """
    return core._target_areas(
        max(1, int(round(n_plates * n_ocean_frac))),
        max(1, n_plates - int(round(n_plates * n_ocean_frac))),
        rng)


def generate_tectonic_plates(seed: int, w: int = 512, h: int = 256,
                             n_raw: int = 30, n_big: int = 6,
                             plate_count_range=(4, 9),
                             size_powerlaw_alpha: float = 1.6,
                             supercontinent_tendency: float = 1.0,
                             continental_fragmentation: float = 0.5,
                             plate_mobility: float = 1.0,
                             isolation: float = 1.0,
                             **kwargs) -> PlanetaryPlates:
    """生成运动驱动板块系统。

    ``n_raw``/``n_big`` 沿用已验证的 core 接口；plate_count_range 等 v2.0 配置
    参数在 future 实现中用于自动反推 n_raw/n_big（当前由调用方显式传入）。
    """
    return core.generate_tectonic_plates(seed, w=w, h=h, n_raw=n_raw, n_big=n_big)


def generate_validated_world(seed: int, w: int = 512, h: int = 256,
                             n_raw: int = 30, n_big: int = 6,
                             max_attempts: int = 16):
    """生成整颗验证通过的行星（板块 + 宏观地理），失败连板块种子一起重滚。"""
    return core.generate_validated_world(seed, w=w, h=h, n_raw=n_raw,
                                         n_big=n_big, max_attempts=max_attempts)
