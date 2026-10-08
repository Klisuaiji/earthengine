"""气候综合：Köppen 分类、冰盖、洋流 (20 / 21 / 22)。"""

from __future__ import annotations

import math

import numpy as np

from earthengine.planet.sphere import _lat_deg

KOPPEN_RGB = {
    "Af": (60, 160, 60), "Am": (90, 190, 70), "Aw": (150, 200, 90),
    "BWh": (225, 205, 130), "BWk": (205, 190, 160), "BS": (220, 200, 150),
    "Cfa": (80, 170, 120), "Cfb": (110, 190, 140), "Csb": (120, 185, 120),
    "Dfb": (70, 150, 170), "Dfc": (90, 140, 190), "Dfa": (60, 160, 150),
    "ET": (200, 220, 230), "EF": (235, 240, 245), "Ocean": (30, 80, 150),
}


def koppen(temp, precip, elev, h):
    """简化 Köppen 气候分类。返回 (code, label) 字符串数组。"""
    lat = _lat_deg(h)
    land = (elev >= 0).astype(np.float32)
    amp = 6.0 + 14.0 * np.abs(lat) / 90.0 + 8.0 * land
    t_cold = temp - amp
    t_warm = temp + amp * 0.6
    code = np.empty(temp.shape, dtype=object)
    code[:] = "Ocean"
    arid_thr = 10.0 * t_warm
    is_arid = precip < arid_thr
    A = t_cold > 18
    E = t_warm < 10
    C = (t_cold > -3) & (t_warm >= 10) & ~A & ~E
    D = (~A) & (~E) & (~C)

    code[A & (precip >= 60)] = "Af"
    code[A & (precip < 60) & (precip >= 25)] = "Am"
    code[A & (precip < 25)] = "Aw"
    code[E & (t_warm < 0)] = "EF"
    code[E & (t_warm >= 0)] = "ET"
    code[C & (precip >= 50)] = "Cfb"
    code[C & (precip < 50) & (t_cold >= 0)] = "Cfa"
    code[C & (precip < 50) & (t_cold < 0)] = "Csb"
    code[D & (t_warm >= 10)] = "Dfb"
    code[D & (t_warm < 10)] = "Dfc"
    code[is_arid & (t_warm >= 18)] = "BWh"
    code[is_arid & (t_warm < 18)] = "BWk"
    code[is_arid & (precip >= 0.5 * arid_thr)] = "BS"
    return code


def ice_layer(temp, elev, h, sst=None):
    """冰盖（极地）+ 冰川（高寒），可选 SST 抑制近岸冰。"""
    lat = _lat_deg(h)
    ice = np.zeros(temp.shape, dtype=bool)
    ice |= (temp < 0) & (np.abs(lat) > 60)
    ice |= (elev > 2500) & (temp < 0)
    if sst is not None:
        ice &= ~(sst > 2)
    return ice


def ocean_currents(u, v, elev, h, w, iterations=2):
    """风驱表层洋流（Ekman ~45° 偏转）+ 西边界强化，返回 (cu, cv, sst)。"""
    land = (elev >= 0).astype(np.float32)
    lat = _lat_deg(h)
    hemi = np.sign(lat)
    cu = u * math.cos(math.radians(45.0)) - hemi * v * math.sin(math.radians(45.0))
    cv = v * math.cos(math.radians(45.0)) + hemi * u * math.sin(math.radians(45.0))
    speed = np.sqrt(cu ** 2 + cv ** 2)
    east_of_land = (np.roll(land, 1, axis=1) > 0.5) & (land < 0.5)
    speed = speed * (1.0 + 0.6 * east_of_land.astype(np.float32))
    speed = speed * (1.0 - land)
    mag = np.sqrt(cu ** 2 + cv ** 2)
    mag = np.where(mag > 1e-6, mag, 1.0)
    cu = cu / mag * speed
    cv = cv / mag * speed
    sst = (30.0 * np.cos(np.deg2rad(lat)) - 2.0).astype(np.float32)
    sst = sst * (1.0 - land)
    for _ in range(iterations):
        adv = (np.roll(sst, 1, axis=1) - sst) * np.sign(cu + 1e-9) * 0.15
        adv = adv + (np.roll(sst, 1, axis=0) - sst) * np.sign(cv + 1e-9) * 0.1
        sst = np.where(land > 0.5, sst, sst + adv)
    return cu.astype(np.float32), cv.astype(np.float32), sst.astype(np.float32)
