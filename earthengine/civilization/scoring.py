"""文明评分 (28)：水可达、温度、地形难度、土壤、资源、河流/海岸可达、宜居性。

现有 ``civilization_index()`` 输出 [0,1] float32、``civilization_points()`` 隔离度
筛选，保留并接入河流/海岸可达性。
"""

from __future__ import annotations

import numpy as np

ARABLE_BIOMES = {
    "Temperate forest", "Temperate broadleaf", "Mediterranean",
    "Humid continental", "Tropical savanna", "Steppe",
}


def civilization_index(temp, precip, elev, koppen_code, biome_name,
                       river_mask=None):
    """宜居性评分 [0,1]。

    arable（温带/草原生物群系）、淡水（降水 400-2000）、舒适（温度 0-30）、
    资源（近岸且不高）+ 河流可达加成。
    """
    land = (elev >= 0).astype(np.float32)
    uniq, inverse = np.unique(biome_name, return_inverse=True)
    arable = np.isin(uniq, list(ARABLE_BIOMES))[inverse.reshape(biome_name.shape)]
    arable = arable.astype(np.float32)
    water = np.clip((precip - 300) / 1200.0, 0, 1) * np.clip((2200 - precip) / 1000.0, 0, 1)
    comfort = np.exp(-((temp - 15.0) ** 2) / (2 * 12.0 ** 2))
    res = np.clip(1.0 - np.abs(elev) / 3000.0, 0, 1)
    score = (0.35 * arable + 0.25 * water + 0.25 * comfort + 0.15 * res) * land
    if river_mask is not None:
        score = score + 0.15 * river_mask.astype(np.float32) * land
    return np.clip(score, 0, 1).astype(np.float32)


def civilization_points(score, min_spacing=24, max_points=60):
    """隔离度筛选的宜居热点。返回 [(y, x, score), ...]。"""
    h, w = score.shape
    pts = []
    cand = score.copy()
    thr = 0.55
    r2 = min_spacing * min_spacing
    while True:
        yx = np.unravel_index(np.argmax(cand), cand.shape)
        s = cand[yx]
        if s < thr:
            break
        y, x = int(yx[0]), int(yx[1])
        pts.append((y, x, float(s)))
        y0, y1 = max(0, y - min_spacing), min(h, y + min_spacing + 1)
        x0, x1 = max(0, x - min_spacing), min(w, x + min_spacing + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        cand[y0:y1, x0:x1][(yy - y) ** 2 + (xx - x) ** 2 <= r2] = 0.0
        if len(pts) > max_points:
            break
    return pts
