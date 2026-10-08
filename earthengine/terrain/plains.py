"""平原 / 高原 / 盆地 / 沿海 生成 (16 / 17 / 18)。"""

from __future__ import annotations

import numpy as np

from earthengine.terrain.noise import fbm


def plains_hills(land_mask, seed, h, w):
    """沿海平原 → 内陆丘陵的渐变压高带。返回 (h,w) float32 高程。"""
    noise = fbm((h, w), base_scale=28, octaves=5, seed=seed + 301)
    inland = _d_ocean_rel(land_mask)
    inland = np.clip((inland - 18.0) / 150.0, 0.0, 1.0)
    plains = 25.0 + noise * 110.0
    hills = 260.0 + noise * 520.0
    return plains + (hills - plains) * inland * (0.35 + 0.65 * noise)


def plateaus(land_mask, seed, h, w):
    """1-2 块大陆内部高原（Tibet 风格）。返回 (h,w) float32 抬升。"""
    noise = fbm((h, w), base_scale=28, octaves=5, seed=seed + 301)
    broad = fbm((h, w), base_scale=96, octaves=3, seed=seed + 302)
    d_ocean = _d_ocean_rel(land_mask)
    rng = np.random.default_rng((seed * 31337) & 0x7FFFFFFF)
    out = np.zeros((h, w), dtype=np.float32)
    for _ in range(2):
        if rng.random() < 0.3:
            continue
        cand = np.argwhere(land_mask & (d_ocean > 50))
        if not len(cand):
            break
        cy, cx = cand[rng.integers(len(cand))]
        r_px = rng.uniform(25.0, 48.0)
        yy, xx = np.ogrid[:h, :w]
        dist2 = (yy - cy) ** 2 + np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) ** 2
        cap = np.clip(1.0 - dist2 / (r_px * r_px), 0.0, 1.0)
        out = out + cap.astype(np.float32) * (1150.0 + broad * 300.0)
    return out


def basins(land_mask, seed, h, w):
    """0-2 块内陆封闭盆地（Sichuan 风格）。返回 (mask, 高程压降)。"""
    broad = fbm((h, w), base_scale=96, octaves=3, seed=seed + 302)
    d_ocean = _d_ocean_rel(land_mask)
    rng = np.random.default_rng((seed * 31337) & 0x7FFFFFFF)
    basin_mask = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=np.float32)
    for _ in range(2):
        if rng.random() < 0.25:
            continue
        cand = np.argwhere(land_mask & (d_ocean > 40))
        if not len(cand):
            break
        cy, cx = cand[rng.integers(len(cand))]
        r_px = rng.uniform(14.0, 26.0)
        yy, xx = np.ogrid[:h, :w]
        dist2 = (yy - cy) ** 2 + np.minimum(np.abs(xx - cx), w - np.abs(xx - cx)) ** 2
        basin = np.clip(1.0 - dist2 / (r_px * r_px), 0.0, 1.0)
        out = out - basin.astype(np.float32) * (420.0 + broad * 160.0)
        basin_mask |= basin > 0.35
    return basin_mask, out


def _d_ocean_rel(land_mask):
    """每像素到最近海洋像素的距离（px）。"""
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(land_mask).astype(np.float32)
    except Exception:  # pragma: no cover
        out = np.zeros(land_mask.shape, dtype=np.float32)
        cur = land_mask.astype(bool)
        d = 0.0
        while cur.any():
            out[cur] = d
            cur = _shrink(cur)
            d += 1.0
        return out


def _shrink(mask):
    out = mask.copy()
    out &= ~(np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0)
             | np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1))
    return out
