"""地形细节 (detail.py)：分形海岸、山脊纹理、程序化高程。

有无 AI 都执行的后处理 (24.3)；Terrain Diffusion 只负责微纹理，地理结构来自
terrain/planner。
"""

from __future__ import annotations

import numpy as np

from earthengine.pipeline.world import (
    TT_MOUNTAIN, TT_PLATEAU, TT_HILL, TT_PLAIN,
)
from earthengine.terrain.noise import fbm
from earthengine.terrain.planner import terrain_region_map


def apply_geography(elev, land_mask, geo_elev, detail_sigma=6.0, detail_gain=0.35):
    """大尺度结构来自 geography，微纹理来自扩散。"""
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover
        return np.maximum(geo_elev, 3.0).astype(np.float32)
    detail = elev - gaussian_filter(elev.astype(np.float32), sigma=detail_sigma)
    out = elev.copy()
    m = land_mask
    out[m] = np.maximum(geo_elev[m] + detail[m] * detail_gain, 3.0)
    return out.astype(np.float32)


def ridge_detail(elev, terrain_class, seed):
    """精致山脊纹理（今怀古风格）：脊状多重分形 + 域扭曲。"""
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover
        return elev
    h, w = elev.shape
    wx = gaussian_filter(fbm((h, w), base_scale=48, octaves=3, seed=seed + 611), 2.0)
    wy = gaussian_filter(fbm((h, w), base_scale=48, octaves=3, seed=seed + 612), 2.0)
    n = fbm((h, w), base_scale=22, octaves=6, seed=seed + 613)
    ys, xs = np.mgrid[0:h, 0:w]
    xs2 = np.clip(xs + wx * w * 0.04, 0, w - 1.001)
    ys2 = np.clip(ys + wy * h * 0.06, 0, h - 1.001)
    x0 = xs2.astype(np.int32); y0 = ys2.astype(np.int32)
    fx = xs2 - x0; fy = ys2 - y0
    x1 = (x0 + 1) % w; y1 = np.minimum(y0 + 1, h - 1)
    n = (n[y0, x0] * (1 - fx) * (1 - fy) + n[y0, x1] * fx * (1 - fy)
         + n[y1, x0] * (1 - fx) * fy + n[y1, x1] * fx * fy)
    ridge = 1.0 - np.abs(2.0 * n - 1.0)
    ridge = ridge * ridge
    amp = np.zeros((h, w), dtype=np.float32)
    amp[terrain_class == TT_MOUNTAIN] = 700.0
    amp[terrain_class == TT_PLATEAU] = 320.0
    amp[terrain_class == TT_HILL] = 130.0
    amp[terrain_class == TT_PLAIN] = 30.0
    return elev + (ridge - 0.55) * 2.0 * amp


def fractalize_coast(land_mask, seed, band=None, target_land=None):
    """多尺度分形海岸线：大湾 → 小湾 → 细糙。

    只细化海岸带，**保留大陆主体布局**（不重写整幅海陆掩膜）。传入的
    ``land_mask`` 若是已验证过的大陆布局，其大陆间距/孤立大陆/大洋结构
    不会被破坏。

    ``target_land`` 给定时，在海岸带内微调陆比使其贴近目标（尽力而为，
    不强锁）：把海岸带内 field 最高的一批海/陆像素翻转。大陆内部与深海
    始终保持不变。
    """
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except Exception:  # pragma: no cover
        return land_mask
    h, w = land_mask.shape
    if band is None:
        band = max(3.0, w / 256.0)
    lm = land_mask.astype(bool)
    sdf = (distance_transform_edt(lm).astype(np.float32)
           - distance_transform_edt(~lm).astype(np.float32))
    wx = gaussian_filter(fbm((h, w), base_scale=48, octaves=3, seed=seed + 811), 2.0)
    wy = gaussian_filter(fbm((h, w), base_scale=48, octaves=3, seed=seed + 812), 2.0)
    n1 = fbm((h, w), base_scale=max(8, w // 10), octaves=3, seed=seed + 813)
    n2 = fbm((h, w), base_scale=max(4, w // 36), octaves=3, seed=seed + 814)
    n3 = fbm((h, w), base_scale=max(2, w // 110), octaves=2, seed=seed + 815)
    ys, xs = np.mgrid[0:h, 0:w]
    xs2 = np.clip(xs + wx * w * 0.03, 0, w - 1.001).astype(np.int32)
    ys2 = np.clip(ys + wy * h * 0.05, 0, h - 1.001).astype(np.int32)
    n1 = n1[ys2, xs2]; n2 = n2[ys2, xs2]; n3 = n3[ys2, xs2]
    warp = ((n1 - 0.5) * band * 3.2 + (n2 - 0.5) * band * 1.6
            + (n3 - 0.5) * band * 0.8).astype(np.float32)
    t = np.clip(1.0 - np.abs(sdf) / (band * 6.0), 0.0, 1.0) ** 0.7
    field = sdf + warp * t
    # 海岸带：sdf 接近 0、扰动权重 t 高的窄带；大陆内部/深海权重 t≈0 保持原样。
    coast = t > 0.08
    out = lm.copy()
    if coast.any():
        out[coast] = (field[coast] > 0.0).astype(bool)
    if target_land is not None and coast.any():
        # 海岸带内微调陆比逼近目标（尽力而为；不足则保持现状）
        need = float(target_land) * out.size - float(out.sum())
        if need > 1.0:
            cand = coast & ~out
            if cand.any():
                idx = np.argwhere(cand)
                order = np.argsort(-field[cand])[: int(need) + 1]
                out[idx[order, 0], idx[order, 1]] = True
        elif need < -1.0:
            cand = coast & out
            if cand.any():
                idx = np.argwhere(cand)
                order = np.argsort(field[cand])[: int(-need) + 1]
                out[idx[order, 0], idx[order, 1]] = False
    return out.astype(bool)


def procedural_elevation(land_mask, boundaries, seed):
    """无扩散模型的完整分辨率高程。

    地理地形 + 细多倍频纹理；绕过神经渲染，CPU-only、高速、内存安全。
    返回 ``(elev, geo_elev, terrain_class)``。
    """
    geo_elev, terrain_class = terrain_region_map(land_mask, boundaries, seed)
    fine = fbm(geo_elev.shape, base_scale=10, octaves=5, seed=seed + 701)
    elev = geo_elev + fine * np.where(land_mask, 60.0, 70.0).astype(np.float32)
    elev = ridge_detail(elev, terrain_class, seed)
    elev[land_mask] = np.maximum(elev[land_mask], 3.0)
    elev[~land_mask] = np.minimum(elev[~land_mask], -1.0)
    return elev.astype(np.float32), geo_elev, terrain_class
