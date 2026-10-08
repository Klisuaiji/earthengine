"""地图渲染 (29)：从 WorldState 渲染所有图层。

支持：physical, plate, boundary, crust, land_sea, elevation, terrain_region,
temperature, precipitation, climate, hydrology, vegetation, civilization。
Pillow 出图（可选依赖）；缺失时纯 numpy 存 PNG。
"""

from __future__ import annotations

import numpy as np

from earthengine.pipeline.world import TERRAIN_RGB, TT_MOUNTAIN, TT_PLATEAU
from earthengine.terrain.noise import fbm
from earthengine.rendering.colors import (
    biome_palette, koppen_palette, terrain_palette, RELIEF_HIRES_LAND, PLATE_PALETTE,
)

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


def save_png(arr, out_path):
    """(h,w,3) uint8 → PNG。优先 Pillow，缺失时纯 numpy 写 PPM→rename。"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if _HAS_PIL:
        _PILImage.fromarray(arr).save(out_path)
        return out_path
    # 纯 numpy 兜底：PNG 用 zlib 手写（简单 RGBA PNG）
    import zlib, struct
    h, w = arr.shape[:2]
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += arr[y].tobytes()
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw)))
    png += chunk(b"IEND", b"")
    with open(out_path, "wb") as f:
        f.write(png)
    return out_path


def _hillshade(elev, az=315.0, alt=45.0, vert_exag=1.0):
    dy, dx = np.gradient(elev.astype(np.float32) * vert_exag)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(dy, -dx)
    azr, altr = np.deg2rad(az), np.deg2rad(alt)
    shade = (np.cos(altr) * np.cos(slope)
             + np.sin(altr) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(shade, 0.0, 1.0)


def _hypsometric(elev, lo=-8000, hi=4000):
    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    sea = elev < 0
    t = np.clip(-elev[sea] / 6000.0, 0, 1)
    rgb[sea, 0] = (20 + t * (60 - 20)) / 255.0
    rgb[sea, 1] = (50 + t * (130 - 50)) / 255.0
    rgb[sea, 2] = (90 + t * (190 - 90)) / 255.0
    land = ~sea
    t = np.clip(elev[land] / 3000.0, 0, 1)
    rgb[land, 0] = np.interp(t, [0, .3, .6, .85, 1], [110, 220, 190, 160, 245]) / 255.0
    rgb[land, 1] = np.interp(t, [0, .3, .6, .85, 1], [200, 225, 170, 145, 245]) / 255.0
    rgb[land, 2] = np.interp(t, [0, .3, .6, .85, 1], [80, 100, 70, 140, 245]) / 255.0
    return rgb


# --------------------------------------------------------------------------
# 各图层渲染器
# --------------------------------------------------------------------------
def render_terrain_types(terrain_class, out_path, seed=97):
    cls = terrain_class.astype(np.int32)
    pal = np.zeros((12, 3), dtype=np.float32)
    for k, col in TERRAIN_RGB.items():
        pal[k] = col
    h, w = cls.shape
    rgb = pal[cls].copy()
    fine = (fbm((h, w), base_scale=6, octaves=4, seed=seed) - 0.5) * 0.20
    broad = (fbm((h, w), base_scale=26, octaves=2, seed=seed + 3) - 0.5) * 0.14
    shade = 1.0 + fine + broad
    try:
        from scipy.ndimage import gaussian_filter
        streak = gaussian_filter(fine, sigma=(0.5, 4.0)) * 1.6
        streaked = (cls == TT_MOUNTAIN) | (cls == TT_PLATEAU)
        shade[streaked] = 1.0 + streak[streaked] + broad[streaked]
    except Exception:
        pass
    rgb *= shade[..., None]
    return save_png(rgb, out_path)


def render_terrain(elev, ice, out_path, vert_exag=2.5):
    base = _hypsometric(elev)
    shade = _hillshade(elev, vert_exag=vert_exag)
    comp = np.clip(base * (0.4 + 0.6 * shade[..., None]), 0, 1)
    comp[ice] = np.array([0.92, 0.95, 0.98])
    return save_png(comp * 255, out_path)


def render_relief_hires(elev, out_path, rivers=None, acc=None, min_acc=1.0,
                        land_mask=None, ice=None, vert_exag=3.0):
    try:
        from scipy.ndimage import gaussian_filter, distance_transform_edt
    except Exception:
        gaussian_filter = distance_transform_edt = None
    elev = elev.astype(np.float32)
    h, w = elev.shape
    land = (elev >= 0) if land_mask is None else land_mask.astype(bool)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    sea = ~land
    if sea.any() and distance_transform_edt is not None:
        d_in = distance_transform_edt(land).astype(np.float32)
        glow = np.exp(-d_in / (w * 0.02))
        depth = np.clip(-elev[sea] / 5500.0, 0, 1)
        rgb[sea, 0] = np.interp(depth, [0, 1], [0.42, 0.09])
        rgb[sea, 1] = np.interp(depth, [0, 1], [0.80, 0.38])
        rgb[sea, 2] = np.interp(depth, [0, 1], [0.86, 0.66])
        rgb[sea] = np.clip(rgb[sea] + glow[sea, None] * np.array([0.10, 0.16, 0.10]), 0, 1)
    if land.any():
        t = np.clip(elev[land] / 5000.0, 0, 1)
        stops = RELIEF_HIRES_LAND[:, 0]
        for c in range(3):
            rgb[land, c] = np.interp(t, stops, RELIEF_HIRES_LAND[:, 1 + c])
    shade = _hillshade(elev, vert_exag=5.0)
    if gaussian_filter is not None:
        shade = gaussian_filter(shade, 0.35)
    rgb *= np.clip(0.22 + 0.95 * shade[..., None], 0.20, 1.35)
    if rivers is not None and rivers.any() and gaussian_filter is not None:
        rmask = rivers & land
        if acc is not None:
            lw = np.zeros((h, w), dtype=np.float32)
            lw[rmask] = np.clip(np.log2(acc[rmask] / max(1.0, min_acc)) * 0.6 + 0.8, 0.5, 2.6)
            lw = gaussian_filter(lw, 0.8)
            strength = np.clip(lw * 0.55 + gaussian_filter(rmask.astype(np.float32), 1.0) * 0.25, 0, 0.92)
            rgb = rgb * (1 - strength[..., None]) + np.array([0.15, 0.33, 0.60]) * strength[..., None]
    if ice is not None:
        rgb[ice] = np.clip(np.array([0.90, 0.93, 0.96]) * np.clip(0.75 + 0.35 * shade[..., None], 0, 1.1)[ice], 0, 1)
    return save_png(np.clip(rgb, 0, 1) * 255, out_path)


def render_satellite(elev, biome_name, ice, out_path):
    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    land = (elev >= 0)
    sea = ~land
    d = np.clip(-elev[sea] / 6000.0, 0, 1)
    rgb[sea, 0] = 10 + (1 - d) * 20
    rgb[sea, 1] = 40 + (1 - d) * 60
    rgb[sea, 2] = 90 + (1 - d) * 90
    pal = biome_palette(biome_name)
    rgb[land] = pal[land]
    shade = _hillshade(elev, vert_exag=2.0)
    rgb[land] = np.clip(rgb[land] * (0.55 + 0.45 * shade[land][..., None]), 0, 1)
    rgb[ice] = np.array([0.95, 0.97, 1.0])
    return save_png(np.clip(rgb, 0, 1) * 255, out_path)


def render_planet(elev, biome_name, ice, koppen_code, out_path):
    rgb = biome_palette(biome_name).copy()
    land = (elev >= 0)
    shade = _hillshade(elev, vert_exag=3.0)
    rgb[land] = np.clip(rgb[land] * (0.6 + 0.4 * shade[land][..., None]), 0, 1)
    rgb[ice] = np.array([0.96, 0.98, 1.0])
    return save_png(np.clip(rgb, 0, 1) * 255, out_path)


def render_plates(plates, out_path):
    """板块图：每块板块一种颜色。"""
    h, w = plates.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    n = int(plates.max()) + 1
    for i in range(n):
        rgb[plates == i] = PLATE_PALETTE[i % len(PLATE_PALETTE)]
    return save_png(rgb * 255, out_path)


def render_boundary(boundary_type, out_path):
    """边界类型图：内部黑、汇聚红、张裂绿、转换蓝、斜向黄。"""
    h, w = boundary_type.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    col = {1: (1, 0.25, 0.25), 2: (0.3, 1, 0.3), 3: (0.3, 0.5, 1), 4: (1, 1, 0.3)}
    for k, c in col.items():
        rgb[boundary_type == k] = c
    return save_png(rgb * 255, out_path)


def render_land_sea(land_mask, out_path):
    h, w = land_mask.shape
    rgb = np.where(land_mask[..., None], np.array([0.55, 0.75, 0.45], dtype=np.float32),
                   np.array([0.25, 0.45, 0.80], dtype=np.float32))
    return save_png(np.clip(rgb, 0, 1) * 255, out_path)


def render_crust(crust_type, out_path):
    """地壳图：陆=棕褐、洋=深蓝、过渡=青。"""
    h, w = crust_type.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[crust_type == 0] = (0.55, 0.42, 0.25)
    rgb[crust_type == 1] = (0.15, 0.25, 0.55)
    rgb[crust_type == 2] = (0.30, 0.70, 0.70)
    return save_png(rgb * 255, out_path)


def render_earth_style(elev, temp, precip, out_path, rivers=None, acc=None,
                       min_acc=1.0, land_mask=None, ice=None, seed=17,
                       vert_exag=4.0):
    """卫星-物理地图风格渲染（真实地球感：沙漠、雨林、苔原、极冰）。"""
    try:
        from scipy.ndimage import gaussian_filter, distance_transform_edt
    except Exception:
        return render_satellite(elev, None, ice if ice is not None else (elev < 0), out_path)
    elev = elev.astype(np.float32)
    h, w = elev.shape
    land = (elev >= 0) if land_mask is None else land_mask.astype(bool)
    t = temp.astype(np.float32)
    p = precip.astype(np.float32)
    tex = fbm((h, w), base_scale=14, octaves=4, seed=seed)
    tex2 = fbm((h, w), base_scale=64, octaves=3, seed=seed + 5)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    sea = ~land
    if sea.any():
        d_in = distance_transform_edt(land).astype(np.float32)
        glow = np.exp(-d_in / (w * 0.030))
        depth = np.clip(-elev[sea] / 5500.0, 0, 1)
        rgb[sea, 0] = np.interp(depth, [0, 0.15, 1], [0.45, 0.18, 0.06])
        rgb[sea, 1] = np.interp(depth, [0, 0.15, 1], [0.75, 0.52, 0.30])
        rgb[sea, 2] = np.interp(depth, [0, 0.15, 1], [0.82, 0.72, 0.55])
        rgb[sea] += glow[sea, None] * np.array([0.05, 0.09, 0.07])
    if land.any():
        t = gaussian_filter(t, 4.0)
        p = gaussian_filter(p, 7.0)
        tl = t[land]; pl = p[land]; xl = tex[land]; x2 = tex2[land]
        from earthengine.planet.sphere import _lat_deg
        lat_deg = _lat_deg(h)
        latband = np.exp(-(((np.abs(lat_deg) - 26.0) / 11.0) ** 2)).astype(np.float32)
        latband = np.broadcast_to(latband, (h, w))
        d_oc = distance_transform_edt(land).astype(np.float32)
        cont = np.clip((d_oc - 70.0) / 130.0, 0.0, 1.0).astype(np.float32)
        dry_zone = np.maximum(latband, 0.85 * cont)
        lb = latband[land]; cz = cont[land]; dz = dry_zone[land]
        cold = np.clip((6.0 - tl) / 14.0, 0, 1)
        peff = pl * (1.0 - 0.75 * lb) * (1.0 - 0.55 * cz)
        arid = np.clip((520.0 - peff) / 480.0, 0, 1) * np.clip((tl + 2.0) / 12.0, 0, 1)
        humid = np.clip((pl - 1100.0) / 900.0, 0, 1) * np.clip((tl - 10.0) / 8.0, 0, 1) * (1 - dz)
        base = np.empty((tl.size, 3), dtype=np.float32)
        base[:, 0] = 0.42 + 0.10 * x2
        base[:, 1] = 0.60 + 0.08 * x2
        base[:, 2] = 0.30 + 0.06 * x2
        sand = np.stack([0.82 + 0.07 * xl, 0.72 + 0.06 * xl, 0.45 + 0.06 * xl], axis=1)
        jungle = np.stack([0.13 + 0.04 * xl, 0.38 + 0.05 * xl, 0.16 + 0.04 * xl], axis=1)
        tundra = np.stack([0.58 + 0.06 * xl, 0.56 + 0.06 * xl, 0.50 + 0.06 * xl], axis=1)
        snow = np.stack([0.93 + 0.015 * xl, 0.94 + 0.015 * xl, 0.96 + 0.015 * xl], axis=1)
        c = base * (1 - arid[:, None]) + sand * arid[:, None]
        c = c * (1 - humid[:, None]) + jungle * humid[:, None]
        c = c * (1 - np.clip(cold, 0, 0.85)[:, None]) + tundra * np.clip(cold, 0, 0.85)[:, None]
        sn = np.clip((-tl - 2.0) / 5.0, 0, 1)
        c = c * (1 - sn[:, None]) + snow * sn[:, None]
        gy, gx = np.gradient(elev)
        slope = np.sqrt(gy * gy + gx * gx)
        rock_f = (np.clip((slope - 55.0) / 90.0, 0.0, 1.0)
                  * np.clip((elev - 500.0) / 900.0, 0.0, 1.0))[land]
        rock = np.stack([0.46 + 0.05 * xl, 0.41 + 0.05 * xl, 0.37 + 0.05 * xl], axis=1)
        c = c * (1 - rock_f[:, None]) + rock * rock_f[:, None]
        rgb[land] = c
    detail = (fbm((h, w), base_scale=5, octaves=3, seed=seed + 91) - 0.5) * 45.0
    detail += (fbm((h, w), base_scale=14, octaves=4, seed=seed + 92) - 0.5) * 120.0
    z_land = elev + detail
    z_sea = gaussian_filter(elev.astype(np.float32), 6.0)
    z_shade = np.where(land, z_land, z_sea).astype(np.float32)
    shade = _mapgen_light(z_shade, az_deg=315.0, overhead=2.5)
    shade = np.where(land, shade, np.float32(1.0)).astype(np.float32)
    rgb *= np.clip(shade[..., None], 0.60, 1.25)
    gray = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip(gray + (rgb - gray) * 1.08, 0.0, 1.0)
    if rivers is not None and rivers.any() and acc is not None:
        rmask = rivers & land
        lw = np.zeros((h, w), dtype=np.float32)
        lw[rmask] = np.clip(np.log2(acc[rmask] / max(1.0, min_acc)) * 0.62 + 1.0, 0.7, 3.2)
        lw = gaussian_filter(lw, 0.9)
        strength = np.clip(lw * 0.72 + gaussian_filter(rmask.astype(np.float32), 1.0) * 0.3, 0, 0.94)
        rgb = rgb * (1 - strength[..., None]) + np.array([0.12, 0.25, 0.42]) * strength[..., None]
    if ice is not None:
        ice_f = gaussian_filter(ice.astype(np.float32), 2.5)
        ice_rgb = np.array([0.89, 0.92, 0.95])
        shade_i = np.clip(0.80 + 0.30 * shade[..., None], 0, 1.1)
        rgb = rgb * (1 - ice_f[..., None]) + ice_rgb * shade_i * ice_f[..., None]
    return save_png(np.clip(rgb, 0, 1) * 255, out_path)


def _mapgen_light(z_m, az_deg=315.0, overhead=2.5, ambient=0.45,
                  slope_z=0.15, flat_z=1.15, h_scale=1.0 / 3000.0):
    z = z_m.astype(np.float32) * np.float32(h_scale)
    h, w = z.shape
    zN = np.empty_like(z); zS = np.empty_like(z)
    zN[1:, :] = z[:-1, :]; zN[0, :] = z[0, :]
    zS[:-1, :] = z[1:, :]; zS[-1, :] = z[-1, :]
    zE = np.roll(z, -1, axis=1)
    zW = np.roll(z, 1, axis=1)
    dzNS = np.float32(0.5) * (zS - zN)
    dzEW = np.float32(0.5) * (zE - zW)
    zb = np.float32(overhead * (1.0 / w + 1.0 / h))
    n = np.sqrt(dzNS * dzNS + dzEW * dzEW + zb * zb)
    nz = zb / n
    a = np.deg2rad(az_deg)
    lx, ly = np.cos(a), np.sin(a)
    lz = slope_z + (flat_z - slope_z) * nz
    ll = np.sqrt(lx * lx + ly * ly + lz * lz)
    light = ambient + np.clip((dzNS / n) * (lx / ll) +
                              (dzEW / n) * (ly / ll) + nz * (lz / ll), 0.0, None)
    return light.astype(np.float32)


def render_hydrology(river_mask, out_path, lakes=None, bg_dark=False):
    """水文图：河蓝、湖深蓝、背景可选黑。"""
    h, w = river_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if bg_dark:
        rgb[:] = 0.08
    rgb[river_mask] = (0.15, 0.35, 0.75)
    if lakes is not None:
        rgb[lakes] = (0.10, 0.25, 0.55)
    return save_png(rgb * 255, out_path)


def render_float_field(field, out_path, lo=0.0, hi=1.0, cmap="viridis"):
    """任意 float 场渲染（温度/降水/宜居性等）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm as _cm
        norm = (np.clip(field, lo, hi) - lo) / (hi - lo)
        rgb = (_cm.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
        return save_png(rgb, out_path)
    except Exception:  # pragma: no cover
        norm = np.clip((field - lo) / (hi - lo), 0, 1)
        rgb = np.stack([norm, norm * 0.6, norm * 0.3], axis=-1)
        return save_png(rgb * 255, out_path)
