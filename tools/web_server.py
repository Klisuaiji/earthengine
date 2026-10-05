#!/usr/bin/env python3
"""Flask web server for interactive Voronoi world generation with Terrain Diffusion.

Plate tectonic skeleton + diffusion-rendered real terrain + physical climate:

    plates  ->  continents  ->  boundary types
    ->  diffusion elevation (real mountains / ocean trenches)
    ->  temperature / precipitation
    ->  land/sea mask

The diffusion models are loaded once at startup and reused across requests.
Run this script in the terrain-diffusion venv (the one with torch/cuda).

Usage: python3 tools/web_server.py [--port PORT]
"""

import io
import os
import sys
import time
import argparse
import base64
from pathlib import Path

import numpy
from flask import Flask, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from worldengine.spherical_voronoi import (
    generate_spherical_world,
)
from worldengine.plate_boundaries import (
    INTERIOR, CONVERGENT, DIVERGENT, TRANSFORM, classify_boundaries,
)

from tools.diffusion_world import (
    build_pipeline,
    conditioning_from_mask,
    generate_world as generate_diffusion_world,
    render_relief,
    _colormap_png,
    compute_physical_climate,
    clamp_land_sea,
)

# Cached diffusion pipeline (loaded once, reused across requests).
# The heavy part is the 1 GB base model; rebuild per-request only changes seed
# and the coarse conditioning store.
_PIPE = None
_PIPE_SEED = None
_PIPE_DEVICE = None

app = Flask(__name__)
# Preserve insertion (pipeline) order in JSON responses instead of alphabetical
# key sorting, so the front-end renders maps in the correct generation order.
app.json.sort_keys = False

# --------------- Palettes ---------------

PLATE_PALETTE = numpy.array(
    [
        (168, 158, 124), (150, 170, 142), (181, 146, 111), (163, 150, 165), (140, 160, 176),
        (192, 175, 140), (128, 152, 130), (172, 132, 108), (154, 164, 186), (186, 168, 152),
        (138, 158, 150), (178, 158, 174), (160, 144, 118), (146, 166, 160), (170, 150, 130),
        (150, 140, 156), (158, 172, 142), (184, 162, 138), (134, 148, 162), (166, 158, 148),
    ],
    dtype=float,
)

MERGED_PALETTE = numpy.array(
    [
        (168, 158, 124), (140, 160, 176), (150, 170, 142),
        (181, 146, 111), (163, 150, 165), (155, 168, 148),
    ],
    dtype=float,
)

CONTINENT_PALETTE = numpy.array([
    (28, 62, 110),   # ocean       (label -1 -> idx 0)
    (46, 120, 52),   # continent 0 (super-continent A)
    (120, 180, 90),  # continent 1 (super-continent B)
    (200, 160, 70),  # continent 2 (mid-latitude continent)
    (90, 140, 180),  # continent 3
    (170, 110, 60),  # continent 4
    (60, 160, 150),  # continent 5
    (150, 90, 140),  # continent 6
], dtype=numpy.uint8)

BOUNDARY_PALETTE = numpy.array([
    (245, 245, 248),   # INTERIOR
    (37, 80, 190),     # CONVERGENT (blue: 海沟/造山带, 图1)
    (220, 50, 50),     # DIVERGENT (red: 海岭/断层, 图1)
    (150, 150, 150),   # TRANSFORM (grey: 转换断层)
], dtype=numpy.uint8)

# Elevation colour stops: ocean (-10 m) -> blue, land (+10 m) -> warm.
ELEV_STOPS = [
    (0.0, (40, 90, 160)), (0.499, (40, 90, 160)),
    (0.501, (150, 190, 110)), (1.0, (235, 225, 185)),
]

# Climate-type map (图4 textbook style): 12 named classes with the standard
# Chinese-atlas hues, classified from annual temp / precip / elevation.
CLIMATE_NAMES = [
    "热带雨林气候", "热带草原气候", "热带沙漠气候",
    "亚热带季风气候", "地中海气候", "温带海洋性气候",
    "温带季风气候", "温带大陆性气候", "亚寒带针叶林气候",
    "苔原气候", "冰原气候", "高原山地气候",
]
CLIMATE_PALETTE = numpy.array([
    (0, 128, 96),      # 热带雨林   teal green
    (150, 195, 115),   # 热带草原   light green
    (245, 170, 70),    # 热带沙漠   orange
    (55, 175, 80),     # 亚热带季风 vivid green
    (195, 200, 95),    # 地中海     olive
    (115, 185, 225),   # 温带海洋性 sky blue
    (35, 120, 170),    # 温带季风   deep teal blue
    (235, 225, 150),   # 温带大陆性 pale yellow
    (70, 140, 185),    # 亚寒带针叶林 steel blue
    (170, 195, 215),   # 苔原       pale blue grey
    (238, 242, 246),   # 冰原       near white
    (175, 125, 85),    # 高原山地   brown
], dtype=numpy.uint8)


def simple_koppen(temp, precip, elev):
    """12-class climate-type map from annual means (annual-only proxy for the
    textbook classification; monsoon/mediterranean seasonality is folded into
    the precip bands)."""
    t = numpy.asarray(temp, dtype=numpy.float32)
    p = numpy.asarray(precip, dtype=numpy.float32)
    out = numpy.zeros(t.shape, dtype=numpy.int32)
    alpine = numpy.asarray(elev) > 2500.0
    ice = t < -10.0
    tundra = (t < 0.0) & ~ice
    boreal = (t < 5.0) & ~tundra & ~ice
    cold_temperate = (t < 18.0) & ~boreal & ~tundra & ~ice
    tropical = t >= 20.0
    subtrop = (t >= 15.0) & ~tropical & ~cold_temperate

    out[ice] = 10
    out[tundra] = 9
    out[alpine] = 11
    out[boreal] = 8
    # cold-temperate belt: maritime vs continental by precipitation
    out[cold_temperate & (p >= 550.0)] = 5     # 温带海洋性
    out[cold_temperate & (p < 300.0)] = 7      # 温带大陆性(干旱内陆)
    out[cold_temperate & ~(p >= 550.0) & (p >= 300.0)] = 6   # 温带季风性
    # subtropics: monsoon (wet) vs mediterranean (moderately dry)
    out[subtrop & (p >= 900.0)] = 3
    out[subtrop & (p < 550.0)] = 4
    out[subtrop & (p >= 550.0) & (p < 900.0)] = 6
    # tropics: rainforest / savanna / desert
    out[tropical & (p >= 1500.0)] = 0
    out[tropical & (p < 300.0)] = 2
    out[tropical & (p >= 300.0) & (p < 1500.0)] = 1
    return out



def _get_cached_pipeline(seed, device="auto"):
    """Return a diffusion pipeline, loading it once at first request."""
    global _PIPE, _PIPE_SEED, _PIPE_DEVICE
    from diffusion_world import resolve_device
    device = resolve_device(device)
    if _PIPE is None:
        _PIPE = build_pipeline(seed=seed, device=device)
        _PIPE_SEED = seed
        _PIPE_DEVICE = device
    # If the requested seed changed, update it (rebuild is cheap once models are loaded).
    if _PIPE_SEED != seed:
        _PIPE.seed = seed
        _PIPE.rebuild()
        # rebuild() resets cond_snr to the config default (often high -> noisy
        # output). Re-assert the "trust conditioning" setting so land/sea stay
        # steered by our Voronoi coarse map instead of drifting back to noise.
        _PIPE.set_cond_snr([0.5, 0.5, 0.5, 0.5, 0.5])
        _PIPE_SEED = seed
    return _PIPE


# --------------- Colour helpers ---------------

def _array_to_rgb(array, stops):
    stops = sorted(stops, key=lambda s: s[0])
    xs = numpy.array([s[0] for s in stops], dtype=float)
    cs = numpy.array([s[1] for s in stops], dtype=float)
    a = numpy.asarray(array, dtype=float)
    amin, amax = a.min(), a.max()
    if amax == amin:
        amax = amin + 1.0
    n = (a - amin) / (amax - amin)
    return numpy.stack([numpy.interp(n, xs, cs[:, i]) for i in range(3)], axis=-1).astype(numpy.uint8)


def _img_to_b64(img_array):
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------- Core generation ---------------

def _arr_to_b64(arr, cmap=None, vmin=None, vmax=None):
    """Convert a numpy array to a base64 PNG; use colormap if provided."""
    if cmap is None:
        rgb = numpy.asarray(arr)
        if rgb.ndim == 2:
            rgb = numpy.stack([rgb] * 3, axis=-1)
        img = Image.fromarray(rgb.astype(numpy.uint8))
    else:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if vmin is None:
                vmin = float(numpy.nanpercentile(arr, 2))
            if vmax is None:
                vmax = float(numpy.nanpercentile(arr, 98))
            fig, ax = plt.subplots(figsize=(arr.shape[1] / 100.0, arr.shape[0] / 100.0), dpi=100)
            ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            buf = io.BytesIO()
            fig.savefig(buf, dpi=100, format="png")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            # matplotlib unavailable or failed: fall back to a safe false-colour ramp
            # so a missing dependency never aborts the whole world generation.
            a = numpy.asarray(arr, dtype=float)
            amin = float(a.min()); amax = float(a.max())
            if amax == amin:
                amax = amin + 1.0
            n = numpy.clip((a - amin) / (amax - amin), 0, 1)
            ramp = numpy.stack(
                [(1 - n) * 30 + n * 220, (1 - n) * 60 + n * 120, (1 - n) * 160 + n * 60],
                axis=-1,
            ).astype(numpy.uint8)
            return _img_to_b64(ramp)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_world(params):
    seed = params.get("seed", 1234567)
    w = params.get("width", 2048)
    h = params.get("height", 1024)
    n_raw = params.get("n_raw", 30)
    n_big = params.get("n_big", 6)
    domain_amp = params.get("domain_amp", 0.22)
    # "procedural": CPU-only high-res relief (default, neural rendering deferred)
    # "diffusion": Terrain Diffusion micro-texture (slow / VRAM-heavy at 2048+)
    detail = params.get("detail", "procedural")

    from worldengine import spherical_voronoi as sv
    sv.DOMAIN_AMP = domain_amp

    t0 = time.time()
    raw, merged, land_mask, continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=n_raw, n_big=n_big
    )
    # generate_spherical_world already absorbs tiny enclaves (min_frac=0.0005);
    # a second pass would be a no-op, so skip it.
    t_voronoi = time.time() - t0
    bnd = classify_boundaries(merged, seed=seed)   # needed by coastal refinement

    from planet_pipeline import (
        terrain_type_map, render_terrain_types, render_relief_hires,
        render_earth_style, _lat as _lat_deg,
    )

    if detail == "diffusion":
        # Terrain Diffusion: cached pipeline, real elevation + physical climate.
        pipe = _get_cached_pipeline(seed)
        # Build the SAME varied coarse conditioning grid that planet_pipeline uses.
        # A binary land/ocean grid gives the model no spatial gradient to steer by
        # (only ~2x4 conditioning cells), so its output collapses toward the prior
        # and looks like noise.  The tectonic coarse elevation (mountains on land,
        # trenches/ridges at sea) supplies the gradient the model needs.
        from planet_pipeline import (
            tectonic_coarse_elevation, coarse_conditioning_grid,
            refine_coastline_and_islands, apply_geography,
        )
        coarse_elev = tectonic_coarse_elevation(land_mask, bnd)
        grid = coarse_conditioning_grid(coarse_elev, land_mask)
        pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=-8000.0)
        elev, _ = generate_diffusion_world(pipe, w, h, tile=256, device=_PIPE_DEVICE or "auto")
        elev = clamp_land_sea(elev, land_mask)
        # Geographic terrain: large-scale structure (plains / hills / plateaus /
        # mountain ranges from plate tectonics) comes from our terrain-type map;
        # the diffusion output only contributes its high-frequency micro-texture.
        # Without this the diffusion model invents geography at random.
        final_land, elev = refine_coastline_and_islands(elev, land_mask, bnd, seed)
        geo_elev, terrain_class = terrain_type_map(land_mask, bnd, seed)
        elev = apply_geography(elev, final_land, geo_elev)
        elev[final_land] = numpy.maximum(elev[final_land], 1.0)
        elev[~final_land] = numpy.minimum(elev[~final_land], -1.0)
    else:
        from planet_pipeline import procedural_elevation, trace_rivers, fractalize_coast
        final_land = fractalize_coast(land_mask, seed)
        elev, geo_elev, terrain_class = procedural_elevation(final_land, bnd, seed)
        river_min_acc = max(90, (w * h) // 12000)
        rivers, river_acc = trace_rivers(elev, final_land, min_acc=river_min_acc)

    ocean_mask = ~final_land
    temp, precip = compute_physical_climate(elev, h, w)
    # polar sea ice + permanent snowcaps (same rule as generate_planet)
    lat_deg = numpy.broadcast_to(_lat_deg(h), (h, w))
    ice = ((temp < -2.0) & (numpy.abs(lat_deg) > 55.0)) | ((elev > 2500.0) & (temp < 0.0))
    t_diffusion = time.time() - t0 - t_voronoi
    t_total = time.time() - t0

    images = {}
    # Terrain-type colouring (the geographic steering map).
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as tmp:
        tt_path = _os.path.join(tmp, "terrain_types.png")
        render_terrain_types(terrain_class, tt_path)
        images["terrain_types"] = _img_to_b64(numpy.asarray(Image.open(tt_path)))

    # Generated continents (图2 silhouette style): clean land/ocean map.
    sil = numpy.full((h, w, 3), (247, 247, 247), dtype=numpy.uint8)
    sil[final_land] = (35, 110, 190)
    # 1px coast stroke for crispness at high zoom
    coast = final_land[:, 1:] != final_land[:, :-1]
    sil[:, 1:][coast] = (20, 60, 120)
    sil[:, :-1][coast] = (20, 60, 120)
    coastv = final_land[1:, :] != final_land[:-1, :]
    sil[1:, :][coastv] = (20, 60, 120)
    sil[:-1, :][coastv] = (20, 60, 120)
    images["continents"] = _img_to_b64(sil)

    # Plate boundary classification (tectonic).
    bt = bnd["boundary_type"]
    images["boundary_types"] = _img_to_b64(BOUNDARY_PALETTE[bt.astype(numpy.int8)])

    # Raw micro-plates (muted earth tones).
    raw_rgb = numpy.resize(PLATE_PALETTE, (n_raw, 3))[raw.astype(int)].astype(numpy.uint8)
    images["raw_plates"] = _img_to_b64(raw_rgb)

    # Merged major plates, textbook-tectonic style (图1): muted fills with
    # RED growth (divergent) / BLUE extinction (convergent) boundary lines.
    n_groups = int(merged.max()) + 1
    pal = numpy.resize(MERGED_PALETTE, (n_groups, 3))
    big_h, big_w = h * 2, w * 2
    labels_big = numpy.asarray(
        Image.fromarray(merged.astype(numpy.uint8)).resize((big_w, big_h), Image.NEAREST)
    )
    rgb_big = pal[labels_big.astype(int)].astype(numpy.uint8).copy()
    boundary = numpy.zeros((big_h, big_w), dtype=bool)
    boundary[:, 1:] |= labels_big[:, 1:] != labels_big[:, :-1]
    boundary[1:, :] |= labels_big[1:, :] != labels_big[:-1, :]
    bt_big = numpy.asarray(
        Image.fromarray(bt.astype(numpy.uint8)).resize((big_w, big_h), Image.NEAREST)
    )
    conv_line = boundary & (bt_big == 1)      # 消亡边界 -> blue
    div_line = boundary & (bt_big == 2)       # 生长边界 -> red
    tr_line = boundary & (bt_big == 3)        # 转换断层 -> grey
    rgb_big[boundary] = [40, 40, 46]
    rgb_big[tr_line] = [150, 150, 150]
    rgb_big[conv_line] = [30, 64, 175]
    rgb_big[div_line] = [220, 38, 38]
    merged_rgb = numpy.asarray(Image.fromarray(rgb_big).resize((w, h), Image.BILINEAR))
    images["merged"] = _img_to_b64(merged_rgb)

    # High-res relief render (今怀古 reference style): hypsometric land,
    # cyan shelf glow, crisp hillshade, river network overlay.
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as tmp:
        rel_path = _os.path.join(tmp, "relief.png")
        if detail == "diffusion":
            render_relief(elev, rel_path, vert_exag=2.5)
        else:
            render_earth_style(elev, temp, precip, rel_path,
                               rivers=rivers, acc=river_acc, min_acc=river_min_acc,
                               land_mask=final_land, ice=ice)
        images["elevation_relief"] = _img_to_b64(numpy.asarray(Image.open(rel_path)))

    # Grayscale shaded-relief heightmap (图7 style; AI super-resolution input).
    from planet_pipeline import _mapgen_light
    shade = _mapgen_light(elev.astype(numpy.float32), az_deg=315.0, overhead=2.5)
    grey = numpy.clip(0.42 + 0.52 * shade, 0.0, 1.0)
    sea = ~final_land
    grey[sea] = numpy.clip(0.30 + 0.20 * numpy.clip(
        (elev[sea] + 6000.0) / 6000.0, 0.0, 1.0), 0.0, 1.0)
    hm = (grey * 255.0).astype(numpy.uint8)
    images["heightmap"] = _img_to_b64(numpy.stack([hm] * 3, axis=-1))

    # Physical temperature and precipitation (standard atlas colour schemes).
    images["temperature"] = _arr_to_b64(temp, cmap="RdYlBu_r", vmin=-30.0, vmax=30.0)
    p_log = numpy.log10(numpy.clip(precip, 60.0, None))
    images["precipitation"] = _arr_to_b64(p_log, cmap="YlGnBu", vmin=1.78, vmax=3.60)

    # Simplified climate-type zones (12 classes, textbook palette; ocean = light blue).
    koppen_idx = simple_koppen(temp, precip, elev)
    koppen_rgb = numpy.full((h, w, 3), (168, 214, 240), dtype=numpy.uint8)
    koppen_rgb[final_land] = CLIMATE_PALETTE[koppen_idx[final_land].astype(numpy.int32)]
    images["koppen"] = _img_to_b64(koppen_rgb)

    # Ocean / land mask.
    om = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    om[:] = (95, 160, 90)       # land
    om[ocean_mask] = (70, 130, 210)  # ocean
    images["ocean_mask"] = _img_to_b64(om)

    # Stats.
    stats = {
        "voronoi_time": round(t_voronoi, 2),
        "diffusion_time": round(t_diffusion, 2),
        "total_time": round(t_total, 2),
        "detail_mode": detail,
        "raw_plates": n_raw,
        "merged_groups": n_groups,
        "ocean_pct": round(float(ocean_mask.sum()) / (w * h) * 100, 1),
        "land_pct": round(float(final_land.sum()) / (w * h) * 100, 1),
        "elev_min_m": round(float(elev.min()), 1),
        "elev_max_m": round(float(elev.max()), 1),
        "temp_min_c": round(float(temp.min()), 1),
        "temp_max_c": round(float(temp.max()), 1),
        "precip_min_mm": round(float(precip.min()), 0),
        "precip_max_mm": round(float(precip.max()), 0),
    }
    if detail == "procedural":
        stats["river_pixels"] = int(rivers.sum())
    for g in range(n_groups):
        mask = merged == g
        stats[f"group_{g}_area_pct"] = round(float(mask.sum()) / (w * h) * 100, 1)

    # ---- Metadata for the interactive front-end ----
    # Layer catalogue grouped by right-rail category (order matters).
    layers_meta = {
        "elevation_relief": {"label": "真实地形", "cat": "地形", "overlay": True,
                              "desc": "气候分区地表色（沙漠/雨林/苔原/冰盖）+ 脊状山脉 + 河网"},
        "heightmap":        {"label": "高程图", "cat": "地形", "overlay": True,
                              "desc": "灰度山体阴影高程图（明 = 高，暗 = 低；AI 超分输入）"},
        "terrain_types":    {"label": "地貌类型", "cat": "地形", "overlay": True,
                              "desc": "地理常理掩膜：平原/丘陵/高原/山脉/盆地（手绘纹理）"},
        "merged":           {"label": "合并大板块", "cat": "板块", "overlay": True,
                              "desc": "红 = 生长边界 · 蓝 = 消亡边界 · 灰 = 转换断层"},
        "raw_plates":       {"label": "原始微板块", "cat": "板块", "overlay": True,
                              "desc": "球面 Voronoi 初始细分区"},
        "boundary_types":   {"label": "板块边界", "cat": "边界", "overlay": True,
                              "desc": "红 = 生长（海岭）· 蓝 = 消亡（海沟/造山）· 灰 = 平移"},
        "continents":       {"label": "大陆轮廓", "cat": "大陆", "overlay": True,
                              "desc": "海陆剪影：白 = 海 · 蓝 = 陆（图2 风格）"},
        "temperature":      {"label": "温度", "cat": "气候", "overlay": True,
                              "desc": "物理模型：纬度温度带 + 高程递减率 6.5 °C/km"},
        "precipitation":    {"label": "降水", "cat": "气候", "overlay": True,
                              "desc": "物理模型：ITCZ/西风带 + 地形性迎风坡（对数毫米）"},
        "koppen":           {"label": "气候类型", "cat": "气候", "overlay": True,
                              "desc": "12 类标准气候分类（雨林/草原/沙漠/季风/地中海/针叶林…）"},
        "ocean_mask":       {"label": "海陆掩膜", "cat": "海陆", "overlay": True,
                              "desc": "蓝 = 海 · 绿 = 陆"},
    }

    # Ordered generation timeline for the "查看生成过程" viewer (spec flow:
    # 板块 → 大陆 → 地貌掩层 → 气候掩层 → 细化地貌 → 着色 → 高程导出).
    process = [
        {"key": "raw_plates",      "label": "① 原始微板块", "desc": "球面 Voronoi 初始细分区"},
        {"key": "merged",          "label": "② 合并大板块", "desc": "中心双海洋板块 + 红生长/蓝消亡边界"},
        {"key": "continents",      "label": "③ 大陆轮廓",   "desc": "超级大陆 + 两对近连 + 澳洲式，海陆剪影"},
        {"key": "boundary_types",  "label": "④ 板块边界",   "desc": "生长/消亡/转换分类"},
        {"key": "terrain_types",   "label": "⑤ 地貌类型",   "desc": "平原/丘陵/高原/山脉/盆地掩层（图3）"},
        {"key": "elevation_relief","label": "⑥ 真实地形",   "desc": "脊状山脉 + D8 河网 + 气候着色（图5/6）"},
        {"key": "koppen",          "label": "⑦ 气候类型",   "desc": "12 类标准气候掩层（图4）"},
        {"key": "temperature",     "label": "⑧ 温度",       "desc": "物理温度模型"},
        {"key": "precipitation",   "label": "⑨ 降水",       "desc": "物理降水模型（对数）"},
        {"key": "heightmap",       "label": "⑩ 高程图",     "desc": "灰度高程导出（AI 超分输入，图7）"},
    ]

    return {
        "success": True,
        "main": "elevation_relief",
        "images": images,
        "layers": layers_meta,
        "process": process,
        "stats": stats,
    }


# --------------- Flask routes ---------------


@app.route("/")
def index():
    app_html = ROOT / "tools" / "earth_app.html"
    return send_file(str(app_html), mimetype="text/html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        params = request.get_json(force=True)
        result = generate_world(params)
        return jsonify(result)
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({"success": False, "error": str(e), "traceback": _tb.format_exc()})


@app.route("/api/diagnose")
def api_diagnose():
    """Lightweight environment probe — helps distinguish environment issues
    (missing torch / CUDA / matplotlib) from generation-time errors."""
    info = {"cwd": str(ROOT)}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_total_mb"] = round(props.total_memory / 1024 ** 2)
            info["gpu_mem_alloc_mb"] = round(torch.cuda.memory_allocated(0) / 1024 ** 2)
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception as e:
        info["matplotlib_error"] = f"{type(e).__name__}: {e}"
    # Is the diffusion pipeline already loaded?
    info["pipeline_loaded"] = _PIPE is not None
    info["resolved_device"] = _PIPE_DEVICE
    return jsonify(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind to; pass 0.0.0.0 to expose the server on the LAN",
    )
    args = parser.parse_args()
    # Startup dependency guard: give a clear, actionable message instead of a
    # confusing ModuleNotFoundError traceback if launched with the wrong python.
    _missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        _missing.append("torch")
    try:
        import numba  # noqa: F401
    except Exception:
        _missing.append("numba")
    if _missing:
        _venv = r"C:/Users/Qq203/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
        _local_venv = str(ROOT / ".venv-torch" / "Scripts" / "python.exe")
        sys.stderr.write(
            "\n"
            "══════════════════════════════════════════════════════════════\n"
            "  启动失败：当前 Python 解释器缺少依赖 " + ", ".join(_missing) + "\n"
            "  本服务必须在包含 torch / numba 的 venv 中运行，例如：\n"
            "    " + _venv + " tools/web_server.py --port " + str(args.port) + "\n"
            "    " + _local_venv + " tools/web_server.py --port " + str(args.port) + "\n"
            "  （不要用裸 python / 托管运行时，它们没有这些依赖）\n"
            "══════════════════════════════════════════════════════════════\n\n"
        )
        sys.exit(2)
    print(f"Starting web server on http://{args.host}:{args.port} ...")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
