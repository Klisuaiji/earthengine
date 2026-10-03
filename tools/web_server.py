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
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
        (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
        (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ],
    dtype=float,
)

MERGED_PALETTE = numpy.array(
    [
        (20, 70, 190), (70, 150, 235),
        (90, 170, 80), (200, 180, 90), (180, 140, 80), (140, 190, 120),
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
    (240, 240, 250),  # INTERIOR
    (160, 32, 240),   # CONVERGENT (purple)
    (30, 180, 60),    # DIVERGENT (green)
    (255, 140, 0),    # TRANSFORM (orange)
], dtype=numpy.uint8)

# Elevation colour stops: ocean (-10 m) -> blue, land (+10 m) -> warm.
ELEV_STOPS = [
    (0.0, (40, 90, 160)), (0.499, (40, 90, 160)),
    (0.501, (150, 190, 110)), (1.0, (235, 225, 185)),
]

# Simplified Köppen climate-zone palette using standard Köppen-Geiger hues
# (A tropical = blues, B arid = red/orange, C temperate = greens,
#  D continental = cyans, E polar = grey, H alpine = brown-grey).
KOPPEN_PALETTE = numpy.array([
    (0, 110, 254),     # A 热带      (Köppen A blue)
    (235, 80, 30),     # B 干旱      (Köppen B red-orange)
    (110, 200, 90),    # C 温带      (Köppen C green)
    (60, 170, 235),    # D 大陆性    (Köppen D cyan)
    (178, 178, 178),   # E 极地      (Köppen E grey)
    (150, 115, 90),    # H 高山      (alpine brown-grey)
], dtype=numpy.uint8)


def simple_koppen(temp, precip, elev):
    """Rough Köppen-style zoning from annual mean temp / precip / elevation.

    Only annual means are available (no monthly series), so this is a coarse
    classification good enough for an overlay legend, not a strict Köppen.
    """
    out = numpy.zeros(temp.shape, dtype=numpy.int32)
    tropical = temp >= 20.0
    polar = temp < 0.0
    arid = precip < 250.0
    alpine = elev > 2500.0
    out[tropical & ~arid] = 0                               # A 热带
    out[arid] = 1                                           # B 干旱
    out[(temp >= 0) & (temp < 20) & ~arid & ~polar] = 2     # C 温带
    out[(~arid) & (~polar) & (~alpine) & (temp < 0)] = 3    # D 大陆性（冬季<0）
    out[polar] = 4                                          # E 极地
    out[alpine] = 5                                         # H 高山
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

    # Generated continents (the actual landmasses).
    cont_idx = numpy.clip(continent_mask + 1, 0, CONTINENT_PALETTE.shape[0] - 1)
    images["continents"] = _img_to_b64(CONTINENT_PALETTE[cont_idx.astype(numpy.int32)])

    # Plate boundary classification (tectonic).
    bt = bnd["boundary_type"]
    images["boundary_types"] = _img_to_b64(BOUNDARY_PALETTE[bt.astype(numpy.int8)])

    # Raw micro-plates.
    raw_rgb = numpy.resize(PLATE_PALETTE, (n_raw, 3))[raw.astype(int)].astype(numpy.uint8)
    images["raw_plates"] = _img_to_b64(raw_rgb)

    # Merged major plates with anti-aliased black boundaries.
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
    rgb_big[boundary] = [0, 0, 0]
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

    # Physical temperature and precipitation (standard atlas colour schemes).
    images["temperature"] = _arr_to_b64(temp, cmap="RdYlBu_r", vmin=-30.0, vmax=30.0)
    p_log = numpy.log10(numpy.clip(precip, 60.0, None))
    images["precipitation"] = _arr_to_b64(p_log, cmap="YlGnBu", vmin=1.78, vmax=3.60)

    # Simplified Köppen climate zones (overlay legend).
    koppen_idx = simple_koppen(temp, precip, elev)
    images["koppen"] = _img_to_b64(KOPPEN_PALETTE[koppen_idx.astype(numpy.int32)])

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
        "terrain_types":    {"label": "地貌类型", "cat": "地形", "overlay": True,
                              "desc": "地理常理掩膜：平原/丘陵/高原/山脉，约束地形大尺度结构"},
        "merged":           {"label": "合并大板块", "cat": "板块", "overlay": True,
                              "desc": "微板块合并为 N 大板块（黑线 = 边界）"},
        "raw_plates":       {"label": "原始微板块", "cat": "板块", "overlay": True,
                              "desc": "球面 Voronoi 初始细分区"},
        "boundary_types":   {"label": "板块边界", "cat": "边界", "overlay": True,
                              "desc": "生长(绿)/消亡(紫)/平移(橙)"},
        "continents":       {"label": "生成大陆", "cat": "大陆", "overlay": True,
                              "desc": "陆地板块即大陆，海洋板块为海"},
        "temperature":      {"label": "温度", "cat": "气候", "overlay": True,
                              "desc": "物理模型：纬度温度带 + 高程递减率 6.5 °C/km"},
        "precipitation":    {"label": "降水", "cat": "气候", "overlay": True,
                              "desc": "物理模型：ITCZ/西风带 + 地形性迎风坡"},
        "koppen":           {"label": "气候带", "cat": "气候", "overlay": True,
                              "desc": "简化 Köppen：热带/干旱/温带/大陆性/极地/高山"},
        "ocean_mask":       {"label": "海陆掩膜", "cat": "海陆", "overlay": True,
                              "desc": "蓝 = 海 · 绿 = 陆"},
    }

    # Ordered generation timeline for the "查看生成过程" viewer.
    process = [
        {"key": "raw_plates",      "label": "① 原始微板块", "desc": "球面 Voronoi 初始细分区"},
        {"key": "merged",          "label": "② 合并大板块", "desc": "微板块合并为 N 大板块"},
        {"key": "continents",      "label": "③ 生成大陆",   "desc": "大陆核生长 + 分形海岸与岛屿"},
        {"key": "boundary_types",  "label": "④ 板块边界",   "desc": "生长/消亡/平移分类"},
        {"key": "terrain_types",   "label": "⑤ 地貌类型",   "desc": "地理常理：平原/丘陵/高原/山脉"},
        {"key": "elevation_relief","label": "⑥ 真实地形",   "desc": "脊状山脉纹理 + D8 河网 + 参考图设色"},
        {"key": "temperature",     "label": "⑦ 温度",       "desc": "物理温度模型"},
        {"key": "precipitation",   "label": "⑧ 降水",       "desc": "物理降水模型"},
        {"key": "koppen",          "label": "⑨ 气候带",     "desc": "简化 Köppen 分类"},
        {"key": "ocean_mask",      "label": "⑩ 海陆掩膜",   "desc": "最终海陆二值掩膜"},
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
