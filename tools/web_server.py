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
    remove_enclaves,
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
    (200, 160, 70),  # continent 2 (continent on ocean plate)
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

# Simplified Köppen climate-zone palette (A/B/C/D/E/H) used by the web layer.
KOPPEN_PALETTE = numpy.array([
    (46, 120, 60),    # A 热带
    (200, 180, 110),  # B 干旱
    (120, 170, 90),   # C 温带
    (200, 120, 60),   # D 大陆性
    (210, 230, 245),  # E 极地
    (150, 150, 150),  # H 高山
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



def _get_cached_pipeline(seed, device="cuda"):
    """Return a diffusion pipeline, loading it once at first request."""
    global _PIPE, _PIPE_SEED
    if _PIPE is None:
        _PIPE = build_pipeline(seed=seed, device=device)
        _PIPE_SEED = seed
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
    w = params.get("width", 1024)
    h = params.get("height", 512)
    n_raw = params.get("n_raw", 30)
    n_big = params.get("n_big", 6)
    domain_amp = params.get("domain_amp", 0.22)

    from worldengine import spherical_voronoi as sv
    sv.DOMAIN_AMP = domain_amp

    t0 = time.time()
    raw, merged, land_mask, continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=n_raw, n_big=n_big
    )
    merged = remove_enclaves(merged, min_frac=0.0005)
    t_voronoi = time.time() - t0
    bnd = classify_boundaries(merged, seed=seed)   # needed by coastal refinement

    # Terrain Diffusion: cached pipeline, real elevation + physical climate.
    pipe = _get_cached_pipeline(seed)
    # Build the SAME varied coarse conditioning grid that planet_pipeline uses.
    # A binary land/ocean grid gives the model no spatial gradient to steer by
    # (only ~2x4 conditioning cells), so its output collapses toward the prior
    # and looks like noise.  The tectonic coarse elevation (mountains on land,
    # trenches/ridges at sea) supplies the gradient the model needs.
    from planet_pipeline import (
        tectonic_coarse_elevation, coarse_conditioning_grid,
        refine_coastline_and_islands,
    )
    coarse_elev = tectonic_coarse_elevation(land_mask, bnd)
    grid = coarse_conditioning_grid(coarse_elev, land_mask)
    pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=-8000.0)
    elev, _ = generate_diffusion_world(pipe, w, h, tile=256, device="cuda")
    elev = clamp_land_sea(elev, land_mask)
    # Reuse planet_pipeline's VERIFIED post-processing so the land gets coherent
    # fractal interior relief (and a fractal coastline + archipelago bumps)
    # instead of the diffusion model's flat / noisy interior. This is the key
    # step that makes the rendered relief look like real terrain, not static.
    final_land, elev = refine_coastline_and_islands(elev, land_mask, bnd, seed)
    elev[final_land] = numpy.maximum(elev[final_land], 1.0)
    elev[~final_land] = numpy.minimum(elev[~final_land], -1.0)
    ocean_mask = ~final_land
    temp, precip = compute_physical_climate(elev, h, w)
    t_diffusion = time.time() - t0 - t_voronoi
    t_total = time.time() - t0

    images = {}

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

    # Diffusion elevation relief (hillshaded + hypsometric).
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as tmp:
        rel_path = _os.path.join(tmp, "relief.png")
        render_relief(elev, rel_path, vert_exag=2.5)
        images["elevation_relief"] = _img_to_b64(numpy.asarray(Image.open(rel_path)))

    # Physical temperature and precipitation.
    images["temperature"] = _arr_to_b64(temp, cmap="turbo")
    images["precipitation"] = _arr_to_b64(precip, cmap="viridis")

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
    for g in range(n_groups):
        mask = merged == g
        stats[f"group_{g}_area_pct"] = round(float(mask.sum()) / (w * h) * 100, 1)

    # ---- Metadata for the interactive front-end ----
    # Layer catalogue grouped by right-rail category (order matters).
    layers_meta = {
        "elevation_relief": {"label": "真实地形", "cat": "地形", "overlay": True,
                              "desc": "Terrain Diffusion 渲染的山海起伏（山体阴影 + 高程设色）"},
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
        {"key": "continents",      "label": "③ 生成大陆",   "desc": "非海洋板块即大陆"},
        {"key": "boundary_types",  "label": "④ 板块边界",   "desc": "生长/消亡/平移分类"},
        {"key": "elevation_relief","label": "⑤ 真实地形",   "desc": "Terrain Diffusion 渲染真实高程"},
        {"key": "temperature",     "label": "⑥ 温度",       "desc": "物理温度模型"},
        {"key": "precipitation",   "label": "⑦ 降水",       "desc": "物理降水模型"},
        {"key": "koppen",          "label": "⑧ 气候带",     "desc": "简化 Köppen 分类"},
        {"key": "ocean_mask",      "label": "⑨ 海陆掩膜",   "desc": "最终海陆二值掩膜"},
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

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldEngine Voronoi + Terrain Diffusion - 交互式世界生成</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:16px}
.header h1{font-size:18px;color:#f0f6fc}
.header .sub{font-size:13px;color:#8b949e}
.layout{display:flex;height:calc(100vh - 60px)}
.sidebar{width:320px;background:#161b22;border-right:1px solid #30363d;padding:16px;overflow-y:auto;flex-shrink:0}
.sidebar h2{font-size:14px;color:#8b949e;margin-bottom:12px;text-transform:uppercase;letter-spacing:1px}
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:13px;color:#c9d1d9;margin-bottom:4px}
.form-group input,.form-group select{width:100%;padding:8px 10px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:14px;font-family:inherit}
.form-group input:focus,.form-group select:focus{outline:none;border-color:#58a6ff}
.form-group input[type=range]{padding:0;height:6px;-webkit-appearance:none;background:#30363d;border-radius:3px;cursor:pointer}
.form-group input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#58a6ff}
.form-group .range-val{font-size:12px;color:#8b949e;float:right;margin-top:2px}
.btn{width:100%;padding:10px 16px;background:#238636;color:#fff;border:none;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer}
.btn:hover{background:#2ea043}
.btn:disabled{background:#30363d;color:#8b949e;cursor:not-allowed}
.main{flex:1;overflow-y:auto;padding:24px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.stat-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;min-width:100px}
.stat-card .val{font-size:22px;font-weight:700;color:#58a6ff}
.stat-card .lbl{font-size:12px;color:#8b949e}
.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.img-card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.img-card img{width:100%;display:block;image-rendering:pixelated}
.img-card .cap{display:flex;flex-direction:column;gap:3px;padding:10px 12px;font-size:13px;color:#8b949e}
.img-card .cap b{color:#c9d1d9;font-size:13px;font-weight:600}
.img-card .cap span{font-size:11px;color:#6e7681;line-height:1.4}
.sec{margin-bottom:26px}
.sec-head{display:flex;align-items:baseline;gap:12px;margin:0 0 12px;padding-bottom:8px;border-bottom:1px solid #30363d}
.sec-head h3{font-size:15px;color:#58a6ff;font-weight:600;margin:0}
.sec-head span{font-size:12px;color:#8b949e}
.loading{text-align:center;padding:40px;color:#8b949e}
.spinner{display:inline-block;width:32px;height:32px;border:3px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error{color:#f85149;font-size:13px;margin-top:8px}
</style>
</head>
<body>
<div class="header">
  <h1>WorldEngine &#8226; 球面 Voronoi + Terrain Diffusion</h1>
  <span class="sub">交互式世界地图生成器 &#8226; 构造骨架 + 扩散真实地形 + 物理气候</span>
</div>
<div class="layout">
  <div class="sidebar">
    <h2>参数</h2>
    <div class="form-group">
      <label>Seed</label>
      <input type="number" id="seed" value="1234567" min="1">
    </div>
    <div class="form-group">
      <label>图像尺寸</label>
      <select id="size">
        <option value="256x256">256 x 256</option>
        <option value="512x512">512 x 512</option>
        <option value="1024x512" selected>1024 x 512</option>
        <option value="1024x1024">1024 x 1024</option>
        <option value="2048x1024">2048 x 1024</option>
        <option value="4096x2048">4096 x 2048</option>
      </select>
    </div>
    <div class="form-group">
      <label>微板块数 (n_raw)</label>
      <input type="range" id="n_raw" min="10" max="100" value="30" oninput="document.getElementById('n_raw_val').textContent=this.value">
      <span class="range-val" id="n_raw_val">30</span>
    </div>
    <div class="form-group">
      <label>大板块数 (n_big)</label>
      <input type="range" id="n_big" min="4" max="10" value="6" oninput="document.getElementById('n_big_val').textContent=this.value">
      <span class="range-val" id="n_big_val">6</span>
    </div>
    <div class="form-group">
      <label>域扭曲幅度 (domain_amp)</label>
      <input type="range" id="domain_amp" min="0" max="50" value="22" oninput="document.getElementById('amp_val').textContent=(this.value/100).toFixed(2)">
      <span class="range-val" id="amp_val">0.22</span>
    </div>
    <button class="btn" id="genBtn" onclick="generate()">生成世界</button>
    <div class="error" id="error"></div>
  </div>
  <div class="main" id="main">
    <div class="loading" id="initial"><p>点击左侧「生成世界」开始</p></div>
  </div>
</div>
<script>
async function generate() {
  const btn = document.getElementById('genBtn');
  const main = document.getElementById('main');
  const err = document.getElementById('error');
  btn.disabled = true;
  btn.textContent = '生成中...';
  err.textContent = '';
  main.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在生成，高分辨率可能需要数分钟...</p></div>';

  const size = document.getElementById('size').value.split('x');
  const params = {
    seed: parseInt(document.getElementById('seed').value),
    width: parseInt(size[0]),
    height: parseInt(size[1]),
    n_raw: parseInt(document.getElementById('n_raw').value),
    n_big: parseInt(document.getElementById('n_big').value),
    domain_amp: parseFloat(document.getElementById('domain_amp').value) / 100,
  };

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params)
    });
    const data = await resp.json();
    if (!data.success) {
      err.textContent = data.error || '未知错误';
      main.innerHTML = '';
      btn.disabled = false;
      btn.textContent = '生成世界';
      return;
    }

    const stats = data.stats || {};
    let statsHtml = '<div class="stats">';
    statsHtml += `<div class="stat-card"><div class="val">${stats.total_time}s</div><div class="lbl">总耗时</div></div>`;
    statsHtml += `<div class="stat-card"><div class="val">${stats.raw_plates}</div><div class="lbl">微板块</div></div>`;
    statsHtml += `<div class="stat-card"><div class="val">${stats.merged_groups}</div><div class="lbl">大板块</div></div>`;
    statsHtml += `<div class="stat-card"><div class="val">${stats.ocean_pct}%</div><div class="lbl">海洋</div></div>`;
    statsHtml += `<div class="stat-card"><div class="val">${stats.land_pct}%</div><div class="lbl">陆地</div></div>`;
    if (stats.diffusion_time !== undefined) {
      statsHtml += `<div class="stat-card"><div class="val">${stats.diffusion_time}s</div><div class="lbl">扩散推理</div></div>`;
    }
    if (stats.elev_max_m !== undefined) {
      statsHtml += `<div class="stat-card"><div class="val">${stats.elev_max_m}m</div><div class="lbl">最高峰</div></div>`;
    }
    for (let g = 0; g < stats.merged_groups; g++) {
      const k = `group_${g}_area_pct`;
      if (stats[k] !== undefined) {
        statsHtml += `<div class="stat-card"><div class="val">${stats[k]}%</div><div class="lbl">板块 ${g}</div></div>`;
      }
    }
    statsHtml += '</div>';

    // 图层元信息：标题 + 一句话说明
    const layers = {
      raw_plates:      {label:'原始微板块', desc:'球面 Voronoi 初始划分'},
      merged:          {label:'合并大板块', desc:'微板块合并为 N 大板块（黑线 = 边界）'},
      continents:      {label:'生成的大陆', desc:'陆地板块即大陆，海洋板块为海'},
      boundary_types:  {label:'板块边界',   desc:'生长(绿)/消亡(紫)/平移(橙)'},
      elevation_relief:{label:'真实地形',   desc:'Terrain Diffusion：山脉、海沟、海陆起伏（颜色+山体阴影）'},
      temperature:     {label:'温度',       desc:'物理模型：纬度温度带 + 高程递减率 6.5 C/km'},
      precipitation:   {label:'降水',       desc:'物理模型：ITCZ/西风带 + 地形性迎风坡增强'},
      ocean_mask:      {label:'海陆掩膜',   desc:'蓝 = 海 · 绿 = 陆'},
    };

    // 按生成管线分组：板块结构 -> 真实地形 -> 气候 -> 海陆
    const sections = [
      { title:'① 板块结构', note:'构造骨架，决定海陆格局', keys:['raw_plates','merged','continents','boundary_types'] },
      { title:'② 真实地形', note:'扩散模型根据大陆骨架渲染的高程', keys:['elevation_relief'] },
      { title:'③ 气候',     note:'由真实地形驱动的温度与降水', keys:['temperature','precipitation'] },
      { title:'④ 海陆掩膜', note:'最终海陆二值掩膜', keys:['ocean_mask'] },
    ];

    const covered = new Set();
    let imgsHtml = '';
    for (const sec of sections) {
      const keys = sec.keys.filter(k => k in data.images);
      if (!keys.length) continue;
      imgsHtml += `<div class="sec"><div class="sec-head"><h3>${sec.title}</h3><span>${sec.note}</span></div><div class="images">`;
      for (const key of keys) {
        covered.add(key);
        const m = layers[key] || {label:key, desc:''};
        imgsHtml += `<div class="img-card"><img src="data:image/png;base64,${data.images[key]}"><div class="cap"><b>${m.label}</b><span>${m.desc}</span></div></div>`;
      }
      imgsHtml += '</div></div>';
    }
    // 兜底：任何未归入分组的图层
    const extras = Object.keys(data.images).filter(k => !covered.has(k));
    if (extras.length) {
      imgsHtml += '<div class="sec"><div class="sec-head"><h3>其他</h3></div><div class="images">';
      for (const key of extras) {
        const m = layers[key] || {label:key, desc:''};
        imgsHtml += `<div class="img-card"><img src="data:image/png;base64,${data.images[key]}"><div class="cap"><b>${m.label}</b><span>${m.desc}</span></div></div>`;
      }
      imgsHtml += '</div></div>';
    }

    main.innerHTML = statsHtml + imgsHtml;
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    main.innerHTML = '';
  } finally {
    btn.disabled = false;
    btn.textContent = '生成世界';
  }
}
</script>
</body>
</html>"""


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
        sys.stderr.write(
            "\n"
            "══════════════════════════════════════════════════════════════\n"
            "  启动失败：当前 Python 解释器缺少依赖 " + ", ".join(_missing) + "\n"
            "  本服务必须在包含 torch / numba 的 venv 中运行：\n"
            "    " + _venv + " tools/web_server.py --port " + str(args.port) + "\n"
            "  （不要用裸 python / 托管运行时，它们没有这些依赖）\n"
            "══════════════════════════════════════════════════════════════\n\n"
        )
        sys.exit(2)
    print(f"Starting web server on http://{args.host}:{args.port} ...")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
