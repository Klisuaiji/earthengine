#!/usr/bin/env python3
"""Flask web server for interactive Voronoi world generation.

Usage: python3 tools/web_server.py [--port PORT]
"""

import io
import json
import os
import sys
import time
import argparse
import base64

import numpy
from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/tools")
from spherical_voronoi import generate_spherical_plates

from worldengine.generation import (
    Step,
    add_noise_to_elevation,
    center_land,
    initialize_ocean_and_thresholds,
    place_oceans_at_map_borders,
)
from worldengine.model.world import GenerationParameters, Size, World
from worldengine.simulations.biome import BiomeSimulation
from worldengine.simulations.humidity import HumiditySimulation
from worldengine.simulations.temperature import TemperatureSimulation
from worldengine.simulations.precipitation import PrecipitationSimulation
from worldengine.simulations.icecap import IcecapSimulation

from generate_stages_voronoi import (
    fractal_noise_field,
    _distance_from_boundary,
    _remove_enclaves,
    OUT as STAGE_DIR,
    TERRAIN,
    TEMPERATURE,
    PRECIPITATION,
    SEA_DEPTH,
    HUMIDITY,
    ICECAP,
    MERGED_PALETTE,
    PLATE_PALETTE,
    save_color,
    save_plates,
)

app = Flask(__name__)
os.makedirs(STAGE_DIR, exist_ok=True)

# --------------- Colour stops helpers ---------------

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


# --------------- Generate elevation ---------------

def _gen_elevation(merged, h, w, seed):
    ocean_mask = merged < 2
    elevation = numpy.where(ocean_mask, -5000.0, -2000.0)
    noise_a = fractal_noise_field((h, w), seed, octaves=4, base_res=64)
    noise_b = fractal_noise_field((h, w), seed + 1, octaves=5, base_res=48)
    noise_c = fractal_noise_field((h, w), seed + 2, octaves=4, base_res=42)
    noise_d = fractal_noise_field((h, w), seed + 3, octaves=6, base_res=36)
    for g in range(2, int(merged.max()) + 1):
        mask = merged == g
        if mask.sum() < 100:
            continue
        dm = _distance_from_boundary(mask)
        elevation += mask.astype(float) * dm * 5000.0
    elevation += noise_a * 1200.0 + noise_b * 600.0 + noise_c * 900.0 + noise_d * 400.0
    rng = numpy.random.RandomState(seed + 999)
    elevation += rng.rand(h, w).astype(numpy.float32) * 0.01
    return elevation.astype(numpy.float32)


# --------------- Core generation ---------------

def generate_world(params):
    seed = params.get("seed", 1)
    w = params.get("width", 512)
    h = params.get("height", 512)
    n_raw = params.get("n_raw", 30)
    n_big = params.get("n_big", 6)
    domain_amp = params.get("domain_amp", 0.06)

    import spherical_voronoi as sv
    sv.DOMAIN_AMP = domain_amp

    t0 = time.time()
    raw, merged = generate_spherical_plates(seed, w, h, n_raw, n_big)
    merged = _remove_enclaves(merged, min_frac=0.0005)
    t_voronoi = time.time() - t0

    elevation = _gen_elevation(merged, h, w, seed)
    elevation -= elevation.min()
    elevation /= elevation.max()
    elevation *= 8000.0
    elevation -= 4000.0

    step = Step.full()
    temps = [0.874, 0.765, 0.594, 0.439, 0.366, 0.124]
    humids = [0.941, 0.778, 0.507, 0.236, 0.073, 0.014, 0.002]
    world = World("web", Size(w, h), seed, GenerationParameters(n_raw, 1.0, step), temps, humids)
    world.elevation = (elevation.astype(numpy.float32), {"ocean": 0.0})
    world.plates = raw

    center_land(world)
    add_noise_to_elevation(world, numpy.random.randint(0, 4096))
    place_oceans_at_map_borders(world)
    initialize_ocean_and_thresholds(world)

    rng = numpy.random.RandomState(seed)
    sub_seeds = rng.randint(0, numpy.iinfo(numpy.int32).max, size=100)

    def _try_sim(name, sim_class, seed_idx):
        try:
            sim_class().execute(world, sub_seeds[seed_idx])
            return True
        except Exception:
            return False

    _try_sim("temperature", TemperatureSimulation, 4)
    _try_sim("precipitation", PrecipitationSimulation, 0)
    _try_sim("biome", BiomeSimulation, 7)
    _try_sim("icecap", IcecapSimulation, 8)

    t_total = time.time() - t0

    images = {}

    # Raw plates
    raw_rgb = numpy.resize(PLATE_PALETTE, (n_raw, 3))[raw.astype(int)].astype(numpy.uint8)
    images["raw_plates"] = _img_to_b64(raw_rgb)

    # Merged plates (with anti-aliased boundaries)
    n_groups = int(merged.max()) + 1
    pal = numpy.resize(MERGED_PALETTE, (n_groups, 3))
    big_h, big_w = h * 2, w * 2
    labels_big = numpy.asarray(Image.fromarray(merged.astype(numpy.uint8)).resize((big_w, big_h), Image.NEAREST))
    rgb_big = pal[labels_big.astype(int)].astype(numpy.uint8).copy()
    boundary = numpy.zeros((big_h, big_w), dtype=bool)
    boundary[:, 1:] |= labels_big[:, 1:] != labels_big[:, :-1]
    boundary[1:, :] |= labels_big[1:, :] != labels_big[:-1, :]
    rgb_big[boundary] = [0, 0, 0]
    merged_rgb = numpy.asarray(Image.fromarray(rgb_big).resize((w, h), Image.BILINEAR))
    images["merged"] = _img_to_b64(merged_rgb)

    # Elevation
    elev = world.layers["elevation"].data
    images["elevation"] = _img_to_b64(_array_to_rgb(elev, TERRAIN))

    # Ocean mask
    ocean = world.layers["ocean"].data
    om = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    om[:] = (95, 160, 90)
    om[ocean] = (70, 130, 210)
    images["ocean_mask"] = _img_to_b64(om)

    # Temperature
    try:
        images["temperature"] = _img_to_b64(_array_to_rgb(world.layers["temperature"].data, TEMPERATURE))
    except Exception:
        pass

    # Precipitation
    try:
        images["precipitation"] = _img_to_b64(_array_to_rgb(world.layers["precipitation"].data, PRECIPITATION))
    except Exception:
        pass

    # Humidity
    try:
        images["humidity"] = _img_to_b64(_array_to_rgb(world.layers["humidity"].data, HUMIDITY))
    except Exception:
        pass

    # Sea depth
    try:
        images["sea_depth"] = _img_to_b64(_array_to_rgb(world.layers["sea_depth"].data, SEA_DEPTH))
    except Exception:
        pass

    # Biome
    try:
        from worldengine.draw import _biome_colors
        bm = world.layers["biome"].data
        brgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        for name_, color in _biome_colors.items():
            brgb[bm == name_] = color[:3]
        brgb[bm == "bare rock"] = (128, 128, 128)
        images["biome"] = _img_to_b64(brgb)
    except Exception:
        pass

    # Icecap
    try:
        images["icecap"] = _img_to_b64(_array_to_rgb(world.layers["icecap"].data, ICECAP))
    except Exception:
        pass

    # Stats
    stats = {
        "voronoi_time": round(t_voronoi, 2),
        "total_time": round(t_total, 2),
        "raw_plates": n_raw,
        "merged_groups": n_groups,
        "ocean_pct": round(float(ocean.sum()) / (w * h) * 100, 1),
        "land_pct": round(float((~ocean).sum()) / (w * h) * 100, 1),
    }
    for g in range(n_groups):
        mask = merged == g
        stats[f"group_{g}_area_pct"] = round(float(mask.sum()) / (w * h) * 100, 1)

    return {"success": True, "images": images, "stats": stats}


# --------------- Flask routes ---------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldEngine Voronoi - 交互式世界生成</title>
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
.btn-danger{background:#da3633}
.btn-danger:hover{background:#f85149}
.presets{margin-bottom:16px}
.presets select{width:100%;padding:8px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:13px}
.main{flex:1;overflow-y:auto;padding:24px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.stat-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;min-width:100px}
.stat-card .val{font-size:22px;font-weight:700;color:#58a6ff}
.stat-card .lbl{font-size:12px;color:#8b949e}
.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.img-card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.img-card img{width:100%;display:block;image-rendering:pixelated}
.img-card .cap{padding:10px 12px;font-size:13px;color:#8b949e}
.loading{text-align:center;padding:40px;color:#8b949e}
.spinner{display:inline-block;width:32px;height:32px;border:3px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error{color:#f85149;font-size:13px;margin-top:8px}
</style>
</head>
<body>
<div class="header">
  <h1>WorldEngine &#8226; 球面 Voronoi</h1>
  <span class="sub">交互式世界地图生成器 &#8226; domain-warped spherical Voronoi</span>
</div>
<div class="layout">
  <div class="sidebar">
    <h2>参数</h2>
    <div class="presets">
      <label style="font-size:13px;color:#c9d1d9;margin-bottom:4px;display:block">预设</label>
      <select id="preset" onchange="applyPreset()">
        <option value="default">默认 (30板 / 6大板)</option>
        <option value="detailed">细节 (60板 / 6大板)</option>
        <option value="coarse">粗放 (15板 / 6大板)</option>
        <option value="eight">8大板 (30板 / 8大板)</option>
      </select>
    </div>
    <div class="form-group">
      <label>Seed</label>
      <input type="number" id="seed" value="1" min="1" max="9999">
    </div>
    <div class="form-group">
      <label>图像尺寸</label>
      <select id="size">
        <option value="256x256">256 x 256</option>
        <option value="512x512" selected>512 x 512</option>
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
      <input type="range" id="domain_amp" min="0" max="30" value="6" oninput="document.getElementById('amp_val').textContent=(this.value/100).toFixed(2)">
      <span class="range-val" id="amp_val">0.06</span>
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
    for (let g = 0; g < stats.merged_groups; g++) {
      const k = `group_${g}_area_pct`;
      if (stats[k] !== undefined) {
        statsHtml += `<div class="stat-card"><div class="val">${stats[k]}%</div><div class="lbl">板块 ${g}${g<2?' 海':' 陆'}</div></div>`;
      }
    }
    statsHtml += '</div>';

    const labels = {
      raw_plates: '原始微板块',
      merged: '合并大板块',
      elevation: '高程图',
      ocean_mask: '海陆掩膜',
      temperature: '温度',
      precipitation: '降水',
      sea_depth: '海洋深度',
      humidity: '湿度',
      biome: '生物群系',
      icecap: '冰盖',
    };

    let imgsHtml = '<div class="images">';
    for (const [key, b64] of Object.entries(data.images)) {
      const label = labels[key] || key;
      imgsHtml += `<div class="img-card"><img src="data:image/png;base64,${b64}"><div class="cap">${label}</div></div>`;
    }
    imgsHtml += '</div>';

    main.innerHTML = statsHtml + imgsHtml;
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    main.innerHTML = '';
  } finally {
    btn.disabled = false;
    btn.textContent = '生成标准世界';
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        params = request.get_json(force=True)
        result = generate_world(params)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"Starting web server on port {args.port}...")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
