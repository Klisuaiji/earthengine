"""Web 地图工作台后端 (40 / 阶段 D)。

职责边界：
- Web 只负责参数配置、生成任务、地图查看、图层切换、局部重新生成、数据导出；
- 所有模拟调用 EarthEngine Core，Web 不得重新实现任何模拟；
- 统一 JSON 响应 envelope：``{success, data|error}``，错误分三类：
  ``input``（参数非法 400）、``generate``（生成失败 500）、``missing_dep``
  （可选依赖缺失 503）。

必须通过 HTTP 访问（禁止 file:// 直开）。未迁移功能在 UI 禁用并标注规划中。
"""

from __future__ import annotations

import base64
import os
import tempfile
import time

import numpy as np

# 后端可提供的工作模式与图层图例（前端据此渲染；字段与 /api/meta 一致）
WORK_MODES = ["world", "terrain", "data"]
LAYERS = {
    "land_sea": {"label": "海陆", "group": "world"},
    "terrain": {"label": "地形", "group": "terrain"},
    "biome": {"label": "植被/生物群系", "group": "data"},
    "temperature": {"label": "温度", "group": "data"},
    "precipitation": {"label": "降水", "group": "data"},
    "plates": {"label": "板块", "group": "world"},
}
TOOLS = ["navigate", "select", "inspect", "plates", "terrain", "rivers", "measure"]


def _ok(data):
    return {"success": True, "data": data, "error": None}


def _err(etype, message, code=None):
    return {"success": False, "data": None,
            "error": {"type": etype, "message": message, "code": code}}


def _validate_params(body):
    """校验并归一化生成参数；非法 → (err_msg, None)。"""
    try:
        seed = int(body.get("seed", 12345))
    except (TypeError, ValueError):
        return "seed 必须为整数", None
    w = int(body.get("width", 1024))
    h = int(body.get("height", 512))
    if not (64 <= w <= 8192 and 64 <= h <= 4096):
        return "分辨率需在 64–8192×4096 之间", None
    if w != 2 * h:
        return "等经纬投影要求 width = 2×height（H:W = 1:2）", None
    detail = body.get("detail", "procedural")
    if detail not in ("procedural", "diffusion"):
        return "detail 只能为 procedural 或 diffusion", None
    return None, {"seed": seed, "width": w, "height": h, "detail": detail}


def _png_b64(fn):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        p = f.name
    fn(p)
    with open(p, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    os.unlink(p)
    return data


def run_server(port=8899, host="127.0.0.1"):
    try:
        from flask import Flask, jsonify, request, send_file
    except Exception as e:
        print(f"ERROR: flask 未安装（pip install 'earthengine[web]'）。{e}")
        return 1

    app = Flask(__name__)
    _HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_app.html")

    @app.route("/")
    def index():
        return send_file(_HTML)

    @app.route("/api/meta", methods=["GET"])
    def api_meta():
        from earthengine import __version__
        return jsonify(_ok({
            "engine": "earthengine", "version": __version__,
            "modes": WORK_MODES, "layers": LAYERS, "tools": TOOLS,
        }))

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        body = request.get_json(force=True, silent=True) or {}
        err, params = _validate_params(body)
        if err:
            return jsonify(_err("input", err, 400)), 400
        seed, w, h, detail = params["seed"], params["width"], params["height"], params["detail"]
        t0 = time.time()
        stages_done = []
        try:
            from earthengine.pipeline.stages import WorldGenerator
            from earthengine.pipeline.validator import WorldValidator
            gen = WorldGenerator(params)
            ws = gen.generate()
            stages_done = list(getattr(gen, "_executed_stages", []))
            v = WorldValidator()
            ok, report = v.validate(ws)
            ws.validation = report
        except Exception as e:                      # 生成失败 → 500
            return jsonify(_err("generate", f"生成失败: {e}")), 500

        from earthengine import rendering as R
        try:
            elev = ws.terrain["elevation"]
            biome = ws.vegetation["biome"]
            ice = ws.climate["ice"]
            koppen = ws.climate["koppen"]
            imgs = {
                "land_sea": _png_b64(lambda p: R.map.render_land_sea(
                    ws.ocean["land_mask"], p)),
                "terrain": _png_b64(lambda p: R.map.render_relief_hires(
                    elev, p, land_mask=ws.ocean["land_mask"], ice=ice)),
                "biome": _png_b64(lambda p: R.map.render_planet(
                    elev, biome, ice, koppen, p)),
                "temperature": _png_b64(lambda p: R.map.render_float_field(
                    ws.climate["temperature"], p, -40, 40)),
                "precipitation": _png_b64(lambda p: R.map.render_float_field(
                    ws.climate["precipitation"], p, 0, 3000)),
                "plates": _png_b64(lambda p: R.map.render_plates(
                    ws.plates.plates, p)),
            }
        except Exception as e:
            return jsonify(_err("generate", f"渲染失败: {e}")), 500

        return jsonify(_ok({
            "seed": seed, "w": w, "h": h,
            "land": float(ws.ocean["land_mask"].mean()),
            "civ_points": len(ws.civilization["points"]),
            "validation": {"passed": ok, "errors": report["errors"],
                           "warnings": report["warnings"]},
            "stages": stages_done,
            "elapsed": round(time.time() - t0, 2),
            "images": imgs,
        }))

    @app.route("/api/diagnose", methods=["GET"])
    def api_diagnose():
        from earthengine import __version__
        return jsonify(_ok({
            "engine": "earthengine", "version": __version__,
            "backend": "procedural",
        }))

    print(f"[web] EarthEngine v2.0 地图工作台 http://{host}:{port}")
    app.run(host=host, port=port)
