"""一键 Web (40)：`earthengine web --port 8899`。

Web 只负责：参数配置、生成任务、地图查看、图层切换、局部重新生成、数据导出。
所有模拟调用 EarthEngine Core，Web 不得重新实现任何模拟。

必须通过 HTTP 访问，不可 file:// 直开。
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np

_PIPE = None
_PIPE_SEED = None


def run_server(port=8899, host="127.0.0.1"):
    try:
        from flask import Flask, jsonify, request, send_file
    except Exception as e:
        print(f"ERROR: flask 未安装（pip install flask）。{e}")
        return 1

    app = Flask(__name__)
    _HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "earth_app.html")

    @app.route("/")
    def index():
        return send_file(_HTML)

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        body = request.get_json(force=True) or {}
        seed = int(body.get("seed", 12345))
        w = int(body.get("width", 1024))
        h = int(body.get("height", 512))
        detail = body.get("detail", "procedural")
        t0 = time.time()
        from earthengine.pipeline.stages import WorldGenerator
        from earthengine.pipeline.validator import WorldValidator
        gen = WorldGenerator({"seed": seed, "width": w, "height": h, "detail": detail})
        ws = gen.generate()
        v = WorldValidator()
        ok, report = v.validate(ws)
        ws.validation = report
        # 返回关键图层 PNG（base64）
        from earthengine import rendering as R
        imgs = {}
        elev = ws.terrain["elevation"]
        bio = ws.vegetation["biome"]
        ice = ws.climate["ice"]
        for name, fn in [
            ("land_sea", lambda: R.map.render_land_sea(ws.ocean["land_mask"], _tmp())),
            ("terrain", lambda: R.map.render_relief_hires(elev, _tmp(), land_mask=ws.ocean["land_mask"], ice=ice)),
            ("biome", lambda: R.map.render_planet(elev, bio, ice, ws.climate["koppen"], _tmp())),
        ]:
            p = fn()
            with open(p, "rb") as fh:
                imgs[name] = base64.b64encode(fh.read()).decode("ascii")
        return jsonify({
            "seed": seed, "w": w, "h": h, "land": float(ws.ocean["land_mask"].mean()),
            "civ_pts": len(ws.civilization["points"]),
            "overall": report["scores"]["overall"], "ok": ok,
            "messages": report["messages"],
            "images": imgs, "elapsed": time.time() - t0,
        })

    @app.route("/api/diagnose", methods=["GET"])
    def api_diagnose():
        return jsonify({
            "engine": "earthengine", "version": "2.0.0",
            "backend": "procedural", "pipeline_loaded": _PIPE is not None,
        })

    print(f"[web] EarthEngine v2.0 server at http://{host}:{port}")
    app.run(host=host, port=port)


def _tmp():
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    return path
