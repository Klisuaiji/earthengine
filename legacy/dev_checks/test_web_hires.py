# -*- coding: utf-8 -*-
"""E2E test: POST /api/generate at 2048x1024 and save the relief PNG."""
import json, base64, time, sys
import urllib.request

payload = {"seed": 2026, "width": 2048, "height": 1024, "detail": "procedural"}
req = urllib.request.Request(
    "http://127.0.0.1:8899/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read().decode())
print("request: %.1fs  success=%s" % (time.time() - t0, data.get("success")))
print("stats:", json.dumps(data.get("stats", {}), ensure_ascii=False))
relief = base64.b64decode(data["images"]["elevation_relief"])
with open("test_out_smoke/web_relief_hires_2048.png", "wb") as f:
    f.write(relief)
tt = base64.b64decode(data["images"]["terrain_types"])
with open("test_out_smoke/web_terrain_types_2048.png", "wb") as f:
    f.write(tt)
print("saved web_relief_hires_2048.png / web_terrain_types_2048.png")
