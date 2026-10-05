# -*- coding: utf-8 -*-
"""Save the new-style layers (merged/koppen/continents/heightmap) for review."""
import json, base64, urllib.request

payload = {"seed": 2026, "width": 2048, "height": 1024, "detail": "procedural"}
req = urllib.request.Request(
    "http://127.0.0.1:8899/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read().decode())
for key in ("merged", "koppen", "continents", "heightmap", "boundary_types"):
    if key in data["images"]:
        with open(f"test_out_smoke/web_{key}_2048.png", "wb") as f:
            f.write(base64.b64decode(data["images"][key]))
        print("saved", key)
print("layers:", sorted(data["images"].keys()))
