"""地质时间演化演示：生成世界 -> 欧拉漂移多个时间截面 -> 每截面渲染高程图。

用法：python3 examples/evolve_timesteps.py [--seed 123456] [--times 0,30,60]
输出 evolution_timesteps.png（多时间截面横向对比）。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from earthengine.pipeline.stages import WorldGenerator
from earthengine.tectonics import geological_evolution as GE
from earthengine.terrain import detail
from earthengine.rendering import map as R
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123456)
    ap.add_argument("--times", default="0,30,60")
    ap.add_argument("--out", default="reports/evolution_timesteps.png")
    args = ap.parse_args()
    times = [float(x) for x in args.times.split(",")]

    ws = WorldGenerator({"seed": args.seed, "width": 384, "height": 192}).generate()
    bounds = {
        "boundary_type": ws.plates.boundary_type,
        "convergent_dist": ws.plates.convergent_dist,
        "divergent_dist": ws.plates.divergent_dist,
    }
    series = GE.evolve_series(ws, times)

    panels = []
    for i, (t, f) in enumerate(series):
        land = f["land"].astype(bool)
        b2 = dict(bounds)
        b2["boundary_type"] = f["boundary"]            # 截面旋转后的边界
        elev, _, _ = detail.procedural_elevation(land, b2, args.seed + int(t))
        tmp = f"/tmp/ev_t{i}.png"
        R.render_relief_hires(elev, tmp, land_mask=land)
        img = Image.open(tmp).convert("RGB")
        # 标签条
        bar = Image.new("RGB", (img.width, 22), (245, 248, 240))
        d = ImageDraw.Draw(bar)
        d.text((6, 4), f"t = {int(t)} Ma", fill=(30, 60, 30))
        panels.append(np.vstack([np.array(bar), np.array(img)]))

    combo = Image.fromarray(np.concatenate(panels, axis=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combo.save(args.out)
    print(f"[evolve] {len(series)} time-sections -> {args.out} "
          f"({combo.width}x{combo.height})")
    for t, f in series:
        print(f"  t={int(t)}Ma land={f['land'].mean():.3f}")


if __name__ == "__main__":
    main()
