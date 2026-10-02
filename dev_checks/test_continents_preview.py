"""Visual self-check for the new continent builder (sketch-style 3-stage pipeline)."""
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from worldengine.spherical_voronoi import generate_spherical_world

OUT = os.path.join(ROOT, "test_out_smoke")
os.makedirs(OUT, exist_ok=True)

CONT_COLORS = [
    (176, 156, 110),  # continent 0: tan
    (120, 160, 90),   # continent 1: green
    (150, 130, 100),  # continent 2: brown-grey
    (100, 150, 120),  # continent 3: teal-green
    (160, 140, 130),  # continent 4
    (140, 150, 100),  # continent 5
    (170, 120, 100),  # continent 6
]
OCEAN_DEEP = (18, 42, 82)
OCEAN_SHALLOW = (52, 96, 150)


def render(seed, w, h, pid, merged, land_mask, continent_mask, path):
    from scipy.ndimage import distance_transform_edt
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # ocean: deep blue -> shallow near coasts
    d = distance_transform_edt(~land_mask)
    t = np.clip(d / 30.0, 0, 1)[..., None]
    ocean_deep = np.array(OCEAN_DEEP, dtype=np.float32)
    ocean_shallow = np.array(OCEAN_SHALLOW, dtype=np.float32)
    rgb[:] = (ocean_shallow * (1 - t) + ocean_deep * t).astype(np.uint8)
    # land coloured by continent with a darker fringe
    for c in np.unique(continent_mask):
        if c < 0:
            continue
        m = continent_mask == c
        col = np.array(CONT_COLORS[int(c) % len(CONT_COLORS)], dtype=np.float32)
        shade = 0.82 + 0.18 * np.clip(distance_transform_edt(~m) / 8.0, 0, 1)
        rgb[m] = (col[None, :] * shade[m, None]).astype(np.uint8)
    # plate boundaries (thin dark lines on ocean and land)
    bnd = (merged != np.roll(merged, -1, 1)) | (merged != np.roll(merged, 1, 1))
    bnd |= merged != np.roll(merged, 1, 0)
    bnd |= merged != np.roll(merged, -1, 0)
    rgb[bnd] = (rgb[bnd] * 0.55).astype(np.uint8)
    Image.fromarray(rgb).save(path)


for seed in (1234567, 42, 2026, 777):
    t0 = time.time()
    pid, merged, land_mask, continent_mask = generate_spherical_world(seed, w=1024, h=512)
    dt = time.time() - t0
    land_pct = land_mask.mean() * 100
    uniq = np.unique(continent_mask[continent_mask >= 0])
    sizes = [(continent_mask == c).sum() / land_mask.sum() * 100 for c in uniq]
    path = os.path.join(OUT, f"earth_preview_{seed}.png")
    render(seed, 1024, 512, pid, merged, land_mask, continent_mask, path)
    print(f"seed {seed}: land {land_pct:.1f}%  continents {len(uniq)} "
          f"(shares {['%.0f%%' % s for s in sizes]})  {dt:.2f}s -> {os.path.basename(path)}")
print("done")
