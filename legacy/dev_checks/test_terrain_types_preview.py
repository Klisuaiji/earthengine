"""Preview the geographic terrain-type map (no diffusion needed)."""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import planet_pipeline as pp
from worldengine.spherical_voronoi import generate_spherical_world

OUT = os.path.join(ROOT, "test_out_smoke")
os.makedirs(OUT, exist_ok=True)

seed, w, h = 2026, 1024, 512
tect = pp.tectonic(seed, w, h)
land_mask = tect["land_mask"]
geo_elev, terrain_class = pp.terrain_type_map(land_mask, tect["boundaries"], seed)

# class colouring
pp.render_terrain_types(terrain_class, os.path.join(OUT, "geo_terrain_types.png"))

# relief from the geographic elevation alone (CPU, no diffusion)
ice = np.zeros((h, w), dtype=bool)
pp.render_terrain(geo_elev, ice, os.path.join(OUT, "geo_relief.png"))

hist = {pp.TERRAIN_NAMES[k]: int((terrain_class == k).sum()) for k in range(6)}
print("terrain histogram:", hist)
land_pct = land_mask.mean() * 100
mountains = (terrain_class == pp.TT_MOUNTAIN).sum() / max(land_mask.sum(), 1) * 100
plateaus = (terrain_class == pp.TT_PLATEAU).sum() / max(land_mask.sum(), 1) * 100
print(f"land {land_pct:.1f}%  mountains {mountains:.0f}% of land  plateaus {plateaus:.0f}% of land")
print("elev range: %.0f .. %.0f m" % (geo_elev.min(), geo_elev.max()))
print("saved geo_terrain_types.png / geo_relief.png")
