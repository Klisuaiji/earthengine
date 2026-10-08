"""CPU smoke test for the 25-step planet pipeline (diffusion step stubbed)."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import planet_pipeline as pp
from worldengine.spherical_voronoi import synthesize_elevation

seed, w, h = 1234567, 512, 256
t0 = time.time()

# Phase 1 (minus step 9, which needs torch): tectonics + coarse conditioning
tect = pp.tectonic(seed, w, h)
coarse = pp.tectonic_coarse_elevation(tect["land_mask"], tect["boundaries"])
grid = pp.coarse_conditioning_grid(coarse, tect["land_mask"])
print("grid:", grid.shape, "range", float(grid.min()), float(grid.max()))

# fake the diffusion output with the analytic elevation, then post-process
elev = synthesize_elevation(tect["merged"], h, w, seed, n_big=6,
                            land_mask=tect["land_mask"], continent_mask=tect["continent_mask"])
final_land, elev = pp.refine_coastline_and_islands(elev, tect["land_mask"], tect["boundaries"], seed)
elev[final_land] = np.maximum(elev[final_land], 1.0)
elev[~final_land] = np.minimum(elev[~final_land], -1.0)

# Phase 2
temp = pp.solar_temperature(elev, h)
pressure = pp.continental_pressure(temp, elev, h)
u, v = pp.wind_field(elev, temp, tect["boundaries"], h, w)
precip = pp.precipitation(u, v, elev, h)
koppen_code = pp.koppen(temp, precip, elev, h)
lat = pp._lat(h)
cu, cv, sst = pp.ocean_currents(u, v, elev, h, w, iterations=2)
eff_temp = np.where(elev >= 0, temp, sst)
ice = ((eff_temp < 0) & (np.abs(lat) > 55)) | ((elev > 2500) & (temp < 0))

# Phase 3
biome_name = pp.biome(koppen_code, ice, h)
civ = pp.civilization_index(temp, precip, elev, koppen_code, biome_name)
civ_pts = pp.civilization_points(civ)

# Rendering (to a temp dir)
out = os.path.join(ROOT, "test_out_smoke")
os.makedirs(out, exist_ok=True)
pp.render_terrain(elev, ice, os.path.join(out, "08_elevation_relief.png"))
pp.render_satellite(elev, biome_name, ice, os.path.join(out, "21_satellite.png"))
pp.render_planet(elev, biome_name, ice, koppen_code, os.path.join(out, "22_planet.png"))
pp._quiver_png(u, v, os.path.join(out, "11_wind.png"), step=24)
pp._render_civ_points(elev, ice, civ_pts, os.path.join(out, "18_civ_points.png"))

# sanity: the Dfc dead-branch fix — no land pixel may stay labelled "Ocean"
# (cold continental land used to fall through to "Ocean" before the fix).
land = elev >= 0
assert land.any()
land_codes = set(koppen_code[land].tolist())
print("land koppen codes:", sorted(land_codes))
assert "Ocean" not in land_codes, f"land mislabelled as Ocean: {land_codes}"

from collections import Counter
print("koppen histogram:", dict(Counter(koppen_code.ravel().tolist()).most_common()))
print("civ points:", len(civ_pts))
print("SMOKE OK in %.1fs" % (time.time() - t0))
