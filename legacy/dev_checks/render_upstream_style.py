# -*- coding: utf-8 -*-
"""Render OUR pipeline data through the UPSTREAM worldengine renderers.

Produces side-by-side comparable images: whatever the original website
would have drawn for the same world.  Data adapters normalise our arrays
into the upstream World model conventions.
"""
import os
import sys

sys.path.insert(0, r"D:\Qq203\Downloads\earthengine-master")

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy

from worldengine.model.world import World, Size, GenerationParameters, Layer
from worldengine.model.world import LayerWithThresholds, LayerWithQuantiles
from worldengine.step import Step
from worldengine.draw import (
    draw_elevation_on_file,
    draw_simple_elevation_on_file,
    draw_temperature_levels_on_file,
    draw_precipitation_on_file,
    draw_biome_on_file,
    draw_satellite_on_file,
    draw_icecaps_on_file,
)

OUT = r"D:\Qq203\Downloads\earthengine-master\test_out_smoke\upstream_compare"
os.makedirs(OUT, exist_ok=True)

# ---- 1. our pipeline at 512x256 (fast) ------------------------------------
from tools import planet_pipeline as pp

w, h = 512, 256
res = pp.generate_planet(seed=1234567, w=w, h=h, out="test_out_smoke/pipe_512",
                         detail="procedural", save_npy=True)
elev = res["elev"]
temp = res["temp"]
precip = res["precip"]
ice = res["ice"]
land = elev >= 0

# ---- 2. build the upstream World ------------------------------------------
world = World(
    "compare", Size(w, h), 1234567,
    GenerationParameters(6, 0.5, Step.get_by_name("full")),
)
# elevation: normalise metres -> 0..1 with sea level at 0.5
e = numpy.clip((elev + 6000.0) / 11000.0, 0.0, 1.0).astype(numpy.float32)
e_th = [("sea", 0.545), ("plain", 0.60), ("hill", 0.68), ("mountain", None)]
world.elevation = (e, e_th)
world.plates = numpy.zeros((h, w), dtype=numpy.int32)
world.ocean = (~land)
world.sea_depth = numpy.clip(-elev / 6000.0, 0.0, 1.0).astype(numpy.float32)

# temperature: C -> normalised so the threshold bands match upstream defaults
t = numpy.clip((temp + 20.0) / 50.0, 0.0, 1.0).astype(numpy.float32)
t_th = [("polar", 0.16), ("alpine", 0.26), ("boreal", 0.38),
        ("cool", 0.50), ("warm", 0.66), ("subtropical", 0.80), ("tropical", None)]
world.temperature = (t, t_th)

# precipitation: mm -> normalised 0..1 with low/med thresholds
p = numpy.clip(precip / 4000.0, 0.0, 1.0).astype(numpy.float32)
p_th = [("low", 0.18), ("med", 0.45), ("hig", None)]
world.precipitation = (p, p_th)

# humidity: reuse precipitation with upstream quantiles for biome drawing
q = {}
for k in (87, 75, 62, 50, 37, 25, 12):
    q[str(k)] = float(numpy.percentile(p, 100 - k))
world.humidity = (p, q)
world.irrigation = numpy.zeros((h, w), dtype=numpy.int32)
world.permeability = (numpy.ones((h, w), dtype=numpy.float32),
                      [("low", 0.3), ("med", 0.6), ("hig", None)])

# icecap: bool array
world.icecap = ice.astype(bool)

# rivers from our D8 network -> upstream river_map conventions (>0 = river)
rivers, acc = pp.trace_rivers(elev, land, min_acc=max(40, (w * h) // 9000))
world.rivermap = (rivers.astype(numpy.float32)
                  * numpy.clip(numpy.log2(acc + 1), 1.0, 3.0)).astype(numpy.float32)
world.lakemap = numpy.zeros((h, w), dtype=numpy.float32)

# biome via upstream classifier
from worldengine.simulations.biome import BiomeSimulation

BiomeSimulation.execute(world, seed=1234567)
biome_matrix = world.biome

# ---- 3. upstream renders on OUR data --------------------------------------
draw_elevation_on_file(world, os.path.join(OUT, "up_elevation.png"), shadow=True)
draw_simple_elevation_on_file(world, os.path.join(OUT, "up_simple_elevation.png"), 0.545)
draw_temperature_levels_on_file(world, os.path.join(OUT, "up_temperature.png"))
draw_precipitation_on_file(world, os.path.join(OUT, "up_precipitation.png"))
draw_biome_on_file(world, os.path.join(OUT, "up_biome.png"))
draw_satellite_on_file(world, os.path.join(OUT, "up_satellite.png"))
draw_icecaps_on_file(world, os.path.join(OUT, "up_icecaps.png"))
print("upstream renders done ->", OUT)
