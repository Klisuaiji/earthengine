#!/usr/bin/env python3
"""WorldEngine pipeline with spherical Voronoi plates (14 raw -> 6 merged).

Replaces the platec tectonic simulation with domain-warped spherical
Voronoi, then generates elevation, runs climate simulation, and saves
all stages as colour PNGs.
"""

import os
import sys

import numpy

sys.path.insert(0, "/workspace")

from worldengine.biome import biome_name_to_index
from worldengine.draw import _biome_colors
from worldengine.generation import (
    Step,
    add_noise_to_elevation,
    center_land,
    initialize_ocean_and_thresholds,
    place_oceans_at_map_borders,
)
from worldengine.image_io import PNGWriter
from worldengine.model.world import GenerationParameters, Size, World
from worldengine.simulations.biome import BiomeSimulation
from worldengine.simulations.erosion import ErosionSimulation
from worldengine.simulations.humidity import HumiditySimulation
from worldengine.simulations.hydrology import WatermapSimulation
from worldengine.simulations.icecap import IcecapSimulation
from worldengine.simulations.irrigation import IrrigationSimulation
from worldengine.simulations.permeability import PermeabilitySimulation
from worldengine.simulations.precipitation import PrecipitationSimulation
from worldengine.simulations.temperature import TemperatureSimulation

from PIL import Image

sys.path.insert(0, "/workspace/tools")
from spherical_voronoi import generate_spherical_plates

OUT = "/workspace/stage_images"
W, H = 4096, 2048
SEED = 1
N_RAW = 30
N_BIG = 6

# ---------------- colour maps (normalised 0..1) ----------------

TERRAIN = [
    (0.00, (20, 40, 110)), (0.18, (40, 90, 160)), (0.32, (80, 145, 200)),
    (0.42, (95, 175, 205)), (0.48, (70, 150, 85)), (0.60, (135, 185, 95)),
    (0.72, (200, 175, 115)), (0.85, (160, 135, 115)), (1.00, (250, 250, 250)),
]
TEMPERATURE = [
    (0.00, (0, 0, 150)), (0.25, (30, 110, 210)), (0.50, (120, 200, 130)),
    (0.75, (255, 200, 50)), (1.00, (220, 30, 20)),
]
PRECIPITATION = [
    (0.00, (255, 255, 210)), (0.33, (190, 230, 160)), (0.66, (70, 180, 210)),
    (1.00, (10, 70, 180)),
]
SEA_DEPTH = [(0.00, (120, 190, 230)), (0.50, (40, 100, 190)), (1.00, (5, 20, 90))]
HUMIDITY = [(0.00, (230, 220, 150)), (0.50, (120, 190, 90)), (1.00, (20, 110, 40))]
PERMEABILITY = [(0.00, (150, 120, 80)), (0.50, (190, 170, 110)), (1.00, (120, 190, 110))]
WATERMAP = [(0.00, (245, 245, 235)), (0.60, (90, 180, 235)), (0.90, (20, 90, 220)), (1.00, (5, 30, 120))]
ICECAP = [(0.00, (150, 190, 225)), (1.00, (255, 255, 255))]

PLATE_PALETTE = numpy.array(
    [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
        (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
        (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ],
    dtype=float,
)

MERGED_PALETTE = numpy.array(
    [
        (20, 70, 190), (70, 150, 235),
        (90, 170, 80), (200, 180, 90), (180, 140, 80), (140, 190, 120),
    ],
    dtype=float,
)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def save_color(array, name, stops):
    path = os.path.join(OUT, name)
    stops = sorted(stops, key=lambda s: s[0])
    xs = numpy.array([s[0] for s in stops], dtype=float)
    cs = numpy.array([s[1] for s in stops], dtype=float)
    a = numpy.asarray(array, dtype=float)
    amin, amax = a.min(), a.max()
    if amax == amin:
        amax = amin + 1.0
    n = (a - amin) / (amax - amin)
    rgb = numpy.stack([numpy.interp(n, xs, cs[:, i]) for i in range(3)], axis=-1).astype(numpy.uint8)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_plates(array, name):
    path = os.path.join(OUT, name)
    n_plates = int(array.max()) + 1
    palette = numpy.resize(PLATE_PALETTE, (n_plates, 3))
    rgb = palette[array.astype(int)].astype(numpy.uint8)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def fractal_noise_field(shape, seed, octaves=4, base_res=64):
    """Low-res FBM field upscaled to (h, w) array shape via PIL BILINEAR."""
    fbm = numpy.zeros((base_res, base_res), dtype=numpy.float32)
    amp = 1.0
    norm = 0.0
    rng = numpy.random.RandomState(seed)
    for o in range(octaves):
        res = max(2, int(base_res * amp))
        small = rng.rand(res, res).astype(numpy.float32) * 2.0 - 1.0
        img = Image.fromarray(((small + 1.0) * 127.5).astype(numpy.uint8))
        fbm += amp * (numpy.asarray(img.resize((base_res, base_res), Image.BILINEAR), dtype=numpy.float32) / 127.5 - 1.0)
        norm += amp
        amp *= 0.5
    fbm /= max(norm, 1e-6)
    img = Image.fromarray(((fbm + 1.0) * 127.5).astype(numpy.uint8))
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return numpy.asarray(img, dtype=numpy.float32) / 127.5 - 1.0


def _distance_from_boundary(mask):
    """Distance transform: distance of each mask pixel to the nearest non-mask pixel."""
    from scipy import ndimage
    h, w = mask.shape
    dm = ndimage.distance_transform_edt(mask)
    dmax = dm.max()
    if dmax <= 0:
        return numpy.zeros((h, w), dtype=numpy.float32)
    return (dm / dmax).astype(numpy.float32)


def _generate_elevation(merged, h, w, seed):
    """Generate elevation from Voronoi merged groups."""
    ocean_mask = merged < 2
    elevation = numpy.where(ocean_mask, -5000.0, -2000.0)

    noise_a = fractal_noise_field((h, w), seed, octaves=4, base_res=64)
    noise_b = fractal_noise_field((h, w), seed + 1, octaves=5, base_res=48)
    noise_c = fractal_noise_field((h, w), seed + 2, octaves=4, base_res=42)
    noise_d = fractal_noise_field((h, w), seed + 3, octaves=6, base_res=36)
    noise_e = fractal_noise_field((h, w), seed + 4, octaves=3, base_res=28)

    for g in range(2, N_BIG):
        mask = merged == g
        if mask.sum() < 100:
            continue
        dm = _distance_from_boundary(mask)
        elevation += mask.astype(float) * dm * 5000.0

    elevation += noise_a * 1200.0
    elevation += noise_b * 600.0
    elevation += noise_c * 900.0
    elevation += noise_d * 400.0

    # Tiny jitter to break elevation ties (prevents droplet recursion)
    rng = numpy.random.RandomState(seed + 999)
    elevation += rng.rand(h, w).astype(numpy.float32) * 0.01

    return elevation.astype(numpy.float32)


def _voronoi_world(name, w, h, seed, n_raw=N_RAW, n_big=N_BIG,
                   temps=None, humids=None, gamma_curve=1.25, curve_offset=0.2,
                   ocean_level=1.0, step=Step.full()):
    """Create a World with spherical Voronoi plates + synthetic elevation."""
    if temps is None:
        temps = [0.874, 0.765, 0.594, 0.439, 0.366, 0.124]
    if humids is None:
        humids = [0.941, 0.778, 0.507, 0.236, 0.073, 0.014, 0.002]

    raw, merged = generate_spherical_plates(seed, w, h, n_raw=n_raw, n_big=n_big)
    elevation = _generate_elevation(merged, h, w, seed)
    elevation -= elevation.min()
    elevation /= elevation.max()
    elevation *= 8000.0
    elevation -= 4000.0

    world = World(
        name,
        Size(w, h),
        seed,
        GenerationParameters(n_raw, ocean_level, step),
        temps, humids, gamma_curve, curve_offset,
    )
    world.elevation = (elevation, {"ocean": 0.0})
    world.plates = raw
    return world, merged


def _label_components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=numpy.ones((3, 3), dtype=int))
    return lab.astype(numpy.int32), int(n)


def _remove_enclaves(plates, min_frac=0.0005):
    h, w = plates.shape
    min_area = int(min_frac * h * w)
    out = plates.astype(int).copy()
    for g in numpy.unique(plates):
        mask = out == g
        lab, n = _label_components(mask)
        for c in range(1, n + 1):
            comp = lab == c
            if int(comp.sum()) < min_area:
                yy, xx = numpy.nonzero(comp)
                neigh = numpy.concatenate([
                    out[numpy.clip(yy + dy, 0, h - 1), numpy.clip(xx + dx, 0, w - 1)]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if not (dy == 0 and dx == 0)
                ])
                neigh = neigh[neigh != g]
                if len(neigh):
                    vals, counts = numpy.unique(neigh, return_counts=True)
                    out[comp] = int(vals[counts.argmax()])
    return out


def save_merged_plates(merged, name):
    path = os.path.join(OUT, name)
    n_groups = int(merged.max()) + 1
    palette = numpy.resize(MERGED_PALETTE, (n_groups, 3))
    h, w = merged.shape
    big_h, big_w = h * 2, w * 2

    labels_img = Image.fromarray(merged.astype(numpy.uint8))
    labels_big = numpy.asarray(labels_img.resize((big_w, big_h), Image.NEAREST))
    rgb_big = palette[labels_big.astype(int)].astype(numpy.uint8).copy()

    boundary = numpy.zeros((big_h, big_w), dtype=bool)
    boundary[:, 1:] |= labels_big[:, 1:] != labels_big[:, :-1]
    boundary[1:, :] |= labels_big[1:, :] != labels_big[:-1, :]
    rgb_big[boundary] = [0, 0, 0]

    img_big = Image.fromarray(rgb_big)
    img_small = img_big.resize((w, h), Image.BILINEAR)
    rgb = numpy.asarray(img_small)

    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)

    for g in range(n_groups):
        area = (merged == g).sum()
        kind = "oceanic" if g < 2 else "continental"
        print(f"  group {g} ({kind}): area {area} px ({area/(h*w)*100:.1f}%)")


def save_ocean_mask(world, name):
    path = os.path.join(OUT, name)
    ocean = world.layers["ocean"].data
    h, w = ocean.shape
    rgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    rgb[:] = (95, 160, 90)
    rgb[ocean] = (70, 130, 210)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_rivers(world, name):
    path = os.path.join(OUT, name)
    ocean = world.layers["ocean"].data
    river = world.layers["river_map"].data
    lake = world.layers["lake_map"].data
    h, w = ocean.shape
    rgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    rgb[:] = (185, 205, 150)
    rgb[ocean] = (80, 135, 205)
    rgb[lake != 0] = (50, 130, 205)
    rgb[river > 0] = (10, 55, 180)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_biome(world, name):
    path = os.path.join(OUT, name)
    bm = world.layers["biome"].data
    h, w = bm.shape
    rgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    for name_, color in _biome_colors.items():
        rgb[bm == name_] = color[:3]
    rgb[bm == "bare rock"] = (128, 128, 128)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def main():
    ensure_out()

    step = Step.full()
    world, merged = _voronoi_world("stages_voronoi", W, H, SEED, n_raw=N_RAW, n_big=N_BIG, step=step)

    # 1. spherical Voronoi raw plates + merged groups
    save_color(world.layers["elevation"].data, "01_sv_elevation.png", TERRAIN)
    save_plates(world.layers["plates"].data, "02_sv_raw_plates.png")
    merged = _remove_enclaves(merged, min_frac=0.0005)
    save_merged_plates(merged, "02b_sv_merged.png")

    # 2. center land
    center_land(world)
    save_color(world.layers["elevation"].data, "03_sv_center_land.png", TERRAIN)

    # 3. add noise
    add_noise_to_elevation(world, numpy.random.randint(0, 4096))
    save_color(world.layers["elevation"].data, "04_sv_noise_elevation.png", TERRAIN)

    # 4. ocean borders + init
    place_oceans_at_map_borders(world)
    save_color(world.layers["elevation"].data, "05_sv_ocean_borders.png", TERRAIN)
    initialize_ocean_and_thresholds(world)
    save_color(world.layers["elevation"].data, "06_sv_ocean_init.png", TERRAIN)
    save_ocean_mask(world, "07_sv_ocean_mask.png")
    save_color(world.layers["sea_depth"].data, "08_sv_sea_depth.png", SEA_DEPTH)

    rng = numpy.random.RandomState(SEED)
    sub_seeds = rng.randint(0, numpy.iinfo(numpy.int32).max, size=100)

    # 5. temperature
    TemperatureSimulation().execute(world, sub_seeds[4])
    save_color(world.layers["temperature"].data, "09_sv_temperature.png", TEMPERATURE)

    # 6. precipitation
    PrecipitationSimulation().execute(world, sub_seeds[0])
    save_color(world.layers["precipitation"].data, "10_sv_precipitation.png", PRECIPITATION)

    # 7. erosion -> rivers + lakes
    ErosionSimulation().execute(world, sub_seeds[1])
    save_rivers(world, "11_sv_rivermap.png")
    save_rivers(world, "12_sv_lakemap.png")
    save_color(world.layers["elevation"].data, "13_sv_eroded_elevation.png", TERRAIN)

    # 8. watermap
    WatermapSimulation().execute(world, sub_seeds[2])
    save_color(world.layers["watermap"].data, "14_sv_watermap.png", WATERMAP)

    # 9. irrigation
    IrrigationSimulation().execute(world, sub_seeds[3])
    save_color(world.layers["irrigation"].data, "15_sv_irrigation.png", PRECIPITATION)

    # 10. humidity
    HumiditySimulation().execute(world, sub_seeds[5])
    save_color(world.layers["humidity"].data, "16_sv_humidity.png", HUMIDITY)

    # 11. permeability
    PermeabilitySimulation().execute(world, sub_seeds[6])
    save_color(world.layers["permeability"].data, "17_sv_permeability.png", PERMEABILITY)

    # 12. biome
    BiomeSimulation().execute(world, sub_seeds[7])
    save_biome(world, "18_sv_biome.png")

    # 13. icecap
    IcecapSimulation().execute(world, sub_seeds[8])
    save_color(world.layers["icecap"].data, "19_sv_icecap.png", ICECAP)

    print("all stages done")


if __name__ == "__main__":
    main()
