#!/usr/bin/env python3
"""WorldEngine stage renderer driven by spherical Voronoi plates + Terrain Diffusion.

Generates the plate layout with :mod:`worldengine.spherical_voronoi`
(domain-warped spherical Voronoi, ``n_raw`` micro-plates merged into
``n_big`` major plates), then feeds the resulting continent/ocean mask into the
Terrain Diffusion model as coarse elevation conditioning. The diffusion model
renders real elevation (mountains, ocean trenches), and a compact physical
climate model produces temperature/precipitation from the new terrain.

Pipeline::

    raw plates -> merged plates -> continents -> boundary types
    -> diffusion elevation (real relief) -> temperature -> precipitation
    -> land/sea mask

Usage (requires the terrain-diffusion venv, default CUDA device)::

    python tools/generate_stages_voronoi.py                  # 1024x512, seed 1
    python tools/generate_stages_voronoi.py -W 2048 -H 1024
    python tools/generate_stages_voronoi.py --seed 7 --out /tmp/stages
"""

import argparse
import os
import sys
from pathlib import Path

import numpy
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldengine.image_io import PNGWriter
from worldengine.plate_boundaries import classify_boundaries
from worldengine.spherical_voronoi import (
    PLATE_AREAS,
    PLATE_OCEAN,
    generate_spherical_world,
    remove_enclaves,
)

# Diffusion terrain pipeline lives in the terrain-diffusion venv; import its helpers.
from tools.diffusion_world import (
    build_pipeline,
    conditioning_from_mask,
    generate_world as generate_diffusion_world,
    render_relief,
    _colormap_png,
    compute_physical_climate,
    clamp_land_sea,
)

DEFAULT_OUT = str(ROOT / "stage_images")
DEFAULT_W, DEFAULT_H = 1024, 512
DEFAULT_SEED = 1
DEFAULT_N_RAW = 30
DEFAULT_N_BIG = 6

# Set by main(); every save_* helper writes here.
OUT = DEFAULT_OUT

# ---------------- palettes ----------------

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

CONTINENT_PALETTE = numpy.array([
    (28, 62, 110),   # ocean       (label -1 -> idx 0)
    (46, 120, 52),   # continent 0
    (120, 180, 90),  # continent 1
    (200, 160, 70),  # continent 2
], dtype=numpy.uint8)

BOUNDARY_PALETTE = numpy.array([
    (240, 240, 250),  # INTERIOR
    (160, 32, 240),   # CONVERGENT (purple)
    (30, 180, 60),    # DIVERGENT (green)
    (255, 140, 0),    # TRANSFORM (orange)
], dtype=numpy.uint8)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)


def save_plates(array, name):
    path = os.path.join(OUT, name)
    n_plates = int(array.max()) + 1
    palette = numpy.resize(PLATE_PALETTE, (n_plates, 3))
    rgb = palette[array.astype(int)].astype(numpy.uint8)
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


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

    n_groups_expected = len(PLATE_AREAS)
    for g in range(n_groups):
        area = (merged == g).sum()
        if n_groups == n_groups_expected:
            kind = "oceanic" if PLATE_OCEAN[g] else "continental"
        else:
            kind = "oceanic" if g < 2 else "continental"
        print(f"  group {g} ({kind}): area {area} px ({area / (h * w) * 100:.1f}%)")


def save_continents(continent_mask, name):
    path = os.path.join(OUT, name)
    palette = CONTINENT_PALETTE
    # shift labels by +1 so -1 (ocean) maps to index 0
    idx = numpy.clip(continent_mask + 1, 0, palette.shape[0] - 1)
    rgb = palette[idx.astype(numpy.int32)]
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    land_pct = float((continent_mask >= 0).sum()) / continent_mask.size * 100.0
    print("wrote", path, f"land={land_pct:.1f}%")


def save_boundary_types(boundary_type, name):
    path = os.path.join(OUT, name)
    rgb = BOUNDARY_PALETTE[boundary_type.astype(numpy.int8)]
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    counts = {int(k): int((boundary_type == k).sum()) for k in numpy.unique(boundary_type)}
    print("wrote", path, "boundary counts:", counts)


def save_elevation_relief(elev, name, vert_exag=2.5):
    """Render the diffusion elevation as a hillshaded hypsometric relief PNG."""
    path = os.path.join(OUT, name)
    render_relief(elev, path, vert_exag=vert_exag)
    print("wrote", path)


def save_colormap(arr, name, cmap="viridis"):
    """Render a single-channel climate field to a PNG."""
    path = os.path.join(OUT, name)
    _colormap_png(arr, path, cmap=cmap)
    print("wrote", path)


def save_ocean_mask(ocean_mask, name):
    path = os.path.join(OUT, name)
    h, w = ocean_mask.shape
    rgb = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    rgb[:] = (95, 160, 90)          # land
    rgb[ocean_mask] = (70, 130, 210)  # ocean
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    print("wrote", path)


def save_terrain_kinds(terrain, name):
    path = os.path.join(OUT, name)
    rgb = TERRAIN_RGB[terrain.astype(numpy.int32)]
    img = PNGWriter.rgb_from_array(rgb, path)
    img.complete()
    counts = {int(k): int((terrain == k).sum()) for k in numpy.unique(terrain)}
    print("wrote", path, "terrain counts:", counts)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-W", "--width", type=int, default=DEFAULT_W, help="map width in pixels")
    ap.add_argument("-H", "--height", type=int, default=DEFAULT_H, help="map height in pixels")
    ap.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="world seed")
    ap.add_argument("--n-raw", type=int, default=DEFAULT_N_RAW, help="number of Voronoi micro-plates")
    ap.add_argument("--n-big", type=int, default=DEFAULT_N_BIG, help="number of merged major plates (2 oceanic)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output directory for the stage PNGs")
    ap.add_argument("--no-diffusion", action="store_true",
                    help="render tectonic stages only (no torch / Terrain Diffusion needed)")
    return ap.parse_args(argv)


def main(argv=None):
    global OUT
    args = parse_args(argv)
    OUT = args.out
    w, h, seed = args.width, args.height, args.seed
    ensure_out()

    print(f"generating {w}x{h}, seed={seed}, {args.n_raw} micro-plates -> {args.n_big} major plates")
    raw, merged, land_mask, continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=args.n_raw, n_big=args.n_big
    )
    merged = remove_enclaves(merged, min_frac=0.0005)
    ocean_mask = ~land_mask

    # 1. raw micro-plates
    save_plates(raw, "01_sv_raw_plates.png")

    # 2. merged major plates (with anti-aliased boundaries)
    save_merged_plates(merged, "02_sv_merged.png")

    # 3. generated continents
    save_continents(continent_mask, "03_sv_continents.png")

    # 4. plate boundary classification
    bnd = classify_boundaries(merged, seed=seed)
    save_boundary_types(bnd["boundary_type"], "04_sv_boundary_types.png")

    # 5-7. Terrain Diffusion: real elevation + physical climate.
    if args.no_diffusion:
        print("skipping diffusion stages (--no-diffusion); tectonic stages complete")
        return
    print("building diffusion pipeline ...")
    pipe = build_pipeline(seed=seed, device="auto")
    print("injecting Voronoi coarse conditioning ...")
    grid = conditioning_from_mask(land_mask)
    pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=-8000.0)
    print(f"generating diffusion elevation {w}x{h} ...")
    elev, climate_raw = generate_diffusion_world(pipe, w, h, tile=256, device="auto")
    pipe.close()
    elev = clamp_land_sea(elev, land_mask)
    temp, precip = compute_physical_climate(elev, h, w)
    print("  land%%=%.1f  elev[%.1f/%.1f]  temp[%.1f/%.1f]  precip[%.0f/%.0f]" %
          ((elev >= 0).mean() * 100, elev.min(), elev.max(), temp.min(), temp.max(),
           precip.min(), precip.max()))

    save_elevation_relief(elev, "05_sv_elevation_relief.png")
    save_colormap(temp, "06_sv_temperature.png", cmap="turbo")
    save_colormap(precip, "07_sv_precipitation.png", cmap="viridis")

    # 8. land / sea mask
    save_ocean_mask(ocean_mask, "08_sv_ocean_mask.png")

    # also save raw arrays for downstream use
    numpy.save(os.path.join(OUT, "elevation.npy"), elev)
    numpy.save(os.path.join(OUT, "temperature.npy"), temp)
    numpy.save(os.path.join(OUT, "precipitation.npy"), precip)
    numpy.save(os.path.join(OUT, "land_mask.npy"), land_mask)

    print("all stages done")


if __name__ == "__main__":
    main()
