#!/usr/bin/env python
"""Complete three-phase planet generation pipeline.

This is the orchestration layer that turns a world seed into a fully realised
planet: tectonics -> climate & rendering -> ecology & civilization.  It ties
together the spherical-Voronoi plate generator, the plate-boundary classifier
and the Terrain-Diffusion elevation refiner that already live in this repo, and
adds every missing stage from the spec:

  Phase 1  Tectonic
    1  plate simulation          (spherical_voronoi)
    2  merge plates              (spherical_voronoi)
    3  classify boundaries       (plate_boundaries: convergent/divergent/transform)
    4  simulate continents       (boundary-stress coarse elevation)
    5  coastline subdivision     (fractal sea-level cut)
    6  island splitting          (archipelago / peninsula noise)
    7  initial colouring         (ocean / plain / hill / mountain palette)
    8  initial grayscale         (coarse elevation, pre-diffusion)
    9  diffusion refinement      (Terrain Diffusion -> real relief)

  Phase 2  Climate & Rendering
    13 solar radiation / temperature bands
    14 continental pressure (land-sea thermal + orographic)
    15 wind belts (planetary + orographic deflection, monsoon)
    16 precipitation (windward orographic + rain shadow + arid interior)
    17 Koppen climate classification (simplified)
    18 ice sheets / glaciers (temp + elevation, iterated with currents)
    19 ocean currents (wind-driven + coastal, coupled to ice)
    20 layer compositing
    21 satellite image
    22 planet map

  Phase 3  Ecology & Civilization
    23 biome (from climate + elevation + currents)
    24 civilization development index (habitability)
    25 civilization seed points (habitability hotspots, isolation-filtered)

Run:
    python tools/generate_planet.py --seed 1234567 --width 1024 --height 512
"""
import os
import sys
import json
import math
import argparse
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.dirname(os.path.abspath(__file__))
for p in (TOOLS, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from worldengine.spherical_voronoi import generate_spherical_world
from worldengine.plate_boundaries import (
    classify_boundaries,
    CONVERGENT,
    DIVERGENT,
    TRANSFORM,
)

# ----- native pixels per diffusion conditioning cell (world_pipeline: S = 256)
CELL = 256

# ----- tectonic coarse-elevation constants (metres) -------------------------
# Tuned so the conditioning elevation, after the pipeline's sign(sqrt)
# normalisation, lands clearly on the land/ocean side of the training mean
# (-37.7 m, std 39.7).  +3500 -> +2.0 std (clear land); -6000 -> -1.9 std
# (clear ocean).  This is what lets the Voronoi continent mask actually steer
# the diffusion output instead of being overwhelmed by its ocean prior.
LAND_BASE = 3500.0        # plateau elevation for interior continents
MOUNTAIN_BONUS = 1500.0   # extra relief near convergent boundaries (-> +5000)
OCEAN_BASE = -6000.0      # open-ocean floor
TRENCH = -8000.0          # subduction trenches (convergent, ocean side)
RIDGE = -4500.0           # mid-ocean ridges (divergent)


# ===========================================================================
#  small helpers
# ===========================================================================
def _lat(h):
    """Latitude in degrees for each row (+90 top -> -90 bottom)."""
    yy = np.arange(h, dtype=np.float32)[:, None]
    return 90.0 - 180.0 * (yy + 0.5) / h


def _smooth_noise(shape, scale, seed):
    """Fractal value noise in [0,1] (vectorised; scipy for upsampling)."""
    from scipy.ndimage import zoom
    h, w = shape
    rng = np.random.default_rng(seed)
    gh, gw = max(2, h // scale), max(2, w // scale)
    coarse = rng.random((gh + 2, gw + 2)).astype(np.float32)
    fine = zoom(coarse, (h / (gh + 2), w / (gw + 2)), order=1)
    return np.clip(fine[:h, :w].astype(np.float32), 0.0, 1.0)


def _fbm(shape, base_scale, octaves, seed):
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        out += amp * _smooth_noise(shape, max(2, base_scale >> o), seed + o * 101)
        tot += amp
        amp *= 0.5
    return out / tot


# ===========================================================================
#  PHASE 1 - TECTONIC
# ===========================================================================
def tectonic(seed, w, h, n_raw=30, n_big=6):
    """Steps 1-3: plates, merge, boundary classification."""
    raw, merged, land_mask, continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=n_raw, n_big=n_big)
    boundaries = classify_boundaries(merged, seed=seed)
    return {
        "raw": raw,
        "merged": merged,
        "land_mask": land_mask.astype(bool),
        "continent_mask": continent_mask,
        "boundaries": boundaries,
    }


def tectonic_coarse_elevation(land_mask, boundaries):
    """Steps 4 & 8: boundary-stress continent/ocean floor, pre-diffusion.

    Convergent boundaries raise mountains on land and dig trenches at sea;
    divergent boundaries build mid-ocean ridges.  Returns (h,w) float32 metres.
    """
    h, w = land_mask.shape
    bt = boundaries["boundary_type"]
    cdist = boundaries["convergent_dist"]
    ddist = boundaries["divergent_dist"]
    lm = land_mask.astype(np.float32)

    elev = np.where(lm > 0.5, LAND_BASE, OCEAN_BASE)
    # mountains near convergent boundaries
    conv = 2200.0 * np.exp(-cdist / 45.0)
    elev = elev + conv * (lm + (cdist < 30).astype(np.float32))
    # trenches (ocean, convergent) and ridges (ocean, divergent)
    trench = (TRENCH - OCEAN_BASE) * np.exp(-cdist / 35.0) * (1.0 - lm)
    ridge = (RIDGE - OCEAN_BASE) * np.exp(-ddist / 45.0) * (1.0 - lm)
    elev = elev + trench + ridge
    return np.clip(elev, TRENCH, LAND_BASE + MOUNTAIN_BONUS)


def coarse_conditioning_grid(coarse_elev, land_mask):
    """Downsample coarse elevation to the diffusion conditioning grid.

    Land cells take the 85th percentile (mountains dominate); ocean cells take
    the 15th percentile (trenches dominate).  Returns (gh, gw) float32.
    """
    h, w = coarse_elev.shape
    gh = (h + CELL - 1) // CELL
    gw = (w + CELL - 1) // CELL
    grid = np.full((gh, gw), OCEAN_BASE, dtype=np.float32)
    for ci in range(gh):
        y0, y1 = ci * CELL, min((ci + 1) * CELL, h)
        for cj in range(gw):
            x0, x1 = cj * CELL, min((cj + 1) * CELL, w)
            block = coarse_elev[y0:y1, x0:x1]
            lblock = land_mask[y0:y1, x0:x1]
            if block.size == 0:
                continue
            if lblock.mean() >= 0.5:
                val = np.percentile(block[lblock], 85) if lblock.any() else LAND_BASE
                grid[ci, cj] = max(val, LAND_BASE)        # guarantee clearly-land
            else:
                val = np.percentile(block, 15)
                grid[ci, cj] = min(val, OCEAN_BASE)       # guarantee clearly-ocean
    return grid


def refine_coastline_and_islands(elev, land_mask, boundaries, seed):
    """Steps 5 & 6: fractal coastline cut + archipelago / peninsula splitting.

    The diffusion model has a strong ocean prior, so its raw elevation is
    negative over much of the Voronoi land.  We therefore keep the *continent
    shape* anchored to the tectonic mask and only perturb the narrow coastal
    fringe with fractal noise (step 5), then drop archipelago / peninsula
    bumps into nearby ocean (step 6), and finally fill continental interiors
    with gentle fractal relief so land is not dead-flat.
    """
    from scipy.ndimage import distance_transform_edt
    h, w = elev.shape
    land = land_mask.astype(bool)
    ocean = ~land
    d_land = distance_transform_edt(ocean)   # 0 on land, grows into ocean
    d_ocean = distance_transform_edt(land)   # 0 on ocean, grows into land

    # 5. coastline subdivision: perturb only the coastal fringe.  The diffusion
    #    elevation is negative over most Voronoi land, so we must NOT cut at its
    #    sign (that would drown the continents).  Instead we keep the tectonic
    #    continent shape and nudge the boundary: coastal land floods where the
    #    sea-level noise is high, coastal ocean emerges where it is low.
    band = 6
    coast = (d_land < band) | (d_ocean < band)
    wander = (_fbm((h, w), base_scale=48, octaves=4, seed=seed + 7) - 0.5) * 280.0
    final_land = land.copy()
    flood = coast & land & (wander > 90.0)     # sea-level rises -> drowns coast
    emerge = coast & ocean & (wander < -90.0)  # sea-level drops -> exposes land
    final_land[flood] = False
    final_land[emerge] = True

    # 6. island splitting: raise archipelago bumps in nearby ocean.
    ddist = boundaries["divergent_dist"]
    near_coast = d_ocean < 70
    near_div = ddist < 90
    arch = _fbm((h, w), base_scale=36, octaves=5, seed=seed + 13)
    island_seed = ocean & (arch > 0.82) & (near_coast | near_div)
    if island_seed.any():
        bump = (arch - 0.82) / 0.18 * 700.0      # up to +700 m
        elev = elev.copy()
        elev[island_seed] = np.maximum(elev[island_seed], bump[island_seed])
        final_land[island_seed] = True

    # continental interior fill: avoid dead-flat lowlands (keeps diffusion
    # mountains where they exist, adds fractal relief elsewhere).
    interior = final_land & (d_ocean > 25) & (elev < 300)
    fill = 150.0 + _fbm((h, w), base_scale=28, octaves=5, seed=seed + 21) * 550.0
    elev[interior] = np.maximum(elev[interior], fill[interior])
    return final_land, elev


def initial_coloring(coarse_elev):
    """Step 7: coarse landform classes (ocean / plain / hill / mountain)."""
    h, w = coarse_elev.shape
    cls = np.zeros((h, w), dtype=np.int8)
    cls[coarse_elev >= 1000] = 3          # mountain
    cls[(coarse_elev >= 300) & (coarse_elev < 1000)] = 2   # hill
    cls[(coarse_elev >= 0) & (coarse_elev < 300)] = 1      # plain
    # 0 = ocean
    return cls


def diffusion_refine(grid, seed, w, h, device="auto", snr0=0.5):
    """Step 9: run Terrain Diffusion with the coarse conditioning grid."""
    from diffusion_world import build_pipeline, generate_world, resolve_device
    device = resolve_device(device)
    pipe = build_pipeline(seed=seed, device=device, snr0=snr0)
    pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=OCEAN_BASE)
    elev, climate_raw = generate_world(pipe, w, h, tile=256, device=device)
    pipe.close()
    return elev, climate_raw


# ===========================================================================
#  PHASE 2 - CLIMATE & RENDERING
# ===========================================================================
def solar_temperature(elev, h):
    """Step 13: latitude temperature band + lapse rate + continentality."""
    lat = _lat(h)
    t0 = 30.0 * np.cos(np.deg2rad(lat)) - 5.0          # ~25 C eq, ~-35 C pole
    # continentality: land warms a little at the equator, cools at high lat
    land = (elev >= 0).astype(np.float32)
    t0 = t0 + land * (4.0 * np.cos(np.deg2rad(lat)) - 2.0 * np.abs(lat) / 90.0 * 6.0)
    temp = t0 - 0.0065 * np.maximum(elev, 0.0)         # lapse rate
    return temp.astype(np.float32)


def continental_pressure(temp, elev, h):
    """Step 14: thermal pressure (warm -> low) + orographic low over highlands."""
    tmean = temp.mean()
    p = -0.9 * (temp - tmean)                          # hPa-ish anomaly
    p = p - 0.02 * np.maximum(elev, 0.0)              # high ground -> lower pressure
    return p.astype(np.float32)


def wind_field(elev, temp, boundaries, h, w):
    """Step 15: planetary wind belts + orographic deflection + monsoon.

    Returns (u, v) wind vector field (m/s-ish, unitless direction matters).
    """
    lat = _lat(h)
    abslat = np.abs(lat)
    # planetary zonal wind: trades (westerly? convention: u<0 = blowing west),
    # westerlies, polar easterlies.  Use a smooth banded function.
    u = np.zeros((h, w), dtype=np.float32)
    # 0-30 trades (eastward flow u<0), 30-60 westerlies (u>0), 60-90 polar (u<0)
    band = np.select(
        [abslat < 30, (abslat >= 30) & (abslat < 60), abslat >= 60],
        [-1.0, 1.0, -1.0], default=0.0)
    u = band * (1.0 - 0.3 * np.cos(np.deg2rad(lat)) ** 2)
    # orographic deflection: wind bends around mountains -> v component from
    # gradient of elevation (flow goes around high ground)
    gy, gx = np.gradient(elev.astype(np.float32))
    v = 0.4 * (np.roll(gx, 1, axis=0) - gx)  # rough divergence around peaks
    # monsoon: low-pressure warm land pulls wind from ocean (toward land).
    # Approximate with v toward lower pressure / land by gradient of temp over land.
    land = (elev >= 0).astype(np.float32)
    gy_t, gx_t = np.gradient(temp * land)
    v = v - 0.5 * gy_t  # flow toward warmer (land) regions in the hemisphere sense
    # normalise magnitude a bit
    mag = np.sqrt(u ** 2 + v ** 2)
    mag = np.where(mag > 0, mag, 1.0)
    u = u / mag * np.clip(mag, 0.3, 1.5)
    v = v / mag * np.clip(mag, 0.3, 1.5)
    return u.astype(np.float32), v.astype(np.float32)


def precipitation(u, v, elev, h):
    """Step 16: zonal base + windward orographic lift + rain shadow + aridity."""
    lat = _lat(h)
    abslat = np.abs(lat)
    base = (2500.0 * np.exp(-((abslat / 12.0) ** 2))
            + 800.0 * np.exp(-(((abslat - 60.0) / 18.0) ** 2))
            + 300.0)
    gy, gx = np.gradient(elev.astype(np.float32))
    # windward slope: uphill in the up-wind direction -> precip enhancement
    # wind (u,v) points the direction air moves TOWARD; upwind = -(u,v)
    windward = -(u * gx + v * gy) / 500.0
    oro = 1.0 + 1.6 * np.clip(windward, 0.0, 1.5)
    shadow = 1.0 - 0.4 * np.clip(-windward, 0.0, 1.5)   # leeward reduction
    arid = 1.0 - 0.35 * np.clip(elev / 4000.0, 0.0, 1.0)
    arid = arid * (1.0 - 0.25 * np.clip(np.abs(lat) - 20.0, 0.0, 1.0) / 50.0)
    precip = base * oro * shadow * arid
    return precip.astype(np.float32)


def koppen(temp, precip, elev, h):
    """Step 17: simplified Koppen climate classification.

    Returns (code, label) arrays.  Groups: Af, Am, Aw (A tropical),
    BWh/BWk/BS (arid), Cfa/Cfb/Csa/Csb (temperate), Dfb/Dfc/Dfa (boreal),
    ET/EF (polar), Ocean.
    """
    lat = _lat(h)
    # seasonal amplitude proxy: larger over land and at high latitude
    land = (elev >= 0).astype(np.float32)
    amp = 6.0 + 14.0 * np.abs(lat) / 90.0 + 8.0 * land
    t_cold = temp - amp          # coldest-month approx
    t_warm = temp + amp * 0.6    # warmest-month approx
    code = np.empty(temp.shape, dtype=object)
    code[:] = "Ocean"
    # aridity threshold (simplified Koppen B)
    arid_thr = 10.0 * t_warm
    is_arid = precip < arid_thr
    # tropical A: coldest month > 18
    A = t_cold > 18
    # polar E: warmest month < 10
    E = t_warm < 10
    # temperate C: coldest > -3 and warmest > 10 and not tropical/polar
    C = (t_cold > -3) & (t_warm >= 10) & ~A & ~E
    # boreal/continental D: everything temperate-cold that is not C.  (Do NOT
    # add `t_warm >= 10` here - it made the Dfc branch below unreachable and
    # mislabelled cold continental land as "Ocean".)
    D = (~A) & (~E) & (~C)

    code[A & (precip >= 60)] = "Af"
    code[A & (precip < 60) & (precip >= 25)] = "Am"
    code[A & (precip < 25)] = "Aw"
    code[E & (t_warm < 0)] = "EF"
    code[E & (t_warm >= 0)] = "ET"
    code[C & (precip >= 50)] = "Cfb"
    code[C & (precip < 50) & (t_cold >= 0)] = "Cfa"
    code[C & (precip < 50) & (t_cold < 0)] = "Csb"
    code[D & (t_warm >= 10)] = "Dfb"
    code[D & (t_warm < 10)] = "Dfc"
    # arid subgroup
    code[is_arid & (t_warm >= 18)] = "BWh"
    code[is_arid & (t_warm < 18)] = "BWk"
    code[is_arid & (precip >= 0.5 * arid_thr)] = "BS"  # steppe-ish fallback
    return code


KOPPEN_RGB = {
    "Af": (60, 160, 60), "Am": (90, 190, 70), "Aw": (150, 200, 90),
    "BWh": (225, 205, 130), "BWk": (205, 190, 160), "BS": (220, 200, 150),
    "Cfa": (80, 170, 120), "Cfb": (110, 190, 140), "Csb": (120, 185, 120),
    "Dfb": (70, 150, 170), "Dfc": (90, 140, 190), "Dfa": (60, 160, 150),
    "ET": (200, 220, 230), "EF": (235, 240, 245), "Ocean": (30, 80, 150),
}


def ice_layer(temp, elev, h, sst=None):
    """Step 18: ice sheets (polar) + glaciers (high & cold). Optional SST warms
    coastal margins.  Returns boolean ice mask."""
    lat = _lat(h)
    ice = np.zeros(temp.shape, dtype=bool)
    # polar caps: cold and high latitude
    ice |= (temp < 0) & (np.abs(lat) > 60)
    # glaciers: cold highlands
    ice |= (elev > 2500) & (temp < 0)
    # coastal moderation from currents (sst) prevents ice where water is warm
    if sst is not None:
        ice &= ~(sst > 2)
    return ice


def ocean_currents(u, v, elev, h, w, iterations=2):
    """Step 19: simplified wind-driven surface currents with coastal steering.

    Ekman-like surface current = wind rotated ~45 deg by the Coriolis force
    (right in N, left in S hemisphere), suppressed on land, with a mild
    western-boundary intensification.  Returns (cu, cv, sst) where ``sst`` is a
    crude advected sea-surface temperature (warm water pushed poleward), used
    by the caller to re-derive ice (the 18<->19 coupling).
    """
    land = (elev >= 0).astype(np.float32)
    lat = _lat(h)
    hemi = np.sign(lat)
    cu = u * math.cos(math.radians(45.0)) - hemi * v * math.sin(math.radians(45.0))
    cv = v * math.cos(math.radians(45.0)) + hemi * u * math.sin(math.radians(45.0))
    speed = np.sqrt(cu ** 2 + cv ** 2)
    # eastern coastline (land pixel whose east neighbour is ocean) -> western
    # boundary intensification on the ocean just west of it.
    east_of_land = (np.roll(land, 1, axis=1) > 0.5) & (land < 0.5)
    speed = speed * (1.0 + 0.6 * east_of_land.astype(np.float32))
    speed = speed * (1.0 - land)  # zero on land
    mag = np.sqrt(cu ** 2 + cv ** 2)
    mag = np.where(mag > 1e-6, mag, 1.0)
    cu = cu / mag * speed
    cv = cv / mag * speed
    # crude SST: latitudinal base, advected by the currents over a few steps.
    sst = (30.0 * np.cos(np.deg2rad(lat)) - 2.0).astype(np.float32)
    sst = sst * (1.0 - land)
    for _ in range(iterations):
        adv = (np.roll(sst, 1, axis=1) - sst) * np.sign(cu + 1e-9) * 0.15
        adv = adv + (np.roll(sst, 1, axis=0) - sst) * np.sign(cv + 1e-9) * 0.1
        sst = np.where(land > 0.5, sst, sst + adv)
    return cu.astype(np.float32), cv.astype(np.float32), sst.astype(np.float32)


# ===========================================================================
#  PHASE 3 - ECOLOGY & CIVILIZATION
# ===========================================================================
BIOME_OF_KOPPEN = {
    "Af": "Tropical rainforest", "Am": "Tropical monsoon", "Aw": "Tropical savanna",
    "BWh": "Hot desert", "BWk": "Cold desert", "BS": "Steppe",
    "Cfa": "Temperate broadleaf", "Cfb": "Temperate forest", "Csb": "Mediterranean",
    "Dfb": "Boreal forest", "Dfc": "Taiga", "Dfa": "Humid continental",
    "ET": "Tundra", "EF": "Ice sheet", "Ocean": "Ocean",
}
BIOME_RGB = {
    "Tropical rainforest": (30, 120, 50), "Tropical monsoon": (50, 150, 60),
    "Tropical savanna": (170, 190, 90), "Hot desert": (220, 200, 130),
    "Cold desert": (200, 185, 155), "Steppe": (210, 195, 140),
    "Temperate broadleaf": (70, 160, 90), "Temperate forest": (90, 175, 110),
    "Mediterranean": (120, 180, 100), "Boreal forest": (60, 140, 150),
    "Taiga": (80, 130, 165), "Humid continental": (75, 155, 130),
    "Tundra": (190, 210, 200), "Ice sheet": (235, 240, 245), "Ocean": (30, 80, 150),
}


def biome(koppen_code, ice, h):
    """Step 23: biome from Koppen + ice override."""
    codes, inverse = np.unique(koppen_code, return_inverse=True)
    names = np.array([BIOME_OF_KOPPEN.get(c, "Ocean") for c in codes])
    out = names[inverse.reshape(koppen_code.shape)]
    out = np.where(ice, "Ice sheet", out)
    out = out.astype(object)
    return out


def civilization_index(temp, precip, elev, koppen_code, biome_name):
    """Step 24: habitability score in [0,1].

    arable (temperate/biome), fresh water (precip 400-2000), comfort
    (temp 0-30), resources (coastal & not too high).
    """
    land = (elev >= 0).astype(np.float32)
    # arable: biome in {Temperate forest, Temperate broadleaf, Mediterranean,
    # Humid continental, Tropical savanna, Steppe}
    arable_biomes = {"Temperate forest", "Temperate broadleaf", "Mediterranean",
                     "Humid continental", "Tropical savanna", "Steppe"}
    # LUT over the (few) unique biome names instead of a per-pixel Python loop.
    uniq, inverse = np.unique(biome_name, return_inverse=True)
    arable = np.isin(uniq, list(arable_biomes))[inverse.reshape(biome_name.shape)]
    arable = arable.astype(np.float32)
    # water
    water = np.clip((precip - 300) / 1200.0, 0, 1) * np.clip((2200 - precip) / 1000.0, 0, 1)
    # comfort
    comfort = np.exp(-((temp - 15.0) ** 2) / (2 * 12.0 ** 2))
    # resources: low elevation (minable/arable) but not floodplain
    res = np.clip(1.0 - np.abs(elev) / 3000.0, 0, 1)
    score = (0.35 * arable + 0.25 * water + 0.25 * comfort + 0.15 * res) * land
    return np.clip(score, 0, 1).astype(np.float32)


def civilization_points(score, min_spacing=24):
    """Step 25: isolation-filtered habitability hotspots."""
    h, w = score.shape
    pts = []
    cand = score.copy()
    thr = 0.55
    r2 = min_spacing * min_spacing
    while True:
        yx = np.unravel_index(np.argmax(cand), cand.shape)
        s = cand[yx]
        if s < thr:
            break
        y, x = int(yx[0]), int(yx[1])
        pts.append((y, x, float(s)))
        # suppress a neighbourhood (windowed: only touches the local disk)
        y0, y1 = max(0, y - min_spacing), min(h, y + min_spacing + 1)
        x0, x1 = max(0, x - min_spacing), min(w, x + min_spacing + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        cand[y0:y1, x0:x1][(yy - y) ** 2 + (xx - x) ** 2 <= r2] = 0.0
        if len(pts) > 60:
            break
    return pts


# ===========================================================================
#  rendering helpers (reuse diffusion_world where possible)
# ===========================================================================
def _discrete_png(arr, palette, out_path):
    from PIL import Image
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for k, col in palette.items():
        rgb[arr == k] = col
    Image.fromarray(rgb).save(out_path)
    return out_path


def _label_png(labels, palette, out_path):
    from PIL import Image
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    flat = labels.flatten()
    for k, col in palette.items():
        rgb.reshape(-1, 3)[flat == k] = col
    Image.fromarray(rgb).save(out_path)
    return out_path


def _grayscale_png(elev, out_path, lo=-8000, hi=4000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(elev.shape[1] / 100.0, elev.shape[0] / 100.0), dpi=100)
    ax.imshow(np.clip(elev, lo, hi), cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def _quiver_png(u, v, out_path, bg=None, step=24, scale=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = u.shape
    fig, ax = plt.subplots(figsize=(w / 100.0, h / 100.0), dpi=100)
    if bg is not None:
        ax.imshow(bg, origin="upper")
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ax.quiver(xs, ys, u[::step, ::step] * scale, -v[::step, ::step] * scale,
              scale=1.0, scale_units="xy", angles="xy", width=0.0025,
              color="white", alpha=0.8)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def _hillshade(elev, az=315.0, alt=45.0, vert_exag=1.0):
    dy, dx = np.gradient(elev.astype(np.float32) * vert_exag)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(dy, -dx)
    azr, altr = np.deg2rad(az), np.deg2rad(alt)
    shade = (np.cos(altr) * np.cos(slope) +
             np.sin(altr) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(shade, 0.0, 1.0)


def _hypsometric(elev, lo=-8000, hi=4000):
    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    # ocean depth tint (lighter than near-black so trenches are visible but
    # not oppressive)
    sea = elev < 0
    t = np.clip(-elev[sea] / 6000.0, 0, 1)
    rgb[sea, 0] = (20 + t * (60 - 20)) / 255.0
    rgb[sea, 1] = (50 + t * (130 - 50)) / 255.0
    rgb[sea, 2] = (90 + t * (190 - 90)) / 255.0
    # land ramp: light-green lowland -> yellow-green -> brown -> grey -> white
    land = ~sea
    t = np.clip(elev[land] / 3000.0, 0, 1)
    rgb[land, 0] = np.interp(t, [0, .3, .6, .85, 1], [110, 220, 190, 160, 245]) / 255.0
    rgb[land, 1] = np.interp(t, [0, .3, .6, .85, 1], [200, 225, 170, 145, 245]) / 255.0
    rgb[land, 2] = np.interp(t, [0, .3, .6, .85, 1], [80, 100, 70, 140, 245]) / 255.0
    return rgb


def render_terrain(elev, ice, out_path, vert_exag=2.5):
    base = _hypsometric(elev)
    shade = _hillshade(elev, vert_exag=vert_exag)
    comp = np.clip(base * (0.4 + 0.6 * shade[..., None]), 0, 1)
    comp[ice] = np.array([0.92, 0.95, 0.98])
    from PIL import Image
    Image.fromarray((comp * 255).astype(np.uint8)).save(out_path)
    return out_path


def _biome_palette(biome_name):
    """(h, w, 3) float32 biome colours via a LUT over unique biome names."""
    uniq, inverse = np.unique(biome_name, return_inverse=True)
    lut = np.array([BIOME_RGB.get(b, (120, 160, 100)) for b in uniq], dtype=np.float32)
    return lut[inverse.reshape(biome_name.shape)]


def render_satellite(elev, biome_name, ice, out_path):
    """Step 21: natural-ish colour composite."""
    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    land = (elev >= 0)
    # ocean: depth tint
    sea = ~land
    d = np.clip(-elev[sea] / 6000.0, 0, 1)
    rgb[sea, 0] = 10 + (1 - d) * 20
    rgb[sea, 1] = 40 + (1 - d) * 60
    rgb[sea, 2] = 90 + (1 - d) * 90
    # land: biome colour brightened by hillshade
    pal = _biome_palette(biome_name)
    rgb[land] = pal[land] / 255.0
    shade = _hillshade(elev, vert_exag=2.0)
    rgb[land] = np.clip(rgb[land] * (0.55 + 0.45 * shade[land][..., None]), 0, 1)
    rgb[ice] = np.array([0.95, 0.97, 1.0])
    from PIL import Image
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return out_path


def render_planet(elev, biome_name, ice, koppen_code, out_path):
    """Step 22: stylised planet map (biome-dominant with terrain shading)."""
    rgb = _biome_palette(biome_name).astype(np.float32) / 255.0
    land = (elev >= 0)
    shade = _hillshade(elev, vert_exag=3.0)
    rgb[land] = np.clip(rgb[land] * (0.6 + 0.4 * shade[land][..., None]), 0, 1)
    rgb[ice] = np.array([0.96, 0.98, 1.0])
    from PIL import Image
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out_path)
    return out_path


# ===========================================================================
#  orchestrator
# ===========================================================================
def generate_planet(seed=1234567, w=1024, h=512, out="planet_out",
                    device="auto", snr0=0.5, save_npy=True):
    """Run the full pipeline and emit every stage as PNG (+ npy)."""
    os.makedirs(out, exist_ok=True)
    t_total = time.time()

    # ---- Phase 1 ----
    print("[P1-1] tectonics ...", flush=True)
    tect = tectonic(seed, w, h)
    raw, merged, land_mask, boundaries = (tect["raw"], tect["merged"],
                                           tect["land_mask"], tect["boundaries"])
    coarse_elev = tectonic_coarse_elevation(land_mask, boundaries)      # steps 4 & 8
    grid = coarse_conditioning_grid(coarse_elev, land_mask)
    coloring = initial_coloring(coarse_elev)                           # step 7

    print("[P1-2] diffusion refinement (step 9) ...", flush=True)
    elev, climate_raw = diffusion_refine(grid, seed, w, h, device=device, snr0=snr0)

    print("[P1-3] coastline + islands (steps 5,6) ...", flush=True)
    final_land, elev = refine_coastline_and_islands(elev, land_mask, boundaries, seed)
    # snap sign to refined land mask (preserve internal relief)
    elev[final_land] = np.maximum(elev[final_land], 1.0)
    elev[~final_land] = np.minimum(elev[~final_land], -1.0)

    # ---- Phase 2 ----
    print("[P2] climate & rendering ...", flush=True)
    temp = solar_temperature(elev, h)                                  # 13
    pressure = continental_pressure(temp, elev, h)                     # 14
    u, v = wind_field(elev, temp, boundaries, h, w)                    # 15
    precip = precipitation(u, v, elev, h)                              # 16
    koppen_code = koppen(temp, precip, elev, h)                        # 17
    # ice + currents coupled (step 19 <-> 18): ocean cells use SST; currents
    # warm polar coasts -> less sea ice.  (The standalone ice_layer() result is
    # intentionally not used - this SST-aware recomputation supersedes it.)
    lat = _lat(h)
    cu, cv, sst = ocean_currents(u, v, elev, h, w, iterations=2)        # 19
    eff_temp = np.where((elev >= 0), temp, sst)                        # ocean uses SST
    ice = ((eff_temp < 0) & (np.abs(lat) > 55)) | ((elev > 2500) & (temp < 0))

    # ---- Phase 3 ----
    print("[P3] ecology & civilization ...", flush=True)
    biome_name = biome(koppen_code, ice, h)                            # 23
    civ = civilization_index(temp, precip, elev, koppen_code, biome_name)  # 24
    civ_pts = civilization_points(civ)                                 # 25

    # ---- rendering ----
    print("[R] rendering stages ...", flush=True)
    _discrete_png(raw, {i: _plate_color(i) for i in range(int(raw.max()) + 1)},
                  os.path.join(out, "01_plates_raw.png"))
    _render_merged(merged, os.path.join(out, "02_plates_merged.png"))
    _discrete_png(tect["continent_mask"].astype(int),
                  {i: _cont_color(i) for i in range(int(tect["continent_mask"].max()) + 2)},
                  os.path.join(out, "03_continents.png"))
    _render_boundaries(boundaries["boundary_type"],
                       os.path.join(out, "04_boundary_types.png"))
    _grayscale_png(coarse_elev, os.path.join(out, "05_coarse_elevation.png"))   # 8
    _discrete_png(coloring, {0: (30, 80, 150), 1: (150, 190, 110),
                             2: (120, 150, 80), 3: (130, 100, 70)},
                  os.path.join(out, "06_initial_coloring.png"))               # 7
    _discrete_png(final_land.astype(int),
                  {0: (40, 90, 160), 1: (150, 190, 110)},
                  os.path.join(out, "07_land_mask_refined.png"))              # 5/6
    render_terrain(elev, ice, os.path.join(out, "08_elevation_relief.png"))   # 9
    _colormap(temp, os.path.join(out, "09_temperature.png"), "turbo")
    _colormap(pressure, os.path.join(out, "10_pressure.png"), "coolwarm")
    _quiver_png(u, v, os.path.join(out, "11_wind.png"),
                bg=_hypsometric(elev), step=28)
    _colormap(precip, os.path.join(out, "12_precipitation.png"), "viridis")
    _label_png(koppen_code, KOPPEN_RGB, os.path.join(out, "13_koppen.png"))   # 17
    _discrete_png(ice.astype(int), {0: (20, 40, 80), 1: (235, 240, 245)},
                  os.path.join(out, "14_ice.png"))                            # 18
    _quiver_png(cu, cv, os.path.join(out, "15_ocean_currents.png"),
                bg=None, step=24)
    _colormap(sst, os.path.join(out, "15b_sst.png"), "coolwarm")
    _label_png(biome_name, BIOME_RGB, os.path.join(out, "16_biome.png"))      # 23
    _colormap(civ, os.path.join(out, "17_civilization_index.png"), "magma")
    _render_civ_points(elev, ice, civ_pts, os.path.join(out, "18_civilization_points.png"))
    render_satellite(elev, biome_name, ice, os.path.join(out, "21_satellite.png"))     # 21
    render_planet(elev, biome_name, ice, koppen_code,
                  os.path.join(out, "22_planet.png"))                          # 22
    # composite (20)
    render_terrain(elev, ice, os.path.join(out, "20_composite.png"))

    if save_npy:
        np.save(os.path.join(out, "elevation.npy"), elev)
        np.save(os.path.join(out, "temperature.npy"), temp)
        np.save(os.path.join(out, "precipitation.npy"), precip)
        np.save(os.path.join(out, "pressure.npy"), pressure)
        np.save(os.path.join(out, "sst.npy"), sst)
        np.save(os.path.join(out, "ice.npy"), ice)
        np.save(os.path.join(out, "koppen.npy"), koppen_code)
        np.save(os.path.join(out, "biome.npy"), biome_name)
        np.save(os.path.join(out, "civilization_index.npy"), civ)
        with open(os.path.join(out, "civilization_points.json"), "w") as f:
            json.dump(civ_pts, f)

    land_pct = float((elev >= 0).mean() * 100)
    print("DONE in %.1fs  land%%=%.1f  elev[%.0f,%.0f]  temp[%.1f,%.1f]  "
          "precip[%.0f,%.0f]  biomes=%d  civ_pts=%d" % (
              time.time() - t_total, land_pct, elev.min(), elev.max(),
              temp.min(), temp.max(), precip.min(), precip.max(),
              len(set(biome_name.flatten().tolist())), len(civ_pts)), flush=True)
    return {
        "elev": elev, "temp": temp, "precip": precip, "pressure": pressure,
        "koppen": koppen_code, "biome": biome_name, "ice": ice,
        "civ": civ, "civ_pts": civ_pts, "land_pct": land_pct,
    }


# ---- extra discrete renderers ----
def _plate_color(i):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.137) % 1.0, 0.6, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


def _cont_color(i):
    import colorsys
    if i == 0:
        return (20, 40, 90)
    r, g, b = colorsys.hsv_to_rgb((i * 0.19) % 1.0, 0.55, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def _render_merged(merged, out_path):
    from PIL import Image
    import colorsys
    n = int(merged.max()) + 1
    pal = {g: _plate_color(g) for g in range(n)}
    rgb = np.zeros((*merged.shape, 3), dtype=np.uint8)
    for g in range(n):
        rgb[merged == g] = pal[g]
    # black boundaries
    bnd = (merged != np.roll(merged, 1, 1)) | (merged != np.roll(merged, -1, 1))
    if merged.shape[0] > 1:
        bnd |= (merged != np.roll(merged, 1, 0)) | (merged != np.roll(merged, -1, 0))
    rgb[bnd] = (0, 0, 0)
    Image.fromarray(rgb).save(out_path)


def _render_boundaries(bt, out_path):
    from PIL import Image
    pal = {0: (60, 60, 70), CONVERGENT: (150, 60, 40),
           DIVERGENT: (60, 170, 80), TRANSFORM: (70, 120, 200)}
    rgb = np.zeros((*bt.shape, 3), dtype=np.uint8)
    for k, c in pal.items():
        rgb[bt == k] = c
    Image.fromarray(rgb).save(out_path)


def _colormap(arr, out_path, cmap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(arr.shape[1] / 100.0, arr.shape[0] / 100.0), dpi=100)
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _render_civ_points(elev, ice, pts, out_path):
    from PIL import Image
    rgb = np.zeros((*elev.shape, 3), dtype=np.float32)
    sea = elev < 0
    d = np.clip(-elev[sea] / 6000.0, 0, 1)
    rgb[sea, 0] = (10 + (1 - d) * 20) / 255.0
    rgb[sea, 1] = (40 + (1 - d) * 60) / 255.0
    rgb[sea, 2] = (90 + (1 - d) * 90) / 255.0
    land = ~sea
    t = np.clip(elev[land] / 3000.0, 0, 1)
    rgb[land, 0] = np.interp(t, [0, .5, 1], [80, 150, 235]) / 255.0
    rgb[land, 1] = np.interp(t, [0, .5, 1], [150, 180, 235]) / 255.0
    rgb[land, 2] = np.interp(t, [0, .5, 1], [80, 130, 240]) / 255.0
    rgb[ice] = (0.95, 0.97, 1.0)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(elev.shape[1] / 100.0, elev.shape[0] / 100.0), dpi=100)
    ax.imshow(np.asarray(img))
    if pts:
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        ax.scatter(xs, ys, s=14, c="red", edgecolors="yellow", linewidths=0.6, zorder=5)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234567)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", type=str, default="planet_out")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--snr0", type=float, default=0.5)
    args = ap.parse_args()
    generate_planet(seed=args.seed, w=args.width, h=args.height, out=args.out,
                    device=args.device, snr0=args.snr0)


if __name__ == "__main__":
    main()
