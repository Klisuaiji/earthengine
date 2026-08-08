#!/usr/bin/env python
"""Integrate Terrain Diffusion into WorldEngine.

Pipeline
--------
1. Build a Voronoi continent / ocean mask with ``worldengine.spherical_voronoi``.
2. Downsample that mask to the diffusion *conditioning grid* (one cell = 256
   native pixels) and inject it as coarse elevation conditioning
   (``WorldPipeline.set_custom_conditioning_import``). Land cells -> +LAND_ELEV
   metres, ocean cells -> deep sea (-1000 m default).
3. Tile the world into 256x256 native windows and ask the diffusion model for
   real (山海起伏) elevation + climate via ``pipe.get(..., with_climate=True)``.
4. Render a hillshaded hypsometric relief map plus temperature / precipitation
   maps, and save the raw elevation + climate arrays for downstream use.

Why the custom loader?
----------------------
The stock ``from_pretrained`` path crashes in this environment (randn init /
whole-file reads blow the per-process RAM & commit ceilings, and mmap->GPU
copies poison the CUDA context). ``_load_st_chunked.load_models_chunked`` builds
each U-Net on the ``meta`` device and streams weights to the GPU in 256 KB
regular-heap chunks -- zero commit, zero mmap. We also cap ``cache_limit`` at
128 MB and run BLAS single-threaded so the decoder stays under the ~2 GB
per-process VRAM cap.

Run (from the project root, with the terrain-diffusion venv):
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python tools/diffusion_world.py --seed 1 --width 2048 --height 1024 --out out_diff
"""
import os, sys, json, argparse, time
import numpy as np
import torch

# ---- paths -------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, ".workbuddy", "参考", "terrain-diffusion-master")
for p in (TOOLS, REPO, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import _load_st_chunked as ld  # meta + chunked GPU loader

# native pixels per conditioning cell (from world_pipeline: S = 32 * scale = 256)
CELL = 256


def build_pipeline(seed=1, cache_limit=128 * 1024 * 1024, device="cuda", snr0=0.5):
    with open(os.path.join(REPO, "config.json")) as f:
        cfg = json.load(f)
    kw = dict(
        native_resolution=cfg["native_resolution"],
        latent_compression=cfg["latent_compression"],
        frequency_mult=cfg["frequency_mult"],
        drop_water_pct=cfg["drop_water_pct"],
        cond_snr=cfg["cond_snr"],
        coarse_pooling=cfg["coarse_pooling"],
        elev_coarse_pool_mode=cfg["elev_coarse_pool_mode"],
        p5_coarse_pool_mode=cfg["p5_coarse_pool_mode"],
        residual_mean=cfg["residual_mean"],
        residual_std=cfg["residual_std"],
        coarse_means=cfg["coarse_means"],
        coarse_stds=cfg["coarse_stds"],
        caching_strategy="direct",
        cache_limit=cache_limit,
        latents_batch_size=[1, 2, 4],
        log_mode="warn",
        torch_compile=False,
        dtype=None,
        decoder_tile_size=256,
        decoder_tile_stride=192,
    )
    # Defensive fallback: if numba is somehow missing from the environment,
    # register a minimal stub so `from numba import njit` keeps working
    # (njit becomes an identity decorator -> pure-Python RNG, just slower).
    try:
        import numba  # noqa: F401
    except Exception:
        import types
        _numba = types.ModuleType("numba")
        def _njit(*_args, **_kwargs):
            def _wrap(f):
                return f
            return _wrap
        _numba.njit = _njit
        import sys as _sys
        _sys.modules.setdefault("numba", _numba)
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    pipe = WorldPipeline(seed=seed, **kw)
    ld.load_models_chunked(pipe, device=device)
    pipe.to(device)
    pipe.bind()
    # cond_snr is read at coarse-stage build time; rebuilding after bind applies it.
    # Channel order: [elev, temp, temp_std, precip, precip_cv]. We trust the
    # elevation channel most (snr0); the rest keep the config default of 0.5.
    # NOTE: higher cond_snr -> MORE noise / LESS conditioning (the pipeline blends
    # conditioning with noise via cos(atan(snr))); 0.5 is the "trust conditioning"
    # setting, which is what lets our Voronoi coarse map actually steer land/sea.
    pipe.set_cond_snr([snr0, 0.5, 0.5, 0.5, 0.5])
    return pipe


def voronoi_land_mask(seed, w, h):
    """Return a boolean land mask (h, w) from the spherical Voronoi generator."""
    from worldengine.spherical_voronoi import generate_spherical_world
    raw, merged, land_mask, continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=30, n_big=6)
    return land_mask.astype(bool)


def conditioning_from_mask(land_mask, land_elev=4000.0, ocean_elev=-8000.0):
    """Downsample land_mask to the conditioning grid.

    Returns (grid_h, grid_w) float32 conditioning elevation in metres, anchored
    at cell (0,0) == native pixel (0,0).

    The pipeline applies ``sign(x)*sqrt(|x|)`` to the elevation conditioning and
    then normalises by the training mean/std (-37.7 / 39.7). A land value of
    +4000 -> +63 -> +2.3 std (clearly land); an ocean value of -8000 -> -89 ->
    -1.3 std (clearly ocean). This large contrast is what lets the coarse model
    separate continents from oceans despite its strong ocean prior.
    """
    h, w = land_mask.shape
    gh = (h + CELL - 1) // CELL
    gw = (w + CELL - 1) // CELL
    grid = np.full((gh, gw), ocean_elev, dtype=np.float32)
    for ci in range(gh):
        y0, y1 = ci * CELL, min((ci + 1) * CELL, h)
        for cj in range(gw):
            x0, x1 = cj * CELL, min((cj + 1) * CELL, w)
            block = land_mask[y0:y1, x0:x1]
            if block.size and block.mean() >= 0.5:
                grid[ci, cj] = land_elev
    return grid


def generate_world(pipe, w, h, tile=256, device="cuda"):
    """Tile the world and collect elevation + climate into full arrays."""
    elev = np.zeros((h, w), dtype=np.float32)
    climate = np.zeros((5, h, w), dtype=np.float32)
    n_rows, n_cols = (h + tile - 1) // tile, (w + tile - 1) // tile
    t0 = time.time()
    for ri in range(n_rows):
        for cj in range(n_cols):
            y0, x0 = ri * tile, cj * tile
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            reg = pipe.get(y0, x0, y1, x1, with_climate=True)
            e = reg["elev"].detach().cpu().numpy().astype(np.float32)
            c = reg["climate"].detach().cpu().numpy().astype(np.float32)
            # crop any padding the pipeline may return (should be exact)
            e = e[: y1 - y0, : x1 - x0]
            c = c[:, : y1 - y0, : x1 - x0]
            elev[y0:y1, x0:x1] = e
            climate[:, y0:y1, x0:x1] = c
            if device == "cuda":
                torch.cuda.empty_cache()
    print("  generated %dx%d in %.1fs" % (w, h, time.time() - t0), flush=True)
    return elev, climate


# ----------------------- post-process & climate ----------------------------
def clamp_land_sea(elev, land_mask, land_floor=1.0, ocean_ceiling=-1.0):
    """Force continent shape to match the Voronoi mask.

    The diffusion coarse conditioning is only 256 px/cell and interpolates
    smoothly, so a single ocean cell can bleed ~30% stray land from its
    neighbours. We snap the sign per Voronoi region: land pixels are lifted to
    at least ``land_floor`` m, ocean pixels pushed to at most ``ocean_ceiling``
    m. The internal relief (mountains / trenches) from the diffusion output is
    preserved -- only the sea-level sign at the boundary is corrected.
    """
    out = elev.copy()
    out[land_mask] = np.maximum(out[land_mask], land_floor)
    out[~land_mask] = np.minimum(out[~land_mask], ocean_ceiling)
    return out


def compute_physical_climate(elev, h, w):
    """Physically-based temperature (C) and precipitation (mm/yr) from elevation.

    Self-contained (no WorldClim / rasterio needed). Uses:
      * latitude temperature band (warm equator, cold poles)
      * elevation lapse rate 6.5 C/km
      * zonal precipitation (ITCZ maxima + mid-latitude maxima, subtropical lows)
      * orographic enhancement on slopes, slight plateau aridity at high elev
    Returns (temp, precip), each (h, w) float32.
    """
    yy = np.arange(h)[:, None].astype(np.float32)
    lat = 90.0 - 180.0 * (yy + 0.5) / h          # +90 top -> -90 bottom
    t0 = 30.0 * np.cos(np.deg2rad(lat)) - 5.0    # ~25 C equator, ~-35 C pole
    temp = t0 - 0.0065 * np.maximum(elev, 0.0)

    abslat = np.abs(lat)
    base = (2500.0 * np.exp(-((abslat / 12.0) ** 2))
            + 800.0 * np.exp(-(((abslat - 60.0) / 18.0) ** 2))
            + 300.0)
    gy, gx = np.gradient(elev.astype(np.float32))
    slope = np.sqrt(gx ** 2 + gy ** 2)
    oro = 1.0 + 1.5 * np.clip(slope / 500.0, 0.0, 1.0)
    arid = 1.0 - 0.3 * np.clip(elev / 4000.0, 0.0, 1.0)
    precip = base * oro * arid
    return temp.astype(np.float32), precip.astype(np.float32)


# ----------------------------- rendering ----------------------------------
def _hillshade(elev, az=315.0, alt=45.0, vert_exag=1.0):
    dy, dx = np.gradient(elev.astype(np.float32) * vert_exag)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(dy, -dx)
    azr, altr = np.deg2rad(az), np.deg2rad(alt)
    shade = (np.cos(altr) * np.cos(slope) +
             np.sin(altr) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(shade, 0.0, 1.0)


def _normalize(elev, land_max=2500.0, ocean_min=-4000.0):
    """Percentile-stretch raw elevation into a display-friendly range."""
    out = elev.copy()
    land = elev >= 0
    if land.any():
        hi = np.percentile(elev[land], 97)
        if hi > 0:
            out[land] = np.clip(elev[land] / hi * land_max, 0, land_max)
    sea = elev < 0
    if sea.any():
        lo = np.percentile(elev[sea], 3)
        if lo < 0:
            out[sea] = np.clip(elev[sea] / lo * ocean_min, ocean_min, 0)
    return out


def _hypsometric(norm):
    """RGB hypsometric tint for a normalized elevation array."""
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # ocean: deep (#03203a) -> shallow (#2a78b5)
    sea = norm < 0
    t = np.clip(-norm[sea] / 4000.0, 0, 1)
    rgb[sea, 0] = 3 + t * (42 - 3)
    rgb[sea, 1] = 32 + t * (120 - 32)
    rgb[sea, 2] = 58 + t * (181 - 58)
    # land ramp: green -> yellow -> brown -> grey -> white
    land = norm >= 0
    t = np.clip(norm[land] / 2500.0, 0, 1)
    # piecewise
    r = np.interp(t, [0.0, 0.3, 0.6, 0.85, 1.0], [60, 150, 150, 130, 235])
    g = np.interp(t, [0.0, 0.3, 0.6, 0.85, 1.0], [140, 180, 130, 120, 235])
    b = np.interp(t, [0.0, 0.3, 0.6, 0.85, 1.0], [60, 90, 70, 110, 240])
    rgb[land, 0] = r
    rgb[land, 1] = g
    rgb[land, 2] = b
    return rgb / 255.0


def render_relief(elev, out_path, vert_exag=2.0):
    norm = _normalize(elev)
    base = _hypsometric(norm)
    shade = _hillshade(elev, vert_exag=vert_exag)
    comp = np.clip(base * (0.4 + 0.6 * shade[..., None]), 0, 1)
    from PIL import Image
    Image.fromarray((comp * 255).astype(np.uint8)).save(out_path)
    return out_path


def _colormap_png(arr, out_path, vmin=None, vmax=None, cmap=None):
    """Render a single-channel array with a matplotlib colormap to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))
    fig, ax = plt.subplots(figsize=(arr.shape[1] / 100.0, arr.shape[0] / 100.0), dpi=100)
    ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234567)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", type=str, default="out_diff")
    ap.add_argument("--land-elev", type=float, default=4000.0)
    ap.add_argument("--ocean-elev", type=float, default=-8000.0)
    ap.add_argument("--snr0", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tile", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("[1/6] building diffusion pipeline ...", flush=True)
    pipe = build_pipeline(seed=args.seed, device=args.device, snr0=args.snr0)

    print("[2/6] generating Voronoi land mask %dx%d ..." % (args.width, args.height), flush=True)
    land_mask = voronoi_land_mask(args.seed, args.width, args.height)

    print("[3/6] building coarse conditioning + injecting ...", flush=True)
    grid = conditioning_from_mask(land_mask, land_elev=args.land_elev, ocean_elev=args.ocean_elev)
    pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=args.ocean_elev)
    print("  conditioning grid %dx%d, land cells=%d" %
          (grid.shape[1], grid.shape[0], int((grid >= 0).sum())), flush=True)

    print("[4/6] generating elevation + climate (tiled) ...", flush=True)
    elev, climate_raw = generate_world(pipe, args.width, args.height, tile=args.tile, device=args.device)
    pipe.close()

    print("[5/6] post-process: snap land/sea to Voronoi + physical climate ...", flush=True)
    elev = clamp_land_sea(elev, land_mask)
    temp, precip = compute_physical_climate(elev, args.height, args.width)
    land_pct = float((elev >= 0).mean() * 100)
    print("  land%%=%.1f  elev[min/max]=%.1f/%.1f  temp[min/max]=%.1f/%.1f  precip[min/max]=%.0f/%.0f"
          % (land_pct, elev.min(), elev.max(), temp.min(), temp.max(), precip.min(), precip.max()), flush=True)

    print("[6/6] rendering ...", flush=True)
    np.save(os.path.join(args.out, "elevation.npy"), elev)
    np.save(os.path.join(args.out, "climate_raw.npy"), climate_raw)   # raw diffusion climate (model units)
    np.save(os.path.join(args.out, "temperature.npy"), temp)
    np.save(os.path.join(args.out, "precipitation.npy"), precip)
    render_relief(elev, os.path.join(args.out, "elevation_relief.png"), vert_exag=2.5)
    _colormap_png(temp, os.path.join(args.out, "temperature.png"), cmap="turbo")
    _colormap_png(precip, os.path.join(args.out, "precipitation.png"), cmap="viridis")
    print("done ->", os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
