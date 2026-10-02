#!/usr/bin/env python3
"""Spherical Voronoi plate generation with domain warping.

This module replaces the former ``platec`` C-extension tectonic simulation.
It generates ``n_raw`` micro-plates on a sphere via domain-warped Voronoi,
selects continental nuclei and grows them into ``n_big`` major plates, then
synthesises an elevation field from the resulting plate layout.

When ``n_big == len(PLATE_AREAS)`` the generator follows the reference
weights from ``world_gen.py``:

* Six major plates with target areas ``PLATE_AREAS`` (percent of sphere).
* The two plates closest to the oceanic anti-center become oceanic; all
  other plates are continental.
* Continents are simply the merged continental plates (no separate
  continent-core placement step).

Design notes
------------
* Only DOMAIN_WARP is applied (smooth curved boundaries) - no per-pixel
  detail noise, otherwise plate borders become fuzzy.
* Continental nuclei are the plates with the greatest graph distance from
  the oceanic anti-center.
* Everything is pure Python / NumPy / SciPy, so there is no C extension to
  build and results are reproducible on every platform.
"""

import math
from collections import deque

import numpy
from PIL import Image

from worldengine.continents import build_continents

TAU = 2.0 * math.pi

# ---------------- Reference weights (from world_gen.py) ----------------
PLATE_AREAS = [23.33, 17.78, 17.78, 15.56, 14.44, 11.11]
PLATE_OCEAN = [0, 0, 0, 0, 1, 1]   # indices 4/5 are oceanic
DEFAULT_N_OCEAN = 2

DOMAIN_AMP = 0.22
DOMAIN_FREQ = 2.5
PARTITION_ITERS = 12  # pressure-relaxation iterations for area balancing


# Equirectangular (cylindrical / plate-carree) projection.  The map's vertical
# extent covers latitudes [-LAT_LIMIT_DEG, +LAT_LIMIT_DEG] only; everything
# poleward of +/-85 deg is *ignored*.  In an equirectangular projection the
# poles are singular (cos(lat) -> 0 collapses every longitude onto one point),
# which smears plates/elevation into a garbled band along the top & bottom
# rows.  Clipping to 85 deg (the common Web-Mercator cap) removes that smear
# and keeps plate boundaries clean.
LAT_LIMIT_DEG = 85.0


# --------------- Noise field (low-res FBM -> PIL upscale) ---------------
def _noise_field(shape, seed, octaves=4, res=64):
    """Low-resolution FBM field upscaled to ``shape`` via PIL BILINEAR.

    ``shape`` is ``(width, height)`` because it is handed straight to PIL.
    Returns values in ``(-1, +1)``.

    Octaves run coarse -> fine: octave ``o`` is sampled on a
    ``res >> (octaves - 1 - o)`` grid with amplitude ``0.5 ** o``, so low
    frequencies dominate (textbook FBM).  With ``octaves=1`` this collapses
    to a single full-resolution hash grid.
    """

    def _hash2(ix, iy, s):
        h = s & 0x7FFFFFFF
        h = (h ^ (ix * 374761393)) & 0x7FFFFFFF
        h = (h ^ (iy * 668265263)) & 0x7FFFFFFF
        h = (h * 1274126177) & 0x7FFFFFFF
        h = (h ^ (h >> 13)) & 0x7FFFFFFF
        return numpy.float32(h) / 2147483647.0

    _hash2v = numpy.vectorize(_hash2, otypes=[numpy.float32])

    fbm = numpy.zeros((res, res), dtype=numpy.float32)
    norm = numpy.float32(0.0)
    for o in range(octaves):
        # coarse -> fine; the finest octave is the full `res` grid
        layer_res = max(2, res >> (octaves - 1 - o))
        amp = numpy.float32(0.5**o)

        y_idx, x_idx = numpy.mgrid[0:layer_res, 0:layer_res].astype(numpy.float32)
        y_idx *= numpy.float32(res)
        x_idx *= numpy.float32(res)
        layer = _hash2v(x_idx.astype(int), y_idx.astype(int), seed + o * 1013) * 2.0 - 1.0

        if layer_res != res:
            img = Image.fromarray(((layer + 1.0) * 127.5).astype(numpy.uint8))
            layer = numpy.asarray(img.resize((res, res), Image.BILINEAR), dtype=numpy.float32) / 127.5 - 1.0

        fbm += amp * layer
        norm += amp
    fbm /= numpy.maximum(norm, 1e-6)

    img = Image.fromarray(((fbm + 1.0) * 127.5).astype(numpy.uint8))
    img = img.resize(shape, Image.BILINEAR)
    return numpy.asarray(img, dtype=numpy.float32) / 127.5 - 1.0


def fractal_noise_field(shape, seed, octaves=4, base_res=64):
    """Random-valued FBM upscaled to ``shape`` (``(height, width)``).

    Unlike :func:`_noise_field` this uses a seeded ``RandomState`` instead
    of an integer hash, which is cheaper for the large octave counts used
    by the elevation synthesiser.  Octaves run coarse -> fine with halving
    amplitude.
    """
    fbm = numpy.zeros((base_res, base_res), dtype=numpy.float32)
    rng = numpy.random.RandomState(seed & 0x7FFFFFFF)
    norm = 0.0
    for o in range(octaves):
        res = max(2, base_res >> (octaves - 1 - o))
        amp = 0.5**o
        small = rng.rand(res, res).astype(numpy.float32) * 2.0 - 1.0
        if res != base_res:
            img = Image.fromarray(((small + 1.0) * 127.5).astype(numpy.uint8))
            small = numpy.asarray(img.resize((base_res, base_res), Image.BILINEAR), dtype=numpy.float32) / 127.5 - 1.0
        fbm += amp * small
        norm += amp
    fbm /= max(norm, 1e-6)

    img = Image.fromarray(((fbm + 1.0) * 127.5).astype(numpy.uint8))
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return numpy.asarray(img, dtype=numpy.float32) / 127.5 - 1.0


# --------------- Spherical geometry ---------------
def _coords_to_vecs(h, w):
    # Cylindrical projection clipped to +/-LAT_LIMIT_DEG.  Pixel *centres* are
    # sampled ((i + 0.5) / h) so the band is symmetric about the equator and no
    # row sits exactly on a pole.  Rows now span [-85 deg, +85 deg] instead of
    # [-90 deg, +90 deg]: cos(85 deg) = 0.087 (not 0), so the polar smear is
    # gone.
    lat_limit = math.radians(LAT_LIMIT_DEG)
    lat = ((numpy.arange(h, dtype=numpy.float32) + 0.5) / h - 0.5) * 2.0 * lat_limit
    lon = ((numpy.arange(w, dtype=numpy.float32) + 0.5) / w) * TAU
    cl = numpy.cos(lat)[:, None]
    sl = numpy.sin(lat)[:, None]
    clon = numpy.cos(lon)[None, :]
    slon = numpy.sin(lon)[None, :]
    P = numpy.zeros((h, w, 3), dtype=numpy.float32)
    P[:, :, 0] = cl * clon
    P[:, :, 1] = numpy.broadcast_to(sl, (h, w))
    P[:, :, 2] = cl * slon
    return P


def _angular_distance(P, seeds):
    dot = numpy.clip(numpy.tensordot(P, seeds, axes=([2], [1])), -1.0, 1.0)
    return numpy.arccos(dot)


def _warp_dirs(P, seed, h, w):
    """Domain-warp pixel directions with 1 smooth FBM component (minimal warping)."""
    n0 = _noise_field((w, h), seed, octaves=1, res=48)
    n1 = _noise_field((w, h), seed + 911, octaves=1, res=48)
    n2 = _noise_field((w, h), seed + 211, octaves=1, res=48)
    W = P + DOMAIN_AMP * numpy.stack([n0, n1, n2], axis=-1)
    norm = numpy.sqrt((W * W).sum(-1))
    norm = numpy.maximum(norm, 1e-12)
    return W / norm[..., None]


def _slerp(a, b, t):
    """Spherical linear interpolation between two unit 3-vectors."""
    d = numpy.clip(numpy.dot(a, b), -1.0, 1.0)
    om = math.acos(d)
    if om < 1e-5:
        return a
    s = math.sin(om)
    return (a * math.sin((1.0 - t) * om) + b * math.sin(t * om)) / s


# --------------- Seed placement on sphere ---------------
def _place_seeds(seed, n_plates):
    """Farthest-point sampling on sphere.

    Two equatorial oceanic seeds are placed first, then the remaining seeds
    are placed at maximum angular distance from every seed already placed.
    """
    if n_plates < 6:
        raise ValueError("n_plates must be >= 6 (two oceanic seeds are reserved)")

    rng = numpy.random.RandomState((seed * 2654435761) & 0xFFFFFFFF)
    seeds = [None] * n_plates

    # Ocean center: random point near equator
    clat_c = rng.rand() * math.radians(20.0)
    lon_c = (rng.rand() * 2.0 - 1.0) * math.pi
    cl, sl = math.cos(clat_c), math.sin(clat_c)
    C = numpy.array([cl * math.cos(lon_c), sl, cl * math.sin(lon_c)])

    # Two ocean seeds at +/- offset from center
    off = math.radians(26.0)
    cos_off, sin_off = math.cos(off), math.sin(off)
    east = numpy.cross(C, numpy.array([0.0, 1.0, 0.0]))
    en = math.sqrt(east.dot(east))
    east = east / en if en > 1e-9 else numpy.array([1.0, 0.0, 0.0])

    def _norm(v):
        n = math.sqrt(v.dot(v))
        return v / n if n > 1e-9 else v

    seeds[4] = _norm(cos_off * C + sin_off * east)
    seeds[5] = _norm(cos_off * C - sin_off * east)

    # Farthest-point sampling for remaining seeds
    placed = [seeds[4], seeds[5]]
    for k in range(n_plates):
        if seeds[k] is not None:
            continue
        best_dir = numpy.array([0.0, 1.0, 0.0])
        best_score = -1.0
        lat_limit = math.radians(LAT_LIMIT_DEG)
        for _ in range(400):
            la = (rng.rand() * 2.0 - 1.0) * lat_limit
            lo = rng.rand() * TAU
            cla = math.cos(la)
            D = numpy.array([cla * math.cos(lo), math.sin(la), cla * math.sin(lo)])
            dC = math.acos(max(-1.0, min(1.0, float(C.dot(D)))))
            if dC < math.radians(77.0):
                continue
            mind = min(math.acos(max(-1.0, min(1.0, float(D.dot(p))))) for p in placed)
            if mind > best_score:
                best_score = mind
                best_dir = D.copy()
        seeds[k] = _norm(best_dir)
        placed.append(seeds[k])
    return seeds, C


# --------------- Raw plate partition (domain-warped Voronoi, NO detail noise) ---------------
# Distance-cache budget: the angular-distance field is invariant across
# pressure-relaxation iterations, so compute it once when it fits in RAM
# (~256 MB cap) instead of 12x.  Above the cap, fall back to per-iteration
# chunked recomputation (constant memory, as before).
_DIST_CACHE_MAX_ELEMS = 64_000_000


def _partition_sphere(w, h, seeds, seed, n_plates, target_weights=None):
    """Numpy-batched spherical Voronoi with domain warping and pressure relaxation.

    Seed distances never change during relaxation, so they are computed once
    and cached when the map fits the distance-cache budget; above the budget
    they are recomputed per-iteration in row chunks so memory stays bounded
    even at very high resolutions (4096x2048+).  No sub-centers - with 30+
    plates, boundaries are already fine-grained.
    """
    s2 = (seed * 2654435761) & 0x7FFFFFFF
    # Local RNG: the empty-plate donation below must not depend on (or perturb)
    # the process-global NumPy RNG, otherwise generation is not reproducible.
    rng = numpy.random.RandomState(s2)

    P = _coords_to_vecs(h, w)
    W = _warp_dirs(P, s2, h, w)
    del P

    # Seeds as flat array (n_plates, 3)
    seeds_arr = numpy.array(seeds, dtype=numpy.float32)

    chunk_rows = 64
    pressure = numpy.zeros(n_plates, dtype=numpy.float32)
    if target_weights is None:
        target = numpy.full(n_plates, 1.0 / n_plates, dtype=numpy.float32)
    else:
        warr = numpy.asarray(target_weights, dtype=numpy.float32)
        target = warr / warr.sum()

    pid = numpy.zeros((h, w), dtype=numpy.int32)

    # The angular distance to every seed never changes during relaxation - only
    # the per-plate pressure bias does.  Cache the (h, w, n_plates) field once
    # when it fits in the budget; each iteration then reduces to a cheap
    # argmin over (distance + pressure).  Bit-identical to recomputing.
    dist_all = None
    if h * w * n_plates <= _DIST_CACHE_MAX_ELEMS:
        dist_all = numpy.empty((h, w, n_plates), dtype=numpy.float32)
        for y0 in range(0, h, chunk_rows):
            y1 = min(y0 + chunk_rows, h)
            dot = numpy.clip(numpy.tensordot(W[y0:y1], seeds_arr, axes=([2], [1])), -1.0, 1.0)
            dist_all[y0:y1] = numpy.arccos(dot)

    for _it in range(PARTITION_ITERS):
        if dist_all is not None:
            # argmax(-dist - pressure) == argmin(dist + pressure); chunked to
            # avoid a full (h, w, n_plates) temporary.
            for y0 in range(0, h, chunk_rows):
                y1 = min(y0 + chunk_rows, h)
                pid[y0:y1] = (dist_all[y0:y1] + pressure).argmin(axis=2).astype(numpy.int32)
        else:
            for y0 in range(0, h, chunk_rows):
                y1 = min(y0 + chunk_rows, h)
                W_chunk = W[y0:y1]  # (chunk, w, 3)
                dot = numpy.clip(numpy.tensordot(W_chunk, seeds_arr, axes=([2], [1])), -1.0, 1.0)
                dist = numpy.arccos(dot)
                score = -dist - pressure[None, None, :]
                pid[y0:y1] = score.argmax(axis=2).astype(numpy.int32)

        counts = numpy.bincount(pid.ravel(), minlength=n_plates).astype(numpy.float32)

        # Fill empty plates with a few pixels from the most over-represented one
        for k in range(n_plates):
            if counts[k] == 0:
                excess = counts - (w * h) / n_plates
                donor = excess.argmax()
                if counts[donor] > 2:
                    donor_mask = pid == donor
                    ys, xs = numpy.where(donor_mask)
                    if len(ys) > 0:
                        n_reassign = max(1, len(ys) // 100)
                        idx = rng.choice(len(ys), n_reassign, replace=False)
                        pid[ys[idx], xs[idx]] = k
                        counts = numpy.bincount(pid.ravel(), minlength=n_plates).astype(numpy.float32)

        frac = counts / (w * h)
        pressure += (frac - target) * 4.0

    return pid


# --------------- Compute raw plate adjacency ---------------
def _build_adjacency(pid, n_raw):
    """Vectorized raw plate adjacency graph from neighbor differences."""
    pairs = []

    hr = pid[:, :-1]
    hr_next = pid[:, 1:]
    mask_h = hr != hr_next
    if mask_h.any():
        pairs.append(numpy.stack([hr[mask_h], hr_next[mask_h]], axis=1))

    vr = pid[:-1, :]
    vr_next = pid[1:, :]
    mask_v = vr != vr_next
    if mask_v.any():
        pairs.append(numpy.stack([vr[mask_v], vr_next[mask_v]], axis=1))

    if not pairs:
        return {p: set() for p in range(n_raw)}

    edge = numpy.concatenate(pairs, axis=0).astype(numpy.int64)
    edge = numpy.sort(edge, axis=1)
    edge = numpy.unique(edge, axis=0)

    adj = {p: set() for p in range(n_raw)}
    for pi, qi in edge:
        adj[int(pi)].add(int(qi))
        adj[int(qi)].add(int(pi))
    return adj


# --------------- Continental nucleus selection + growth ---------------
def _grow_continents(pid, ocean_center, n_ocean=2, n_continent=4, target_areas=None):
    """Select continental nuclei and grow into large regions.

    Ocean seeds: plates whose 3D centroid is closest to the ocean center.
    Continental nuclei: plates farthest from ocean seeds via graph distance.
    Growth: multi-source area-balanced BFS along adjacency graph.  When
    ``target_areas`` is supplied, growth tries to match those relative areas
    instead of equalising them.
    """
    h, w = pid.shape
    n_raw = int(pid.max()) + 1
    adj = _build_adjacency(pid, n_raw)

    # Compute per-plate area and 3D centroid in one pass via bincount
    pid_flat = pid.ravel().astype(numpy.int64)
    plate_area = numpy.bincount(pid_flat, minlength=n_raw).astype(numpy.float64)
    P_all = _coords_to_vecs(h, w)
    P_flat = P_all.reshape(-1, 3)
    del P_all
    centroids_3d = numpy.zeros((n_raw, 3), dtype=numpy.float64)
    for c in range(3):
        centroids_3d[:, c] = numpy.bincount(pid_flat, weights=P_flat[:, c], minlength=n_raw)
    plate_present = plate_area > 0
    with numpy.errstate(invalid="ignore", divide="ignore"):
        centroids_3d[plate_present] /= plate_area[plate_present, None]
    norms = numpy.linalg.norm(centroids_3d, axis=1)
    norms = numpy.maximum(norms, 1e-9)
    centroids_3d = centroids_3d / norms[:, None]

    n_present = int(plate_present.sum())
    n_ocean = max(1, min(n_ocean, n_present - 1))
    n_continent = max(1, min(n_continent, n_present - n_ocean))

    # Ocean seeds: n_ocean plates with pixels closest to ocean_center
    dots = numpy.clip(centroids_3d @ ocean_center, -1.0, 1.0)
    ang = numpy.arccos(dots)
    ang[~plate_present] = numpy.inf
    ocean_seeds = [int(p) for p in numpy.argsort(ang)[:n_ocean]]

    # Continental nuclei: plates with pixels, farthest graph distance from ocean seeds
    remaining = [p for p in range(n_raw) if plate_present[p] and p not in ocean_seeds]
    dist_from_ocean = {}
    for p in remaining:
        visited = {p}
        q = deque([(p, 0)])
        md = n_raw
        while q:
            cur, d = q.popleft()
            if cur in ocean_seeds:
                md = d
                break
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, d + 1))
        dist_from_ocean[p] = md
    reachable = [p for p in remaining if dist_from_ocean.get(p, n_raw) < n_raw]
    land_nuclei = sorted(reachable, key=lambda p: dist_from_ocean[p], reverse=True)[:n_continent]
    if len(land_nuclei) < n_continent:
        land_nuclei.extend([p for p in remaining if p not in land_nuclei][: n_continent - len(land_nuclei)])

    seeds_all = ocean_seeds + land_nuclei
    assigned = {s: i for i, s in enumerate(seeds_all)}
    areas = [int(plate_area[s]) for s in seeds_all]

    # Normalised area targets for the merged groups
    if target_areas is not None and len(target_areas) >= len(seeds_all):
        tarr = numpy.asarray(target_areas, dtype=numpy.float64)[: len(seeds_all)]
        tarr = tarr / tarr.sum()
    else:
        tarr = numpy.full(len(seeds_all), 1.0 / len(seeds_all), dtype=numpy.float64)

    border = set()
    for s in seeds_all:
        for nxt in adj[s]:
            if nxt not in assigned:
                border.add(nxt)

    while border:
        p = border.pop()
        neighbor_areas = {}
        for nxt in adj[p]:
            if nxt in assigned:
                g = assigned[nxt]
                neighbor_areas[g] = areas[g]
        if neighbor_areas:
            # Pick the neighbour whose current area / target is smallest
            best_g = min(neighbor_areas, key=lambda g: areas[g] / max(tarr[g], 1e-9))
            assigned[p] = best_g
            areas[best_g] += int(plate_area[p])
            for nxt in adj[p]:
                if nxt not in assigned and nxt not in border:
                    border.add(nxt)

    for p in range(n_raw):
        if p not in assigned:
            assigned[p] = 0

    # Scatter via lookup table: merged[y, x] = assigned[pid[y, x]]
    lut = numpy.zeros(n_raw, dtype=pid.dtype)
    for p, g in assigned.items():
        lut[p] = g
    return lut[pid]


# --------------- Connected components ---------------
def _label_components(mask):
    """8-connected component labelling (vectorized with scipy)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=numpy.ones((3, 3), dtype=int))
    return lab.astype(numpy.int32), int(n)


# --------------- Raw plate cleanup ---------------
def _cleanup_raw_plates(pid, n_raw):
    """Keep only the largest 8-connected component of each raw plate.

    Small components are reassigned to their most common neighbouring plate.
    """
    h, w = pid.shape
    min_frac = 0.001
    out = pid.copy()

    for p in range(n_raw):
        mask = out == p
        total = mask.sum()
        if total < 10:
            continue
        lab, n_comp = _label_components(mask)
        if n_comp <= 1:
            continue
        comp_areas = numpy.bincount(lab.ravel(), minlength=n_comp + 1)
        largest = 1 + int(comp_areas[1:].argmax())
        min_area = max(int(min_frac * h * w), 8)
        for c in range(1, n_comp + 1):
            if c == largest:
                continue
            if int(comp_areas[c]) < min_area:
                comp = lab == c
                ys, xs = numpy.where(comp)
                neigh = numpy.concatenate(
                    [
                        out[numpy.clip(ys + dy, 0, h - 1), numpy.clip(xs + dx, 0, w - 1)]
                        for dy in (-1, 0, 1)
                        for dx in (-1, 0, 1)
                        if not (dy == 0 and dx == 0)
                    ]
                )
                neigh = neigh[neigh != p]
                if len(neigh):
                    vals, counts = numpy.unique(neigh, return_counts=True)
                    out[comp] = int(vals[counts.argmax()])
    return out


def remove_enclaves(plates, min_frac=0.0005):
    """Absorb tiny disconnected islands of a merged-plate map into their neighbours."""
    h, w = plates.shape
    min_area = int(min_frac * h * w)
    out = plates.astype(int).copy()
    for g in numpy.unique(plates):
        mask = out == g
        lab, n = _label_components(mask)
        areas = numpy.bincount(lab.ravel(), minlength=n + 1)
        for c in range(1, n + 1):
            if int(areas[c]) < min_area:
                comp = lab == c
                yy, xx = numpy.nonzero(comp)
                neigh = numpy.concatenate(
                    [
                        out[numpy.clip(yy + dy, 0, h - 1), numpy.clip(xx + dx, 0, w - 1)]
                        for dy in (-1, 0, 1)
                        for dx in (-1, 0, 1)
                        if not (dy == 0 and dx == 0)
                    ]
                )
                neigh = neigh[neigh != g]
                if len(neigh):
                    vals, counts = numpy.unique(neigh, return_counts=True)
                    out[comp] = int(vals[counts.argmax()])
    return out


# --------------- Continent mask (derived from merged plates) ---------------
def _build_continents(merged, seed, h, w):
    """Derive land / continent masks directly from the merged plate layout.

    This is the previous (pre-core-placement) approach: a plate is land when
    it is a continental plate (``merged >= DEFAULT_N_OCEAN``), and each
    continental plate becomes its own continent.  There is no separate
    continent-core placement step, so a core can never drift onto a pole.
    """
    land_mask = merged >= DEFAULT_N_OCEAN
    continent_mask = numpy.where(land_mask, merged - DEFAULT_N_OCEAN, -1).astype(numpy.int32)
    return land_mask, continent_mask

# --------------- Pole / border feathering ---------------
def _pole_feather(h, w, feather_deg=5.0):
    """Return a weight mask that is 1 in the interior and fades to 0 at +/-85 deg."""
    lat_limit = math.radians(LAT_LIMIT_DEG)
    lat = ((numpy.arange(h, dtype=numpy.float32) + 0.5) / h - 0.5) * 2.0 * lat_limit
    feather = math.radians(feather_deg)
    # linear ramp over the last `feather` degrees at each pole
    t = numpy.clip((lat_limit - numpy.abs(lat)) / feather, 0.0, 1.0)
    # smoothstep
    t = t * t * (3.0 - 2.0 * t)
    return numpy.broadcast_to(t[:, None], (h, w))


# --------------- Elevation synthesis ---------------
def _distance_from_boundary(mask):
    """Normalised distance of each mask pixel to the nearest non-mask pixel."""
    from scipy import ndimage

    h, w = mask.shape
    dm = ndimage.distance_transform_edt(mask)
    dmax = dm.max()
    if dmax <= 0:
        return numpy.zeros((h, w), dtype=numpy.float32)
    return (dm / dmax).astype(numpy.float32)


def synthesize_elevation(merged, h, w, seed, n_big=6, n_ocean=2, land_mask=None,
                         continent_mask=None):
    """Build a raw elevation field (metres, roughly -5000..+5000).

    If ``land_mask`` is provided it determines land vs ocean directly
    (reference-weighted continent mode).  Otherwise the legacy rule
    ``merged >= n_ocean`` is used.
    """
    if land_mask is None:
        ocean_mask = merged < n_ocean
    else:
        ocean_mask = ~land_mask

    elevation = numpy.where(ocean_mask, -5000.0, -2000.0)

    noise_a = fractal_noise_field((h, w), seed, octaves=4, base_res=64)
    noise_b = fractal_noise_field((h, w), seed + 1, octaves=5, base_res=48)
    noise_c = fractal_noise_field((h, w), seed + 2, octaves=4, base_res=42)
    noise_d = fractal_noise_field((h, w), seed + 3, octaves=6, base_res=36)

    land_labels = continent_mask if continent_mask is not None else merged
    unique_labels = numpy.unique(land_labels[~ocean_mask]) if (~ocean_mask).any() else []
    for g in unique_labels:
        if g < 0:
            continue
        mask = land_labels == g
        if mask.sum() < 100:
            continue
        dm = _distance_from_boundary(mask)
        # float64 before the multiply to keep bit-identical rounding with the
        # former full-map expression `elevation += mask.astype(float) * dm * 5000`.
        elevation[mask] += dm[mask].astype(numpy.float64) * 5000.0

    elevation += noise_a * 1200.0
    elevation += noise_b * 600.0
    elevation += noise_c * 900.0
    elevation += noise_d * 400.0

    # Note: we intentionally do NOT pole-feather the elevation.  The row-to-pole
    # mapping already clips to +/- LAT_LIMIT_DEG (85 deg), which removes the
    # polar point smear.  A previous pole-feather blend was being rolled to an
    # arbitrary latitude by center_land(), producing a visible horizontal "cut"
    # band across continents.

    # Tiny jitter to break elevation ties (prevents droplet recursion in erosion)
    rng = numpy.random.RandomState((seed + 999) & 0x7FFFFFFF)
    elevation += rng.rand(h, w).astype(numpy.float32) * 0.01

    return elevation.astype(numpy.float32)


# --------------- Main entry points ---------------
def generate_spherical_world(seed, w=512, h=512, n_raw=30, n_big=6,
                             land_fraction=None):
    """Return ``(raw_plates, merged_plates, land_mask, continent_mask)``.

    ``raw_plates`` / ``merged_plates`` come from the spherical-Voronoi plate
    simulation (used for boundaries, mountains and trenches).  ``land_mask`` /
    ``continent_mask`` come from :mod:`worldengine.continents`: a handful of
    continental cores grow into irregular geometric continents that are
    deliberately decoupled from the plate shapes, then receive fractal
    coastlines and Earth-like islands.  Land covers ~30% of the sphere
    (Earth-like) regardless of the plate layout.

    ``land_mask`` is a boolean array (True = land) and ``continent_mask``
    labels each land pixel with its continent index (-1 = ocean).
    """
    kwargs = {}
    if land_fraction is not None:
        kwargs["land_fraction"] = land_fraction

    def _plates_and_merge():
        seeds, ocean_center = _place_seeds(seed, n_raw)
        pid = _partition_sphere(w, h, seeds, seed, n_raw)
        pid = _cleanup_raw_plates(pid, n_raw)
        if n_big != len(PLATE_AREAS):
            # Legacy / non-reference mode: keep previous behaviour
            merged = _grow_continents(pid, ocean_center, n_ocean=DEFAULT_N_OCEAN,
                                      n_continent=n_big - DEFAULT_N_OCEAN)
        else:
            n_continent = n_big - DEFAULT_N_OCEAN
            # target order is ocean seeds first, then land nuclei (matches _grow_continents)
            ocean_targets = [PLATE_AREAS[i] for i in range(len(PLATE_AREAS)) if PLATE_OCEAN[i]]
            land_targets = [PLATE_AREAS[i] for i in range(len(PLATE_AREAS)) if not PLATE_OCEAN[i]]
            target_for_grow = ocean_targets + land_targets
            merged = _grow_continents(pid, ocean_center, n_ocean=DEFAULT_N_OCEAN,
                                      n_continent=n_continent,
                                      target_areas=target_for_grow)
        merged = remove_enclaves(merged, min_frac=0.0005)
        return pid, merged

    pid, merged = _plates_and_merge()
    plate_is_ocean = [1 if g < DEFAULT_N_OCEAN else 0
                      for g in range(int(merged.max()) + 1)]
    land_mask, continent_mask = build_continents(seed, w, h, plates=merged,
                                                 plate_is_ocean=plate_is_ocean,
                                                 **kwargs)
    return pid, merged, land_mask, continent_mask


def generate_spherical_plates(seed, w=512, h=512, n_raw=30, n_big=6):
    """Return ``(raw_plates, merged_plates)`` - backwards-compatible wrapper."""
    pid, merged, _land_mask, _continent_mask = generate_spherical_world(
        seed, w=w, h=h, n_raw=n_raw, n_big=n_big
    )
    return pid, merged
