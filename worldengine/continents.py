"""Continental crust generation: cores -> irregular growth -> fractal coasts & islands.

Three-stage layout following the reference sketch:

  Stage 1  Tectonic plates (built in :mod:`worldengine.spherical_voronoi`;
           this module only receives the plate map for island-arc placement).
  Stage 2  A handful of continental *cores* (dots on the sketch) grow into
           irregular geometric continents by thresholding a noise-warped
           angular-distance field.  Land is decoupled from plate shapes -
           a continent may span several plates; everything not land is ocean.
  Stage 3  Fractal coastline refinement plus Earth-like islands: large shelf
           islands off coasts, volcanic arcs along ocean-continent plate
           boundaries, hotspot chains in the open ocean and polar fragments.

The global land fraction is enforced by thresholding at a quantile of the
growth score, so a seed always yields the requested land percentage
(Earth-like default: ~30% land / ~70% ocean).

Public API: ``build_continents(seed, w, h, plates=None)`` ->
``(land_mask, continent_mask)`` where ``land_mask`` is bool and
``continent_mask`` labels land pixels by continent (-1 = ocean).  Islands are
attached to their nearest continent so the label count stays ``n_continents``.
"""
from __future__ import annotations

import math

import numpy

TAU = 2.0 * math.pi

# Earth-like defaults: ~30% land in total.  Mainlands get LAND_FRACTION of the
# sphere, islands add the rest.
LAND_FRACTION = 0.275
ISLAND_FRACTION = 0.04

# Continent count histogram (weights) and share-of-land templates, modelled on
# Earth: one or two dominant continents, several medium ones, one small.
_COUNT_W = {5: 0.25, 6: 0.50, 7: 0.25}
_SHARE_TEMPLATES = {
    5: (0.30, 0.24, 0.20, 0.16, 0.10),
    6: (0.28, 0.22, 0.18, 0.14, 0.11, 0.07),
    7: (0.26, 0.20, 0.16, 0.13, 0.10, 0.09, 0.06),
}

WARP_AMP = 0.85     # stage-2: noise-warped growth -> irregular geometric shape
FINE_AMP = 0.20     # stage-3: shared fine fractal -> ragged coastline detail

_MIN_CORE_SEPARATION_DEG = 28.0
_MAX_SHARE = 0.32   # no single continent above ~1/3 of all land (Eurasia-like)


# ---------------------------------------------------------------------------
# Horizontally tileable value-noise FBM (no seam at the map's x edges).
# ---------------------------------------------------------------------------
def _fbm(h, w, seed, octaves=5, base_res=24):
    """Value-noise FBM in [-1, 1]; wraps in x, clamps in y.

    ``base_res`` is the finest grid (cells across the whole map); octaves run
    coarse -> fine so low frequencies dominate the displacement.
    """
    rng = numpy.random.RandomState((seed * 2654435761) & 0x7FFFFFFF)
    out = numpy.zeros((h, w), dtype=numpy.float32)
    norm = 0.0
    for o in range(octaves):
        res = max(2, base_res >> (octaves - 1 - o))
        grid = rng.rand(res, res).astype(numpy.float32)
        amp = 0.5**o
        ys = (numpy.arange(h, dtype=numpy.float32) + 0.5) * (res / h)
        xs = (numpy.arange(w, dtype=numpy.float32) + 0.5) * (res / w)
        y0 = numpy.clip(numpy.floor(ys).astype(numpy.int64), 0, res - 1)
        y1 = numpy.minimum(y0 + 1, res - 1)
        wy = (ys - numpy.floor(ys))[:, None]
        x0 = numpy.floor(xs).astype(numpy.int64) % res
        x1 = (x0 + 1) % res
        wx = (xs - numpy.floor(xs))[None, :]
        g00 = grid[numpy.ix_(y0, x0)]
        g01 = grid[numpy.ix_(y0, x1)]
        g10 = grid[numpy.ix_(y1, x0)]
        g11 = grid[numpy.ix_(y1, x1)]
        out += amp * ((g00 * (1 - wx) + g01 * wx) * (1 - wy) +
                      (g10 * (1 - wx) + g11 * wx) * wy)
        norm += amp
    out /= max(norm, 1e-6)
    return (out * 2.0 - 1.0).astype(numpy.float32)


# ---------------------------------------------------------------------------
# Stage 2: continental cores + noise-warped growth
# ---------------------------------------------------------------------------
def _sphere_coords(h, w):
    """Unit vectors for an equirectangular grid clipped to +/-85 deg latitude."""
    lat_limit = math.radians(85.0)
    lat = ((numpy.arange(h, dtype=numpy.float32) + 0.5) / h - 0.5) * 2.0 * lat_limit
    lon = ((numpy.arange(w, dtype=numpy.float32) + 0.5) / w) * TAU
    cl, sl = numpy.cos(lat)[:, None], numpy.sin(lat)[:, None]
    clon, slon = numpy.cos(lon)[None, :], numpy.sin(lon)[None, :]
    P = numpy.empty((h, w, 3), dtype=numpy.float32)
    P[:, :, 0] = cl * clon
    P[:, :, 1] = numpy.broadcast_to(sl, (h, w))
    P[:, :, 2] = cl * slon
    return P


def _core_radius(fraction):
    """Angular radius of a spherical cap covering ``fraction`` of the sphere."""
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * fraction)))


def _place_cores(rng, n, shares, total_land, polar_first=False):
    """Farthest-point sampling of continental cores with size-aware separation.

    Two caps may not overlap by more than ~55% of the smaller radius, so big
    continents stay distinct while small ones can sit in the gaps.  With
    ``polar_first`` the first core sits near a pole (Antarctica-like); the
    caller is expected to give it the smallest share.
    """
    radii = [_core_radius(s * total_land) for s in shares]

    def needed_sep(i, j):
        return max(math.radians(26.0), 0.5 * (radii[i] + radii[j]))

    cores = []
    if polar_first:
        lat = math.radians(rng.uniform(74.0, 80.0)) * (1 if rng.rand() < 0.5 else -1)
        lon = rng.rand() * TAU
        cl = math.cos(lat)
        cores.append(numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                                 dtype=numpy.float32))

    candidates = []
    for _ in range(400):
        lat = rng.uniform(-1.309, 1.309)  # +/-75 deg in radians
        lon = rng.rand() * TAU
        cl = math.cos(lat)
        candidates.append(numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                                      dtype=numpy.float32))

    while len(cores) < n:
        idx = len(cores)
        best_ok = None
        best_ok_score = -1.0
        best_any = None
        best_any_score = -1.0
        for cand in candidates:
            score = 1e9
            ok = True
            for ci, c in enumerate(cores):
                ang = math.acos(max(-1.0, min(1.0, float(c.dot(cand)))))
                need = needed_sep(idx, ci)
                score = min(score, ang - need)
                if ang < need:
                    ok = False
            if score > best_any_score:
                best_any_score = score
                best_any = cand
            if ok and score > best_ok_score:
                best_ok_score = score
                best_ok = cand
        cores.append(best_ok if best_ok is not None else best_any)
    return cores


def build_continents(seed, w, h, plates=None, plate_is_ocean=None,
                     land_fraction=LAND_FRACTION, island_fraction=ISLAND_FRACTION):
    """Return ``(land_mask, continent_mask)`` for an ``h x w`` equirect map."""
    rng = numpy.random.RandomState((seed * 69069) & 0x7FFFFFFF)

    # --- continent count + size shares (Earth-like templates, jittered) ---
    counts = sorted(_COUNT_W)
    probs = [_COUNT_W[c] for c in counts]
    n = counts[int(rng.choice(len(counts), p=probs))]
    shares = numpy.asarray(_SHARE_TEMPLATES[n], dtype=numpy.float64)
    shares *= rng.uniform(0.85, 1.15, size=n)
    shares = numpy.sort(shares)[::-1]          # descending
    shares = numpy.minimum(shares, _MAX_SHARE)
    shares /= shares.sum()

    total_land = land_fraction + island_fraction
    # ~half the worlds get one polar continent (Antarctica-like -> ice cap);
    # it takes the smallest share so the polar cap stays small.
    polar = n >= 5 and rng.rand() < 0.55
    if polar:
        shares = numpy.concatenate([[shares[-1]], shares[:-1]])
    cores = _place_cores(rng, n, shares, total_land, polar_first=polar)
    P = _sphere_coords(h, w)
    P_flat = P.reshape(-1, 3)

    warp_res = max(64, min(256, w // 8))
    fine_res = max(160, min(512, w // 4))
    # stage 2: each continent grows as a union of caps - a main body plus
    # 2-4 offset "lobes" that become peninsulas / protrusions (the sketch's
    # irregular geometric continents).  Fields are cached so the size
    # calibration below only rescales lobe radii.
    continent_lobes = []   # per continent: [(unit_vec, lobe_radius_rad), ...]
    continent_bays = []    # per continent: [(unit_vec, bay_radius_rad), ...]
    continent_axes = []    # per continent: (east_v, north_v, elong, axis_angle)
    warps = []
    for ci, core in enumerate(cores):
        main_frac = shares[ci] * total_land
        main_r = _core_radius(main_frac * 0.68)
        east = numpy.array([-core[2], 0.0, core[0]], dtype=numpy.float32)
        ne = float(math.sqrt(float(east.dot(east))))
        east = east / ne if ne > 1e-9 else numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        north = numpy.cross(east, core)

        def _offset(dist, bearing):
            return (core * math.cos(dist)
                    + (east * math.cos(bearing) + north * math.sin(bearing)) * math.sin(dist)
                    ).astype(numpy.float32)

        lobes = [(core, main_r)]
        for _l in range(rng.randint(2, 4)):
            lobe_frac = main_frac * rng.uniform(0.12, 0.26)
            lobe_r = _core_radius(lobe_frac)
            lobes.append((_offset(main_r * rng.uniform(0.85, 1.15), rng.rand() * TAU), lobe_r))
        # bays: caps straddling the coastline that carve open gulfs
        bays = []
        for _b in range(rng.randint(1, 3)):
            bay_r = _core_radius(main_frac * rng.uniform(0.14, 0.26))
            bays.append((_offset(main_r * rng.uniform(0.95, 1.2), rng.rand() * TAU), bay_r))
        continent_lobes.append(lobes)
        continent_bays.append(bays)
        continent_axes.append((east, north, rng.uniform(0.18, 0.32), rng.rand() * TAU))
        warp = 1.0 + WARP_AMP * _fbm(h, w, seed + 977 * ci + 1, octaves=5,
                                     base_res=warp_res).reshape(-1)
        warps.append(numpy.maximum(warp, 0.05))

    # stage 3: shared fine fractal detail on the coastline
    fine = FINE_AMP * _fbm(h, w, seed + 4242, octaves=3,
                           base_res=fine_res).reshape(-1)

    # Size calibration: measure each continent's realised share of the land
    # and rescale its lobe radii (area ~ r^2), so overlap with neighbours or a
    # strong warp cannot steal a continent's target size.
    calib = numpy.ones(n, dtype=numpy.float64)
    s_min = owner = None
    for _it in range(3):
        s_min = numpy.full(h * w, numpy.inf, dtype=numpy.float32)
        owner = numpy.zeros(h * w, dtype=numpy.int32)
        for ci in range(n):
            warp = warps[ci]
            east_v, north_v, elong, axis = continent_axes[ci]
            s_c = numpy.full(h * w, numpy.inf, dtype=numpy.float32)
            for li, (centre, lobe_r) in enumerate(continent_lobes[ci]):
                dot = numpy.clip(P_flat @ centre.astype(numpy.float32), -1.0, 1.0)
                s_lobe = numpy.arccos(dot) / warp / (lobe_r * calib[ci])
                if li == 0:
                    # anisotropic main body: elongated along a random axis
                    ang = numpy.arctan2(P_flat @ north_v, P_flat @ east_v)
                    s_lobe = s_lobe / (1.0 + elong * numpy.cos(2.0 * (ang - axis)))
                s_c = numpy.minimum(s_c, s_lobe)
            # bays push the score above the land threshold near their centres,
            # carving gulfs into the outline
            for centre, bay_r in continent_bays[ci]:
                dot = numpy.clip(P_flat @ centre.astype(numpy.float32), -1.0, 1.0)
                bay = numpy.clip(1.0 - numpy.arccos(dot) / warp / (bay_r * calib[ci]), 0.0, 1.0)
                s_c = s_c + 1.5 * bay * bay
            better = s_c < s_min
            s_min = numpy.where(better, s_c, s_min)
            owner = numpy.where(better, ci, owner)
        s_min = s_min + fine
        thr = numpy.quantile(s_min, min(max(total_land, 0.02), 0.9))
        land_flat = s_min < thr
        if _it == 2:
            break
        sphere = float(h * w)
        for ci in range(n):
            area = float(((owner == ci) & land_flat).sum())
            if area < 50:
                calib[ci] *= 0.85
                continue
            target_frac = shares[ci] * total_land * sphere
            calib[ci] *= min(1.5, max(0.67, math.sqrt(target_frac / max(area, 1.0))))

    land_mask = land_flat.reshape(h, w)
    continent_mask = numpy.where(land_flat, owner, -1).astype(numpy.int32).reshape(h, w)

    island_mask = _grow_islands(seed, w, h, land_mask, cores, shares,
                                island_fraction, plates, plate_is_ocean)
    if island_mask is not None and island_mask.any():
        new_land = island_mask & ~land_mask
        if new_land.any():
            # attach new islands to the nearest continent core
            nearest = numpy.zeros(h * w, dtype=numpy.int32)
            best_d = numpy.full(h * w, numpy.inf, dtype=numpy.float32)
            for ci, core in enumerate(cores):
                dot = numpy.clip(P_flat @ core.astype(numpy.float32), -1.0, 1.0)
                d = numpy.arccos(dot)
                closer = d < best_d
                best_d = numpy.where(closer, d, best_d)
                nearest = numpy.where(closer, ci, nearest)
            land_mask = (land_mask | new_land)
            continent_mask = numpy.where(new_land.reshape(h, w),
                                         nearest.reshape(h, w), continent_mask)
    return land_mask, continent_mask


# ---------------------------------------------------------------------------
# Stage 3: Earth-like islands
# ---------------------------------------------------------------------------
def _blob_mask(h, w, P_flat, center, fraction, seed, warp_amp=0.30):
    """One noise-warped cap island covering ~``fraction`` of the sphere."""
    r = _core_radius(fraction)
    dot = numpy.clip(P_flat @ center.astype(numpy.float32), -1.0, 1.0)
    d = numpy.arccos(dot)
    warp = 1.0 + warp_amp * _fbm(h, w, seed, octaves=3,
                                 base_res=max(48, min(160, w // 12))).reshape(-1)
    return (d / numpy.maximum(warp, 0.05)) < r


def _random_ocean_vec(rng, land_mask):
    """Random unit vector comfortably inside the open ocean."""
    h, w = land_mask.shape
    for _ in range(200):
        lat = rng.uniform(-1.22, 1.22)
        lon = rng.rand() * TAU
        cl = math.cos(lat)
        v = numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                        dtype=numpy.float32)
        y = int((0.5 - lat / math.pi) * h)
        x = int(lon / TAU * w) % w
        y0, y1 = max(0, y - 6), min(h, y + 7)
        x0, x1 = (x - 6) % w, (x + 7) % w
        if y0 >= y1:
            continue
        if x0 < x1:
            patch = land_mask[y0:y1, x0:x1]
        else:
            patch = numpy.hstack([land_mask[y0:y1, x0:], land_mask[y0:y1, :x1]])
        if not patch.any():
            return v
    return None


def _add_local(result, h, w, vec, fraction, seed, warp_amp=0.4):
    """Tiny island via a noise-warped cap, evaluated only in a local window.

    ``fraction`` must be small (<= ~3e-4 of the sphere) so the window stays
    cheap; far cheaper than a full-map blob for the many 1-3 px islands.
    """
    r = _core_radius(fraction)
    r_px = r / (math.pi / h)
    half = int(r_px * (1 + warp_amp)) + 3
    lat = math.asin(max(-1.0, min(1.0, float(vec[1]))))
    lon = math.atan2(float(vec[2]), float(vec[0])) % TAU
    y = int((0.5 - lat / math.pi) * h)
    x = int(lon / TAU * w) % w
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    if y1 <= y0:
        return
    xs = numpy.arange(x - half, x + half + 1) % w
    yy = (numpy.arange(y0, y1, dtype=numpy.float32) + 0.5) / h
    la = math.pi / 2 - yy * math.pi
    lo = xs.astype(numpy.float32) / w * TAU
    cl = numpy.cos(la)[:, None]
    pl = numpy.empty((y1 - y0, len(xs), 3), dtype=numpy.float32)
    pl[:, :, 0] = cl * numpy.cos(lo)[None, :]
    pl[:, :, 1] = numpy.broadcast_to(numpy.sin(la)[:, None], pl.shape[:2])
    pl[:, :, 2] = cl * numpy.sin(lo)[None, :]
    dot = numpy.clip(pl @ vec.astype(numpy.float32), -1.0, 1.0)
    warp = 1.0 + warp_amp * _fbm(y1 - y0, len(xs), seed, octaves=2,
                                 base_res=max(12, len(xs))).reshape(y1 - y0, len(xs))
    mask = numpy.arccos(dot) / numpy.maximum(warp, 0.05) < r
    if xs[0] < xs[-1]:
        view = result[y0:y1, xs[0]:xs[-1] + 1]
        view |= mask
    else:  # window wraps the x edge
        cut = int(numpy.searchsorted(xs, 0)) if xs[-1] == 0 else int(numpy.nonzero(numpy.diff(xs) < 0)[0][0]) + 1
        view = numpy.hstack([result[y0:y1, xs[:cut]], result[y0:y1, xs[cut:]]])
        view |= mask
        result[y0:y1, xs[:cut]] = view[:, :cut]
        result[y0:y1, xs[cut:]] = view[:, cut:]


def _grow_islands(seed, w, h, land_mask, cores, shares, island_fraction,
                  plates, plate_is_ocean):
    """Scatter shelf islands, volcanic arcs, hotspot chains, polar fragments."""
    rng = numpy.random.RandomState((seed * 8191) & 0x7FFFFFFF)
    P = _sphere_coords(h, w)
    P_flat = P.reshape(-1, 3)
    result = numpy.zeros((h, w), dtype=bool)

    def add(vec, fraction, s, warp_amp=0.30):
        m = _blob_mask(h, w, P_flat, vec, fraction, seed + 31 * s, warp_amp)
        result[:] |= m.reshape(h, w)

    # --- 1. large shelf islands off continental coasts ----------------------
    land_yx = numpy.argwhere(land_mask)
    shelf_islands = rng.randint(3, 6)
    for s in range(shelf_islands):
        base = land_yx[rng.randint(len(land_yx))]
        # step away from land along a random bearing until in open water
        lat = math.pi / 2 - (base[0] + 0.5) / h * math.pi
        lon = (base[1] + 0.5) / w * TAU
        cl = math.cos(lat)
        v = numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                        dtype=numpy.float32)
        # slide the centre a few degrees sideways off the coast
        off = rng.uniform(2.5, 6.0) * (1 if rng.rand() < 0.5 else -1)
        w2 = numpy.array([-v[2], 0.0, v[0]], dtype=numpy.float32)
        nw = float(math.sqrt(float(w2.dot(w2))))
        w2 = w2 / nw if nw > 1e-9 else numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        nvec = numpy.cross(w2, v)
        centre = v * math.cos(math.radians(off)) + nvec * math.sin(math.radians(off))
        centre = centre / float(math.sqrt(float(centre.dot(centre))))
        add(centre, rng.uniform(0.0004, 0.0014), 100 + s)

    # --- 2. volcanic arcs along ocean-side plate boundaries -----------------
    arc_seeds = _ocean_boundary_points(seed, w, h, plates, plate_is_ocean)
    for s, (point, direction) in enumerate(arc_seeds):
        n_blobs = rng.randint(4, 8)
        step = rng.uniform(2.0, 4.5)
        f0 = rng.uniform(0.00008, 0.00025)
        cur = numpy.array(point, dtype=numpy.float32)
        for k in range(n_blobs):
            _add_local(result, h, w, cur, f0 * (1.0 - 0.12 * k),
                       seed + 200 + 10 * s + k, warp_amp=0.15)
            cur = cur * math.cos(math.radians(step)) + direction * math.sin(math.radians(step))
            cur = cur / float(math.sqrt(float(cur.dot(cur))))
            # lateral jitter so the chain reads as an arc, not a pearl string
            side = numpy.cross(cur, direction)
            ns = float(math.sqrt(float(side.dot(side))))
            if ns > 1e-9:
                side = side / ns
                cur = cur + side * rng.uniform(-0.35, 0.35) * math.sin(math.radians(step))
                cur = cur / float(math.sqrt(float(cur.dot(cur))))

    # --- 3. hotspot chains in the open ocean --------------------------------
    chains = rng.randint(2, 4)
    for s in range(chains):
        start = _random_ocean_vec(rng, land_mask)
        if start is None:
            continue
        bearing = rng.rand() * TAU
        east = numpy.array([math.cos(bearing), 0.0, math.sin(bearing)], dtype=numpy.float32)
        north = numpy.cross(start, east)
        nn = float(math.sqrt(float(north.dot(north))))
        if nn < 1e-9:
            continue
        north = north / nn
        n_blobs = rng.randint(4, 8)
        f0 = rng.uniform(0.00008, 0.0002)
        cur = start
        for k in range(n_blobs):
            _add_local(result, h, w, cur, f0 * (1.0 - 0.1 * k),
                       seed + 300 + 10 * s + k, warp_amp=0.12)
            step_deg = rng.uniform(3.0, 5.0)
            cur = cur * math.cos(math.radians(step_deg)) + north * math.sin(math.radians(step_deg))
            cur = cur / float(math.sqrt(float(cur.dot(cur))))
            side = numpy.cross(cur, north)
            ns = float(math.sqrt(float(side.dot(side))))
            if ns > 1e-9:
                side = side / ns
                cur = cur + side * rng.uniform(-0.3, 0.3) * math.sin(math.radians(step_deg))
                cur = cur / float(math.sqrt(float(cur.dot(cur))))

    # --- 4. polar fragments --------------------------------------------------
    frags = rng.randint(6, 14)
    for s in range(frags):
        lat = math.radians(rng.uniform(56.0, 80.0)) * (1 if rng.rand() < 0.5 else -1)
        lon = rng.rand() * TAU
        cl = math.cos(lat)
        v = numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                        dtype=numpy.float32)
        _add_local(result, h, w, v, rng.uniform(0.00002, 0.0001),
                   seed + 400 + s, warp_amp=0.1)

    # --- 5. coastal fringe: scatter small islands off every coast ------------
    try:
        from scipy.ndimage import distance_transform_edt
        d_land = distance_transform_edt(~land_mask)
        band_y, band_x = numpy.nonzero((d_land > 4) & (d_land < 20))
        fringe = rng.randint(70, 110)
        if len(band_y):
            chosen = []
            tries = 0
            while len(chosen) < fringe and tries < 900:
                tries += 1
                i = rng.randint(len(band_y))
                cy, cx = int(band_y[i]), int(band_x[i])
                if any((cy - qy) ** 2 + min(abs(cx - qx), w - abs(cx - qx)) ** 2
                       < 8 ** 2 for qy, qx in chosen):
                    continue
                chosen.append((cy, cx))
                lat = math.pi / 2 - (cy + 0.5) / h * math.pi
                lon = (cx + 0.5) / w * TAU
                cl = math.cos(lat)
                v = numpy.array([cl * math.cos(lon), math.sin(lat), cl * math.sin(lon)],
                                dtype=numpy.float32)
                _add_local(result, h, w, v, rng.uniform(0.000012, 0.00008),
                           seed + 500 + tries, warp_amp=0.5)
    except ImportError:
        pass

    return result


def _ocean_boundary_points(seed, w, h, plates, plate_is_ocean, n=3):
    """Points on ocean/continent plate boundaries + the boundary direction.

    Returns up to ``n`` ``(unit_vector, tangent_direction)`` pairs used to lay
    volcanic island arcs (Indonesia/Japan style) along subduction zones.
    """
    if plates is None:
        return []
    plate_is_ocean = plate_is_ocean if plate_is_ocean is not None else [0] * (int(plates.max()) + 1)
    ocean_of_plate = numpy.zeros(int(plates.max()) + 1, dtype=bool)
    for i, flag in enumerate(plate_is_ocean):
        if i < len(ocean_of_plate):
            ocean_of_plate[i] = bool(flag)
    oc = ocean_of_plate[plates.astype(numpy.int32)]
    # ocean plate cells adjacent (4-neighbour, x wraps) to a continental plate
    b = numpy.zeros((h, w), dtype=bool)
    b |= oc & (oc != numpy.roll(oc, -1, axis=1))
    b |= oc & (oc != numpy.roll(oc, 1, axis=1))
    if h > 1:
        b[:-1] |= oc[:-1] & (oc[:-1] != oc[1:])
        b[1:] |= oc[1:] & (oc[1:] != oc[:-1])
    ys, xs = numpy.nonzero(b)
    if len(ys) == 0:
        return []
    rng = numpy.random.RandomState((seed * 104729) & 0x7FFFFFFF)
    lat = math.pi / 2 - (ys + 0.5) / h * math.pi
    lon = (xs + 0.5) / w * TAU
    cl = numpy.cos(lat)
    pts = numpy.stack([cl * numpy.cos(lon), numpy.sin(lat), cl * numpy.sin(lon)], axis=1)
    picked = []
    tries = 0
    while len(picked) < n and tries < 60:
        tries += 1
        i = rng.randint(len(ys))
        v = pts[i].astype(numpy.float32)
        if any(float(v.dot(p[0])) > math.cos(math.radians(25)) for p in picked):
            continue
        # tangent along the boundary: neighbour boundary pixel direction
        j = rng.randint(len(ys))
        v2 = pts[j].astype(numpy.float32)
        t = v2 - v * float(v2.dot(v))
        nt = float(math.sqrt(float(t.dot(t))))
        if nt < 1e-6:
            continue
        picked.append((v, t / nt))
    return picked
