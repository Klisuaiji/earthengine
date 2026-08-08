"""Plate boundary classification (convergent / divergent / transform).

Given a 2D plate-id map produced by :mod:`worldengine.spherical_voronoi`, this
module classifies every boundary cell into one of three tectonic interaction
types by comparing the *velocity vectors* of the two plates that meet there.

Why velocity vectors?
---------------------
A pure Voronoi partition is purely geometric - every plate is a region and
nothing is moving.  To get a physical-looking topology we give each plate a
deterministic but seed-dependent velocity vector (in 3D, on the unit sphere)
and let the relative velocity of two adjacent plates decide the boundary type:

* **Convergent (消亡边界)**: plates move *toward* each other -> mountains /
  subduction zones.  The relative velocity has a positive component along the
  boundary normal that points inward from both sides.
* **Divergent (生长边界)**: plates move *apart* -> rift valleys / mid-ocean
  ridges.  The relative velocity component along the boundary normal is
  negative (apart).
* **Transform (转换边界)**: plates slide *past* each other -> transform
  faults.  The relative velocity is mostly tangential to the boundary.

Implementation
--------------
* Each plate's velocity is a 3D unit vector drawn from a hashed PRNG keyed
  by ``(seed, plate_id)`` - the same seed always yields the same directions.
* Velocity magnitude is uniform (``PLATE_SPEED = 1.0``); the *direction* is
  what differentiates boundaries.
* Boundary cells are detected with a 4-neighbour scan on the plate map.  At
  each boundary cell the six-neighbour radial spread of the two plates is
  used to compute the boundary normal (in 3D, on the sphere).
* Convergence / divergence is the sign of the relative velocity projected
  onto the normal; tangential dominance marks transform.

The output arrays carry one value per pixel:

* ``boundary_type``:  0 = interior, 1 = convergent, 2 = divergent, 3 = transform
* ``boundary_dist``:  pixels-to-nearest-boundary (cheap distance transform)
* ``convergent_dist``: pixels-to-nearest-convergent-boundary
* ``divergent_dist``:  pixels-to-nearest-divergent-boundary
* ``plate_velocity``:  ``(n_plates, 3)`` ro
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy

# Boundary-type codes (kept stable so downstream tools can match them).
INTERIOR = 0
CONVERGENT = 1   # 消亡边界 (subduction / mountain range)
DIVERGENT = 2    # 生长边界 (rift / mid-ocean ridge)
TRANSFORM = 3    # 转换边界 (strike-slip)

BOUNDARY_NAMES = {
    INTERIOR: "interior",
    CONVERGENT: "convergent",
    DIVERGENT: "divergent",
    TRANSFORM: "transform",
}

# Classification thresholds (relative to the magnitude of the relative
# velocity).  These are fractions of |v_rel|, not absolute pixel values, so
# they work the same at any resolution.
#
# Let ``share_tan = |tangent| / |v_rel|``.  When share_tan is large the
# plates are sliding past each other (transform).  When share_tan is small
# the component along the normal dominates - sign decides convergent vs
# divergent.
_TRANSFORM_TAN_SHARE = 0.65     # above this share -> transform
_CONVERGE_DOT_SHARE = 0.05      # below this absolute dot share -> interior / fence


def _hash01(k: int, seed: int) -> float:
    """Stable 32-bit hash -> [0, 1)."""
    h = (seed * 2654435761) & 0x7FFFFFFF
    h = (h ^ (k * 374761393)) & 0x7FFFFFFF
    h = (h * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 13)) & 0x7FFFFFFF
    return h / 2147483647.0


def _make_velocity(k: int, seed: int) -> numpy.ndarray:
    """Return a deterministic 3D unit velocity vector for plate ``k``."""
    # Uniform random point on the sphere from a hashed PRNG.
    u = _hash01(k * 2 + 1, seed)
    v = _hash01(k * 2 + 2, seed)
    z = 1.0 - 2.0 * u
    phi = 2.0 * math.pi * v
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return numpy.array([r * math.cos(phi), z, r * math.sin(phi)], dtype=numpy.float32)


def _pixel_vec(h: int, w: int) -> numpy.ndarray:
    """Unit-sphere vector for each pixel of an (h, w) map (equirectangular, ±85°)."""
    lat = ((numpy.arange(h, dtype=numpy.float32) + 0.5) / h - 0.5) * math.radians(170.0)
    lon = ((numpy.arange(w, dtype=numpy.float32) + 0.5) / w) * (2.0 * math.pi)
    cl = numpy.cos(lat)[:, None]
    sl = numpy.sin(lat)[:, None]
    co = numpy.cos(lon)[None, :]
    so = numpy.sin(lon)[None, :]
    P = numpy.zeros((h, w, 3), dtype=numpy.float32)
    P[:, :, 0] = cl * co
    P[:, :, 1] = sl
    P[:, :, 2] = cl * so
    return P


def classify_boundaries(
    merged: numpy.ndarray,
    seed: int,
) -> Dict[str, numpy.ndarray]:
    """Classify every boundary cell of ``merged`` into convergent/divergent/transform.

    Parameters
    ----------
    merged : (h, w) int array
        Plate-id map produced by ``spherical_voronoi`` (use the *merged* map,
        not the raw plate map, so we classify continental-scale boundaries).
    seed : int
        World seed.  Each plate-id derives its velocity from this seed so the
        result is fully reproducible.

    Returns
    -------
    dict with keys:
        * ``boundary_type``   -- (h, w) int8  (0/1/2/3)
        * ``boundary_dist``   -- (h, w) float32  distance to nearest boundary
        * ``convergent_dist`` -- (h, w) float32  distance to nearest convergent
        * ``divergent_dist``  -- (h, w) float32  distance to nearest divergent
        * ``transform_dist``  -- (h, w) float32  distance to nearest transform
        * ``plate_velocity``  -- (n_plates, 3) float32
    """
    h, w = merged.shape
    n_plates = int(merged.max()) + 1

    # Per-plate velocity vectors (deterministic, reproducible).
    plate_velocity = numpy.zeros((n_plates, 3), dtype=numpy.float32)
    for k in range(n_plates):
        plate_velocity[k] = _make_velocity(k, seed)

    # -------------------- 1. Boundary mask (4-neighbour wrap in x) -----------
    m = merged.astype(numpy.int32)
    is_bnd = numpy.zeros((h, w), dtype=bool)
    right = numpy.roll(m, -1, axis=1)
    left = numpy.roll(m, 1, axis=1)
    is_bnd |= m != right
    is_bnd |= m != left
    if h > 1:
        is_bnd[1:] |= m[1:] != m[:-1]
        is_bnd[:-1] |= m[:-1] != m[1:]

    if not is_bnd.any():
        # Trivial: no boundaries at all (single-plate map).  Return zeros.
        return {
            "boundary_type": numpy.zeros((h, w), dtype=numpy.int8),
            "boundary_dist": numpy.zeros((h, w), dtype=numpy.float32),
            "convergent_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "divergent_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "transform_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "junctions": {},
            "junction_count": 0,
            "plate_velocity": plate_velocity,
        }

    # -------------------- 2. For each boundary cell, classify ----------------
    # The "normal" of the boundary at (y, x) is the average outward direction
    # from the two (or more) neighbouring plate-centroid vectors.  We project
    # the relative velocity of the two main plates onto that normal.
    P = _pixel_vec(h, w)

    # Build neighbour-id arrays once (cost: 4 * h * w bytes).
    nR = numpy.roll(m, -1, axis=1)
    nL = numpy.roll(m, 1, axis=1)
    nU = numpy.zeros_like(m) if h == 1 else numpy.zeros_like(m)
    nD = numpy.zeros_like(m) if h == 1 else numpy.zeros_like(m)
    if h > 1:
        nU[1:] = m[:-1]
        nD[:-1] = m[1:]

    boundary_type = numpy.zeros((h, w), dtype=numpy.int8)

    bnd_y, bnd_x = numpy.where(is_bnd)
    if len(bnd_y) == 0:
        return {
            "boundary_type": boundary_type,
            "boundary_dist": numpy.zeros((h, w), dtype=numpy.float32),
            "convergent_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "divergent_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "transform_dist": numpy.full((h, w), -1.0, dtype=numpy.float32),
            "junctions": {},
            "junction_count": 0,
            "plate_velocity": plate_velocity,
        }

    # Relative velocity at each boundary cell: V[other] - V[me], where 'me' is
    # the plate at the cell and 'other' is the most common neighbour.
    own = m[bnd_y, bnd_x]
    cands = numpy.stack([nR[bnd_y, bnd_x], nL[bnd_y, bnd_x],
                          nU[bnd_y, bnd_x], nD[bnd_y, bnd_x]], axis=1)
    # Pick the first neighbour that differs from own.
    diff_mask = cands != own[:, None]
    has_diff = diff_mask.any(axis=1)
    other = own.copy()
    if has_diff.any():
        first_diff = numpy.argmax(diff_mask, axis=1)
        other = cands[numpy.arange(len(bnd_y)), first_diff]
    rel_v = plate_velocity[other] - plate_velocity[own]  # (N, 3)

    # Boundary normal: average of own + other outward direction in 3D.
    own_dir = P[bnd_y, bnd_x]
    other_dir = P[bnd_y, bnd_x]  # placeholder; we want the neighbour's vector
    # Re-derive neighbour's P correctly:
    # nR[-1, x] wraps to x=0 so we have to handle the wrap with modular x.
    nr_x = (bnd_x + 1) % w
    nl_x = (bnd_x - 1 + w) % w
    nu_y = numpy.clip(bnd_y + 1, 0, h - 1)
    nd_y = numpy.clip(bnd_y - 1, 0, h - 1)

    # Pull the neighbour's P for whichever direction we actually used.
    other_x = numpy.where(first_diff == 0, nr_x,
                  numpy.where(first_diff == 1, nl_x,
                  numpy.where(first_diff == 2, bnd_x, bnd_x)))
    other_y = numpy.where(first_diff == 0, bnd_y,
                  numpy.where(first_diff == 1, bnd_y,
                  numpy.where(first_diff == 2, nu_y, nd_y)))
    other_P = P[other_y, other_x]

    normal = own_dir + other_P
    n_norm = numpy.linalg.norm(normal, axis=1)
    safe = n_norm > 1e-9
    normal[safe] /= n_norm[safe, None]
    normal[~safe] = own_dir[~safe]

    # Project relative velocity onto the normal (positive = approach).
    dot = (rel_v * normal).sum(axis=1)
    rel_v_norm = numpy.linalg.norm(rel_v, axis=1)
    tan = numpy.sqrt(numpy.maximum(rel_v_norm ** 2 - dot ** 2, 0.0))

    # Decide type per boundary cell.  Both dot and tan are in the same units
    # as |v_rel|, so we work in *shares* of that magnitude.
    safe = rel_v_norm > 1e-6
    dot_share = numpy.zeros_like(dot)
    tan_share = numpy.zeros_like(tan)
    dot_share[safe] = dot[safe] / rel_v_norm[safe]
    tan_share[safe] = tan[safe] / rel_v_norm[safe]

    types = numpy.zeros(len(bnd_y), dtype=numpy.int8)
    # Tangential dominance -> transform regardless of sign.
    transform_mask = (tan_share > _TRANSFORM_TAN_SHARE)
    convergent_mask = (~transform_mask) & (dot_share > _CONVERGE_DOT_SHARE)
    divergent_mask = (~transform_mask) & (dot_share < -_CONVERGE_DOT_SHARE)
    types[convergent_mask] = CONVERGENT
    types[divergent_mask] = DIVERGENT
    types[transform_mask] = TRANSFORM

    boundary_type[bnd_y, bnd_x] = types

    # -------------------- 3. Distance maps ------------------------------------
    boundary_dist = _distance_to(mask=is_bnd, h=h, w=w)
    convergent_dist = _distance_to(mask=(boundary_type == CONVERGENT), h=h, w=w)
    divergent_dist = _distance_to(mask=(boundary_type == DIVERGENT), h=h, w=w)
    transform_dist = _distance_to(mask=(boundary_type == TRANSFORM), h=h, w=w)

    # Replace "no-target" sentinel with +inf for clarity.
    convergent_dist[convergent_dist < 0] = numpy.inf
    divergent_dist[divergent_dist < 0] = numpy.inf
    transform_dist[transform_dist < 0] = numpy.inf

    # -------------------- 4. Triple-junction (jarcs) detection ---------------
    # A *triple junction* is a vertex where three or more distinct boundary
    # arcs meet.  We first label every connected boundary segment as an "arc",
    # then scan every 2x2 corner: if the union of the 4 corner cells and their
    # 4-neighbours touches >=3 distinct arcs, every boundary cell in that
    # corner is recorded as a junction carrying the list of arc ids.
    # arc_of: each boundary cell carries the id of the plate-PAIR it separates
    # (an "arc" = a maximal run of boundary cells between the same two plates).
    # This is what makes a *triple* junction detectable: three plates meeting at
    # one vertex yield three distinct pair ids in the same neighbourhood.
    arc_of = numpy.full(h * w, -1, dtype=numpy.int32)
    if len(bnd_y) > 0:
        pa = numpy.minimum(own, other)
        pb = numpy.maximum(own, other)
        pairs = numpy.stack([pa, pb], axis=1)
        _uniq, inv = numpy.unique(pairs, axis=0, return_inverse=True)
        arc_of[bnd_y * w + bnd_x] = inv.astype(numpy.int32)
    is_jcell = is_bnd.reshape(-1).astype(numpy.int8)  # 1 = pixel lies on a boundary

    # Candidate junction corners: a 2x2 block that touches >=2 boundary pixels.
    jc = is_bnd.astype(numpy.int32)
    c00 = jc[:-1, :]
    c10 = jc[:-1, (numpy.arange(w) + 1) % w]
    c01 = jc[1:, :]
    c11 = jc[1:, (numpy.arange(w) + 1) % w]
    corner_count = c00 + c10 + c01 + c11
    is_jcor = (corner_count >= 2).reshape(-1).astype(numpy.int8)

    jarcs = {}
    for cy in range(h - 1):
        for cx in range(w):
            if is_jcor[cy * w + cx] == 0:
                continue
            seen = set()
            # 检查角点周围 4 个格及其 4-邻域的并集
            for dy in [0, 1]:
                for dx in [0, 1]:
                    cell_x = (cx + dx) % w
                    cell_y = cy + dy
                    if cell_y < 0 or cell_y >= h:
                        continue
                    for ndx, ndy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx = (cell_x + ndx) % w
                        ny = cell_y + ndy
                        if ny < 0 or ny >= h:
                            continue
                        ni = ny * w + nx
                        a = arc_of[ni]
                        if a >= 0:
                            seen.add(a)
            # 只有真正的三联点才记录
            if len(seen) >= 3:
                for dy in [0, 1]:
                    for dx in [0, 1]:
                        cell_x = (cx + dx) % w
                        cell_y = cy + dy
                        if cell_y < 0 or cell_y >= h:
                            continue
                        cell_idx = cell_y * w + cell_x
                        if is_jcell[cell_idx] == 1:
                            jarcs[cell_idx] = list(seen)

    return {
        "boundary_type": boundary_type,
        "boundary_dist": boundary_dist,
        "convergent_dist": convergent_dist,
        "divergent_dist": divergent_dist,
        "transform_dist": transform_dist,
        "junctions": jarcs,
        "junction_count": len(jarcs),
        "plate_velocity": plate_velocity,
    }


def _distance_to(mask: numpy.ndarray, h: int, w: int) -> numpy.ndarray:
    """Cheap distance to the nearest ``True`` pixel, with +inf inside the mask.

    Falls back to a BFS when scipy is unavailable (it almost always is in this
    project, but we guard for headless test environments).
    """
    out = numpy.full((h, w), -1.0, dtype=numpy.float32)
    if not mask.any():
        return out
    try:
        from scipy import ndimage
        dm = ndimage.distance_transform_edt(~mask)
        out[:] = dm.astype(numpy.float32)
    except Exception:
        # BFS fallback (slower but correct).
        from collections import deque
        q = deque()
        out[mask] = 0.0
        for y, x in zip(*numpy.where(mask)):
            q.append((y, x))
        while q:
            y, x = q.popleft()
            d = out[y, x] + 1.0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, (x + dx) % w if (dx != 0) else x
                if 0 <= ny < h and 0 <= nx < w and out[ny, nx] < 0:
                    out[ny, nx] = d
                    q.append((ny, nx))
    return out
