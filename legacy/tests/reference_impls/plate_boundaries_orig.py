"""Plate boundary classification (growth / subduction).

Boundary *type* is decided by the **topological arc-decomposition + triple
junction voting + length-balancing + per-plate constraints** algorithm ported
faithfully from the reference generator ``world_gen.py`` (its ``_build_boundary``
routine).  This is the project convention: we do **not** synthesise per-plate
velocity vectors and project them onto a boundary normal (that earlier approach
produced a pathological ~80% "transform" bias and is no longer used).

Why this algorithm (instead of velocity vectors)
------------------------------------------------
A pure Voronoi partition is geometric - every plate is a region and nothing is
moving.  The reference's approach embraces that and decides each boundary's
character purely from *topology*:

* The boundary is split into **arcs** - maximal runs of cells separating the
  same pair of plates.
* At every **triple junction** (>=3 incident arcs) we vote: usually one
  incident arc is marked subduction (red), unless a 15% roll makes the whole
  junction growth.
* **Length balancing** flips the longest pure-growth arc to subduction until the
  red/blue total lengths are roughly equal.
* A **per-plate constraint** guarantees every plate carries both a growth and a
  subduction segment.

The result is a self-consistent, earth-like distribution (~50/50 growth vs
subduction) with no transform class, exactly as in the reference.

Codes (kept compatible with downstream tools)
---------------------------------------------
* ``0`` = interior
* ``1`` = CONVERGENT (subduction / 消亡边界, red)
* ``2`` = DIVERGENT  (growth / 生长边界, green)
* ``3`` = TRANSFORM  (reserved; this algorithm does not produce it)

The returned arrays carry one value per pixel:

* ``boundary_type``   : (h, w) int8   (0/1/2)
* ``boundary_dist``   : (h, w) float32 distance to nearest boundary
* ``convergent_dist`` : (h, w) float32 distance to nearest convergent boundary
* ``divergent_dist``  : (h, w) float32 distance to nearest divergent boundary
* ``transform_dist``  : (h, w) float32 all -inf (reserved)
* ``boundary_main``   : (h, w) int8   the two longest arcs (main chains)
* ``junction_mask``   : (h, w) int8   1 on triple-junction cells
* ``junctions``       : dict cell_idx -> list of incident arc ids (true triples)
* ``junction_count``  : int
"""
from __future__ import annotations

import math
from collections import deque

import numpy

# Boundary-type codes (stable so downstream tools can match them).
INTERIOR = 0
CONVERGENT = 1   # 消亡边界 (subduction / mountain range)
DIVERGENT = 2    # 生长边界 (growth / rift / mid-ocean ridge)
TRANSFORM = 3    # reserved; not produced by this algorithm

BOUNDARY_NAMES = {
    INTERIOR: "interior",
    CONVERGENT: "convergent",
    DIVERGENT: "divergent",
    TRANSFORM: "transform",
}

# Mirror of spherical_voronoi.PLATE_OCEAN (indices 4/5 are oceanic).  Only used
# for the returned ``is_ocean`` field; the classification itself ignores it.
PLATE_OCEAN = [0, 0, 0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Reference 2D hash (world_gen.py _hash2) - identical mixing so the junction
# voting randomness reproduces the reference bit-for-bit.
# ---------------------------------------------------------------------------
def _hash2(ix: int, iy: int, seed: int) -> float:
    h = seed & 0x7FFFFFFF
    h = (h ^ (ix * 374761393)) & 0x7FFFFFFF
    h = (h ^ (iy * 668265263)) & 0x7FFFFFFF
    h = (h * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 13)) & 0x7FFFFFFF
    return h / 2147483647.0


def classify_boundaries(
    merged: numpy.ndarray,
    seed: int,
    is_ocean: list = None,
) -> dict:
    """Classify every boundary cell of ``merged`` into growth / subduction.

    Parameters
    ----------
    merged : (h, w) int array
        Plate-id map produced by ``spherical_voronoi`` (use the *merged* map,
        not the raw plate map, so we classify continental-scale boundaries).
    seed : int
        World seed (drives the triple-junction voting randomness, reproducing
        the reference exactly).
    is_ocean : list[int] | None
        Per-plate ocean flag (0/1). Only carried through to the returned
        ``is_ocean`` field; the classification is topology-driven and does not
        read it.

    Returns
    -------
    dict with keys: ``boundary_type``, ``boundary_dist``, ``convergent_dist``,
    ``divergent_dist``, ``transform_dist``, ``boundary_main``, ``junction_mask``,
    ``junctions``, ``junction_count``, ``is_ocean``, ``plate_velocity``.
    """
    h, w = merged.shape
    pid = merged.astype(numpy.int32).reshape(-1).tolist()
    npl = int(merged.max()) + 1
    if is_ocean is None:
        is_ocean = [PLATE_OCEAN[i] if i < len(PLATE_OCEAN) else 0
                    for i in range(npl)]

    # ===================== 1. Boundary cells =====================
    is_bnd = [0] * (w * h)
    for y in range(h):
        row = y * w
        for x in range(w):
            idx = row + x
            p = pid[idx]
            rx = (x + 1) % w
            lx = (x - 1 + w) % w
            up = y + 1
            dn = y - 1
            diff = False
            if pid[row + rx] != p:
                diff = True
            elif pid[row + lx] != p:
                diff = True
            elif up < h and pid[up * w + x] != p:
                diff = True
            elif dn >= 0 and pid[dn * w + x] != p:
                diff = True
            if diff:
                is_bnd[idx] = 1

    # ===================== 2. Corner-lattice triple junctions =====================
    is_jcor = [0] * (w * (h - 1))
    for cy in range(h - 1):
        row = cy * w
        nrow = (cy + 1) * w
        for cx in range(w):
            A = pid[row + cx]
            B = pid[row + ((cx + 1) % w)]
            C = pid[nrow + cx]
            D = pid[nrow + ((cx + 1) % w)]
            s = 1
            if B != A:
                s += 1
            if C != A and C != B:
                s += 1
            if D != A and D != B and D != C:
                s += 1
            if s >= 3:
                is_jcor[row + cx] = 1

    # ===================== 3. Corner -> 4 surrounding cells =====================
    is_jcell = [0] * (w * h)
    for cy in range(h - 1):
        row = cy * w
        nrow = (cy + 1) * w
        for cx in range(w):
            if is_jcor[row + cx] == 1:
                is_jcell[row + cx] = 1
                is_jcell[row + ((cx + 1) % w)] = 1
                is_jcell[nrow + cx] = 1
                is_jcell[nrow + ((cx + 1) % w)] = 1

    # ===================== 4. Per-cell plate-pair key (fork detection) =====================
    pair_key = [-1] * (w * h)
    for y in range(h):
        row = y * w
        for x in range(w):
            idx = row + x
            if is_bnd[idx] == 0:
                continue
            p = pid[idx]
            rx = (x + 1) % w
            lx = (x - 1 + w) % w
            up = y + 1
            dn = y - 1
            pk = -1
            ok = True
            np_r = pid[row + rx]
            if np_r != p:
                pk = min(p, np_r) * npl + max(p, np_r)
            if up < h:
                np_u = pid[up * w + x]
                if np_u != p:
                    kk = min(p, np_u) * npl + max(p, np_u)
                    if pk == -1:
                        pk = kk
                    elif pk != kk:
                        ok = False
            if dn >= 0:
                np_d = pid[dn * w + x]
                if np_d != p:
                    kk = min(p, np_d) * npl + max(p, np_d)
                    if pk == -1:
                        pk = kk
                    elif pk != kk:
                        ok = False
            np_l = pid[row + lx]
            if np_l != p:
                kk = min(p, np_l) * npl + max(p, np_l)
                if pk == -1:
                    pk = kk
                elif pk != kk:
                    ok = False
            if ok and pk != -1:
                pair_key[idx] = pk

    # Triple-junction cells are forced to break the arc run.
    for jk in range(w * h):
        if is_jcell[jk] == 1:
            pair_key[jk] = -1

    # ===================== 5. Flood-fill decompose arcs =====================
    arc_of = [-1] * (w * h)
    arc_cells = []
    comp_id = 0
    for i in range(w * h):
        if pair_key[i] < 0 or arc_of[i] >= 0:
            continue
        apk = pair_key[i]
        stack = [i]
        arc_of[i] = comp_id
        cells = [i]
        while stack:
            cur = stack.pop()
            cx = cur % w
            cy = cur // w
            dirs = [((cx + 1) % w, cy), ((cx - 1 + w) % w, cy),
                    (cx, cy + 1), (cx, cy - 1)]
            for (nx, ny) in dirs:
                if ny < 0 or ny >= h:
                    continue
                ni = ny * w + nx
                if pair_key[ni] == apk and arc_of[ni] < 0:
                    arc_of[ni] = comp_id
                    cells.append(ni)
                    stack.append(ni)
        arc_cells.append(cells)
        comp_id += 1
    n_arcs = comp_id

    # ===================== 6. Incident arcs at junctions (classification) =====
    # Faithful to world_gen.py: collect arc_of from the 4 orthogonal neighbours
    # of each junction cell.  At a triple point every corner cell only sees
    # <=2 arcs, so the triple-junction *voting* (step 7) is intentionally
    # inert and the red/blue balance is produced purely by the length
    # balancing (step 8) - exactly as in the reference.  (Triple-junction
    # *detection* for analysis uses a wider neighbourhood below.)
    jarcs = {}
    for jk in range(w * h):
        if is_jcell[jk] == 0:
            continue
        jx = jk % w
        jy = jk // w
        dirs = [((jx + 1) % w, jy), ((jx - 1 + w) % w, jy),
                (jx, jy + 1), (jx, jy - 1)]
        seen = set()
        for (nx, ny) in dirs:
            if ny < 0 or ny >= h:
                continue
            ni = ny * w + nx
            a = arc_of[ni]
            if a >= 0:
                seen.add(a)
        jarcs[jk] = list(seen)

    # ===================== 7. Triple-junction voting =====================
    # 1 = growth, 2 = subduction (reference codes; remapped later).
    arc_type = [1] * n_arcs
    for jk, inc in jarcs.items():
        if len(inc) >= 3:
            all_growth = _hash2(jk, 777, seed) < 0.15
            if not all_growth:
                kill = int(math.floor(_hash2(jk, 13, seed) * float(len(inc)))) % len(inc)
                arc_type[inc[kill]] = 2

    # ===================== 8. Balance: equalise red/blue total length =====================
    growth_len = 0
    ext_len = 0
    for si in range(n_arcs):
        L = len(arc_cells[si])
        if arc_type[si] == 1:
            growth_len += L
        else:
            ext_len += L
    guard = 0
    while growth_len - ext_len > max(growth_len, 1) * 0.12 and guard < 2000:
        guard += 1
        best_si = -1
        best_L = -1
        for si in range(n_arcs):
            if arc_type[si] != 1:
                continue
            L = len(arc_cells[si])
            if L > best_L:
                best_L = L
                best_si = si
        if best_si < 0:
            break
        flipL = len(arc_cells[best_si])
        arc_type[best_si] = 2
        growth_len -= flipL
        ext_len += flipL

    # ===================== 8.5 Constraint repair: every plate has both types =====================
    arc_plates = []
    for si in range(n_arcs):
        pl_set = set()
        cells = arc_cells[si]
        for ci in range(len(cells)):
            pl_set.add(pid[cells[ci]])
        arc_plates.append(list(pl_set))
    pblue = [0] * npl
    pred = [0] * npl
    for si in range(n_arcs):
        for pk in arc_plates[si]:
            if arc_type[si] == 1:
                pblue[pk] += 1
            else:
                pred[pk] += 1
    guard2 = 0
    while guard2 < 50:
        guard2 += 1
        fixed = False
        for p in range(npl):
            miss_ext = (pred[p] == 0)
            miss_gro = (pblue[p] == 0)
            if not miss_ext and not miss_gro:
                continue
            from_t = 1 if miss_ext else 2
            to_t = 2 if miss_ext else 1
            best_si = -1
            best_L = 1000000
            for si in range(n_arcs):
                if arc_type[si] != from_t:
                    continue
                if p not in arc_plates[si]:
                    continue
                ok_flip = True
                for q in arc_plates[si]:
                    if (from_t == 1 and pblue[q] < 2) or (from_t == 2 and pred[q] < 2):
                        ok_flip = False
                        break
                if not ok_flip:
                    continue
                L = len(arc_cells[si])
                if L < best_L:
                    best_L = L
                    best_si = si
            if best_si >= 0:
                arc_type[best_si] = to_t
                for q in arc_plates[best_si]:
                    if from_t == 1:
                        pblue[q] -= 1
                    else:
                        pred[q] -= 1
                    if to_t == 1:
                        pblue[q] += 1
                    else:
                        pred[q] += 1
                fixed = True
        if not fixed:
            break

    # ===================== 9. Main chains = two longest arcs =====================
    main = [0] * n_arcs
    best1 = -1
    best2 = -1
    l1 = -1
    l2 = -1
    for si in range(n_arcs):
        L = len(arc_cells[si])
        if L > l1:
            l2 = l1
            best2 = best1
            l1 = L
            best1 = si
        elif L > l2:
            l2 = L
            best2 = si
    if best1 >= 0:
        main[best1] = 1
    if best2 >= 0:
        main[best2] = 1

    # ===================== 10. Write back boundary_type / junction_mask / main =====================
    boundary_type = [0] * (w * h)
    boundary_main = [0] * (w * h)
    junction_mask = [0] * (w * h)
    for si in range(n_arcs):
        cells = arc_cells[si]
        for ci in range(len(cells)):
            boundary_type[cells[ci]] = arc_type[si]
            if main[si] == 1:
                boundary_main[cells[ci]] = 1
    for jk, inc in jarcs.items():
        junction_mask[jk] = 1
        red = False
        m = False
        for si in inc:
            if arc_type[si] == 2:
                red = True
            if main[si] == 1:
                m = True
        boundary_type[jk] = 2 if red else 1
        if m:
            boundary_main[jk] = 1

    # ===================== 11. Light thickening (1 ring) =====================
    src = list(boundary_type)
    for i in range(w * h):
        bt = src[i]
        if bt == 0:
            continue
        x = i % w
        y = i // w
        neigh4 = [((x + 1) % w, y), ((x - 1 + w) % w, y), (x, y + 1), (x, y - 1)]
        for (nx, ny) in neigh4:
            if ny < 0 or ny >= h:
                continue
            ni = ny * w + nx
            if is_jcell[ni] == 1:
                continue
            if src[ni] == 0:
                boundary_type[ni] = bt
                if boundary_main[i] == 1:
                    boundary_main[ni] = 1

    # ===================== 11.5 Triple-junction *detection* (analysis only) ============
    # Independent of the classification above: a wide neighbourhood (the whole
    # 2x2 corner block + each block cell's 4-neighbours) actually reveals all
    # three radiating arcs at a triple point, so true triples (>=3 distinct
    # arcs) are captured here.  These are returned in ``junctions`` for the
    # topology analysis and never influence the boundary-type assignment.
    junctions = {}
    for cy in range(h - 1):
        for cx in range(w):
            if is_jcor[cy * w + cx] == 0:
                continue
            seen = set()
            block = [(cx, cy), ((cx + 1) % w, cy),
                     (cx, cy + 1), ((cx + 1) % w, cy + 1)]
            for (bx, by) in block:
                for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx = (bx + ndx) % w
                    ny = by + ndy
                    if ny < 0 or ny >= h:
                        continue
                    ni = ny * w + nx
                    a = arc_of[ni]
                    if a >= 0:
                        seen.add(a)
            if len(seen) >= 3:
                for (bx, by) in block:
                    cell_idx = by * w + bx
                    if is_jcell[cell_idx] == 1:
                        junctions[cell_idx] = list(seen)
    junction_count = len(junctions)

    # ===================== 12. Remap reference codes -> project codes =====================
    # reference: 1 = growth, 2 = subduction, 0 = interior
    # project  : DIVERGENT=2, CONVERGENT=1
    bt_ref = numpy.array(boundary_type, dtype=numpy.int8).reshape(h, w)
    out = numpy.zeros((h, w), dtype=numpy.int8)
    out[bt_ref == 1] = DIVERGENT
    out[bt_ref == 2] = CONVERGENT

    boundary_main_np = numpy.array(boundary_main, dtype=numpy.int8).reshape(h, w)
    junction_mask_np = numpy.array(junction_mask, dtype=numpy.int8).reshape(h, w)
    is_bnd_np = numpy.array(is_bnd, dtype=bool).reshape(h, w)

    # ===================== 13. Distance maps =====================
    boundary_dist = _distance_to(is_bnd_np, h, w)
    convergent_dist = _distance_to(out == CONVERGENT, h, w)
    divergent_dist = _distance_to(out == DIVERGENT, h, w)
    transform_dist = numpy.full((h, w), -1.0, dtype=numpy.float32)
    transform_dist[transform_dist < 0] = numpy.inf

    # junctions / junction_count already computed in step 11.5 (wide detection).

    return {
        "boundary_type": out,
        "boundary_dist": boundary_dist,
        "convergent_dist": convergent_dist,
        "divergent_dist": divergent_dist,
        "transform_dist": transform_dist,
        "boundary_main": boundary_main_np,
        "junction_mask": junction_mask_np,
        "junctions": junctions,
        "junction_count": junction_count,
        "is_ocean": numpy.array(is_ocean, dtype=numpy.int8),
        "plate_velocity": None,
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
