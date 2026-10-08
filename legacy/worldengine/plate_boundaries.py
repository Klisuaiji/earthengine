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

Implementation note
-------------------
The per-cell stages (boundary detection, corner-lattice junctions, pair keys,
arc flood fill, write-back, ring thickening) are vectorised with NumPy (+ SciPy
``connected_components`` for the arc flood fill).  The output is bit-identical
to the former pure-Python loop implementation: arc ids keep the reference's
BFS discovery order (ascending minimum cell index), and junction voting /
thickening keep the original processing order.  The junction-level and
arc-level stages (voting, balancing, constraint repair) stay in Python - they
only touch the (few) junction cells and arcs, never the whole map.

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
* ``transform_dist``  : (h, w) float32 all +inf (reserved)
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


def _horizontal_diff(m: numpy.ndarray) -> numpy.ndarray:
    """Cells whose left or right (wrapping) neighbour differs."""
    return (m != numpy.roll(m, -1, axis=1)) | (m != numpy.roll(m, 1, axis=1))


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
    m = merged.astype(numpy.int32)
    npl = int(merged.max()) + 1
    if is_ocean is None:
        is_ocean = [PLATE_OCEAN[i] if i < len(PLATE_OCEAN) else 0
                    for i in range(npl)]

    # ===================== 1. Boundary cells =====================
    # right/left neighbours wrap around x; up/down are bounded (like the
    # reference: up = y+1 guarded by up < h, dn = y-1 guarded by dn >= 0).
    is_bnd = _horizontal_diff(m)
    if h > 1:
        is_bnd[:-1] |= m[:-1] != m[1:]
        is_bnd[1:] |= m[1:] != m[:-1]

    # ===================== 2. Corner-lattice triple junctions =====================
    # Distinct plate count in each 2x2 corner block >= 3.
    is_jcor = numpy.zeros((h, w), dtype=bool)
    if h > 1:
        A = m[:-1]
        B = numpy.roll(m[:-1], -1, axis=1)
        C = m[1:]
        D = numpy.roll(m[1:], -1, axis=1)
        s = 1 + (B != A) + ((C != A) & (C != B)) + ((D != A) & (D != B) & (D != C))
        is_jcor[:-1] = s >= 3

    # ===================== 3. Corner -> 4 surrounding cells =====================
    is_jcell = numpy.zeros((h, w), dtype=bool)
    if h > 1:
        jy, jx = numpy.nonzero(is_jcor[:-1])
        is_jcell[jy, jx] = True
        is_jcell[jy, (jx + 1) % w] = True
        is_jcell[jy + 1, jx] = True
        is_jcell[jy + 1, (jx + 1) % w] = True

    # ===================== 4. Per-cell plate-pair key (fork detection) =====================
    # A boundary cell joins an arc only when *all* of its differing neighbours
    # bound the same plate pair (min*npl+max key).  Any disagreement -> -1.
    n_r = numpy.roll(m, -1, axis=1)
    n_l = numpy.roll(m, 1, axis=1)
    n_dn = numpy.zeros_like(m)   # neighbour at y-1
    n_up = numpy.zeros_like(m)   # neighbour at y+1
    if h > 1:
        n_up[:-1] = m[1:]
        n_dn[1:] = m[:-1]
    v_up = numpy.zeros((h, w), dtype=bool); v_up[:-1] = True
    v_dn = numpy.zeros((h, w), dtype=bool); v_dn[1:] = True

    def _pair(a, b):
        lo = numpy.minimum(a, b).astype(numpy.int64)
        hi = numpy.maximum(a, b).astype(numpy.int64)
        return lo * npl + hi

    pk = numpy.full((h, w), -1, dtype=numpy.int64)
    ok = numpy.ones((h, w), dtype=bool)
    # direction order matches the reference (right, up, down, left); the
    # outcome is order-independent but keep it for clarity.
    for valid, nk in ((numpy.ones((h, w), dtype=bool), n_r),
                      (v_up, n_up), (v_dn, n_dn),
                      (numpy.ones((h, w), dtype=bool), n_l)):
        kk = _pair(m, nk)
        diff = valid & (m != nk)
        newly = diff & (pk < 0)
        ok &= ~(diff & (pk >= 0) & (kk != pk))
        pk = numpy.where(newly, kk, pk)
    pair_key = numpy.where(ok & (pk >= 0), pk, numpy.int64(-1))
    # Triple-junction cells are forced to break the arc run.
    pair_key[is_jcell] = -1

    # ===================== 5. Flood-fill decompose arcs =====================
    arc_of_flat, arc_cells = _flood_fill_arcs(pair_key, h, w)
    n_arcs = len(arc_cells)

    # ===================== 6. Incident arcs at junctions (classification) =====
    # Faithful to world_gen.py: collect arc_of from the 4 orthogonal neighbours
    # of each junction cell.  At a triple point every corner cell only sees
    # <=2 arcs, so the triple-junction *voting* (step 7) is intentionally
    # inert and the red/blue balance is produced purely by the length
    # balancing (step 8) - exactly as in the reference.  (Triple-junction
    # *detection* for analysis uses a wider neighbourhood below.)
    jarcs = {}
    for jk in numpy.nonzero(is_jcell.reshape(-1))[0]:
        jk = int(jk)
        jx = jk % w
        jy = jk // w
        dirs = [((jx + 1) % w, jy), ((jx - 1 + w) % w, jy),
                (jx, jy + 1), (jx, jy - 1)]
        seen = set()
        for (nx, ny) in dirs:
            if ny < 0 or ny >= h:
                continue
            ni = ny * w + nx
            a = int(arc_of_flat[ni])
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
    pid_flat = m.reshape(-1)
    arc_plates = []
    for si in range(n_arcs):
        cells = arc_cells[si]
        pl_set = set()
        for v in pid_flat[cells]:
            pl_set.add(int(v))
        arc_plates.append(list(pl_set))
    pblue = [0] * npl
    pred = [0] * npl
    for si in range(n_arcs):
        for pk_plate in arc_plates[si]:
            if arc_type[si] == 1:
                pblue[pk_plate] += 1
            else:
                pred[pk_plate] += 1
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
    bt_ref_flat = numpy.zeros(w * h, dtype=numpy.int8)
    main_flat = numpy.zeros(w * h, dtype=numpy.int8)
    if n_arcs:
        lens = [len(c) for c in arc_cells]
        cells_cat = numpy.concatenate(arc_cells)
        bt_ref_flat[cells_cat] = numpy.repeat(numpy.asarray(arc_type, dtype=numpy.int8), lens)
        main_flat[cells_cat] = numpy.repeat(numpy.asarray(main, dtype=numpy.int8), lens)
    junction_mask_flat = numpy.zeros(w * h, dtype=numpy.int8)
    for jk, inc in jarcs.items():
        junction_mask_flat[jk] = 1
        red = False
        mk = False
        for si in inc:
            if arc_type[si] == 2:
                red = True
            if main[si] == 1:
                mk = True
        bt_ref_flat[jk] = 2 if red else 1
        if mk:
            main_flat[jk] = 1

    # ===================== 11. Light thickening (1 ring) =====================
    # Snapshot semantics of the reference loop: the source cells are visited in
    # raster order and each pushes its value onto its interior neighbours, so
    # for every target cell the *largest source index* wins.  Horizontal
    # wrapping makes that winning index column-dependent (the left neighbour of
    # x=0 is w-1, the right neighbour of x=w-1 is 0), hence explicit source
    # index arrays instead of a fixed direction order.
    src_bt = bt_ref_flat.reshape(h, w)
    src_mn = main_flat.reshape(h, w)
    out_bt = src_bt.copy()
    out_mn = src_mn.copy()

    def _shift_y(arr, down):
        out = numpy.zeros_like(arr)
        if down:      # neighbour at y+1, invalid on the last row
            out[:-1] = arr[1:]
        else:         # neighbour at y-1, invalid on the first row
            out[1:] = arr[:-1]
        return out

    idx = numpy.arange(w * h, dtype=numpy.int64).reshape(h, w)
    src_right = numpy.empty((h, w), dtype=numpy.int64)
    src_right[:, :w - 1] = idx[:, 1:]
    if w > 1:
        src_right[:, w - 1] = idx[:, w - 1] + 1 - w
    src_left = numpy.empty((h, w), dtype=numpy.int64)
    src_left[:, 1:] = idx[:, :-1]
    if w > 1:
        src_left[:, 0] = idx[:, 0] - 1 + w
    else:
        src_left[:, 0] = idx[:, 0]  # self-loop: neighbour == cell itself, never a boundary push

    interior = (src_bt == 0) & (~is_jcell)
    cand_idx = numpy.full((h, w), -1, dtype=numpy.int64)
    mn_or = numpy.zeros((h, w), dtype=numpy.int8)
    for s_idx, nbt, nmn in (
        (src_right, numpy.roll(src_bt, -1, axis=1), numpy.roll(src_mn, -1, axis=1)),
        (src_left, numpy.roll(src_bt, 1, axis=1), numpy.roll(src_mn, 1, axis=1)),
        (idx + w, _shift_y(src_bt, True), _shift_y(src_mn, True)),
        (idx - w, _shift_y(src_bt, False), _shift_y(src_mn, False)),
    ):
        take = interior & (nbt != 0) & (s_idx > cand_idx)
        out_bt[take] = nbt[take]
        cand_idx[take] = s_idx[take]
        # boundary_main uses sticky OR semantics in the reference: it is only
        # ever written *to 1* (`if boundary_main[i] == 1: boundary_main[ni] = 1`),
        # so a target is 1 when ANY boundary neighbour is a main chain.
        mn_or = numpy.maximum(mn_or, nmn)
    out_mn[interior & (mn_or != 0)] = 1

    # ===================== 11.5 Triple-junction *detection* (analysis only) ============
    # Independent of the classification above: a wide neighbourhood (the whole
    # 2x2 corner block + each block cell's 4-neighbours) actually reveals all
    # three radiating arcs at a triple point, so true triples (>=3 distinct
    # arcs) are captured here.  These are returned in ``junctions`` for the
    # topology analysis and never influence the boundary-type assignment.
    junctions = {}
    if h > 1:
        for cy, cx in numpy.argwhere(is_jcor[:-1]):
            cy = int(cy)
            cx = int(cx)
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
                    a = int(arc_of_flat[ni])
                    if a >= 0:
                        seen.add(a)
            if len(seen) >= 3:
                for (bx, by) in block:
                    cell_idx = by * w + bx
                    if is_jcell[by, bx]:
                        junctions[cell_idx] = list(seen)
    junction_count = len(junctions)

    # ===================== 12. Remap reference codes -> project codes =====================
    # reference: 1 = growth, 2 = subduction, 0 = interior
    # project  : DIVERGENT=2, CONVERGENT=1
    out = numpy.zeros((h, w), dtype=numpy.int8)
    out[out_bt == 1] = DIVERGENT
    out[out_bt == 2] = CONVERGENT

    is_bnd_np = is_bnd

    # ===================== 13. Distance maps =====================
    boundary_dist = _distance_to(is_bnd_np, h, w)
    convergent_dist = _distance_to(out == CONVERGENT, h, w)
    divergent_dist = _distance_to(out == DIVERGENT, h, w)
    transform_dist = numpy.full((h, w), numpy.inf, dtype=numpy.float32)

    return {
        "boundary_type": out,
        "boundary_dist": boundary_dist,
        "convergent_dist": convergent_dist,
        "divergent_dist": divergent_dist,
        "transform_dist": transform_dist,
        "boundary_main": out_mn.astype(numpy.int8),
        "junction_mask": junction_mask_flat.reshape(h, w),
        "junctions": junctions,
        "junction_count": junction_count,
        "is_ocean": numpy.array(is_ocean, dtype=numpy.int8),
        "plate_velocity": None,
    }


def _flood_fill_arcs(pair_key: numpy.ndarray, h: int, w: int):
    """Group boundary cells into arcs: 4-connected cells sharing one pair key.

    Returns ``(arc_of_flat, arc_cells)`` where ``arc_of_flat`` maps every flat
    cell index to its arc id (-1 = not an arc cell) and ``arc_cells[si]`` is the
    (sorted) flat cell indices of arc ``si``.

    Arc ids follow the reference BFS discovery order (an arc is numbered by the
    rank of its minimum cell index), so junction voting reproduces the pure
    Python implementation bit-for-bit.  Falls back to the reference BFS loop
    when SciPy is unavailable.
    """
    mask = pair_key >= 0
    if not mask.any():
        return numpy.full(h * w, -1, dtype=numpy.int32), []
    try:
        cells = numpy.nonzero(mask.reshape(-1))[0]
        keys = pair_key.reshape(-1)[cells]
        n = cells.size

        # Edges between 4-adjacent cells with the same pair key.  Horizontal
        # neighbours wrap (like the reference flood fill), vertical do not.
        pos = numpy.full(h * w, -1, dtype=numpy.int64)
        pos[cells] = numpy.arange(n, dtype=numpy.int64)
        y = cells // w
        x = cells % w
        key_flat = pair_key.reshape(-1)

        n_r = y * w + (x + 1) % w
        same_r = key_flat[n_r] == keys
        n_d = cells + w
        valid_d = (y + 1) < h
        same_d = numpy.zeros(n, dtype=bool)
        if h > 1:
            same_d[valid_d] = key_flat[n_d[valid_d]] == keys[valid_d]

        ei = numpy.concatenate([numpy.nonzero(same_r)[0], numpy.nonzero(same_d)[0]])
        ej = numpy.concatenate([pos[n_r[same_r]], pos[n_d[same_d]]])

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        graph = coo_matrix((numpy.ones(ei.size, dtype=numpy.int8), (ei, ej)),
                           shape=(n, n))
        n_comp, labels = connected_components(graph, directed=False)

        # Renumber: arc id = rank of the arc's minimum cell index (this equals
        # the reference's raster-order BFS discovery numbering).
        first = numpy.full(max(n_comp, 1), n, dtype=numpy.int64)
        numpy.minimum.at(first, labels, numpy.arange(n, dtype=numpy.int64))
        order = numpy.argsort(first, kind="stable")
        rank = numpy.empty(n_comp, dtype=numpy.int64)
        rank[order] = numpy.arange(n_comp, dtype=numpy.int64)

        arc_of_flat = numpy.full(h * w, -1, dtype=numpy.int32)
        arc_of_flat[cells] = rank[labels].astype(numpy.int32)

        arc_rank = rank[labels]
        sort_idx = numpy.lexsort((cells, arc_rank))
        sorted_rank = arc_rank[sort_idx]
        change = numpy.nonzero(numpy.diff(sorted_rank))[0] + 1
        arc_cells = [numpy.sort(cells[g]) for g in numpy.split(sort_idx, change)]
        return arc_of_flat, arc_cells
    except ImportError:
        return _flood_fill_arcs_python(pair_key, h, w)


def _flood_fill_arcs_python(pair_key: numpy.ndarray, h: int, w: int):
    """Reference iterative flood fill (fallback when SciPy is unavailable)."""
    keys = pair_key.reshape(-1).tolist()
    arc_of = [-1] * (w * h)
    arc_cells = []
    comp_id = 0
    for i in range(w * h):
        if keys[i] < 0 or arc_of[i] >= 0:
            continue
        apk = keys[i]
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
                if keys[ni] == apk and arc_of[ni] < 0:
                    arc_of[ni] = comp_id
                    cells.append(ni)
                    stack.append(ni)
        cells.sort()
        arc_cells.append(numpy.asarray(cells, dtype=numpy.int64))
        comp_id += 1
    return numpy.asarray(arc_of, dtype=numpy.int32), arc_cells


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
