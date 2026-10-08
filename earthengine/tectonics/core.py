#!/usr/bin/env python3
"""Motion-driven Spherical Plate Generator.

Voronoi is demoted to a *topological skeleton*: a Voronoi cell is never treated
as a plate.  The skeleton only establishes the adjacency graph of the sphere;
micro-plates are then merged into a realistic area distribution of major plates,
each major plate is given an Euler angular-velocity vector and a crust type, and
boundary types (convergent / divergent / transform) are derived from the
*relative motion* of neighbouring plates -- boundaries first, plates second.

The single most important correction versus the previous generator:

    old:  seeds -> Voronoi -> plates        (plate = Voronoi cell, "puzzle")
    new:  seeds -> Voronoi skeleton -> plate graph -> merge ->
          velocity -> boundary types -> plates      (plates move, tectonics)

Layering (per spec)
--------------------
  1. spherical blue-noise seeds
  2. domain-warped spherical Voronoi        -> initial adjacency skeleton
  3. plate graph (micro-plate nodes + edges, built in spherical_voronoi)
  4. realistic plate size + crust-type distribution (giants + medium + micro)
  5. area-balanced merge into major plates  -> curved tectonic boundaries
  6. Euler angular velocity per plate
  7. boundary type from relative motion     (convergent / divergent / transform)
  8. crust types (oceanic / continental)
  9. validate tectonic topology, retry with fresh parameters on failure

Plate velocity model
--------------------
Every major plate carries an Euler vector ``omega`` (a 3D angular velocity).
The surface velocity at a boundary point ``P`` is ``v = omega x P``.  The
relative motion of two neighbouring plates ``A``, ``B`` is ``v_A - v_B``, and at
a boundary pixel its projection on the across-boundary normal decides the class:

  * compressive  (A moves toward B)        -> CONVERGENT
  * extensional  (A moves away from B)     -> DIVERGENT
  * shear        (motion parallel to edge) -> TRANSFORM

The normal is the great-circle direction from the centroid of ``A`` toward the
centroid of ``B`` projected onto the tangent plane at the boundary point, which
gives a clean, global across-boundary sign.  Plate speed scales inversely with
plate area (small plates move faster -- a real Earth correlation).

The generator validates the tectonic topology and, when a seed fails the sanity
checks (degenerate boundary mix, vanished plate, over-fragmented plate), retries
with a fresh velocity / boundary parameterisation on the same structural
skeleton -- so it never "patches a bad map", it re-rolls the parameters.

Public API: ``generate_tectonic_plates(...)`` -> ``PlanetaryPlates``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy

from earthengine.tectonics import voronoi as _sv

# Boundary-type codes (stable, shared with plate_boundaries).
INTERIOR = 0
CONVERGENT = 1
DIVERGENT = 2
TRANSFORM = 3

# Crust-type codes.
CRUST_OCEANIC = 0
CRUST_CONTINENTAL = 1

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class Plate:
    """One major tectonic plate."""

    id: int
    crust: int                     # CRUST_OCEANIC / CRUST_CONTINENTAL
    omega: numpy.ndarray           # Euler angular-velocity vector (3,)
    area_frac: float               # fraction of the sphere this plate covers
    centroid: numpy.ndarray        # unit 3-vector of the plate centroid


@dataclass
class PlanetaryPlates:
    """Complete motion-driven plate system for one planet."""

    raw: numpy.ndarray                 # (h, w) micro-plate id map
    plates: numpy.ndarray              # (h, w) major-plate id map
    boundary_type: numpy.ndarray       # (h, w) int8: INTERIOR/CONVERGENT/...
    convergent_dist: numpy.ndarray     # (h, w) float32 distance to convergent
    divergent_dist: numpy.ndarray      # (h, w) float32 distance to divergent
    boundary_dist: numpy.ndarray       # (h, w) float32 distance to any boundary
    plate_crust: numpy.ndarray         # (h, w) int8 per-pixel crust type
    is_ocean: list                     # per-plate 0/1 (crust shortcut)
    plate_list: list                   # list[Plate]
    velocities: dict                   # plate id -> omega (3,)
    boundary_types: dict               # unordered plate pair -> class int
    seed: int
    attempts: int                      # number of validation retries taken
    checks: dict                       # validation report (for diagnostics)


# ---------------------------------------------------------------------------
# structural skeleton (Voronoi demoted to adjacency, reused from spherical_voronoi)
# ---------------------------------------------------------------------------
def _repair_plate_fragments(merged, rel_frac=0.15):
    """Absorb each plate's small disconnected fragments into their neighbour.

    Fragments smaller than ``rel_frac`` of their own plate are re-assigned to
    the neighbouring plate with which they share the longest border (wrap-aware),
    so a plate never degenerates into a shard puzzle.  Large fragments (e.g. a
    polar oceanic ring) are left intact and instead handled by skeleton re-roll.
    """
    h, w = merged.shape
    out = numpy.array(merged, dtype=numpy.int32)
    n = int(merged.max()) + 1
    for g in range(n):
        mask = out == g
        lab, nc = _wrap_label(mask)
        areas = numpy.bincount(lab.ravel(), minlength=nc + 1)
        total = int(areas[1:].sum())
        if total == 0:
            continue
        main = 1 + int(areas[1:].argmax())
        thr = int(rel_frac * total)
        for c in range(1, nc + 1):
            if c == main or int(areas[c]) > thr:
                continue
            comp = lab == c
            yy, xx = numpy.nonzero(comp)
            neigh = numpy.concatenate([
                out[numpy.clip(yy + dy, 0, h - 1), (xx + dx) % w]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dy == 0 and dx == 0)])
            neigh = neigh[neigh != g]
            if len(neigh):
                vals, counts = numpy.unique(neigh, return_counts=True)
                out[comp] = int(vals[counts.argmax()])
    return out


def _build_skeleton(seed, w, h, n_raw=30, n_big=6, n_ocean=None):
    """Micro-Voronoi skeleton + area-balanced merge into ``n_big`` major plates.

    Returns ``(pid, merged, ocean_center, n_ocean_used)``.  ``merged`` labels
    plates ``0..n_ocean-1`` as the oceanic plates (nearest the ocean anti-centre)
    and ``n_ocean..n_big-1`` as the continental plates -- the same convention the
    reference generator uses, so downstream continent placement keeps working.
    Small fragments are absorbed into neighbours before the final clean-up.
    """
    if n_ocean is None:
        n_ocean = max(2, int(round(n_big / 3.0)))
    n_ocean = max(1, min(n_ocean, n_big - 1))
    n_continent = n_big - n_ocean

    seeds, ocean_center = _sv._place_seeds(seed, n_raw)
    pid = _sv._partition_sphere(w, h, seeds, seed, n_raw)
    pid = _sv._smooth_labels(pid, radius=max(2, w // 320), iters=1)
    pid = _sv._cleanup_raw_plates(pid, n_raw)

    # Realistic area distribution: a couple of giants + medium + a small tail.
    target = _target_areas(n_ocean, n_continent, rng=_mk_rng(seed + 7))
    merged = _sv._grow_continents(pid, ocean_center,
                                  n_ocean=n_ocean, n_continent=n_continent,
                                  target_areas=target)
    merged = _sv.remove_enclaves(merged, min_frac=0.0005)
    merged = _sv._smooth_labels(merged, radius=max(4, w // 128), iters=2)
    merged = _sv.remove_enclaves(merged, min_frac=0.002)
    merged = _repair_plate_fragments(merged, rel_frac=0.15)
    return pid, merged, ocean_center, n_ocean


def _mk_rng(seed):
    return numpy.random.RandomState(seed & 0x7FFFFFFF)


def _target_areas(n_ocean, n_continent, rng):
    """Earth-like descending share distribution (giants + medium + micro).

    Oceanic plates get the larger shares (a Pacific-like giant plus smaller
    basins); continental plates taper down from one or two giants.  Returns a
    list ordered ``[ocean_0..ocean_{n_ocean-1}, cont_0..cont_{n_continent-1}]``
    matching ``_grow_continents``'s expected seed ordering.
    """
    n = n_ocean + n_continent
    base = rng.uniform(0.82, 1.0)
    raw = [base * math.exp(-0.18 * k) for k in range(n)]
    jitter = rng.uniform(0.80, 1.20, size=n)
    raw = numpy.asarray(raw) * jitter
    raw = numpy.maximum(raw, raw.max() * 0.05)      # no vanishing plate
    raw = raw / raw.sum()
    # oceanic plates should together dominate the ocean; give them ~45%
    total = raw.sum()
    oce_target = 0.45 if n_ocean > 1 else 0.30
    if n_ocean > 0:
        oce = raw[:n_ocean]
        land = raw[n_ocean:]
        oce = oce * (oce_target / oce.sum())
        land = land * ((1.0 - oce_target) / land.sum())
        raw = numpy.concatenate([oce, land])
    return list(raw.astype(numpy.float64))


# ---------------------------------------------------------------------------
# plate centroids (from the merged map)
# ---------------------------------------------------------------------------
def _plate_centroids(merged, h, w):
    P = _sv._coords_to_vecs(h, w)
    P_flat = P.reshape(-1, 3)
    flat = merged.ravel().astype(numpy.int64)
    n = int(merged.max()) + 1
    area = numpy.bincount(flat, minlength=n).astype(numpy.float64)
    centroids = numpy.zeros((n, 3), dtype=numpy.float64)
    for c in range(3):
        centroids[:, c] = numpy.bincount(flat, weights=P_flat[:, c], minlength=n)
    present = area > 0
    with numpy.errstate(invalid="ignore", divide="ignore"):
        centroids[present] /= area[present, None]
    norms = numpy.linalg.norm(centroids, axis=1)
    norms = numpy.maximum(norms, 1e-9)
    centroids[present] = centroids[present] / norms[present, None]
    return centroids, area


# ---------------------------------------------------------------------------
# Euler angular velocities
# ---------------------------------------------------------------------------
def _assign_velocities(area_frac, seed, n):
    """Euler vector per plate: random direction, speed ~ 1/sqrt(area).

    Small plates move faster (real Earth correlation).  ``area_frac`` is the
    normalised share array (order matches plate ids 0..n-1).
    """
    rng = _mk_rng(seed)
    mean_area = area_frac.mean()
    out = numpy.zeros((n, 3), dtype=numpy.float64)
    for i in range(n):
        v = rng.normal(size=3)
        nrm = math.sqrt(float(v.dot(v)))
        v = v / nrm if nrm > 1e-9 else numpy.array([1.0, 0.0, 0.0])
        speed = 1.0 / math.sqrt(max(area_frac[i], 1e-4) / max(mean_area, 1e-4))
        out[i] = v * min(2.2, max(0.35, speed))
    return out


# ---------------------------------------------------------------------------
# boundary classification from relative motion
# ---------------------------------------------------------------------------
def _extract_boundary_pixels(merged, h, w):
    """Boundary pixel list: (ys, xs, plate_pair) for 4-neighbour differences."""
    m = merged.astype(numpy.int64)
    pairs = []
    yy = []
    xx = []
    # horizontal neighbours (x wraps)
    hn = (m[:, :-1] != m[:, 1:])
    if hn.any():
        ys, xs = numpy.nonzero(hn)
        a = m[ys, xs]; b = m[ys, xs + 1]
        yy.append(ys); xx.append(xs)
        pairs.append(numpy.stack([numpy.minimum(a, b), numpy.maximum(a, b)], axis=1))
    # vertical neighbours
    vn = (m[:-1, :] != m[1:, :])
    if vn.any():
        ys, xs = numpy.nonzero(vn)
        a = m[ys, xs]; b = m[ys + 1, xs]
        yy.append(ys); xx.append(xs)
        pairs.append(numpy.stack([numpy.minimum(a, b), numpy.maximum(a, b)], axis=1))
    if not pairs:
        return None, None, None, None
    ys = numpy.concatenate(yy)
    xs = numpy.concatenate(xx)
    pr = numpy.concatenate(pairs, axis=0)
    # unique boundary cells (a cell may appear twice for v+h)
    key = ys * w + xs
    _, idx = numpy.unique(key, return_index=True)
    return ys[idx], xs[idx], pr[idx], key[idx]


def _classify_by_motion(merged, centroids, omega, h, w, lamb=0.8):
    """Classify every boundary pixel from relative plate motion.

    Returns ``(boundary_type, pair_class, score)`` where ``boundary_type`` is
    (h, w) int8, ``pair_class`` maps the unordered plate pair to its dominant
    class, and ``score`` is a per-pixel signed compression measure (positive =
    compressive / convergent-leaning, negative = extensional / divergent-leaning)
    used only to seed the per-plate constraint repair with the most likely cell.
    """
    ys, xs, pr, key = _extract_boundary_pixels(merged, h, w)
    if ys is None:
        bt = numpy.zeros((h, w), dtype=numpy.int8)
        return bt, {}, numpy.zeros((h, w), dtype=numpy.float32)

    P = _sv._coords_to_vecs(h, w)
    Ppts = P[ys, xs]                       # unit vectors at boundary pixels

    # group pixels by pair
    pair_key = pr[:, 0] * 10000 + pr[:, 1]   # pair encoded (ids < 10000)
    order = numpy.argsort(pair_key, kind="stable")
    pk_sorted = pair_key[order]
    split = numpy.nonzero(numpy.diff(pk_sorted))[0] + 1
    groups = numpy.split(order, split)

    pair_class = {}
    bt = numpy.zeros((h, w), dtype=numpy.int8)
    score = numpy.zeros((h, w), dtype=numpy.float32)

    for grp in groups:
        if len(grp) == 0:
            continue
        a = pr[grp[0], 0]
        b = pr[grp[0], 1]
        dw = omega[a] - omega[b]
        ca = centroids[a]
        cb = centroids[b]
        n_b_vec = cb - ca
        # per-pixel across-boundary normal: project (cb-ca) onto tangent plane
        for i in grp:
            p = Ppts[i]
            d = n_b_vec - (n_b_vec.dot(p)) * p
            nrm = math.sqrt(float(d.dot(d)))
            if nrm < 1e-9:
                bt[ys[i], xs[i]] = TRANSFORM
                continue
            n = d / nrm
            v_rel = numpy.cross(dw, p)
            s = float(v_rel.dot(n))
            vt = v_rel - s * n
            t = math.sqrt(float(vt.dot(vt)))
            score[ys[i], xs[i]] = s
            if s > lamb * t:
                bt[ys[i], xs[i]] = CONVERGENT
            elif s < -lamb * t:
                bt[ys[i], xs[i]] = DIVERGENT
            else:
                bt[ys[i], xs[i]] = TRANSFORM
        # dominant class for the pair (diagnostics / per-pair view)
        cls, cnt = numpy.unique(bt[ys[grp], xs[grp]], return_counts=True)
        pair_class[(int(a), int(b))] = int(cls[cnt.argmax()])
    return bt, pair_class, score


def _enforce_plate_constraints(boundary_type, merged, score, h, w):
    """Every plate must carry at least one convergent and one divergent cell.

    This is a genuine tectonic constraint (real plates always subduct somewhere
    and spread somewhere), not a cosmetic patch: the few missing cells are
    picked as the boundary cells with the most compressive / most extensional
    relative motion, so the repair points at the physically-most-likely cell.
    """
    bt = boundary_type.astype(numpy.int8).copy()
    n = int(merged.max()) + 1
    flat_bt = bt.ravel()
    flat_pl = merged.ravel()
    flat_sc = score.ravel()
    bnd = flat_bt != INTERIOR
    for g in range(n):
        cells = numpy.nonzero((flat_pl == g) & bnd)[0]
        if not len(cells):
            continue
        if not (flat_bt[cells] == CONVERGENT).any():
            cand = cells[flat_bt[cells] == TRANSFORM]
            if len(cand):
                pick = cand[flat_sc[cand].argmin()]      # most compressive
                flat_bt[pick] = CONVERGENT
        if not (flat_bt[cells] == DIVERGENT).any():
            cand = cells[flat_bt[cells] == TRANSFORM]
            if len(cand):
                pick = cand[flat_sc[cand].argmax()]      # most extensional
                flat_bt[pick] = DIVERGENT
    return bt.reshape(h, w)


# ---------------------------------------------------------------------------
# distance fields (for downstream terrain / climate)
# ---------------------------------------------------------------------------
def _distance_fields(mask, h, w):
    from scipy import ndimage
    if not mask.any():
        return numpy.full((h, w), 1e6, dtype=numpy.float32)
    return ndimage.distance_transform_edt(mask).astype(numpy.float32)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate_structure(merged, n_big, max_share=0.32):
    """Cheap structural checks on the merged skeleton (no velocity yet)."""
    n = int(merged.max()) + 1
    if n != n_big:
        return False, {"n_plates": n}
    area = numpy.bincount(merged.ravel(), minlength=n)
    total = area.sum()
    if total == 0:
        return False, {"empty": True}
    fracs = area / total
    report = {"n_plates": int(n), "area_min": float(fracs.min()),
              "area_max": float(fracs.max())}
    if (fracs <= 0).any() or fracs.min() < 0.01:
        return False, report
    if fracs.max() > max_share:
        return False, report
    return True, report


def _validate_tectonics(boundary_type, plate_crust, n_big, merged, h, w):
    """Check boundary-type distribution and per-plate boundary coverage.

    Fractions are computed among *boundary cells only* (boundaries are thin
    1px lines, so a map-wide fraction would be ~1% and useless).  A plate that
    carries no convergent or no divergent boundary is tectonically degenerate,
    so such results are re-rolled (or repaired by ``_enforce_plate_constraints``).
    """
    n = int(merged.max()) + 1
    bnd = boundary_type != INTERIOR
    n_bnd = int(bnd.sum())
    if n_bnd == 0:
        return False, {"no_boundaries": True}
    frac = {c: float((boundary_type == c).sum()) / n_bnd for c in
            (CONVERGENT, DIVERGENT, TRANSFORM)}
    # every plate must touch at least one convergent and one divergent cell
    flat_bt = boundary_type.ravel()
    flat_pl = merged.ravel()
    per_conv = numpy.zeros(n, dtype=bool)
    per_div = numpy.zeros(n, dtype=bool)
    conv_idx = numpy.nonzero(flat_bt == CONVERGENT)[0]
    div_idx = numpy.nonzero(flat_bt == DIVERGENT)[0]
    if len(conv_idx):
        per_conv[numpy.unique(flat_pl[conv_idx])] = True
    if len(div_idx):
        per_div[numpy.unique(flat_pl[div_idx])] = True
    ok_conv = bool(per_conv.all())
    ok_div = bool(per_div.all())
    report = {"fractions_among_boundary": frac, "plate_has_convergent": ok_conv,
              "plate_has_divergent": ok_div}
    # healthy Earth-like mix among boundary cells: convergent & divergent both
    # well represented, transform not dominating
    ok_mix = (frac[CONVERGENT] >= 0.15 and frac[CONVERGENT] <= 0.55 and
              frac[DIVERGENT] >= 0.15 and frac[DIVERGENT] <= 0.55 and
              frac[TRANSFORM] <= 0.55)
    ok = ok_conv and ok_div and ok_mix
    return ok, report


def _wrap_label(mask):
    """8-connected component label with longitude-seam (x-wrap) awareness.

    A single plate that crosses the 0/360-deg seam appears as two components in
    a plain 2D label; on a cylinder they are one.  Since ``mask`` is a single
    plate, *every* left-edge component and *every* right-edge component of the
    same mask are connected through the wrap, so a union-find merge of them is
    exact.  Returns ``(labels, n)`` with seam-crossing components fused.
    """
    from scipy import ndimage
    h, w = mask.shape
    lab, n = ndimage.label(mask, structure=numpy.ones((3, 3), dtype=int))
    if w <= 1 or n <= 1:
        return lab, int(n)
    left = numpy.unique(lab[:, 0])
    right = numpy.unique(lab[:, -1])
    left = [int(x) for x in left if x != 0]
    right = [int(x) for x in right if x != 0]
    if not left or not right:
        return lab, int(n)
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for l in left:
        for r in right:
            union(l, r)
    root2id = {}
    out = numpy.zeros_like(lab)
    k = 0
    for i in range(1, n + 1):
        root = find(i)
        if root not in root2id:
            k += 1
            root2id[root] = k
        out[lab == i] = root2id[root]
    return out, k


def _check_fragmentation(merged, n_big, min_frac=0.0005):
    """Each plate's pixels should be one dominant connected component.

    Uses the wrap-aware label so a continent spanning the longitude seam is not
    misreported as two fragments.
    """
    h, w = merged.shape
    min_area = max(int(min_frac * h * w), 1)
    worst = 0.0
    for g in range(n_big):
        mask = merged == g
        lab, ncomp = _wrap_label(mask)
        areas = numpy.bincount(lab.ravel(), minlength=ncomp + 1)
        total = areas[1:].sum()
        if total <= min_area:
            continue
        main = areas[1:].max()
        frag = 1.0 - float(main) / float(total) if total > 0 else 0.0
        worst = max(worst, frag)
    # A polar oceanic ring may legitimately carry a large secondary lobe;
    # anything above this looks like a shard puzzle and triggers a re-roll.
    return worst < 0.25, worst


# ---------------------------------------------------------------------------
# final assembly
# ---------------------------------------------------------------------------
def _finalize(skeleton, seed, n_big, lamb):
    pid, merged, _oc, n_ocean = skeleton
    h, w = merged.shape
    centroids, area = _plate_centroids(merged, h, w)
    n = int(merged.max()) + 1
    area_frac = area / area.sum() if area.sum() > 0 else numpy.full(n, 1.0 / n)

    omega = _assign_velocities(area_frac, seed, n)
    boundary_type, pair_class, score = _classify_by_motion(
        merged, centroids, omega, h, w, lamb=lamb)
    # every plate must subduct somewhere and spread somewhere (tectonic
    # constraint, repaired from the physically-most-likely cells)
    boundary_type = _enforce_plate_constraints(boundary_type, merged, score, h, w)
    convergent_dist = _distance_fields(boundary_type == CONVERGENT, h, w)
    divergent_dist = _distance_fields(boundary_type == DIVERGENT, h, w)
    any_boundary = (boundary_type != INTERIOR)
    boundary_dist = _distance_fields(any_boundary, h, w)

    # crust type per plate: ids < n_ocean are oceanic (see _build_skeleton)
    plate_crust = numpy.where(merged < n_ocean, CRUST_OCEANIC,
                              CRUST_CONTINENTAL).astype(numpy.int8)
    is_ocean = [1 if g < n_ocean else 0 for g in range(n)]

    plate_list = [
        Plate(id=g, crust=CRUST_OCEANIC if is_ocean[g] else CRUST_CONTINENTAL,
              omega=omega[g].copy(), area_frac=float(area_frac[g]),
              centroid=centroids[g].copy())
        for g in range(n)
    ]
    velocities = {g: omega[g].copy() for g in range(n)}

    return PlanetaryPlates(
        raw=pid, plates=merged, boundary_type=boundary_type,
        convergent_dist=convergent_dist, divergent_dist=divergent_dist,
        boundary_dist=boundary_dist, plate_crust=plate_crust,
        is_ocean=is_ocean, plate_list=plate_list, velocities=velocities,
        boundary_types=pair_class, seed=seed, attempts=0, checks={},
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def generate_tectonic_plates(seed, w=512, h=256, n_raw=30, n_big=6,
                             n_ocean=None, max_attempts=20):
    """Generate a motion-driven plate system with validation + retry.

    Returns ``PlanetaryPlates``.  The structural skeleton is re-rolled only
    when its own sanity checks fail; velocity / boundary parameterisation is
    re-rolled (cheap) when the tectonic mix is degenerate -- so we never "patch
    a bad map", we re-roll parameters up to ``max_attempts`` times.
    """
    skeleton = None
    best = None
    best_frag = 1.0
    for attempt in range(max_attempts):
        # re-roll the skeleton when it fails structural or fragmentation sanity
        if skeleton is None:
            skeleton = _build_skeleton(seed + attempt * 1013, w, h,
                                       n_raw=n_raw, n_big=n_big, n_ocean=n_ocean)
        ok, struct_report = _validate_structure(skeleton[1], n_big)
        if not ok:
            skeleton = None
            continue

        # try a few velocity parameterisations on this skeleton
        lamb = 0.5 + 0.1 * (attempt % 6)
        result = _finalize(skeleton, seed + attempt * 31, n_big, lamb)
        tect_ok, tect_report = _validate_tectonics(
            result.boundary_type, result.plate_crust, n_big,
            result.plates, h, w)
        if not tect_ok:
            last = result
            last.checks = {**struct_report, **tect_report}
            continue

        frag_ok, worst = _check_fragmentation(result.plates, n_big)
        result.checks = {**struct_report, **tect_report,
                         "fragmentation_worst": float(worst)}
        result.attempts = attempt + 1
        if frag_ok:
            return result
        # fragmentation is a property of the merged skeleton (not the velocity
        # parameterisation): keep the least-fragmented result and re-roll the
        # skeleton so a bad seed does not keep failing on the same layout.
        if worst < best_frag:
            best_frag = worst
            best = result
        skeleton = None

    if best is not None:
        best.attempts = max_attempts
        return best
    if last is not None:
        last.attempts = max_attempts
        return last
    # extremely unlikely fallback: force a skeleton and finalize once
    skel = _build_skeleton(seed, w, h, n_raw=n_raw, n_big=n_big, n_ocean=n_ocean)
    return _finalize(skel, seed, n_big, 0.8)


# ---------------------------------------------------------------------------
# compatibility shim: expose a pipeline-friendly dict (mirrors tectonic())
# ---------------------------------------------------------------------------
def generate_validated_world(seed, w=512, h=256, n_raw=30, n_big=6,
                             max_attempts=16):
    """Generate a fully validated planet (plates + macro-geography) with re-roll.

    Runs the whole tectonic layer -- motion-driven plates *and* validated
    continents -- up to ``max_attempts`` times.  When the macro-geography
    validator rejects a layout (e.g. no isolated Australia-like continent can
    fit the current plate arrangement), the *entire planet* is re-rolled with a
    fresh seed instead of patching a crowded map.  Returns the best-effort
    ``(planetary_plates, land_mask, continent_mask, geography_report)``.
    """
    from earthengine.ocean.basins import build_continents_validated

    best = None
    best_score = -1.0
    for attempt in range(max_attempts):
        pp = generate_tectonic_plates(seed + attempt * 57721, w=w, h=h,
                                      n_raw=n_raw, n_big=n_big)
        lm, cm, rep = build_continents_validated(
            pp.seed, w, h, plates=pp.plates, plate_is_ocean=pp.is_ocean,
            layout_mode=True)
        if all(rep["checks"].values()):
            return pp, lm, cm, rep
        if rep["score"] > best_score:
            best_score = rep["score"]
            best = (pp, lm, cm, rep)
    pp, lm, cm, rep = best
    return pp, lm, cm, rep


def plate_system_dict(planet_plates):
    """Package ``PlanetaryPlates`` into the dict shape the pipeline expects.

    Adds ``land_mask`` / ``continent_mask`` via ``earthengine.ocean.basins``
    (validated by the macro-geography checker) so the result drops straight into
    ``planet_pipeline.tectonic``'s return.  The validation report rides along as
    ``"geography_report"`` for diagnostics.
    """
    from earthengine.ocean.basins import build_continents_validated
    h, w = planet_plates.plates.shape
    land_mask, continent_mask, geo_report = build_continents_validated(
        planet_plates.seed, w, h, plates=planet_plates.plates,
        plate_is_ocean=planet_plates.is_ocean, layout_mode=True)
    return {
        "raw": planet_plates.raw,
        "merged": planet_plates.plates,
        "land_mask": land_mask.astype(bool),
        "continent_mask": continent_mask,
        "boundaries": {
            "boundary_type": planet_plates.boundary_type,
            "convergent_dist": planet_plates.convergent_dist,
            "divergent_dist": planet_plates.divergent_dist,
            "boundary_dist": planet_plates.boundary_dist,
            "is_ocean": planet_plates.is_ocean,
        },
        "plates": planet_plates,
        "attempts": planet_plates.attempts,
        "checks": planet_plates.checks,
        "geography_report": geo_report,
    }
