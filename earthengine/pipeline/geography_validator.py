#!/usr/bin/env python3
"""Macro-geography validator for global continents.

After continents are generated they must pass a set of Earth-like structural
checks *before* they are handed to terrain / climate.  If a layout fails, the
caller re-rolls the layout parameters (fresh seed / share jitter) instead of
patching the map -- generate -> validate -> fail -> re-roll -> regenerate.

Checks (from the spec's P0 list)
--------------------------------
1. Land fraction within ``[lo, hi]`` (default 25-35 %, Earth-like).
2. Main-continent count within ``[4, 7]``.
3. A dominant supercontinent exists: ``largest_continent >= 25 %`` of land.
4. At least one *isolated* continent (Australia-like): its nearest-neighbour
   coast-to-coast angular gap to every other continent exceeds a threshold,
   i.e. it is surrounded by real open ocean, not a peninsular stub.
5. A large ocean basin exists: the largest connected ocean component covers a
   meaningful share of the sphere (no fully land-locked tiny seas).
6. Continents are cohesive (no shard puzzles): each continent is one dominant
   connected component, measured with longitude-seam awareness.

Public API: ``validate_global_geography(land_mask, continent_mask)`` ->
``(ok, report)``.  ``report`` carries every measured metric plus a normalised
``score`` (higher = closer to Earth-like) so a re-roll loop can pick the
best-effort layout when a seed keeps failing.
"""

from __future__ import annotations

import math

import numpy

from earthengine.tectonics.core import _wrap_label

TAU = 2.0 * math.pi

# ---- Earth-like thresholds ------------------------------------------------
LAND_LO, LAND_HI = 0.25, 0.35
CONT_MIN, CONT_MAX = 4, 7
LARGEST_MIN = 0.25          # largest continent must be >= 25% of land
ISOLATION_DEG = 5.0         # min coast-to-coast gap (deg) to all others
OCEAN_MIN = 0.25            # largest ocean basin >= 25% of the sphere
FRAG_MAX = 0.03             # mainland may not be more than 3% sharded
ISLET_FRAC = 0.04           # components <4% of a continent are islets, not shards

_SUB = 3                    # cell subsampling step for the isolation tree


def _sphere_pts(h, w, ys, xs):
    """Unit vectors for grid cells (lat/lon convention matching continents)."""
    lat = (0.5 - (ys + 0.5) / h) * math.pi
    lon = (xs + 0.5) / w * TAU
    cl = numpy.cos(lat)
    return numpy.stack([cl * numpy.cos(lon), numpy.sin(lat), cl * numpy.sin(lon)],
                       axis=1).astype(numpy.float32)


def _largest_ocean_fraction(land_mask):
    """Fraction of the sphere in the single largest connected ocean basin."""
    ocean = ~land_mask
    lab, n = _wrap_label(ocean)
    areas = numpy.bincount(lab.ravel(), minlength=n + 1)
    return float(areas[1:].max()) / float(ocean.size) if n else 0.0


def _continent_fragmentation(continent_mask, labels):
    """Worst *mainland* fragmentation per continent.

    Small disconnected pieces below ``ISLET_FRAC`` of a continent's own area are
    deliberate islands (shelf islets, volcanic arcs, hotspot chains) and are
    ignored -- the metric measures only whether the mainland body itself is split
    into multiple large shards.
    """
    worst = 0.0
    for g in labels:
        lab, n = _wrap_label(continent_mask == g)
        areas = numpy.bincount(lab.ravel(), minlength=n + 1)
        total = float(areas[1:].sum())
        if total <= 0:
            continue
        # significant components: those not tiny islets
        sig = areas[1:]
        significant = sig[sig >= ISLET_FRAC * total]
        if significant.size == 0:
            continue
        main = significant.max()
        frag = 1.0 - float(main) / float(significant.sum())
        worst = max(worst, frag)
    return worst


def _mainland_cells(continent_mask, g, h, w):
    """Subsampled unit-vector cells of continent ``g``'s *mainland* body.

    Islets below ``ISLET_FRAC`` of the continent (shelf islets, arc fragments,
    polar flecks) are excluded so coast-to-coast gaps measure the mainland body
    only -- otherwise a tiny islet near a neighbour drags the gap to ~0.
    """
    lab, n = _wrap_label(continent_mask == g)
    areas = numpy.bincount(lab.ravel(), minlength=n + 1)
    total = float(areas[1:].sum())
    if total <= 0:
        return numpy.zeros((0, 3), dtype=numpy.float32)
    thr = ISLET_FRAC * total
    keep = numpy.zeros(lab.shape, dtype=bool)
    for c in range(1, n + 1):
        if areas[c] >= thr:
            keep |= lab == c
    yy, xx = numpy.nonzero(keep)
    if len(yy) == 0:
        return numpy.zeros((0, 3), dtype=numpy.float32)
    yy, xx = yy[::_SUB], xx[::_SUB]
    return _sphere_pts(h, w, yy, xx)


def _isolation_scores(land_mask, continent_mask, labels):
    """Per-continent mainland coast-to-coast angular gap (rad) to the nearest
    other continent.  Returns ``{label: min_angular_gap}`` and the global max
    (the Australia-like island continent).
    """
    from scipy.spatial import cKDTree
    h, w = land_mask.shape
    cells = {g: _mainland_cells(continent_mask, g, h, w)
             for g in labels}
    gaps = {}
    for g, pts in cells.items():
        if len(pts) == 0:
            gaps[g] = 0.0
            continue
        others = numpy.concatenate([cells[o] for o in cells if o != g], axis=0)
        if len(others) == 0:
            gaps[g] = math.pi
            continue
        tree = cKDTree(others)
        d, _ = tree.query(pts, k=1)
        gaps[g] = float(d.min())
    return gaps, max(gaps.values()) if gaps else 0.0


def validate_global_geography(land_mask, continent_mask, land_target=None,
                              verbose_report=True):
    """Return ``(ok, report)``.

    ``land_target`` (optional) is the intended land fraction; when supplied the
    land-fraction window is tightened around it so the re-roll aims at the
    requested coverage instead of the generic 25-35 % range.
    """
    h, w = land_mask.shape
    land = land_mask.astype(bool)
    total_land = int(land.sum())
    land_frac = float(total_land) / float(land.size)

    labels = numpy.unique(continent_mask[land])
    labels = labels[labels >= 0]
    n_cont = int(len(labels))

    # 只对非负标签计数（-1 = 未分配 / 海洋）
    valid = continent_mask[land] >= 0
    shares = numpy.zeros(max(1, int(continent_mask.max()) + 1), dtype=float)
    if total_land > 0 and valid.any():
        shares = numpy.bincount(continent_mask[land][valid].astype(numpy.int64),
                                minlength=shares.size)
        shares = shares / total_land
    largest = float(shares.max()) if shares.size else 0.0

    frag = _continent_fragmentation(continent_mask, labels)
    gaps, iso_max = _isolation_scores(land_mask, continent_mask, labels)
    ocean = _largest_ocean_fraction(land_mask)

    # per-continent gap to every OTHER continent (exclude self from the min)
    iso_others = {}
    for g, gap in gaps.items():
        others = [v for k, v in gaps.items() if k != g]
        iso_others[g] = min(others) if others else math.pi

    lo, hi = LAND_LO, LAND_HI
    if land_target is not None:
        lo = max(LAND_LO, land_target - 0.035)
        hi = min(LAND_HI, land_target + 0.035)

    # ---- checks -----------------------------------------------------------
    c_land = lo <= land_frac <= hi
    c_count = CONT_MIN <= n_cont <= CONT_MAX
    c_largest = largest >= LARGEST_MIN
    c_iso = iso_max >= math.radians(ISOLATION_DEG)
    c_ocean = ocean >= OCEAN_MIN
    c_frag = frag <= FRAG_MAX

    ok = c_land and c_count and c_largest and c_iso and c_ocean and c_frag

    # normalised score (each sub-metric 0..1, higher = more Earth-like)
    def _clip01(x):
        return max(0.0, min(1.0, x))

    s_land = 1.0 - _clip01(abs(land_frac - (land_target if land_target else 0.30)) / 0.08)
    s_count = 1.0 if c_count else 0.3
    s_largest = _clip01(largest / LARGEST_MIN)
    s_iso = _clip01(math.degrees(iso_max) / ISOLATION_DEG)
    s_ocean = _clip01(ocean / OCEAN_MIN)
    s_frag = 1.0 - _clip01(frag / FRAG_MAX * 0.5)
    score = 0.22 * s_land + 0.10 * s_count + 0.20 * s_largest + 0.22 * s_iso + \
            0.16 * s_ocean + 0.10 * s_frag

    report = {
        "land_fraction": land_frac, "n_continents": n_cont,
        "largest_share": largest, "isolation_deg": math.degrees(iso_max),
        "isolation_by_continent": {int(k): round(math.degrees(v), 1)
                                   for k, v in iso_others.items()},
        "largest_ocean_fraction": ocean, "fragmentation_worst": frag,
        "checks": {"land": c_land, "count": c_count, "largest": c_largest,
                   "isolation": c_iso, "ocean": c_ocean, "fragmentation": c_frag},
        "score": score,
    }
    return ok, report


def best_geography(reports):
    """Pick the highest-scoring layout from a list of ``(ok, report)``."""
    if not reports:
        return None, None
    return max(reports, key=lambda r: r[1]["score"])
