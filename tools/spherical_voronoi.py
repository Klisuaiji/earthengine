#!/usr/bin/env python3
# Spherical Voronoi plate generation with domain warping.
# Generates n_small plates on a sphere via domain-warped Voronoi,
# then selects continental nuclei and grows them into n_big plates.
#
# Key: only DOMAIN_WARP (smooth curved boundaries), NO per-pixel detail noise.
#      Continental nuclei selected as cells farthest from ocean anti-center.

import math
import numpy
from PIL import Image
from collections import deque

TAU = 2.0 * math.pi

DOMAIN_AMP = 0.06
DOMAIN_FREQ = 2.5
PARTITION_ITERS = 12  # sub-center angular distance from main seed


# --------------- Noise field (low-res FBM -> PIL upscale) ---------------
def _noise_field(shape, seed, octaves=4, res=64):
    """Low-resolution FBM field upscaled to shape via PIL BILINEAR. Returns (-1, +1)."""
    def _hash2(ix, iy, s):
        h = s & 0x7fffffff
        h = (h ^ (ix * 374761393)) & 0x7fffffff
        h = (h ^ (iy * 668265263)) & 0x7fffffff
        h = (h * 1274126177) & 0x7fffffff
        h = (h ^ (h >> 13)) & 0x7fffffff
        return numpy.float32(h) / 2147483647.0
    _hash2v = numpy.vectorize(_hash2, otypes=[numpy.float32])

    fbm = numpy.zeros((res, res), dtype=numpy.float32)
    amp = numpy.float32(1.0)
    fr = numpy.float32(res)
    norm = numpy.float32(0.0)
    for o in range(octaves):
        y_idx, x_idx = numpy.mgrid[0:res, 0:res].astype(numpy.float32)
        y_idx *= fr
        x_idx *= fr
        n = _hash2v(x_idx.astype(int), y_idx.astype(int), seed + o * 1013) * 2.0 - 1.0
        fbm += amp * n
        norm += amp
        amp *= 0.5
        fr *= 2.0
    fbm /= numpy.maximum(norm, 1e-6)

    img = Image.fromarray(((fbm + 1.0) * 127.5).astype(numpy.uint8))
    img = img.resize(shape, Image.BILINEAR)
    return numpy.asarray(img, dtype=numpy.float32) / 127.5 - 1.0


# --------------- Spherical geometry ---------------
def _coords_to_vecs(h, w):
    lat = (numpy.arange(h, dtype=numpy.float32) / h - 0.5) * math.pi
    lon = (numpy.arange(w, dtype=numpy.float32) / w) * TAU
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


# --------------- Seed placement on sphere ---------------
def _place_seeds(seed, n_plates):
    """Farthest-point sampling on sphere. 2 equatorial seeds (plates 4,5) placed first,
    then remaining seeds placed at maximum distance from all existing seeds."""
    rng = numpy.random.RandomState((seed * 2654435761) & 0xFFFFFFFF)
    seeds = [None] * n_plates

    # Ocean center: random point near equator
    clat_c = rng.rand() * math.radians(20.0)
    lon_c = (rng.rand() * 2.0 - 1.0) * math.pi
    cl, sl = math.cos(clat_c), math.sin(clat_c)
    C = numpy.array([cl * math.cos(lon_c), sl, cl * math.sin(lon_c)])

    # Two ocean seeds at ±offset from center
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
        for _ in range(400):
            la = (rng.rand() * 2.0 - 1.0) * math.pi * 0.5
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
def _partition_sphere(w, h, seeds, seed, n_plates):
    """Numpy-batched spherical Voronoi with domain warping and pressure relaxation.

    Distances are recomputed per-iteration in row chunks so memory stays bounded
    even at very high resolutions (4096x2048+). No sub-centers — with 30+ plates,
    boundaries are already fine-grained.
    """
    s2 = (seed * 2654435761) & 0x7fffffff

    P = _coords_to_vecs(h, w)
    W = _warp_dirs(P, s2, h, w)
    del P

    # Seeds as flat array (n_plates, 3)
    seeds_arr = numpy.array(seeds, dtype=numpy.float32)

    chunk_rows = 64
    pressure = numpy.zeros(n_plates, dtype=numpy.float32)
    target = numpy.full(n_plates, 1.0 / n_plates, dtype=numpy.float32)
    pid = numpy.zeros((h, w), dtype=numpy.int32)

    for it in range(PARTITION_ITERS):
        # Recompute distances chunk by chunk (memory bounded, no D_all cache)
        for y0 in range(0, h, chunk_rows):
            y1 = min(y0 + chunk_rows, h)
            W_chunk = W[y0:y1]  # (chunk, w, 3)
            dot = numpy.clip(numpy.tensordot(W_chunk, seeds_arr, axes=([2], [1])), -1.0, 1.0)
            dist = numpy.arccos(dot)
            score = -dist - pressure[None, None, :]
            pid[y0:y1] = score.argmax(axis=2).astype(numpy.int32)

        counts = numpy.bincount(pid.ravel(), minlength=n_plates).astype(numpy.float32)

        # Fill empty plates with a few pixels from largest neighbor
        for k in range(n_plates):
            if counts[k] == 0:
                # Find the most overrepresented plate
                excess = counts - (w * h) / n_plates
                donor = excess.argmax()
                if counts[donor] > 2:
                    # Get donor region boundary and reassign 1% to empty plate
                    donor_mask = pid == donor
                    ys, xs = numpy.where(donor_mask)
                    if len(ys) > 0:
                        n_reassign = max(1, len(ys) // 100)
                        idx = numpy.random.choice(len(ys), n_reassign, replace=False)
                        pid_copy = pid.copy()
                        pid_copy[ys[idx], xs[idx]] = k
                        pid = pid_copy
                        counts = numpy.bincount(pid.ravel(), minlength=n_plates).astype(numpy.float32)

        frac = counts / (w * h)
        pressure += (frac - target) * 4.0

    return pid


# --------------- Compute raw plate adjacency ---------------
def _build_adjacency(pid, n_raw):
    """Vectorized raw plate adjacency graph from neighbor differences."""
    h, w = pid.shape
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
def _grow_continents(pid, ocean_center, n_ocean=2, n_continent=4):
    """Select continental nuclei and grow into large regions.

    Ocean seeds: plates whose 3D centroid is closest to the ocean center.
    Continental nuclei: plates farthest from ocean seeds via graph distance.
    Growth: multi-source area-balanced BFS along adjacency graph.
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

    # Ocean seeds: n_ocean plates with pixels closest to ocean_center
    dots = numpy.clip(centroids_3d @ ocean_center, -1.0, 1.0)
    ang = numpy.arccos(dots)
    ang[~plate_present] = numpy.inf
    ocean_seeds = [int(p) for p in numpy.argsort(ang)[:n_ocean]]
    assert len(ocean_seeds) == n_ocean, f"Not enough valid oceanic plates: {len(ocean_seeds)}"

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
        land_nuclei.extend([p for p in remaining if p not in land_nuclei][:n_continent - len(land_nuclei)])

    seeds_all = ocean_seeds + land_nuclei
    assigned = {s: i for i, s in enumerate(seeds_all)}
    areas = [int(plate_area[s]) for s in seeds_all]

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
            best_g = min(neighbor_areas, key=neighbor_areas.get)
            assigned[p] = best_g
            areas[best_g] += int(plate_area[p])
            for nxt in adj[p]:
                if nxt not in assigned and nxt not in border:
                    border.add(nxt)

    for p in range(n_raw):
        if p not in assigned:
            assigned[p] = 0

    merged = numpy.empty_like(pid)
    for p, g in assigned.items():
        merged[pid == p] = g
    return merged


# --------------- Connected components ---------------
def _label_components(mask):
    """8-connected component labelling (vectorized with scipy)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=numpy.ones((3, 3), dtype=int))
    return lab.astype(numpy.int32), int(n)


# --------------- Raw plate cleanup ---------------
def _cleanup_raw_plates(pid, n_raw):
    """Keep only the largest 8-connected component of each raw plate.
    Reassign small components to the most common neighboring plate."""
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
        comp_areas = [int((lab == c).sum()) for c in range(1, n_comp + 1)]
        largest = 1 + comp_areas.index(max(comp_areas))
        min_area = max(int(min_frac * h * w), 8)
        for c in range(1, n_comp + 1):
            if c == largest:
                continue
            comp = lab == c
            if int(comp.sum()) < min_area:
                ys, xs = numpy.where(comp)
                neigh = numpy.concatenate([
                    out[numpy.clip(ys + dy, 0, h - 1), numpy.clip(xs + dx, 0, w - 1)]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if not (dy == 0 and dx == 0)
                ])
                neigh = neigh[neigh != p]
                if len(neigh):
                    vals, counts = numpy.unique(neigh, return_counts=True)
                    out[comp] = int(vals[counts.argmax()])
    return out


# --------------- Main entry ---------------
def generate_spherical_plates(seed, w=512, h=512, n_raw=30, n_big=6):
    seeds, ocean_center = _place_seeds(seed, n_raw)
    pid = _partition_sphere(w, h, seeds, seed, n_raw)
    pid = _cleanup_raw_plates(pid, n_raw)
    merged = _grow_continents(pid, ocean_center, n_ocean=2, n_continent=n_big - 2)
    return pid, merged
