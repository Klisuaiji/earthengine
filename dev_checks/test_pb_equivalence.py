"""Equivalence test: vectorized plate_boundaries vs the original pure-Python one.

The pristine pre-optimization implementation is kept in
``tests/reference_impls/plate_boundaries_orig.py``. Every output field must
match bit-for-bit.
"""
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import worldengine.plate_boundaries as new_mod

_ref = os.path.join(ROOT, "tests", "reference_impls", "plate_boundaries_orig.py")
if not os.path.isfile(_ref):
    _ref = os.path.join(os.environ["TEMP"], "plate_boundaries_orig.py")
spec = importlib.util.spec_from_file_location("pb_old", _ref)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)


def compare(merged, seed, tag):
    o = old_mod.classify_boundaries(merged, seed=seed)
    n = new_mod.classify_boundaries(merged, seed=seed)
    ok = True
    for key in ("boundary_type", "boundary_main", "junction_mask"):
        if not np.array_equal(o[key], n[key]):
            ok = False
            d = np.argwhere(o[key] != n[key])
            print(f"  [{tag}] seed={seed} {key} DIFFERS at {len(d)} cells, first={d[:3].tolist()}")
    for key in ("boundary_dist", "convergent_dist", "divergent_dist", "transform_dist"):
        if not np.array_equal(o[key], n[key], equal_nan=True):
            ok = False
            print(f"  [{tag}] seed={seed} {key} DIFFERS max|d|={np.abs(o[key]-n[key]).max()}")
    if set(o["junctions"].keys()) != set(n["junctions"].keys()):
        ok = False
        print(f"  [{tag}] seed={seed} junctions keys differ")
    else:
        for k in o["junctions"]:
            if list(o["junctions"][k]) != list(n["junctions"][k]):
                ok = False
                print(f"  [{tag}] seed={seed} junctions[{k}] differs: {o['junctions'][k]} vs {n['junctions'][k]}")
    if o["junction_count"] != n["junction_count"]:
        ok = False
        print(f"  [{tag}] seed={seed} junction_count {o['junction_count']} vs {n['junction_count']}")
    if not ok:
        print(f"FAIL [{tag}] seed={seed} shape={merged.shape}")
    return ok


def gen_map(h, w, k, rng):
    return rng.randint(0, k, size=(h, w)).astype(np.int32)


def main():
    rng = np.random.RandomState(20260101)
    failures = 0
    total = 0

    # ---- synthetic random plate maps (stress: junctions everywhere) ----
    for shape in [(1, 20), (5, 9), (16, 16), (37, 53), (64, 96)]:
        for k in (1, 2, 3, 5, 8, 12):
            for seed in (0, 1, 1234567):
                m = gen_map(*shape, k, rng)
                total += 1
                if not compare(m, seed, f"rand{shape}k{k}"):
                    failures += 1

    # ---- structured maps (few long arcs, like real Voronoi output) ----
    from scipy.ndimage import zoom
    for shape in [(48, 96), (64, 128), (100, 200)]:
        for k in (4, 6, 10):
            base = rng.randint(0, k, size=(shape[0] // 8, shape[1] // 8))
            m = np.round(zoom(base, (shape[0] / (shape[0] // 8), shape[1] / (shape[1] // 8)),
                              order=0)).astype(np.int32)
            for seed in (1, 42, 1234567):
                total += 1
                if not compare(m, seed, f"smooth{shape}k{k}"):
                    failures += 1

    # ---- real spherical-Voronoi merged maps ----
    from worldengine.spherical_voronoi import generate_spherical_world
    for (w, h) in [(64, 32), (128, 64)]:
        for seed in (1, 42, 777, 1234567):
            _, merged, _, _ = generate_spherical_world(seed, w=w, h=h, n_raw=30, n_big=6)
            total += 1
            if not compare(merged.astype(np.int32), seed, f"voronoi{w}x{h}"):
                failures += 1
    # legacy mode (n_big != 6)
    for seed in (1, 1234567):
        _, merged, _, _ = generate_spherical_world(seed, w=96, h=48, n_raw=10, n_big=4)
        total += 1
        if not compare(merged.astype(np.int32), seed, "voronoi-legacy"):
            failures += 1

    # ---- single plate / two plates ----
    total += 1
    if not compare(np.zeros((32, 64), dtype=np.int32), 5, "uniform"):
        failures += 1
    total += 1
    if not compare(np.where(np.arange(32 * 64).reshape(32, 64) % 64 < 32, 0, 1).astype(np.int32), 5, "halfsplit"):
        failures += 1

    print(f"\n{total - failures}/{total} cases bit-identical" if failures else
          f"\nALL {total} cases bit-identical")
    return 1 if failures else 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"elapsed {time.time() - t0:.1f}s")
    sys.exit(rc)
