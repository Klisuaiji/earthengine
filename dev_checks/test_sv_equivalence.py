"""Equivalence test: optimized spherical_voronoi vs the pristine GitHub version."""
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import worldengine.spherical_voronoi as new_mod

_ref = os.path.join(ROOT, "tests", "reference_impls", "spherical_voronoi_orig.py")
if not os.path.isfile(_ref):
    _ref = os.path.join(os.environ["TEMP"], "spherical_voronoi_orig.py")
spec = importlib.util.spec_from_file_location("sv_old", _ref)
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)


def compare(seed, w, h, n_raw, n_big, tag):
    o = old_mod.generate_spherical_world(seed, w=w, h=h, n_raw=n_raw, n_big=n_big)
    n = new_mod.generate_spherical_world(seed, w=w, h=h, n_raw=n_raw, n_big=n_big)
    names = ("pid/raw", "merged", "land_mask", "continent_mask")
    ok = True
    for name, a, b in zip(names, o, n):
        if not np.array_equal(a, b):
            ok = False
            print(f"  [{tag}] seed={seed} {name} DIFFERS ({int((np.asarray(a) != np.asarray(b)).sum())} cells)")
    # elevation synthesis as well
    oe = old_mod.synthesize_elevation(o[1], h, w, seed, n_big=n_big, land_mask=o[2], continent_mask=o[3])
    ne = new_mod.synthesize_elevation(n[1], h, w, seed, n_big=n_big, land_mask=n[2], continent_mask=n[3])
    if not np.array_equal(oe, ne):
        ok = False
        print(f"  [{tag}] seed={seed} elevation DIFFERS max|d|={np.abs(oe - ne).max()}")
    if not ok:
        print(f"FAIL [{tag}] {w}x{h} n_raw={n_raw} n_big={n_big}")
    return ok


def main():
    cases = [
        (1, 64, 32, 30, 6), (42, 64, 32, 30, 6), (1234567, 128, 64, 30, 6),
        (777, 128, 64, 30, 6), (2026, 256, 128, 30, 6),
        (1, 96, 48, 10, 4), (9, 96, 48, 12, 8),      # legacy / non-reference modes
        (5, 64, 32, 6, 3),                            # minimum plate counts
    ]
    failures = 0
    for seed, w, h, n_raw, n_big in cases:
        t0 = time.perf_counter()
        ok = compare(seed, w, h, n_raw, n_big, "voronoi")
        print(f"{'ok  ' if ok else 'FAIL'} {w}x{h} seed={seed} n_raw={n_raw} n_big={n_big} "
              f"({time.perf_counter() - t0:.2f}s)")
        failures += 0 if ok else 1

    # speed check at 1024x512 (cache path active: 1024*512*30 = 15.7M elems)
    t0 = time.perf_counter()
    old_mod.generate_spherical_world(1234567, w=1024, h=512, n_raw=30, n_big=6)
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    new_mod.generate_spherical_world(1234567, w=1024, h=512, n_raw=30, n_big=6)
    t_new = time.perf_counter() - t0
    print(f"1024x512 full world: old {t_old:.2f}s -> new {t_new:.2f}s ({t_old / t_new:.1f}x)")
    print("ALL OK" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
