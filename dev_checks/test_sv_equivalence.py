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
    # The tectonic plate maps must stay bit-identical to the pre-optimization
    # implementation.  land_mask / continent_mask were intentionally redesigned
    # (worldengine/continents.py: core-growth continents decoupled from plate
    # shapes, ~30% land), so they are no longer compared here.
    names = ("pid/raw", "merged")
    ok = True
    for name, a, b in zip(names, o, n):
        if not np.array_equal(a, b):
            ok = False
            print(f"  [{tag}] seed={seed} {name} DIFFERS ({int((np.asarray(a) != np.asarray(b)).sum())} cells)")
    # land sanity on the new continents: fraction and label count
    land_mask, continent_mask = n[2], n[3]
    frac = float(land_mask.mean())
    n_cont = len(np.unique(continent_mask[continent_mask >= 0]))
    if not (0.24 <= frac <= 0.36):
        ok = False
        print(f"  [{tag}] seed={seed} land fraction {frac:.3f} outside Earth-like range")
    if not (3 <= n_cont <= 8):
        ok = False
        print(f"  [{tag}] seed={seed} continent count {n_cont} outside 3..8")
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
