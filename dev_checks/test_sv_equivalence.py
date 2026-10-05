"""Structural regression for the redesigned spherical_voronoi pipeline.

The plate layout is now intentionally spec-driven (two oceanic plates at the
map centre, five Earth-like continents with a super continent, two
near-connected pairs and an Australia-like continent on an oceanic plate),
so bit-equality against the pre-redesign reference implementation no longer
holds.  This test locks the NEW structural invariants instead:

* merged plate count == n_big
* Earth-like land fraction (0.24..0.36)
* reference mode (n_big == 6): exactly 5 continents, and the map centre
  (the Pacific analogue) stays open ocean
* legacy modes (n_big != 6): 3..8 continents
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import worldengine.spherical_voronoi as sv


def check(seed, w, h, n_raw, n_big, tag):
    pid, merged, land, cont = sv.generate_spherical_world(
        seed, w=w, h=h, n_raw=n_raw, n_big=n_big)
    ok = True
    n_groups = int(merged.max()) + 1
    if n_groups != n_big:
        ok = False
        print(f"  [{tag}] merged groups {n_groups} != n_big {n_big}")
    frac = float(land.mean())
    if not (0.24 <= frac <= 0.36):
        ok = False
        print(f"  [{tag}] land fraction {frac:.3f} outside Earth-like range")
    n_cont = len(np.unique(cont[cont >= 0]))
    if n_big == 6:
        if n_cont != 5:
            ok = False
            print(f"  [{tag}] reference mode expects 5 continents, got {n_cont}")
        # the central pure-ocean zone: a disk around the map centre
        h2, w2 = h // 2, w // 2
        r = max(8, int(w * 0.06))
        yy, xx = np.ogrid[:h, :w]
        disk = (yy - h2) ** 2 + np.minimum(np.abs(xx - w2), w - np.abs(xx - w2)) ** 2 <= r * r
        central_land = float(land[disk].mean())
        if central_land > 0.15:
            ok = False
            print(f"  [{tag}] central ocean zone {central_land*100:.0f}% land (should stay open)")
    else:
        if not (3 <= n_cont <= 8):
            ok = False
            print(f"  [{tag}] continent count {n_cont} outside 3..8")
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
        ok = check(seed, w, h, n_raw, n_big, "voronoi")
        print(f"{'ok  ' if ok else 'FAIL'} {w}x{h} seed={seed} n_raw={n_raw} n_big={n_big} "
              f"({time.perf_counter() - t0:.2f}s)")
        failures += 0 if ok else 1

    t0 = time.perf_counter()
    sv.generate_spherical_world(1234567, w=1024, h=512, n_raw=30, n_big=6)
    print(f"1024x512 full world: {time.perf_counter() - t0:.2f}s")
    print("ALL OK" if failures == 0 else f"{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
