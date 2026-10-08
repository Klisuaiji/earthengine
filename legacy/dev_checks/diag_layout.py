# -*- coding: utf-8 -*-
"""Verify the spec-driven layout: plate oceanic flags, continent count,
host plates, and pairwise separation of the near-connected pairs."""
import sys, math
sys.path.insert(0, r"D:\Qq203\Downloads\earthengine-master")
import numpy
from worldengine import spherical_voronoi as sv

pid, merged, land, cont = sv.generate_spherical_world(2026, w=1024, h=512)
n_groups = int(merged.max()) + 1
print("plates:", n_groups)
# oceanic flags: recompute centre rule
gys, gxs = numpy.indices(merged.shape)
rank = []
for g in range(n_groups):
    m = merged == g
    dx = numpy.minimum(gxs[m].mean(), 1024 - gxs[m].mean()) - 512.0
    dy = gys[m].mean() - 256.0
    rank.append((dx * dx + dy * dy, g, gxs[m].mean(), gys[m].mean(), m.sum()))
rank.sort()
ocean = [g for _, g, _, _, _ in rank[:2]]
print("centre plates (oceanic):", [(g, f"({x:.0f},{y:.0f})", f"{a/1024/512*100:.1f}%")
                                    for _, g, x, y, a in rank[:2]])

h, w = land.shape
ids = [c for c in numpy.unique(cont) if c >= 0]
print("continents:", len(ids))
for c in ids:
    m = cont == c
    ys, xs = numpy.nonzero(m)
    area = m.sum()
    # which plate hosts the continent core (deepest cell)
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(m)
    cy, cx = numpy.unravel_index(numpy.argmax(d), d.shape)
    host = merged[cy, cx]
    print(f"  cont {c}: area {area/(h*w)*100:5.2f}%  centroid ({xs.mean():.0f},{ys.mean():.0f})"
          f"  host plate {host} ({'OCEAN' if host in ocean else 'cont'})")
# pairwise min distances (bbox of pixels, sampled)
pts = {c: numpy.argwhere(cont == c)[::37] for c in ids}
print("pairwise min distances (px):")
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = pts[ids[i]], pts[ids[j]]
        d2 = numpy.min((a[:, None, 0] - b[None, :, 0]) ** 2
                       + ((a[:, None, 1] - b[None, :, 1] + 512) % 1024 - 512) ** 2)
        print(f"  {ids[i]}-{ids[j]}: {math.sqrt(d2):.0f}")
