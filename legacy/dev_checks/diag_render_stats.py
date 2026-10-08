# -*- coding: utf-8 -*-
"""Diagnose why web render looks flatter than the offline preview."""
import numpy
from PIL import Image

a = numpy.asarray(Image.open("test_out_smoke/hires_2048/08c_earth_style.png"), dtype=float)
b = numpy.asarray(Image.open("test_out_smoke/web_relief_hires_2048.png"), dtype=float)
print("preview shape", a.shape, "web shape", b.shape)
if a.shape != b.shape:
    from PIL import Image as I
    b = numpy.asarray(I.open("test_out_smoke/web_relief_hires_2048.png").resize(
        (a.shape[1], a.shape[0])), dtype=float)
land_a = a.mean(axis=2) > 80   # anything not ocean-blue-ish
for name, img in (("preview", a), ("web", b)):
    lum = img.mean(axis=2)
    print(f"{name}: mean={lum.mean():.1f} std={lum.std():.1f} p5={numpy.percentile(lum,5):.0f} p95={numpy.percentile(lum,95):.0f}")
# local contrast: gradient magnitude mean on land half
from scipy.ndimage import gaussian_filter
for name, img in (("preview", a), ("web", b)):
    lum = img.mean(axis=2)
    gy, gx = numpy.gradient(gaussian_filter(lum, 0.5))
    gm = numpy.sqrt(gx * gx + gy * gy)
    print(f"{name}: grad mean={gm.mean():.2f} p95={numpy.percentile(gm,95):.2f}")
