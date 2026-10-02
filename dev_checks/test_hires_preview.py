# -*- coding: utf-8 -*-
"""High-res procedural preview (2048x1024, no diffusion) + stats."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy
from tools import planet_pipeline as pp

t0 = time.time()
res = pp.generate_planet(seed=2026, w=2048, h=1024, out="test_out_smoke/hires_2048",
                         detail="procedural", save_npy=True)
rivers = numpy.load("test_out_smoke/hires_2048/rivers.npy")
land = res["elev"] >= 0
print("river pixels: %d (%.2f%% of land)" % (rivers.sum(), 100.0 * rivers.sum() / max(1, land.sum())))
print("total: %.1fs" % (time.time() - t0))
