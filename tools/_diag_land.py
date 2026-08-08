"""Diagnostic: per-tile land fraction vs conditioning strength.

Usage: _diag_land.py SEED SNR0 LAND_ELEV
"""
import os, sys, json, time, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, ".workbuddy", "参考", "terrain-diffusion-master")
for p in (REPO, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import importlib.util
spec = importlib.util.spec_from_file_location("ld", os.path.join(ROOT, "tools", "_load_st_chunked.py"))
ld = importlib.util.module_from_spec(spec); spec.loader.exec_module(ld)
from tools.diffusion_world import build_pipeline, voronoi_land_mask, conditioning_from_mask, CELL

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
snr0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
land_elev = float(sys.argv[3]) if len(sys.argv) > 3 else 2000.0
ocean_elev = float(sys.argv[4]) if len(sys.argv) > 4 else -4000.0
W, H = 1024, 512
pipe = build_pipeline(seed=seed, cache_limit=128*1024*1024, device="cuda")
pipe.set_cond_snr([snr0, 0.5, 0.5, 0.5, 0.5])
lm = voronoi_land_mask(seed, W, H)
grid = conditioning_from_mask(lm, land_elev=land_elev, ocean_elev=ocean_elev)
print("seed=%d snr0=%.1f land=%.0f ocean=%.0f" % (seed, snr0, land_elev, ocean_elev))
print("Voronoi land fraction: %.1f%%" % (lm.mean()*100))
pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=-1000.0)

rows, cols = (H+256-1)//256, (W+256-1)//256
torch.cuda.empty_cache()
tot_land = 0; tot_px = 0; emin, emax = 1e9, -1e9
for ri in range(rows):
    for cj in range(cols):
        y0, x0 = ri*256, cj*256
        y1, x1 = min(y0+256, H), min(x0+256, W)
        reg = pipe.get(y0, x0, y1, x1, with_climate=True)
        e = reg["elev"].detach().cpu().numpy().astype(np.float32)
        e = e[:y1-y0, :x1-x0]
        cond_val = grid[min(ri, grid.shape[0]-1), min(cj, grid.shape[1]-1)]
        land = (e >= 0).mean()*100
        tot_land += (e >= 0).sum(); tot_px += e.size
        emin, emax = min(emin, e.min()), max(emax, e.max())
        print("tile(%d,%d) cond=%.0f mean_elev=%.1f land%%=%.1f" % (ri, cj, cond_val, e.mean(), land), flush=True)
        torch.cuda.empty_cache()
print("WORLD land%%=%.1f elev[min/max]=%.1f/%.1f" % (tot_land/tot_px*100, emin, emax))
pipe.close()
print("DONE")
