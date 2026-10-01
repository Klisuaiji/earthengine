"""Minimal end-to-end CUDA inference test with conservative memory settings."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from diffusion_world import build_pipeline, resolve_device
import torch

device = resolve_device("auto")
print("device:", device, flush=True)

pipe = build_pipeline(seed=1234567, cache_limit=64 * 1024 * 1024,
                      device=device, snr0=0.5)

coarse = np.full((2, 4), -1000.0, np.float32)
coarse[0, 0] = 800
coarse[0, 1] = 800
coarse[1, 2] = 800
pipe.set_custom_conditioning_import(0, coarse, 0, 0, default_value=-1000.0)

W = H = 128
t0 = time.time()
region = pipe.get(0, 0, W, H, with_climate=True)
elev = region["elev"].detach().cpu().numpy().astype(np.float32)
climate = region["climate"].detach().cpu().numpy().astype(np.float32)
print("elev", elev.shape, "min/max", float(elev.min()), float(elev.max()),
      "land%%", float((elev >= 0).mean() * 100), flush=True)
print("climate", climate.shape, flush=True)
print("inference wall time: %.1fs" % (time.time() - t0))
print("VRAM allocated: %.2f GB" % (torch.cuda.memory_allocated(0) / 1024**3))
pipe.close()
print("CUDA E2E OK", flush=True)
