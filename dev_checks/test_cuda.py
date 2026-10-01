"""CUDA verification for this machine (RTX 5050, Blackwell sm_120)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch

print("torch:", torch.__version__, " cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
cap = f"sm_{props.major}{props.minor}"
print("GPU:", name, " compute capability:", cap,
      " VRAM: %.1f GB" % (props.total_memory / 1024**3))

# 1. real compute: big matmul on GPU vs CPU
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
import time
t0 = time.perf_counter()
c = a @ b
torch.cuda.synchronize()
t_gpu = time.perf_counter() - t0
c_cpu = a.cpu() @ b.cpu()
ok = torch.allclose(c.cpu(), c_cpu, atol=1e-2, rtol=1e-2)
print(f"matmul 2048x2048 on GPU: {t_gpu*1000:.1f} ms, matches CPU: {ok}")
assert ok

# 2. bf16 / fp16 support (used by the diffusion pipeline)
h = a.half() @ b.half()
print("fp16 matmul ok:", h.shape, h.dtype)

# 3. the project's chunked safetensors loader straight to GPU
import _load_st_chunked as ld
print("resolved terrain-diffusion REPO:", ld.REPO, "exists:", os.path.isdir(ld.REPO))
st = os.path.join(ld.REPO, "base_model", "diffusion_pytorch_model.safetensors")
print("checkpoint:", st, "exists:", os.path.isfile(st))
t0 = time.perf_counter()
sd = ld.load_safetensors_chunked(st, device="cuda")
t_load = time.perf_counter() - t0
n_params = sum(t.numel() for t in sd.values())
vram = torch.cuda.memory_allocated(0) / 1024**3
print(f"loaded {len(sd)} tensors ({n_params/1e6:.1f}M params) to CUDA in {t_load:.1f}s; "
      f"VRAM allocated {vram:.2f} GB")
sample = next(iter(sd.values()))
print("sample tensor:", tuple(sample.shape), sample.dtype, "device:", sample.device)
# sanity: tensor really usable on GPU
_ = (sample.float() * 2).sum().item()

# 4. device auto-resolution from the project helpers
from diffusion_world import resolve_device
print("resolve_device('auto') ->", resolve_device("auto"))
assert resolve_device("auto") == "cuda"

print("CUDA VERIFICATION PASSED")
