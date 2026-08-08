"""Chunked safetensors loader that NEVER commits process RAM.

Strategy: mmap the safetensors file (file-backed pages, zero commit charge),
create a numpy *view* over each tensor's exact byte range (no copy), wrap it
with torch.from_numpy (shares the mmap, no copy), then copy_ directly into a
GPU tensor. The only allocation is the GPU tensor itself (VRAM, not system RAM).

This bypasses two independent blockers we hit:
  * Windows commit-limit / pagefile: even with 7+GB physical free, a 2MB heap
    buffer (.read + .copy) fails once the process has committed a lot of memory.
  * Per-process RAM caps: mmap'd pages are file-backed and do not count against
    the commit ceiling.

Also provides a patched WorldPipeline model loader that uses this for the
three EDM U-Nets (coarse/base/decoder) by building them on the `meta` device
(no CPU param allocation) and assigning the GPU-loaded state dict.
"""
import json, struct, sys, os, mmap, numpy as np, torch

REPO = r"D:\Qq203\Downloads\earthengine-master\.workbuddy\参考\terrain-diffusion-master"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_TORCH_DTYPE = {
    'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
    'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16,
    'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool,
}
_NP_DTYPE = {
    'F64': '<f8', 'F32': '<f4', 'F16': '<f2',
    'I64': '<i8', 'I32': '<i4', 'I16': '<i2', 'I8': '<i1', 'U8': '<u1', 'BOOL': '?',
}


_CHUNK = 256 * 1024  # 256 KB CPU staging buffer (small -> no commit ceiling hit)


def load_safetensors_chunked(path, device='cuda'):
    """Load every tensor in a safetensors file directly to `device`.

    Reads the file in small (256 KB) regular heap chunks and streams each chunk
    into a pre-allocated GPU tensor via copy_. The peak CPU RAM is one 256 KB
    buffer (well under the per-process commit ceiling), and we never copy from
    memory-mapped file pages into the GPU (that poisoned the CUDA context).
    """
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n).decode('utf-8'))
        data_start = 8 + n
        out = {}
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dt = meta['dtype']
            shape = meta['shape']
            s, e = meta['data_offsets']
            npdt = np.dtype(_NP_DTYPE[dt])
            tdt = _TORCH_DTYPE[dt]
            numel = int(np.prod(shape)) if shape else 1
            itemsize = npdt.itemsize
            gpu_t = torch.empty(shape, dtype=tdt, device=device)
            flat = gpu_t.flatten()
            f.seek(data_start + s)
            eoff = 0
            while eoff < numel:
                cb = min(_CHUNK, (numel - eoff) * itemsize)
                buf = f.read(cb)
                if not buf:
                    break
                cpu = np.frombuffer(buf, dtype=npdt)
                flat[eoff:eoff + cpu.size].copy_(torch.from_numpy(cpu))
                eoff += cpu.size
            out[name] = gpu_t
    return out


def load_models_chunked(pipe, device='cuda'):
    """Attach the three U-Nets to an (already constructed) WorldPipeline by
    building them on the `meta` device and loading weights straight to GPU."""
    import terrain_diffusion.models.edm_unet as eu
    specs = [('coarse_model', 'coarse_model'),
             ('base_model', 'base_model'),
             ('decoder_model', 'decoder_model')]
    for idx, (attr, folder) in enumerate(specs):
        cfg = json.load(open(os.path.join(REPO, folder, 'config.json')))
        with torch.device('meta'):
            model = eu.EDMUnet2D.from_config(cfg)
        sd = load_safetensors_chunked(
            os.path.join(REPO, folder, 'diffusion_pytorch_model.safetensors'), device=device)
        missing, unexpected = model.load_state_dict(sd, assign=True)
        if missing or unexpected:
            print('  [warn] %s missing=%d unexpected=%d' % (attr, len(missing), len(unexpected)), flush=True)
        setattr(pipe, attr, model)
        print('  [ok] loaded %s to %s' % (attr, device), flush=True)
        # Release reserved caching-allocator blocks so the next (large) model's
        # first allocation isn't blocked by fragmentation of freed blocks.
        torch.cuda.empty_cache()
    pipe._apply_dtype_and_compile()
    return pipe


if __name__ == '__main__':
    import faulthandler, time
    faulthandler.enable()
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    import numpy as np
    with open(os.path.join(REPO, 'config.json')) as fh:
        cfg = json.load(fh)
    kw = dict(
        native_resolution=cfg["native_resolution"], latent_compression=cfg["latent_compression"],
        frequency_mult=cfg["frequency_mult"], drop_water_pct=cfg["drop_water_pct"],
        cond_snr=cfg["cond_snr"], coarse_pooling=cfg["coarse_pooling"],
        elev_coarse_pool_mode=cfg["elev_coarse_pool_mode"], p5_coarse_pool_mode=cfg["p5_coarse_pool_mode"],
        residual_mean=cfg["residual_mean"], residual_std=cfg["residual_std"],
        coarse_means=cfg["coarse_means"], coarse_stds=cfg["coarse_stds"],
        caching_strategy="direct", cache_limit=512*1024*1024,
        latents_batch_size=[1,2,4,8,16], log_mode="info", torch_compile=False, dtype=None,
    )
    t0 = time.time()
    pipe = WorldPipeline(seed=1234567, **kw)
    load_models_chunked(pipe, device='cuda')
    print('models loaded %.1fs' % (time.time() - t0), flush=True)
    pipe.to('cuda'); pipe.bind()
    print('bind ok %.1fs' % (time.time() - t0), flush=True)
    coarse = np.full((2, 4), -1000.0, np.float32); coarse[0,0]=800; coarse[0,1]=800; coarse[1,2]=800
    pipe.set_custom_conditioning_import(0, coarse, 0, 0, default_value=-1000.0)
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    region = pipe.get(0, 0, W, H, with_climate=True)
    elev = region['elev'].detach().cpu().numpy().astype(np.float32)
    climate = region['climate'].detach().cpu().numpy().astype(np.float32)
    print('elev', elev.shape, 'min/max', float(elev.min()), float(elev.max()),
          'land%%', float((elev>=0).mean()*100), flush=True)
    print('climate', climate.shape, flush=True)
    pipe.close()
    print('ALL OK', flush=True)
