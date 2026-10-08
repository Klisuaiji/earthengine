"""POC: verify terrain-diffusion custom conditioning injection works end-to-end
on the local RTX 5050 (envs/default, torch CUDA)."""
import os, sys, time, json

REPO = r"D:\Qq203\Downloads\earthengine-master\.workbuddy\参考\terrain-diffusion-master"
sys.path.insert(0, REPO)

import numpy as np

def main():
    with open(os.path.join(REPO, "config.json")) as f:
        cfg = json.load(f)
    kw = dict(
        native_resolution=cfg["native_resolution"],
        latent_compression=cfg["latent_compression"],
        frequency_mult=cfg["frequency_mult"],
        drop_water_pct=cfg["drop_water_pct"],
        cond_snr=cfg["cond_snr"],
        coarse_pooling=cfg["coarse_pooling"],
        elev_coarse_pool_mode=cfg["elev_coarse_pool_mode"],
        p5_coarse_pool_mode=cfg["p5_coarse_pool_mode"],
        residual_mean=cfg["residual_mean"],
        residual_std=cfg["residual_std"],
        coarse_means=cfg["coarse_means"],
        coarse_stds=cfg["coarse_stds"],
        caching_strategy="direct",
        cache_limit=512*1024*1024,
        latents_batch_size=[1,2,4,8,16],
        log_mode="info", torch_compile=False, dtype=None,
    )
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    t0=time.time()
    pipe = WorldPipeline.from_local_models(
        coarse_model_path=os.path.join(REPO,"coarse_model"),
        base_model_path=os.path.join(REPO,"base_model"),
        decoder_model_path=os.path.join(REPO,"decoder_model"),
        seed=1234567, **kw)
    pipe.to("cuda")
    pipe.bind()
    print("build+bind %.1fs" % (time.time()-t0))

    # Coarse elevation: 4x2 grid, land = +800, ocean = -1000 deep sea
    coarse = np.full((2,4), -1000.0, dtype=np.float32)
    coarse[0,0]=800.0; coarse[0,1]=800.0; coarse[1,2]=800.0  # 3 land cells
    pipe.set_custom_conditioning_import(0, coarse, 0, 0, default_value=-1000.0)

    # native tile covering coarse [0:4]x[0:2] -> 4*256 x 2*256 = 1024x512
    t0=time.time()
    region = pipe.get(0,0,1024,512, with_climate=True)
    print("get %.1fs" % (time.time()-t0))
    elev = region["elev"].detach().cpu().numpy().astype(np.float32)
    climate = region["climate"].detach().cpu().numpy().astype(np.float32)
    print("elev shape", elev.shape, "min/max", float(elev.min()), float(elev.max()))
    print("land%% (elev>=0):", float((elev>=0).mean()*100))
    print("climate shape", climate.shape, "(expect 5,H,W)")
    print("climate per-channel min/max:")
    for c in range(climate.shape[0]):
        print("  ch%d: %.3f .. %.3f" % (c, climate[c].min(), climate[c].max()))
    pipe.close()

if __name__ == "__main__":
    main()
