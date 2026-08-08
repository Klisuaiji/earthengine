import os, sys, json, faulthandler, traceback
faulthandler.enable()
REPO = r"D:\Qq203\Downloads\earthengine-master\.workbuddy\参考\terrain-diffusion-master"
sys.path.insert(0, REPO)
import numpy as np

def kw():
    with open(os.path.join(REPO,"config.json")) as f: cfg=json.load(f)
    return dict(native_resolution=cfg["native_resolution"],latent_compression=cfg["latent_compression"],
        frequency_mult=cfg["frequency_mult"],drop_water_pct=cfg["drop_water_pct"],cond_snr=cfg["cond_snr"],
        coarse_pooling=cfg["coarse_pooling"],elev_coarse_pool_mode=cfg["elev_coarse_pool_mode"],
        p5_coarse_pool_mode=cfg["p5_coarse_pool_mode"],residual_mean=cfg["residual_mean"],residual_std=cfg["residual_std"],
        coarse_means=cfg["coarse_means"],coarse_stds=cfg["coarse_stds"],caching_strategy="direct",
        cache_limit=512*1024*1024,latents_batch_size=[1,2,4,8,16],log_mode="info",torch_compile=False,dtype=None)

def try_path(device):
    import torch
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    print(f"[step] from_local_models device={device}", flush=True)
    pipe = WorldPipeline.from_local_models(
        coarse_model_path=os.path.join(REPO,"coarse_model"),
        base_model_path=os.path.join(REPO,"base_model"),
        decoder_model_path=os.path.join(REPO,"decoder_model"),
        seed=1234567, **kw())
    print("[step] built (CPU). moving to", device, flush=True)
    if device!="cpu":
        pipe.to(device)
    print("[step] to() done. bind()...", flush=True)
    pipe.bind()
    print("[step] bind done. set_custom...", flush=True)
    coarse = np.full((2,4),-1000.0,np.float32); coarse[0,0]=800;coarse[0,1]=800;coarse[1,2]=800
    pipe.set_custom_conditioning_import(0, coarse, 0,0, default_value=-1000.0)
    print("[step] get(0,0,256,256)...", flush=True)
    region = pipe.get(0,0,256,256,with_climate=True)
    elev = region["elev"].detach().cpu().numpy()
    print(f"[OK] device={device} elev {elev.shape} min/max {elev.min():.1f}/{elev.max():.1f} land% {(elev>=0).mean()*100:.1f}", flush=True)
    pipe.close()

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "cpu"
    try:
        try_path(which)
    except Exception as e:
        traceback.print_exc()
        print("EXC", type(e).__name__, e, flush=True)
