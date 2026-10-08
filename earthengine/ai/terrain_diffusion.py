"""TerrainDiffusionAdapter (24 / Phase 6)。

Terrain Diffusion 是**细化层**，不是世界的地理逻辑层 (25)。缺失 torch 时自动
回落 ProceduralEnhancer (6/49.2)。适配映射见 24.3：

    build_pipeline / set_custom_conditioning_import / generate_world / clamp_land_sea
    → adapter.encode_conditioning / generate / postprocess
"""

from __future__ import annotations

import numpy as np

from earthengine.ai.base import TerrainEnhancer, ProceduralEnhancer

_PIPE_SINGLETON = {"pipe": None, "seed": None, "device": None}


class TerrainDiffusionAdapter(TerrainEnhancer):
    """可选 AI 高分辨率地形增强后端。torch 缺失时回退 procedural。"""

    name = "terrain_diffusion"

    def __init__(self, seed: int = 0, device: str = "auto", snr0: float = 0.5,
                 tile: int = 256):
        self.seed = int(seed)
        self.device = device
        self.snr0 = snr0
        self.tile = tile
        try:
            import torch  # noqa: F401
            self._torch_ok = True
        except Exception:
            self._torch_ok = False
        self._fallback = ProceduralEnhancer()

    # ------------------------------------------------------------------
    def generate(self, conditioning: dict, region, resolution: int,
                 seed: int) -> np.ndarray:
        """返回细化后高程 (h,w) float32 米。

        有 torch 且 conditioning 含 macro_elevation → 走扩散分块推理；
        否则回落程序化增强（契约：同 dtype、同量纲、同海陆符号）。
        """
        if not self._torch_ok:
            return self._fallback.generate(conditioning, region, resolution, seed)
        macro = conditioning.get("macro_elevation")
        if macro is None:
            return self._fallback.generate(conditioning, region, resolution, seed)
        try:
            return self._diffuse(macro, conditioning.get("land_sea"))
        except Exception:
            return self._fallback.generate(conditioning, region, resolution, seed)

    # ------------------------------------------------------------------
    def _build_pipeline(self):
        if (_PIPE_SINGLETON["pipe"] is not None
                and _PIPE_SINGLETON["seed"] == self.seed
                and _PIPE_SINGLETON["device"] == self.device):
            return _PIPE_SINGLETON["pipe"]
        from earthengine.ai.terrain_diffusion_impl import build_pipeline
        pipe = build_pipeline(seed=self.seed, device=self.device, snr0=self.snr0)
        _PIPE_SINGLETON.update(pipe=pipe, seed=self.seed, device=self.device)
        return pipe

    def _diffuse(self, macro, land_mask):
        from earthengine.ai.terrain_diffusion_impl import (
            generate_world, OCEAN_BASE, LAND_BASE,
        )
        h, w = macro.shape
        # 粗条件网格：陆地取高百分位、海洋取低百分位，保证海陆符号清晰
        gh = (h + self.tile - 1) // self.tile
        gw = (w + self.tile - 1) // self.tile
        grid = np.full((gh, gw), OCEAN_BASE, dtype=np.float32)
        lm = land_mask
        for ci in range(gh):
            y0, y1 = ci * self.tile, min((ci + 1) * self.tile, h)
            for cj in range(gw):
                x0, x1 = cj * self.tile, min((cj + 1) * self.tile, w)
                block = macro[y0:y1, x0:x1]
                lb = lm[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                if lb.mean() >= 0.5:
                    val = np.percentile(block[lb], 85) if lb.any() else LAND_BASE
                    grid[ci, cj] = max(val, LAND_BASE)
                else:
                    grid[ci, cj] = min(np.percentile(block, 15), OCEAN_BASE)
        pipe = self._build_pipeline()
        pipe.set_custom_conditioning_import(0, grid, 0, 0, default_value=OCEAN_BASE)
        elev, _ = generate_world(pipe, w, h, tile=self.tile, device=self.device)
        pipe.close()
        elev[land_mask] = np.maximum(elev[land_mask], 1.0)
        elev[~land_mask] = np.minimum(elev[~land_mask], -1.0)
        return elev.astype(np.float32)


def _noop(*a, **k):
    return None
