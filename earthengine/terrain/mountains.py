"""山脉生成 (17)：沿汇聚边界 + 走向粗糙度。"""

from __future__ import annotations

import numpy as np

from earthengine.terrain.noise import fbm


def mountain_belt(land_mask, convergent_dist, seed, h, w):
    """沿汇聚边界的窄高山脉带（Andes/Himalaya 风格）。

    返回 (h,w) float32 高程抬升。
    """
    noise = fbm((h, w), base_scale=28, octaves=5, seed=seed + 301)
    broad = fbm((h, w), base_scale=96, octaves=3, seed=seed + 302)
    rugged = 0.6 + 0.7 * np.clip(noise * 1.6, 0.0, 1.0)
    conv = 2900.0 * np.exp(-convergent_dist / 17.0) * land_mask.astype(np.float32) * rugged
    return conv + broad * 180.0
