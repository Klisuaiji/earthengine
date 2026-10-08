"""地形噪声（值噪声 + FBM）。

水平可平铺（粗网格末列镜像首列），无接缝；用 numpy 实现，可选 scipy 上采样。
替换旧依赖 ``noise==1.2.2``（C 扩展）。
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import zoom as _zoom
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def smooth_noise(shape, scale, seed):
    """分形值噪声 [0,1]，水平可平铺（粗网格末列镜像首列）。"""
    h, w = shape
    rng = np.random.default_rng(seed)
    gh, gw = max(2, h // scale), max(2, w // scale)
    coarse = rng.random((gh + 2, gw + 1)).astype(np.float32)
    coarse[:, -1] = coarse[:, 0]
    if _HAS_SCIPY:
        fine = _zoom(coarse, (h / (gh + 2), w / (gw + 1)), order=1)
    else:
        fine = _nn_zoom(coarse, (h, w))
    return np.clip(fine[:h, :w].astype(np.float32), 0.0, 1.0)


def _nn_zoom(a, shape):
    """纯 numpy 双线性上采样（无 scipy 时兜底）。"""
    ih, iw = a.shape
    oh, ow = shape
    ys = (np.arange(oh) + 0.5) * ih / oh - 0.5
    xs = (np.arange(ow) + 0.5) * iw / ow - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ih - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, iw - 1)
    y1 = np.clip(y0 + 1, 0, ih - 1)
    x1 = np.clip(x0 + 1, 0, iw - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    return (a[y0][:, x0] * (1 - fy) * (1 - fx) + a[y0][:, x1] * (1 - fy) * fx
            + a[y1][:, x0] * fy * (1 - fx) + a[y1][:, x1] * fy * fx)


def fbm(shape, base_scale, octaves, seed):
    """多倍频 FBM，幅度逐倍减半。"""
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        out += amp * smooth_noise(shape, max(2, base_scale >> o), seed + o * 101)
        tot += amp
        amp *= 0.5
    return out / tot
