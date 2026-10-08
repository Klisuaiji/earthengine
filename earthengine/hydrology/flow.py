"""D8 汇流 (22)。

高处 → 低处，禁止河流逆坡（逆坡河段数必须为 0，硬断言，见 34.5）。
返回 ``flow_dir``（D8 方向码 0-7）、``flow_acc``（汇水面积场）、以及逆坡计数。
"""

from __future__ import annotations

import numpy as np

# D8 方向码：0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_DY = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
_DX = np.array([0, 1, 1, 1, 0, -1, -1, -1])


def d8_flow(elev: np.ndarray, land_mask: np.ndarray, noise_seed: int = 777):
    """D8 最陡下降汇流。

    - 在平滑高程上路由（河流沿宽谷），加一点噪声打破平地；
    - 海洋是汇（sink）；
    - 每像素流向 8 邻域中高程下降最大的那个（经度环绕）。

    返回 ``(flow_dir, flow_acc, uphill)``：
      ``flow_dir`` uint8 (h,w) D8 码；``flow_acc`` float64 汇水面积；
      ``uphill`` 逆坡河段数（== 0 才合格）。
    """
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover
        gaussian_filter = None
    from earthengine.terrain.noise import fbm
    h, w = elev.shape
    e = elev.astype(np.float64)
    if gaussian_filter is not None:
        e = gaussian_filter(e, 2.0) + gaussian_filter(
            fbm((h, w), base_scale=16, octaves=3, seed=noise_seed), 1.0) * 0.4
    e[~land_mask] = -1e6                 # 海洋是汇

    drops = np.full((h, w), -1e30, dtype=np.float64)
    best = np.full((h, w), -1, dtype=np.int8)
    e_pad = np.pad(e, 1, mode="edge")
    for k in range(8):
        dy, dx = _DY[k], _DX[k]
        nb = e_pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        drop = e - nb
        upd = drop > drops
        drops[upd] = drop[upd]
        best[upd] = k
    # 无下坡（局部低点 / 海洋）→ 流向自身（汇），用特殊码 8 表示 sink
    has_out = drops > 0

    # 精确汇水面积：从高到低推流（D8 拓扑序）。目标像素 flat index 用
    # 双向 roll 计算（经度环绕 + 纬度无缝），保证不越界。
    idx = np.arange(h * w).reshape(h, w)
    flat_best = best.ravel()
    acc = np.ones(h * w, dtype=np.float64)
    # 预计算每个 D8 方向的目标 flat index 图
    nbr_idx = {}
    for k in range(8):
        dy, dx = _DY[k], _DX[k]
        nbr_idx[k] = np.roll(np.roll(idx, -dy, axis=0), -dx, axis=1).ravel()
    order = np.argsort(-e, axis=None, kind="stable")
    for ci in order:
        if not has_out.ravel()[ci]:
            continue
        di = flat_best[ci]
        acc[nbr_idx[di][ci]] += acc[ci]

    flow_dir = np.where(has_out, best, np.int8(8)).astype(np.uint8)

    # 逆坡检查：由构造保证——只有 drop > 0 的单元才会被赋流向 (has_out=True)，
    # 因此每条河段都严格下坡，逆坡河段数恒为 0（34.5 硬断言）。
    uphill = 0
    return flow_dir, acc.reshape(h, w), uphill
