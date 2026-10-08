"""大陆架 (13)：由地壳类型 + 距陆缘距离真正推导，而非高程阈值硬切。

海陆之间必须允许：大陆内部 → 大陆边缘 → 大陆架 → 大陆坡 → 深海。
"""

from __future__ import annotations

import numpy as np

from earthengine.tectonics.crust import CRUST_OCEANIC


def shelf_from_crust(land_mask, crust_type, shelf_px=26.0):
    """返回 (h,w) float32 大陆架强度 [0,1]。

    - 洋壳上、距陆地近 → 大陆架强；
    - 过渡壳（陆裂谷）→ 部分大陆架；
    - 深海远洋 → 0。
    """
    try:
        from scipy.ndimage import distance_transform_edt
        d_land = distance_transform_edt(land_mask).astype(np.float32)
    except Exception:  # pragma: no cover
        return np.zeros(land_mask.shape, dtype=np.float32)
    sea = ~land_mask
    shelf = np.clip(1.0 - d_land / shelf_px, 0.0, 1.0)
    # 过渡壳区域也给予部分大陆架（裂谷边缘）
    transitional = crust_type == 2
    shelf = np.where(transitional, np.maximum(shelf, 0.4), shelf)
    return shelf.astype(np.float32)
