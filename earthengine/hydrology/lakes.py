"""湖泊 / 内流盆地 (22)：D8 汇的积水区。

内流盆地：陆地局部低点（无下坡、流向自身）且上游汇水面积超过阈值 → 湖。
"""

from __future__ import annotations

import numpy as np


def lakes(flow_dir, flow_acc, land_mask, h, w, min_acc=None):
    """陆地局部低点 + 足够汇水面积 → 湖。返回 (h,w) bool。"""
    if min_acc is None:
        min_acc = max(60, (h * w) // 20000)
    # sink = flow_dir 8（无下坡，流向自身）
    sinks = (flow_dir == 8) & land_mask
    # 只有汇水面积足够大的 sink 才成湖（否则是平地噪声点）
    return sinks & (flow_acc >= min_acc)


def closed_basins(flow_dir, land_mask, h, w):
    """内流盆地：sink 的汇水区域（可达性），近似为 sink 像素 + 上游。"""
    basins = (flow_dir == 8) & land_mask
    return basins
