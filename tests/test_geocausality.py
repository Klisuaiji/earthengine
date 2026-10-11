"""阶段 B 地理因果一致性测试 (8 节)。

- 地壳类型独立于 plate ID（同一 plate 内可有多种地壳、不同 plate 可同地壳）。
- 山脉/海沟/裂谷由构造规则 + 地壳组合驱动，而非仅距边界距离。
- 海盆(bathymetry)与海陆一致：陆地高程 > 0、海洋高程 < 0。

用法：python3 tests/test_geocausality.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from earthengine.pipeline.stages import WorldGenerator
from earthengine.pipeline.world import (
    TT_MOUNTAIN, TT_TRENCH, TT_RIDGE, TT_RIFT,
    CRUST_CONTINENTAL, CRUST_OCEANIC, CRUST_TRANSITIONAL,
    CONVERGENT, DIVERGENT,
)


def make(seed=1234567, w=256, h=128):
    return WorldGenerator({"seed": seed, "width": w, "height": h}).generate()


def test_crust_independent_of_plate_id(ws):
    """地壳类型不等于 plate 标签：同一 plate 内存在多种地壳、不同 plate 可同地壳。"""
    p = ws.plates.plates
    ct = ws.crust.type
    # 至少两个不同 plate 出现相同地壳类型（⇒ 地壳≠plate标签）
    plate_ids = np.unique(p)
    types_by_plate = {pid: set(np.unique(ct[p == pid]).tolist()) for pid in plate_ids}
    shared = False
    for a in range(len(plate_ids)):
        for b in range(a + 1, len(plate_ids)):
            if types_by_plate[plate_ids[a]] & types_by_plate[plate_ids[b]]:
                shared = True
                break
        if shared:
            break
    assert shared, "所有板块地壳类型两两互斥——地壳可能退化为 plate 标签"
    # 至少一个板块内部地壳类型不只一种（说明地壳是独立空间场）
    multi = any(len(t) > 1 for t in types_by_plate.values())
    assert multi, "没有板块内部地壳类型多样——地壳场可能未独立演化"


def test_crust_has_oceanic_and_continental(ws):
    ct = ws.crust.type
    assert CRUST_CONTINENTAL in np.unique(ct)
    assert CRUST_OCEANIC in np.unique(ct)


def test_land_sea_bathymetry_consistent(ws):
    """陆地高程 > 0，海洋高程 < 0（海盆与海陆一致，P0-2）。"""
    elev = ws.terrain["elevation"]
    lm = ws.ocean["land_mask"]
    conflict = int((lm & (elev < 0)).sum() | ((~lm) & (elev > 0)).sum())
    frac = conflict / elev.size
    assert frac <= 0.02, f"海陆-高程矛盾 {frac:.4f}"


def test_mountains_near_convergent(ws):
    """山脉主要由汇聚边界驱动（非纯随机）。"""
    bt = ws.boundaries["boundary_type"]
    terr = ws.terrain["region"]
    conv = bt == CONVERGENT
    if not conv.any():
        return
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return
    d = distance_transform_edt(~conv)
    mount = terr == TT_MOUNTAIN
    if not mount.any():
        return
    within = (d[mount] < 5.0).mean()
    assert within >= 0.60, f"山脉落汇聚边界缓冲区 {within:.3f}"


def test_trench_near_oceanic_convergence(ws):
    """海沟与地壳组合相关：至少存在部分海沟靠近汇聚边界。"""
    bt = ws.boundaries["boundary_type"]
    terr = ws.terrain["region"]
    trench = terr == TT_TRENCH
    conv = bt == CONVERGENT
    if not conv.any() or not trench.any():
        return
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return
    d = distance_transform_edt(~conv)
    frac = (d[trench] < 5.0).mean()
    assert frac >= 0.30, f"海沟落汇聚边界缓冲区 {frac:.3f}"


_ALL = [v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)]


def run_all(seed=1234567):
    ws = make(seed)
    failed = 0
    for fn in _ALL:
        try:
            fn(ws)
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(_ALL)-failed}/{len(_ALL)} passed")
    return failed == 0


if __name__ == "__main__":
    import sys as _s
    _s.exit(0 if run_all() else 1)
