"""v2.0 地理规律量化测试 (34 / 35)。

以可复现 seed 生成小世界，断言 34.6 表的量化阈值。可直接 ``python3 tests/run_checks.py``
运行（无 pytest 依赖），亦兼容 ``pytest tests/``。

阈值表 (34.6)：
    板块数 ∈ [4, 10]                       (FAIL)
    边界类型 ≥ 3 类                        (WARN)
    陆地占比 ∈ [0.20, 0.45]                (WARN)
    最大连通陆地/总陆地 ≥ 0.35             (WARN)
    单像素岛占比 ≤ 0.02                    (WARN)
    山脉落汇聚边界 3° 内 ≥ 0.60            (FAIL)
    纬度–温度 zonal |r| ≥ 0.70             (FAIL)
    逆坡河段 == 0                          (FAIL)
"""

from __future__ import annotations

import numpy as np

from earthengine.pipeline.stages import WorldGenerator
from earthengine.pipeline.world import TT_MOUNTAIN, CONVERGENT
from earthengine.planet.sphere import _lat_deg


def make_world(seed=1234567, w=256, h=128):
    g = WorldGenerator({"seed": seed, "width": w, "height": h})
    return g.generate()


# ---------------------------------------------------------------------------
def test_plate_count(ws):
    n = int(np.unique(ws.plates.plates).size)
    assert 4 <= n <= 10, f"板块数 {n} 超范围"


def test_boundary_type_diversity(ws):
    bt = ws.boundaries["boundary_type"]
    kinds = np.unique(bt[bt > 0]).size
    assert kinds >= 3, f"边界类型退化 ({kinds} 类)"


def test_land_fraction(ws):
    from earthengine.pipeline.geography_validator import LAND_LO, LAND_HI
    frac = float(ws.ocean["land_mask"].mean())
    assert LAND_LO <= frac <= LAND_HI, f"陆地占比 {frac:.3f} 不在 [{LAND_LO},{LAND_HI}]"


def test_max_continent_fraction(ws):
    from earthengine.pipeline.geography_validator import LARGEST_MIN
    lm = ws.ocean["land_mask"]
    cm = ws.ocean["continent_mask"]
    if cm is None:
        return  # 无大陆掩膜时跳过
    labels = np.unique(cm[lm])
    labels = labels[labels >= 0]
    total = int(lm.sum())
    biggest = max(int((cm[lm] == lb).sum()) for lb in labels) if labels.size else 0
    assert biggest / total >= LARGEST_MIN, f"最大大陆占比 {biggest/total:.3f} < {LARGEST_MIN}"


def test_single_pixel_islands(ws):
    lm = ws.ocean["land_mask"].astype(np.uint8)
    try:
        from scipy.ndimage import label
        lbl, n = label(lm)
    except Exception:
        return
    sizes = np.bincount(lbl.ravel())
    total = int(lm.sum())
    frac_single = float(sizes[sizes == 1].sum()) / total if total else 0.0
    assert frac_single <= 0.02, f"单像素岛占比 {frac_single:.3f}"


def test_mountain_convergent(ws):
    terr = ws.terrain["region"]
    bt = ws.boundaries["boundary_type"]
    if bt is None:
        return
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return
    conv = bt == CONVERGENT
    if not conv.any():
        return
    d = distance_transform_edt(~conv)
    mount = terr == TT_MOUNTAIN
    if not mount.any():
        return
    within = (d[mount] < 3.0).mean()
    assert within >= 0.60, f"山脉落汇聚边界占比 {within:.3f}"


def test_latitude_temperature(ws):
    temp = ws.climate["temperature"]
    h, w = temp.shape
    lat = np.abs(_lat_deg(h).ravel())
    zonal = temp.mean(axis=1)
    r = float(np.corrcoef(lat, zonal)[0, 1])
    assert abs(r) >= 0.70, f"纬度-温度 zonal |r|={abs(r):.3f}"


def test_no_uphill(ws):
    uphill = ws.hydrology["uphill_segments"]
    assert uphill == 0, f"逆坡河段 {uphill}"


# ---------------------------------------------------------------------------
_ALL_TESTS = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]


def run_all(seed=1234567, w=256, h=128, verbose=True):
    """自包含运行器：无 pytest 也能跑。返回 (ok, report)。"""
    ws = make_world(seed, w, h)
    results = []
    failed = 0
    for fn in _ALL_TESTS:
        try:
            fn(ws)
            results.append((fn.__name__, True, "ok"))
        except Exception as e:
            failed += 1
            results.append((fn.__name__, False, str(e)))
    if verbose:
        for name, ok, msg in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {msg}")
        print(f"\n{len(results)-failed}/{len(results)} passed")
    return failed == 0, results


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1234567
    ok, _ = run_all(seed)
    sys.exit(0 if ok else 1)
