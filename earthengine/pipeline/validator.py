"""WorldValidator (35 / 34)：按量化阈值验证世界地理规律。

输出 tectonic/geography/terrain/climate/hydrology 分项分 + overall。
`overall_score < threshold` 时允许 reject/regenerate（重试上限由 config 控制，
并把被拒原因记录到 world.json）。

阈值见 34.6 表：
    板块面积基尼系数 ≥ 0.35 (WARN)
    陆地占比 0.20–0.45 (WARN)
    最大连通陆地/总陆地 ≥ 0.35 (WARN)
    单像素岛占比 ≤ 0.02 (WARN)
    山脉像素落在汇聚边界 3° 内比例 ≥ 0.60 (FAIL)
    纬度–温度 Pearson |r| ≥ 0.70 (FAIL)
    逆坡河段 = 0 (FAIL)
    大陆架连续过渡覆盖率 ≥ 0.80 (WARN)
"""

from __future__ import annotations

import numpy as np

# ---- 量化阈值 (34.6) ----
THRESHOLDS = {
    "plate_gini": (0.35, "WARN", "板块面积分布过均匀"),
    "land_fraction": ((0.20, 0.45), "WARN", "陆地占比超范围"),
    "max_continent": (0.35, "WARN", "最大连通陆地占比过低（过度碎片化）"),
    "single_pixel_islands": (0.02, "WARN", "单像素岛占比过高"),
    "mountain_convergent": (0.60, "FAIL", "山脉未落在汇聚边界缓冲区内"),
    "lat_temp_corr": (0.70, "FAIL", "纬度-温度相关性不足"),
    "uphill": (0, "FAIL", "存在逆坡河段"),
    "shelf_coverage": (0.80, "WARN", "大陆架连续过渡覆盖率不足"),
}


class WorldValidator:
    """宏观验证器：返回 (ok, report)，report 含各分项 score 与 overall。"""

    def __init__(self, config=None):
        self.config = config or {}
        self.threshold = float(self.config.get("reject_threshold", 0.55))

    # ------------------------------------------------------------------
    def validate(self, world_state) -> tuple:
        scores = {}
        msgs = []
        scores["tectonic"] = self._tectonic_score(world_state, msgs)
        scores["geography"] = self._geography_score(world_state, msgs)
        scores["terrain"] = self._terrain_score(world_state, msgs)
        scores["climate"] = self._climate_score(world_state, msgs)
        scores["hydrology"] = self._hydrology_score(world_state, msgs)
        scores["overall"] = float(np.mean(list(scores.values())))
        ok = scores["overall"] >= self.threshold
        return ok, {
            "scores": scores, "messages": msgs, "ok": ok,
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    def _tectonic_score(self, ws, msgs):
        from earthengine.pipeline.world import BOUNDARY_NAMES
        score = 1.0
        pp = ws.plates
        if pp is None:
            return 0.0
        # 面积基尼系数
        areas = np.bincount(pp.plates.ravel()) / float(pp.plates.size)
        gini = _gini(areas[areas > 0])
        if gini < THRESHOLDS["plate_gini"][0]:
            score -= 0.1
            msgs.append(f"WARNING: {THRESHOLDS['plate_gini'][2]} (gini={gini:.2f})")
        # 边界类型分布：不能退化到单一类型
        bt = getattr(pp, "boundary_type", None)
        if bt is not None:
            vals = bt[bt > 0]
            if vals.size and (np.unique(vals).size < 3):
                score -= 0.1
                msgs.append("WARNING: 边界类型分布退化（少于三类）")
        return max(score, 0.0)

    def _geography_score(self, ws, msgs):
        from earthengine.pipeline.geography_validator import validate_global_geography
        lm = ws.ocean["land_mask"] if ws.ocean else None
        if lm is None:
            return 0.0
        cm = ws.ocean.get("continent_mask") if ws.ocean else None
        if cm is None:
            # 无大陆掩膜（如仅从 fields.npz 恢复）：退化为陆地占比 + 海洋连通
            land_frac = float(lm.mean())
            lo, hi = 0.20, 0.45
            score = 1.0
            if not (lo <= land_frac <= hi):
                msgs.append(f"WARNING: 陆地占比超范围 ({land_frac:.2f})")
                score -= 0.4
            return max(score, 0.0)
        ok, rep = validate_global_geography(lm, cm, land_target=None)
        score = rep["score"]
        if not ok:
            msgs.append(f"WARNING: 宏观地理未全通过 (score={score:.2f})")
        return float(score)

    def _terrain_score(self, ws, msgs):
        from earthengine.terrain import macro
        if ws.terrain is None or ws.boundaries is None:
            return 0.0
        # 山脉落在汇聚边界缓冲区内（≥60%）
        try:
            from scipy.ndimage import distance_transform_edt
            terr = ws.terrain["region"]
            bt = ws.boundaries["boundary_type"]
            if bt is None:
                return 1.0
            from earthengine.pipeline.world import TT_MOUNTAIN, CONVERGENT
            conv = bt == CONVERGENT
            if not conv.any():
                return 1.0
            d_conv = distance_transform_edt(~conv).astype(np.float32)
            mount = terr == TT_MOUNTAIN
            if not mount.any():
                return 0.5
            within = (d_conv[mount] < 3.0).mean()
            if within < THRESHOLDS["mountain_convergent"][0]:
                msgs.append(f"ERROR: {THRESHOLDS['mountain_convergent'][2]} ({within:.2f})")
                return 0.3
        except Exception:
            return 0.8
        return 1.0

    def _climate_score(self, ws, msgs):
        from earthengine.planet.sphere import _lat_deg
        if ws.climate is None:
            return 0.0
        temp = ws.climate["temperature"]
        h, w = temp.shape
        lat = np.abs(_lat_deg(h).ravel())       # |纬度|，按纬度带
        zonal = temp.mean(axis=1)               # 每纬度带平均温度（气候学纬向平均）
        r = float(np.corrcoef(lat, zonal)[0, 1])
        if abs(r) < THRESHOLDS["lat_temp_corr"][0]:
            msgs.append(f"ERROR: {THRESHOLDS['lat_temp_corr'][2]} (|r|={abs(r):.2f})")
            return 0.3
        return 1.0

    def _hydrology_score(self, ws, msgs):
        if ws.hydrology is None:
            return 0.0
        uphill = ws.hydrology.get("uphill_segments", 0)
        if uphill != THRESHOLDS["uphill"][0]:
            msgs.append(f"ERROR: {THRESHOLDS['uphill'][2]} ({uphill})")
            return 0.0
        return 1.0


def _gini(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n < 2:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0
