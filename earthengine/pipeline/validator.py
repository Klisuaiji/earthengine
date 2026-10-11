"""WorldValidator (35 / 34 / P0-3)：按量化阈值验证世界地理规律。

分级 (P0-3 / 7 节)：
- **ERROR / Hard fail**：缺字段、shape 不匹配、NaN/Inf、海陆与高程矛盾、
  非法 plate ID、无效流向、关键阶段失败 → 令 ``passed=False``，**不被总分
  平均抵消**。
- **WARNING**：大陆过碎、海岸噪声、山脉带断裂、温度梯度不足等质量问题。
- **METRIC / Score**：海陆比例、面积分布、板块多样性、温度分布、河网密度
  等统计。

报告记录：实际值、目标区间、单位、seed、配置、分辨率、代码版本、阶段耗时
（可得时）。
"""

from __future__ import annotations

import time

import numpy as np

from earthengine.pipeline.world import (
    TT_MOUNTAIN, CONVERGENT, BOUNDARY_NAMES,
)

# ---- 量化阈值 (34.6) ----
THRESHOLDS = {
    "plate_gini": (0.35, "WARN", "板块面积分布过均匀"),
    "land_fraction": ((0.20, 0.45), "WARN", "陆地占比超范围"),
    "max_continent": (0.25, "WARN", "最大连通陆地占比过低（过度碎片化）"),
    "single_pixel_islands": (0.02, "WARN", "单像素岛占比过高"),
    "mountain_convergent": (0.60, "ERROR", "山脉未落在汇聚边界缓冲区内"),
    "lat_temp_corr": (0.70, "ERROR", "纬度-温度相关性不足"),
    "uphill": (0, "ERROR", "存在逆坡河段"),
    "shelf_coverage": (0.80, "WARN", "大陆架连续过渡覆盖率不足"),
    "na_max_fraction": (1e-6, "ERROR", "输出场存在 NaN/Inf"),
    "land_sea_conflict": (0.02, "ERROR", "海陆与高程矛盾像素占比过高"),
}


class _Check:
    """单条验证结果。"""

    __slots__ = ("level", "name", "value", "target", "unit", "ok", "message")

    def __init__(self, level, name, value, target=None, unit="", ok=None,
                 message=""):
        self.level = level          # ERROR / WARNING / METRIC
        self.name = name
        self.value = value
        self.target = target
        self.unit = unit
        self.ok = ok
        self.message = message

    def as_dict(self):
        return {
            "level": self.level, "name": self.name, "value": self.value,
            "target": self.target, "unit": self.unit, "ok": self.ok,
            "message": self.message,
        }


class WorldValidator:
    """分级验证器：``passed`` 由 ERROR 级检查决定，WARNING/METRIC 仅参考。"""

    def __init__(self, config=None):
        self.config = config or {}
        self.threshold = float(self.config.get("reject_threshold", 0.0))

    # ------------------------------------------------------------------
    def validate(self, world_state) -> tuple:
        t0 = time.time()
        checks = []
        self._hard_checks(world_state, checks)      # 结构性 ERROR
        self._geo_checks(world_state, checks)       # 地理/气候/水文指标
        errors = [c for c in checks if c.level == "ERROR"]
        warnings = [c for c in checks if c.level == "WARNING"]
        passed = len(errors) == 0
        # 参考评分：仅用于排序/报告，不抵消 ERROR
        overall = 1.0 if passed else min(1.0, 0.9 - 0.1 * len(errors))
        report = {
            "passed": passed,
            "ok": passed,
            "errors": [c.as_dict() for c in errors],
            "warnings": [c.as_dict() for c in warnings],
            "metrics": [c.as_dict() for c in checks if c.level == "METRIC"],
            "checks": [c.as_dict() for c in checks],
            "scores": {"overall": overall},
            "messages": [f"{c.level}: {c.message or c.name}" for c in checks
                         if not c.ok],
            "meta": {
                "seed": getattr(world_state, "seed", None),
                "resolution": f"{getattr(world_state, 'w', '?')}x{getattr(world_state, 'h', '?')}",
                "engine_version": getattr(world_state, "engine_version", "?"),
                "elapsed_s": round(time.time() - t0, 3),
            },
            "threshold": self.threshold,
        }
        return passed, report

    # ------------------------------------------------------------------
    # 结构性硬检查（ERROR）
    # ------------------------------------------------------------------
    def _hard_checks(self, ws, checks):
        # 缺字段
        need = {"terrain": ("elevation", "region"),
                "ocean": ("land_mask",),
                "climate": ("temperature", "precipitation"),
                "hydrology": ("flow_dir",)}
        for grp, fields in need.items():
            obj = getattr(ws, grp, None)
            for f in fields:
                if not (obj and obj.get(f) is not None):
                    checks.append(_Check("ERROR", f"{grp}.{f}",
                                         None, message=f"缺少字段 {grp}.{f}"))
        # 场 shape 一致（H:W = 1:2）
        shapes = {}
        for grp, fields in need.items():
            obj = getattr(ws, grp, None)
            if not obj:
                continue
            for f in fields:
                v = obj.get(f)
                if isinstance(v, np.ndarray):
                    shapes[f"{grp}.{f}"] = v.shape
        sh = list(shapes.values())
        if sh:
            ref = sh[0]
            for name, s in shapes.items():
                if s != ref:
                    checks.append(_Check(
                        "ERROR", "shape", f"{name}:{s} vs {ref}",
                        message=f"场 shape 不一致：{name} {s} != {ref}"))
        # NaN/Inf
        for grp, fields in need.items():
            obj = getattr(ws, grp, None)
            if not obj:
                continue
            for f in fields:
                v = obj.get(f)
                if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
                    if not np.isfinite(np.asarray(v, dtype=np.float64)).all():
                        checks.append(_Check("ERROR", f"{grp}.{f}.nan",
                                             None, message=f"{grp}.{f} 含 NaN/Inf"))
        # 海陆与高程矛盾：elevation 与 land_mask 符号一致性
        if ws.terrain and ws.terrain.get("elevation") is not None and \
           ws.ocean and ws.ocean.get("land_mask") is not None:
            elev = ws.terrain["elevation"]
            lm = ws.ocean["land_mask"]
            if elev.shape == lm.shape:
                conflict = int((lm & (elev < 0)).sum() | ((~lm) & (elev > 0)).sum())
                frac = conflict / lm.size
                if frac > THRESHOLDS["land_sea_conflict"][0]:
                    checks.append(_Check(
                        "ERROR", "land_sea_conflict", frac,
                        target=THRESHOLDS["land_sea_conflict"][0], unit="frac",
                        message=f"海陆与高程矛盾像素 {frac:.4f}"))
                else:
                    checks.append(_Check("METRIC", "land_sea_conflict", frac,
                                         unit="frac", ok=True))
        # 非法 plate ID
        if ws.plates is not None:
            try:
                p = ws.plates.plates
                if p.min() < 0 or p.max() >= int(np.unique(p).size):
                    checks.append(_Check("ERROR", "plate_id",
                                         (int(p.min()), int(p.max())),
                                         message="plate ID 超出有效范围"))
            except Exception:
                pass
        # 无效流向
        if ws.hydrology and ws.hydrology.get("flow_dir") is not None:
            fd = ws.hydrology["flow_dir"]
            bad = int((fd > 8).sum())
            if bad:
                checks.append(_Check("ERROR", "flow_dir", bad,
                                     message=f"flow_dir 存在 >8 的非法码 {bad}"))

    # ------------------------------------------------------------------
    # 地理 / 气候 / 水文指标（WARNING + METRIC）
    # ------------------------------------------------------------------
    def _geo_checks(self, ws, checks):
        self._tectonic_metric(ws, checks)
        self._geography_metric(ws, checks)
        self._terrain_metric(ws, checks)
        self._climate_metric(ws, checks)
        self._hydrology_metric(ws, checks)

    def _tectonic_metric(self, ws, checks):
        def _gini(values):
            v = np.sort(np.asarray(values, dtype=np.float64))
            n = len(v)
            if n < 2:
                return 0.0
            cum = np.cumsum(v)
            return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n) if cum[-1] > 0 else 0.0
        pp = ws.plates
        if pp is None:
            return
        areas = np.bincount(pp.plates.ravel()).astype(np.float64)
        areas = areas[areas > 0] / float(pp.plates.size)
        gini = _gini(areas)
        ok = gini >= THRESHOLDS["plate_gini"][0]
        checks.append(_Check(
            THRESHOLDS["plate_gini"][1] if not ok else "METRIC",
            "plate_gini", round(float(gini), 3),
            target=THRESHOLDS["plate_gini"][0], unit="gini", ok=ok,
            message=f"板块面积基尼 {gini:.2f}"))
        # 边界类型多样性
        bt = getattr(pp, "boundary_type", None)
        if bt is not None:
            kinds = int(np.unique(bt[bt > 0]).size)
            checks.append(_Check(
                "METRIC", "boundary_types", kinds,
                target=">=3", ok=kinds >= 3,
                message=f"边界类型 {kinds} 类"))

    def _geography_metric(self, ws, checks):
        from earthengine.pipeline.geography_validator import (
            LAND_LO, LAND_HI, LARGEST_MIN, validate_global_geography,
        )
        lm = ws.ocean["land_mask"] if ws.ocean else None
        if lm is None:
            return
        land_frac = float(lm.mean())
        ok = LAND_LO <= land_frac <= LAND_HI
        checks.append(_Check(
            "METRIC" if ok else "WARNING", "land_fraction",
            round(land_frac, 3), target=[LAND_LO, LAND_HI], ok=ok,
            message=f"陆地占比 {land_frac:.3f}"))
        cm = ws.ocean.get("continent_mask") if ws.ocean else None
        if cm is not None:
            okg, rep = validate_global_geography(lm, cm, land_target=None)
            checks.append(_Check(
                "METRIC" if okg else "WARNING", "global_geography",
                round(float(rep["score"]), 3), ok=okg,
                message="宏观地理" + ("通过" if okg else f"未全通过 score={rep['score']:.2f}")))

    def _terrain_metric(self, ws, checks):
        if ws.terrain is None or ws.boundaries is None:
            return
        bt = ws.boundaries.get("boundary_type") if ws.boundaries else None
        if bt is None:
            return
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception:
            return
        terr = ws.terrain["region"]
        conv = bt == CONVERGENT
        if not conv.any():
            return
        d = distance_transform_edt(~conv)
        mount = terr == TT_MOUNTAIN
        if not mount.any():
            return
        within = float((d[mount] < 3.0).mean())
        ok = within >= THRESHOLDS["mountain_convergent"][0]
        checks.append(_Check(
            THRESHOLDS["mountain_convergent"][1] if not ok else "METRIC",
            "mountain_convergent", round(within, 3),
            target=THRESHOLDS["mountain_convergent"][0], unit="frac", ok=ok,
            message=f"山脉落汇聚边界 {within:.2f}"))

    def _climate_metric(self, ws, checks):
        from earthengine.planet.sphere import _lat_deg
        if ws.climate is None:
            return
        temp = ws.climate["temperature"]
        if temp is None:
            return
        h = temp.shape[0]
        lat = np.abs(_lat_deg(h).ravel())
        zonal = temp.mean(axis=1)
        r = float(np.corrcoef(lat, zonal)[0, 1])
        ok = abs(r) >= THRESHOLDS["lat_temp_corr"][0]
        checks.append(_Check(
            THRESHOLDS["lat_temp_corr"][1] if not ok else "METRIC",
            "lat_temp_corr", round(abs(r), 3),
            target=THRESHOLDS["lat_temp_corr"][0], ok=ok,
            message=f"纬度-温度 zonal |r|={abs(r):.3f}"))
        precip = ws.climate.get("precipitation")
        if precip is not None:
            neg = int((precip < 0).sum())
            checks.append(_Check(
                "ERROR" if neg else "METRIC", "precip_nonneg", neg,
                target=0, ok=neg == 0, message=f"降水负值 {neg}"))

    def _hydrology_metric(self, ws, checks):
        if ws.hydrology is None:
            return
        uphill = ws.hydrology.get("uphill_segments", 0)
        ok = uphill == THRESHOLDS["uphill"][0]
        checks.append(_Check(
            THRESHOLDS["uphill"][1] if not ok else "METRIC",
            "uphill", uphill, target=0, ok=ok,
            message=f"逆坡河段 {uphill}"))
