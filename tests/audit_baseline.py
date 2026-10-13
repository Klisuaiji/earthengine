"""阶段 0：当前版本基准与数据流审计（可运行、产出明确失败项）。

固定 seed 生成标准世界，分层检查：
- 数据流一致性：ocean.land_mask 与 terrain 高程符号是否同源；plate/crust 是否进高程
- 两个优先验证点：①best-effort 失败报告是否可见  ②地壳演化是否与高程耦合
- 宏观指标：大陆数量/最大连通陆块/孤立小岛/海陆比例/高程范围/geography 失败项

用法：python3 tests/audit_baseline.py [--seeds 123456,42,777]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import ndimage

from earthengine.pipeline.stages import WorldGenerator
from earthengine.pipeline.validator import WorldValidator


def audit_one(seed, w=256, h=128):
    """对单个 seed 生成世界并做数据流审计，返回结构化结果 + 失败项。"""
    findings = []           # 明确失败项（不带"疑似"，只有坐实/未实测两类）
    g = WorldGenerator({"seed": seed, "width": w, "height": h})
    ws = g.generate()
    # 复现 CLI 真实行为：WorldValidator + 合并大陆验证报告
    ok, report = WorldValidator().validate(ws)
    geo_rep = getattr(g, "_tect", {}).get("geography_report")
    if geo_rep:
        g_checks = geo_rep.get("checks") or {}
        report.setdefault("geography", {})
        report["geography"].update({"passed": bool(g_checks) and all(g_checks.values()),
                                    "checks": g_checks, "score": geo_rep.get("score")})
    ws.validation = report
    # ---- 数据流：ocean.land_mask 与 terrain 是否同一份 ----
    lm = ws.ocean["land_mask"]
    elev = ws.terrain["elevation"]
    # terrain 高程符号应遵守同一 land_mask（陆地>0 海洋<0）
    conflict = int((lm & (elev < 0)).sum() + ((~lm) & (elev > 0)).sum())
    # ---- 地壳-高程耦合（优先验证点②）----
    cs = ws.crust
    b0 = getattr(g, "_crust_bias0", None)
    has_bias = hasattr(cs, "elevation_bias") and b0 is not None
    bias = getattr(cs, "elevation_bias", None)
    r_bias = 0.0
    eff_present = False
    n_eff = 0
    dh_construct = 0.0
    if has_bias and bias is not None:
        deff = bias - b0                       # 演化驱动的构造偏差场
        mask = np.abs(deff) > 50.0
        n_eff = int(mask.sum())
        hi = deff > 1500.0                     # 造山带(+)
        lo = deff < -1500.0                    # 海沟带(-)
        if n_eff > 50:
            eff_present = True
            r_bias = float(np.corrcoef(deff[mask], elev[mask])[0, 1])
            if hi.sum() > 20 and lo.sum() > 20:
                # 方向判据：造山带平均高程应显著高于海沟带（>500m）
                dh_construct = float(elev[hi].mean() - elev[lo].mean())
    # ---- best-effort 失败报告可见性（优先验证点①）----
    validation = ws.validation
    # geography_report 是否进入最终输出（world.json 的 validation）
    geo_visible = bool(validation) and ("geography" in validation)
    geo_ok = bool(geo_rep and geo_rep.get("checks") and all(geo_rep["checks"].values()))
    if geo_rep and not geo_ok and not geo_visible:
        findings.append({
            "id": "best_effort_failure_invisible",
            "level": "ERROR",
            "msg": "大陆验证 best-effort 未通过但失败项未进入 WorldState.validation",
            "detail": {k: v for k, v in (geo_rep.get("checks") or {}).items()},
        })
    if eff_present and dh_construct > 0 and dh_construct < 500:
        findings.append({
            "id": "crust_elevation_uncoupled",
            "level": "WARN",
            "msg": "造山带/海沟带高程差过小(<500m)——构造效应未有效进入高程",
            "detail": {"dh_construct_m": round(dh_construct, 1),
                       "r_construct": round(r_bias, 3), "n_eff_px": n_eff},
        })
    # ---- 宏观指标 ----
    lab, n = ndimage.label(lm.astype(bool), structure=np.ones((3, 3)))
    sizes = ndimage.sum(lm, lab, range(1, n + 1))
    tot = lm.sum()
    cc_main = float(sizes.max()) / tot if n else 0.0
    d_coast = ndimage.distance_transform_edt(~lm.astype(bool))
    lone = int(sum(1 for i in range(n) if sizes[i] < 4 and d_coast[lab == (i + 1)].min() > 10))
    # geography 失败项明细
    geo_fails = []
    if geo_rep:
        geo_fails = [k for k, v in (geo_rep.get("checks") or {}).items() if not v]
    return {
        "seed": seed, "w": w, "h": h,
        "land_frac": float(lm.mean()),
        "elev_range_m": [float(elev.min()), float(elev.max())],
        "land_sea_conflict_px": conflict,
        "n_cc": n, "cc_mainland": round(cc_main, 3),
        "lone_islets_lt4px": lone,
        "crust_elevation_corr": round(r_bias, 3),
        "geo_checks_passed": geo_ok,
        "geo_fail_items": geo_fails,
        "geography_report_visible": geo_visible,
        "findings": findings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="123456,42,777")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--out", default="reports/baseline_audit.json")
    args = ap.parse_args()
    rows = []
    for s in [int(x) for x in args.seeds.split(",")]:
        r = audit_one(s, args.width, args.height)
        rows.append(r)
        fids = [f["id"] for f in r["findings"]]
        print(f"seed={s} land={r['land_frac']:.3f} cc_main={r['cc_mainland']} "
              f"geo_ok={r['geo_checks_passed']} visible={r['geography_report_visible']} "
              f"crust_corr={r['crust_elevation_corr']} findings={fids or '无'}")
    # 汇总所有失败项（跨 seed 去重）
    all_f = []
    seen = set()
    for r in rows:
        for f in r["findings"]:
            if f["id"] not in seen:
                seen.add(f["id"]); all_f.append(f)
    summary = {
        "engine_version": "audit-baseline",
        "seed_count": len(rows), "rows": rows,
        "consolidated_findings": all_f,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n审计报告 -> {args.out}")
    print(f"确认失败项 {len(all_f)} 个：{[f['id'] for f in all_f]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
