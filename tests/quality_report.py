"""多 seed 地理质量报告 (11 节 / 阶段 B)。

对一组 seed 生成固定分辨率世界，跑分级验证器，输出每 seed 指标 +
汇总 JSON 到 reports/quality_report.json。记录 seed、配置、分辨率、
代码版本、实际值/目标/单位。

用法：
    python3 tests/quality_report.py [--seeds 1,2,3] [--width 256] [--height 128]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from earthengine.pipeline.stages import WorldGenerator
from earthengine.pipeline.validator import WorldValidator


def run_one(seed, w, h):
    t0 = time.time()
    gen = WorldGenerator({"seed": seed, "width": w, "height": h})
    ws = gen.generate()
    v = WorldValidator()
    ok, rep = v.validate(ws)
    metrics = {c["name"]: c for c in rep["metrics"]}
    errors = [e["name"] for e in rep["errors"]]
    warnings = [w0["name"] for w0 in rep["warnings"]]
    return {
        "seed": seed, "width": w, "height": h,
        "passed": ok, "errors": errors, "warnings": warnings,
        "land_fraction": float(ws.ocean["land_mask"].mean()),
        "n_plates": int(np.unique(ws.plates.plates).size),
        "civ_points": len(ws.civilization["points"]),
        "metrics": {k: {"value": m["value"], "target": m["target"],
                        "unit": m["unit"], "ok": m["ok"]}
                    for k, m in metrics.items()},
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1234567,42,777,28070,99,2024,555")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--out", default="reports/quality_report.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = []
    for s in seeds:
        row = run_one(s, args.width, args.height)
        rows.append(row)
        flag = "OK" if row["passed"] else "FAIL"
        print(f"[{flag}] seed={s} land={row['land_fraction']:.3f} "
              f"n_plates={row['n_plates']} errors={row['errors']} "
              f"warnings={row['warnings']} ({row['elapsed_s']}s)")

    from earthengine import __version__
    summary = {
        "engine_version": __version__,
        "resolution": f"{args.width}x{args.height}",
        "seeds": args.seeds,
        "passed_seeds": sum(1 for r in rows if r["passed"]),
        "total_seeds": len(rows),
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    n_ok = summary["passed_seeds"]
    print(f"\nquality report -> {args.out}  ({n_ok}/{len(rows)} passed)")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
