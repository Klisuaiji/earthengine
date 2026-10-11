"""阶段 A 回归测试 (P0-1/P0-3)：依赖闭包 + 验证器分级 + 确定性。

可直接 ``python3 tests/test_pipeline.py`` 运行（无 pytest 依赖）。
"""

from __future__ import annotations

import os
import sys

# 允许直接 `python3 tests/test_pipeline.py`：把仓库根加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from earthengine.pipeline.stages import WorldGenerator
from earthengine.pipeline.validator import WorldValidator
from earthengine.pipeline import dependencies as deps


def test_acyclic():
    assert deps.detect_cycle() == [], "依赖图存在环"


def test_dependency_closure():
    g = WorldGenerator({"seed": 1, "width": 128, "height": 64})
    g.generate(stages=["terrain"])
    # terrain 必须自动补齐 planet/tectonics/crust/ocean 前置
    assert g._executed_stages == [
        "planet", "tectonics", "crust", "ocean", "terrain"], g._executed_stages
    assert "land_mask" in g._ocean
    assert "elevation" in g._terrain


def test_climate_prereqs():
    g = WorldGenerator({"seed": 1, "width": 128, "height": 64})
    g.generate(stages="climate")
    assert g._executed_stages[:3] == ["planet", "tectonics", "crust"]
    assert "terrain" in g._executed_stages and "atmosphere" in g._executed_stages
    assert "temperature" in g._climate


def test_unknown_stage_error():
    g = WorldGenerator({"seed": 1, "width": 64, "height": 32})
    try:
        g.generate(stages=["bogus"])
        raise AssertionError("应抛 StageDependencyError")
    except deps.StageDependencyError as e:
        assert "bogus" in str(e)


def test_validator_grades():
    g = WorldGenerator({"seed": 1234567, "width": 256, "height": 128})
    ws = g.generate()
    v = WorldValidator()
    ok, rep = v.validate(ws)
    assert ok is True
    assert len(rep["errors"]) == 0
    # 报告含 meta
    assert rep["meta"]["resolution"] == "256x128"


def test_hard_error_not_averaged():
    """硬性 ERROR 必须令 passed=False，不得被平均分抵消。"""
    import copy
    g = WorldGenerator({"seed": 1234567, "width": 256, "height": 128})
    ws = g.generate()
    v = WorldValidator()
    ok, rep = v.validate(ws)
    assert ok is True
    # 注入海陆矛盾（全部反转 mask → 大量矛盾像素）
    bad = copy.deepcopy(ws)
    bad.ocean["land_mask"] = ~bad.ocean["land_mask"]
    ok2, rep2 = v.validate(bad)
    assert ok2 is False
    names = [e["name"] for e in rep2["errors"]]
    assert "land_sea_conflict" in names


def test_deterministic():
    g1 = WorldGenerator({"seed": 42, "width": 128, "height": 64})
    ws1 = g1.generate()
    g2 = WorldGenerator({"seed": 42, "width": 128, "height": 64})
    ws2 = g2.generate()
    assert np.allclose(ws1.terrain["elevation"], ws2.terrain["elevation"])


_ALL = [v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)]


def run_all():
    failed = 0
    for fn in _ALL:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(_ALL)-failed}/{len(_ALL)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
