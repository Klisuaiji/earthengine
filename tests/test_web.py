"""阶段 D：Web 前后端参数契约测试 (10 节)。

验证统一 JSON envelope、参数校验、错误分类（input/generate/missing_dep），
以及生成路径能产出图层。不需要 flask 即可跑（直接测校验/封包逻辑）；
有 flask 时另行做真实 HTTP smoke。

用法：python3 tests/test_web.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earthengine.web import web_server as W


def test_envelope_ok():
    e = W._ok({"k": 1})
    assert e["success"] is True and e["data"] == {"k": 1} and e["error"] is None


def test_envelope_error_types():
    for t in ("input", "generate", "missing_dep"):
        e = W._err(t, "msg")
        assert e["success"] is False
        assert e["error"]["type"] == t


def test_param_validation_valid():
    err, p = W._validate_params({"seed": 5, "width": 256, "height": 128,
                                 "detail": "procedural"})
    assert err is None and p == {"seed": 5, "width": 256, "height": 128,
                                 "detail": "procedural"}


def test_param_validation_ratio():
    err, _ = W._validate_params({"width": 100, "height": 100})
    assert err and "2×height" in err


def test_param_validation_range():
    err, _ = W._validate_params({"width": 10, "height": 5})
    assert err


def test_param_validation_detail():
    err, _ = W._validate_params({"width": 256, "height": 128, "detail": "bogus"})
    assert err and "procedural" in err


def test_meta_contract():
    assert W.WORK_MODES == ["world", "terrain", "data"]
    assert W.TOOLS == ["navigate", "select", "inspect", "plates",
                       "terrain", "rivers", "measure"]
    # 图层按工作模式分组：每个 mode 至少一层
    groups = set(l["group"] for l in W.LAYERS.values())
    assert set(W.WORK_MODES) <= groups


def test_generate_end_to_end():
    """生成路径产图层（不经 HTTP，直接调用核心 + 渲染）。"""
    from earthengine.pipeline.stages import WorldGenerator
    from earthengine.pipeline.validator import WorldValidator
    from earthengine import rendering as R
    import numpy as np, base64, tempfile, os
    g = WorldGenerator({"seed": 7, "width": 128, "height": 64})
    ws = g.generate()
    v = WorldValidator(); ok, rep = v.validate(ws)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        p = f.name
    R.map.render_land_sea(ws.ocean["land_mask"], p)
    with open(p, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    os.unlink(p)
    assert len(b64) > 100 and ok is True
    assert rep["meta"]["resolution"] == "128x64"


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
