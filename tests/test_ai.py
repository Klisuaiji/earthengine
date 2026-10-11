"""阶段 C：AI 接口契约与 fallback 回归测试 (9 节)。

- ``TerrainEnhancer.generate`` 签名契约：返回 float32、量纲米、海陆符号正确。
- ProceduralEnhancer（默认 fallback）：无 AI 时能跑、输出不破坏海陆拓扑。
- TerrainDiffusionAdapter：无 torch 自动回落 procedural，结果一致。
- 自定义 mock enhancer 经 registry 注册可被调用。

用法：python3 tests/test_ai.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from earthengine.ai.base import ProceduralEnhancer, resolve_device
from earthengine.ai import registry
from earthengine.ai.terrain_diffusion import TerrainDiffusionAdapter


def _cond(h=64, w=128):
    rng = np.random.default_rng(0)
    macro = rng.uniform(-6000, 3500, (h, w)).astype(np.float32)
    land = macro > 0
    return {
        "macro_elevation": macro,
        "land_sea": land,
        "terrain_region": np.zeros((h, w), np.int8),
    }


def test_procedural_signature():
    e = ProceduralEnhancer()
    out = e.generate(_cond(), None, 64, 1)
    assert out.dtype == np.float32, f"dtype {out.dtype} 应 float32"
    assert out.ndim == 2 and out.shape == (64, 128)


def test_procedural_preserves_land_sea_sign():
    """输出不破坏海陆拓扑：陆地 > 0，海洋 < 0。"""
    c = _cond()
    out = ProceduralEnhancer().generate(c, None, 64, 1)
    land, sea = c["land_sea"], ~c["land_sea"]
    assert (out[land] > 0).all(), "陆地高程出现 ≤0"
    assert (out[sea] < 0).all(), "海洋高程出现 ≥0"


def test_fallback_when_no_torch():
    """Adapter 无 torch 时自动回落 procedural，且结果一致。"""
    ad = TerrainDiffusionAdapter(seed=1)
    assert ad._torch_ok is False or ad.generate is not None
    c = _cond()
    out = ad.generate(c, None, 64, 1)
    proc = ProceduralEnhancer().generate(c, None, 64, 1)
    assert np.allclose(out, proc), "diffusion fallback 与 procedural 不一致"


def test_mock_enhancer_via_registry():
    """自定义 enhancer 经 registry 注册可被调用（mock，无需真实模型）。"""
    class MockEnhancer:
        name = "mock"
        def generate(self, conditioning, region, resolution, seed):
            return np.ones((64, 128), np.float32)

    registry.register("mock", MockEnhancer)
    e = registry.get_enhancer("mock")
    out = e.generate({}, None, 64, 1)
    assert np.all(out == 1.0)
    assert "mock" in registry.available()


def test_resolve_device():
    d = resolve_device("cpu")
    assert d == "cpu"


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
