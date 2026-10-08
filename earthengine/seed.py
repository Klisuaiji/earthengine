"""可复现性核心：SeedManager。

统一管理各阶段随机流。v2.0 可复现性要求 (38)：随机性只由 seed + 阶段/板块标识决定，
与像素坐标、分辨率无关；各阶段不再混用全局 np.random。

用法：
    rng = seed_manager.derive("tectonics.motion", seed)   # 每阶段独立子流
    rng = seed_manager.derive(("boundary", a, b), seed)  # 每板块对独立子流
"""

from __future__ import annotations

import numpy as np

_INT64_MAX = np.iinfo(np.int64).max


def _mix(parts) -> int:
    """Deterministic 64-bit hash of an arbitrary key (str / int / tuple)."""
    if isinstance(parts, tuple):
        h = 1469598103934665603
        for p in parts:
            h = (h * 1099511628211) & _INT64_MAX
            h ^= _mix(p) & _INT64_MAX
        return h
    if isinstance(parts, str):
        return _mix(tuple(ord(c) for c in parts))
    return int(parts) & _INT64_MAX


class SeedManager:
    """Deterministic, resolution-independent RNG derivation.

    ``rng(scope, base_seed)`` returns a fresh ``np.random.Generator`` whose state
    depends only on ``base_seed`` and the scope key — never on pixel coordinates
    or grid resolution, so the same seed reproduces the same world at any size.
    """

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed) & _INT64_MAX

    def derive(self, scope, salt: int = 0) -> np.random.Generator:
        key = _mix((self.base_seed, _mix(scope), int(salt)))
        return np.random.default_rng(key)

    def seed_for(self, scope, salt: int = 0) -> int:
        return _mix((self.base_seed, _mix(scope), int(salt)))
