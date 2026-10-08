"""阶段缓存 (31 / pipeline/cache.py)：跨阶段复用中间结果。

简化实现：进程内 dict 缓存 key=阶段名；可选磁盘 npz 缓存。
"""

from __future__ import annotations

import os

import numpy as np


class StageCache:
    """简单阶段缓存。"""

    def __init__(self, root=None):
        self.root = root
        self._mem = {}

    def get(self, key):
        if key in self._mem:
            return self._mem[key]
        if self.root and os.path.exists(os.path.join(self.root, f"{key}.npy")):
            val = np.load(os.path.join(self.root, f"{key}.npy"))
            self._mem[key] = val
            return val
        return None

    def put(self, key, value):
        self._mem[key] = value
        if self.root:
            os.makedirs(self.root, exist_ok=True)
            np.save(os.path.join(self.root, f"{key}.npy"), value)
        return value

    def clear(self):
        self._mem.clear()
