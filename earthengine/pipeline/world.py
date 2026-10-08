"""WorldState：模块之间只通过它 / 明确数据对象交换 (30)。

所有场为 (H,W) 数组、等经纬投影、H:W = 1:2、缺省 1024×512。字段形状与 dtype
规格见 30.1 表。

``World`` 类迁移：保留 ``elevation/plates/precipitation/humidity/temperature/biome``
属性作为只读快捷方式，避免破坏现有渲染代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --- 地形类别枚举 (19 / A.2) ---
# v0：7 类；v2.0 升级为 12 类（补 ridge/trench/volcanic/rift/coastal）
TT_DEEP, TT_SHELF, TT_COASTAL, TT_PLAIN, TT_HILL, TT_PLATEAU, TT_MOUNTAIN, \
    TT_BASIN, TT_RIDGE, TT_TRENCH, TT_VOLCANIC, TT_RIFT = range(12)

TERRAIN_NAMES = [
    "深海", "大陆架", "沿海", "平原", "丘陵", "高原",
    "山脉", "盆地", "洋中脊", "海沟", "火山", "裂谷",
]
TERRAIN_RGB = {
    TT_DEEP: (16, 38, 74), TT_SHELF: (56, 110, 168), TT_COASTAL: (130, 160, 110),
    TT_PLAIN: (118, 168, 88), TT_HILL: (196, 184, 106), TT_PLATEAU: (156, 116, 74),
    TT_MOUNTAIN: (168, 168, 174), TT_BASIN: (150, 100, 140), TT_RIDGE: (110, 90, 130),
    TT_TRENCH: (30, 30, 60), TT_VOLCANIC: (210, 90, 70), TT_RIFT: (170, 120, 60),
}

# --- 边界类型 (7.4) ---
INTERIOR, CONVERGENT, DIVERGENT, TRANSFORM, OBLIQUE = 0, 1, 2, 3, 4
BOUNDARY_NAMES = ["内部", "汇聚", "张裂", "转换", "斜向"]

# --- 地壳类型 (11) ---
CRUST_CONTINENTAL, CRUST_OCEANIC, CRUST_TRANSITIONAL = 0, 1, 2
CRUST_NAMES = ["陆壳", "洋壳", "过渡壳"]


@dataclass
class WorldState:
    """一颗已生成行星的全部状态。各阶段写入自己的子对象。"""

    seed: int
    planet: object = None            # planet.parameters.Planet
    sphere: object = None            # planet.sphere.LatLonGrid
    plates: object = None            # tectonics 产物 dict / object
    boundaries: object = None
    crust: object = None
    ocean: object = None
    terrain: object = None
    atmosphere: object = None
    climate: object = None
    hydrology: object = None
    geology: object = None
    vegetation: object = None
    civilization: object = None
    terrain_conditioning: dict = field(default_factory=dict)
    render: object = None
    validation: object = None
    engine_version: str = "2.0.0"

    # ---- 只读快捷方式（旧渲染代码兼容） ----
    @property
    def elevation(self):
        return self.terrain.get("elevation") if self.terrain else None

    @property
    def land_mask(self):
        return self.ocean.get("land_mask") if self.ocean else None

    @property
    def temperature(self):
        return self.climate.get("temperature") if self.climate else None

    @property
    def precipitation(self):
        return self.climate.get("precipitation") if self.climate else None

    @property
    def humidity(self):
        return self.climate.get("humidity") if self.climate else None

    @property
    def biome(self):
        return self.vegetation.get("biome") if self.vegetation else None

    @property
    def h(self):
        if self.terrain and self.terrain.get("elevation") is not None:
            return self.terrain["elevation"].shape[0]
        if self.sphere is not None:
            return getattr(self.sphere, "H", None)
        return None

    @property
    def w(self):
        if self.terrain and self.terrain.get("elevation") is not None:
            return self.terrain["elevation"].shape[1]
        if self.sphere is not None:
            return getattr(self.sphere, "W", None)
        return None

    # ---- 汇总字段表（30.1） ----
    def field_summary(self) -> dict:
        """返回字段规格快照，供 world.json 记录与调试。"""
        shape = (None, None)
        if self.terrain and self.terrain.get("elevation") is not None:
            shape = self.terrain["elevation"].shape
        return {
            "shape": shape,
            "plates.id": "int16" if self.plates else None,
            "boundaries.type": "int8",
            "crust.type": "int8",
            "crust.age": "float32",
            "ocean.land_mask": "bool",
            "terrain.elevation_macro": "float32",
            "terrain.region": "int8",
            "climate.temperature": "float32",
            "climate.precipitation": "float32",
            "hydrology.flow_dir": "uint8",
            "hydrology.river_mask": "bool",
            "vegetation.biome": "int8",
            "civilization.score": "float32",
        }
