"""阶段编排 (31)：WorldGenerator.generate(config)。

内部顺序：
    initialize_planet → simulate_tectonics → generate_crust → generate_ocean
    → generate_macro_terrain → simulate_atmosphere → simulate_climate
    → simulate_hydrology → simulate_geology → build_terrain_conditioning
    → enhance_terrain → simulate_vegetation → score_civilization → render_maps

允许选择阶段：generate(stages=["planet","tectonics","ocean","terrain"])。
每个阶段用独立 SeedManager 子流，保证可复现 (38)。
"""

from __future__ import annotations

import numpy as np

from earthengine.seed import SeedManager
from earthengine.planet.parameters import Planet
from earthengine.planet.sphere import LatLonGrid
from earthengine.pipeline.world import (
    WorldState, TERRAIN_NAMES, BOUNDARY_NAMES, CRUST_NAMES,
)

# 阶段别名 → 执行函数列表（按依赖顺序）
_STAGE_ORDER = [
    "planet", "tectonics", "crust", "ocean", "terrain",
    "atmosphere", "climate", "hydrology", "geology",
    "conditioning", "enhance", "vegetation", "civilization", "render",
]
_ALIASES = {
    "tectonic": "tectonics", "ocean": "ocean", "terrain": "terrain",
    "climate": "climate", "hydrology": "hydrology",
}


class WorldGenerator:
    """世界生成器：核心无 AI 也能跑完整世界 (26)。"""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.seed = int(config.get("seed", 12345))
        self.w = int(config.get("width", 1024))
        self.h = int(config.get("height", 512))
        self.detail = config.get("detail", "procedural")
        self.plate_model = config.get("plate_model", "motion")
        self.device = config.get("device", "auto")
        self.snr0 = config.get("snr0", 0.5)
        self.land_target = config.get("land_fraction", 0.30)
        self.ocean_target = config.get("ocean_fraction", 0.70)
        self.sm = SeedManager(self.seed)
        self.planet = None
        self.sphere = None
        self._stages = {}

    # ------------------------------------------------------------------
    def generate(self, stages=None) -> WorldState:
        """运行（或选择部分）阶段，返回完整 WorldState。"""
        if stages is None:
            stages = _STAGE_ORDER
        elif isinstance(stages, str):
            stages = [_ALIASES.get(s, s) for s in stages.split(",")]
        want = set(_ALIASES.get(s, s) for s in stages)
        for s in _STAGE_ORDER:
            if s in want:
                getattr(self, f"stage_{s}")()
        return self.finalize()

    # ------------------------------------------------------------------
    # 各阶段
    # ------------------------------------------------------------------
    def stage_planet(self):
        self.planet = Planet(
            radius_km=self.config.get("radius_km", 6371.0),
            rotation_period_h=self.config.get("rotation_period_h", 24.0),
            axial_tilt_deg=self.config.get("axial_tilt_deg", 23.4),
            star_distance_au=self.config.get("star_distance_au", 1.0),
            star_luminosity=self.config.get("star_luminosity", 1.0),
            ocean_fraction=self.ocean_target,
            initial_temperature_c=self.config.get("initial_temperature", 15.0),
        )
        self.sphere = LatLonGrid(self.h, self.w)

    def stage_tectonics(self):
        from earthengine.tectonics import plates as P
        if self.plate_model == "motion":
            pp, lm, cm, geo_rep = P.generate_validated_world(
                self.seed, w=self.w, h=self.h)
            self._tect = {
                "raw": pp.raw, "merged": pp.plates,
                "land_mask": lm, "continent_mask": cm,
                "boundary_type": pp.boundary_type,
                "convergent_dist": pp.convergent_dist,
                "divergent_dist": pp.divergent_dist,
                "boundary_dist": pp.boundary_dist,
                "plate_crust": pp.plate_crust, "is_ocean": pp.is_ocean,
                "plates_meta": pp, "geography_report": geo_rep,
            }
        else:
            from earthengine.tectonics import voronoi as SV
            raw, merged, lm, cm = SV.generate_spherical_world(
                self.seed, w=self.w, h=self.h)
            self._tect = {
                "raw": raw, "merged": merged,
                "land_mask": lm, "continent_mask": cm,
            }
        self._stages["tectonics"] = self._tect

    def stage_crust(self):
        from earthengine.tectonics import crust as C, evolution as E
        t = self._tect
        cs = C.generate_crust(t["merged"], t.get("is_ocean", []),
                              self.h, self.w, self.seed)
        if "boundary_type" in t:
            cs = E.evolve_crust(cs, t["boundary_type"], self.h, self.w)
        self._crust = cs
        self._stages["crust"] = cs

    def stage_ocean(self):
        from earthengine.ocean import sea_level
        t = self._tect
        if "land_mask" in t and t.get("land_mask") is not None:
            lm = t["land_mask"]
        else:
            lm = np.zeros((self.h, self.w), dtype=bool)
        self._ocean = {
            "land_mask": lm.astype(bool),
            "continent_mask": t.get("continent_mask"),
        }
        self._stages["ocean"] = self._ocean

    def stage_terrain(self):
        from earthengine.terrain import macro, planner, detail
        from earthengine.ocean import sea_level
        t = self._tect
        # 板块 land_mask（大陆布局）作为地理形状底
        if t.get("land_mask") is not None:
            geo_lm = t["land_mask"]
        else:
            geo_lm = np.zeros((self.h, self.w), dtype=bool)
        bounds = {
            "boundary_type": t.get("boundary_type",
                                   np.zeros((self.h, self.w), np.int8)),
            "convergent_dist": t.get("convergent_dist",
                                     np.full((self.h, self.w), 1e6, np.float32)),
            "divergent_dist": t.get("divergent_dist",
                                    np.full((self.h, self.w), 1e6, np.float32)),
        }
        coarse = macro.coarse_elevation(geo_lm, bounds)
        # 用海平面强制陆地占比（flood 低海岸地，确定性、非随机抖动）
        target_land = max(0.25, min(0.35, self.land_target))
        sea_level_m = sea_level.ocean_fraction_to_sea_level(coarse, 1.0 - target_land)
        geo_elev, terrain_class = planner.terrain_region_map(geo_lm, bounds, self.seed)
        # 细化海岸线（分形场分位数阈值精确锁定 target_land）+ 程序化高程
        final_land = detail.fractalize_coast(geo_lm, self.seed, target_land=target_land)
        elev, geo_elev, terrain_class = detail.procedural_elevation(final_land, bounds, self.seed)
        self._ocean["land_mask"] = final_land
        self._ocean["sea_level_m"] = sea_level_m
        self._terrain = {
            "elevation": elev, "elevation_macro": geo_elev,
            "region": terrain_class, "coarse": coarse,
        }
        self._stages["terrain"] = self._terrain

    def stage_atmosphere(self):
        from earthengine.atmosphere.temperature import solar_temperature, continental_pressure
        from earthengine.atmosphere.circulation import wind_field
        elev = self._terrain["elevation"]
        temp = solar_temperature(self.planet, elev, self.h)
        pressure = continental_pressure(temp, elev, self.h)
        u, v = wind_field(self.planet, elev, temp, self.h, self.w)
        self._atmosphere = {
            "temperature": temp, "pressure": pressure, "u": u, "v": v,
        }
        self._stages["atmosphere"] = self._atmosphere

    def stage_climate(self):
        from earthengine.climate.precipitation import precipitation
        from earthengine.climate.climate import koppen, ice_layer, ocean_currents
        elev = self._terrain["elevation"]
        a = self._atmosphere
        precip = precipitation(a["u"], a["v"], elev, self.h)
        code = koppen(a["temperature"], precip, elev, self.h)
        cu, cv, sst = ocean_currents(a["u"], a["v"], elev, self.h, self.w)
        ice = ice_layer(a["temperature"], elev, self.h, sst=sst)
        self._climate = {
            "temperature": a["temperature"], "precipitation": precip,
            "koppen": code, "current_u": cu, "current_v": cv, "sst": sst,
            "ice": ice,
        }
        self._stages["climate"] = self._climate

    def stage_hydrology(self):
        from earthengine.hydrology.flow import d8_flow
        from earthengine.hydrology.rivers import rivers_from_acc, ocean_outlets
        from earthengine.hydrology.lakes import lakes, closed_basins
        elev = self._terrain["elevation"]
        lm = self._ocean["land_mask"]
        flow_dir, acc, uphill = d8_flow(elev, lm)
        min_acc = max(90, (self.w * self.h) // 12000)
        rivers, _ = rivers_from_acc(acc, lm, min_acc=min_acc, h=self.h, w=self.w)
        lake = lakes(flow_dir, acc, lm, self.h, self.w)
        outlets = ocean_outlets(flow_dir, acc, lm, self.h, self.w)
        self._hydrology = {
            "flow_dir": flow_dir, "flow_acc": acc, "river_mask": rivers,
            "lakes": lake, "outlets": outlets, "uphill_segments": uphill,
            "min_acc": min_acc,
        }
        self._stages["hydrology"] = self._hydrology

    def stage_geology(self):
        from earthengine.geology.erosion import erosion, rock_hardness
        from earthengine.geology.faults import fault_map
        elev = self._terrain["elevation"]
        lm = self._ocean["land_mask"]
        precip = self._climate["precipitation"]
        eroded = erosion(elev, lm, precip, seed=self.seed)
        self._terrain["elevation"] = eroded
        t = self._tect
        bt = t.get("boundary_type", np.zeros((self.h, self.w), np.int8))
        self._geology = {
            "faults": fault_map(bt, self.h, self.w),
            "rock": rock_hardness(self._terrain["region"], self.seed),
        }
        self._stages["geology"] = self._geology

    def stage_conditioning(self):
        from earthengine.terrain.planner import build_conditioning
        t = self._tect
        bounds = {
            "boundary_type": t.get("boundary_type",
                                   np.zeros((self.h, self.w), np.int8)),
            "convergent_dist": t.get("convergent_dist",
                                     np.full((self.h, self.w), 1e6, np.float32)),
            "divergent_dist": t.get("divergent_dist",
                                    np.full((self.h, self.w), 1e6, np.float32)),
        }
        self._conditioning = build_conditioning(
            self._terrain["elevation_macro"], self._terrain["region"],
            self._ocean["land_mask"], bounds,
            climate=self._climate)
        self._stages["conditioning"] = self._conditioning

    def stage_enhance(self):
        from earthengine.ai.registry import get_enhancer
        from earthengine.ai.terrain_diffusion import TerrainDiffusionAdapter
        if self.detail == "diffusion":
            enh = TerrainDiffusionAdapter(seed=self.seed, device=self.device,
                                          snr0=self.snr0)
        else:
            enh = get_enhancer("procedural")
        elev = enh.generate(self._conditioning, None, self.h, self.seed)
        lm = self._ocean["land_mask"]
        elev[lm] = np.maximum(elev[lm], 1.0)
        elev[~lm] = np.minimum(elev[~lm], -1.0)
        self._terrain["elevation"] = elev.astype(np.float32)
        self._stages["enhance"] = elev

    def stage_vegetation(self):
        from earthengine.biology.vegetation import biome, biome_index
        elev = self._terrain["elevation"]
        ice = self._climate["ice"]
        names = biome(self._climate["koppen"], ice)
        self._vegetation = {
            "biome": names, "biome_index": biome_index(names),
        }
        self._stages["vegetation"] = self._vegetation

    def stage_civilization(self):
        from earthengine.civilization.scoring import (
            civilization_index, civilization_points,
        )
        elev = self._terrain["elevation"]
        a = self._atmosphere
        c = self._climate
        rivers = self._hydrology["river_mask"]
        score = civilization_index(a["temperature"], c["precipitation"], elev,
                                   c["koppen"], self._vegetation["biome"],
                                   river_mask=rivers)
        pts = civilization_points(score)
        self._civilization = {"score": score, "points": pts}
        self._stages["civilization"] = self._civilization

    def stage_render(self):
        from earthengine import rendering as R
        self._stages["render"] = {}

    # ------------------------------------------------------------------
    def finalize(self) -> WorldState:
        from earthengine.tectonics import plates as P
        ws = WorldState(
            seed=self.seed, planet=self.planet, sphere=self.sphere,
            plates=self._tect.get("plates_meta"),
            boundaries={
                "boundary_type": self._tect.get("boundary_type"),
                "convergent_dist": self._tect.get("convergent_dist"),
                "divergent_dist": self._tect.get("divergent_dist"),
            } if "boundary_type" in self._tect else None,
            crust=self._crust if hasattr(self, "_crust") else None,
            ocean=self._ocean if hasattr(self, "_ocean") else None,
            terrain=self._terrain if hasattr(self, "_terrain") else None,
            atmosphere=self._atmosphere if hasattr(self, "_atmosphere") else None,
            climate=self._climate if hasattr(self, "_climate") else None,
            hydrology=self._hydrology if hasattr(self, "_hydrology") else None,
            geology=self._geology if hasattr(self, "_geology") else None,
            vegetation=self._vegetation if hasattr(self, "_vegetation") else None,
            civilization=self._civilization if hasattr(self, "_civilization") else None,
            terrain_conditioning=self._conditioning if hasattr(self, "_conditioning") else {},
            engine_version="2.0.0",
        )
        return ws
