"""World Package 导出 (37)。

world/
├── world.json          # 元数据 + 验证
├── fields.npz          # 全部 (H,W) 场
├── plates.png … civilization.png   # 各图层地图
└── stages/             # 阶段图（若生成）
"""

from __future__ import annotations

import os

import numpy as np

from earthengine.io.json import save_json, world_meta
from earthengine.io.npz import save_npz, fields_dict
from earthengine import rendering as R
from earthengine.pipeline.world import BOUNDARY_NAMES, CRUST_NAMES, TERRAIN_NAMES


def export_world(world_state, out_dir, render=True, stages=None):
    """导出整包到 ``out_dir``。返回 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)
    meta = world_meta(world_state)
    meta["fields"] = sorted(fields_dict(world_state).keys())
    save_json(os.path.join(out_dir, "world.json"), meta)
    save_npz(os.path.join(out_dir, "fields.npz"), world_state)

    if render:
        _render_layers(world_state, out_dir)
    if stages:
        _render_stages(world_state, out_dir, stages)
    return out_dir


def _render_layers(ws, out_dir):
    lm = ws.ocean["land_mask"] if ws.ocean else None
    if lm is not None:
        R.map.render_land_sea(lm, os.path.join(out_dir, "land_sea.png"))
    if ws.plates is not None:
        R.map.render_plates(ws.plates.plates, os.path.join(out_dir, "plates.png"))
    if ws.boundaries is not None and ws.boundaries.get("boundary_type") is not None:
        R.map.render_boundary(ws.boundaries["boundary_type"],
                              os.path.join(out_dir, "boundaries.png"))
    if ws.crust is not None:
        R.map.render_crust(ws.crust.type, os.path.join(out_dir, "crust.png"))
    if ws.terrain is not None:
        R.map.render_terrain_types(ws.terrain["region"],
                                   os.path.join(out_dir, "terrain_regions.png"))
        ice = ws.climate["ice"] if ws.climate else None
        R.map.render_relief_hires(ws.terrain["elevation"],
                                  os.path.join(out_dir, "elevation_macro.png"),
                                  land_mask=lm, ice=ice)
    if ws.climate is not None:
        R.map.render_float_field(ws.climate["temperature"],
                                 os.path.join(out_dir, "temperature.png"), -40, 40)
        R.map.render_float_field(ws.climate["precipitation"],
                                 os.path.join(out_dir, "precipitation.png"), 0, 3000)
        kp = ws.climate["koppen"]
        R.map.save_png(R.colors.koppen_palette(kp) * 255,
                       os.path.join(out_dir, "climate.png"))
    if ws.hydrology is not None:
        R.map.render_hydrology(ws.hydrology["river_mask"],
                               os.path.join(out_dir, "rivers.png"),
                               lakes=ws.hydrology["lakes"])
    if ws.vegetation is not None and ws.terrain is not None:
        R.map.render_satellite(ws.terrain["elevation"], ws.vegetation["biome"],
                               ws.climate["ice"] if ws.climate else None,
                               os.path.join(out_dir, "vegetation.png"))
        R.map.render_planet(ws.terrain["elevation"], ws.vegetation["biome"],
                            ws.climate["ice"] if ws.climate else None,
                            ws.climate["koppen"] if ws.climate else None,
                            os.path.join(out_dir, "planet.png"))
    if ws.civilization is not None:
        R.map.render_float_field(ws.civilization["score"],
                                 os.path.join(out_dir, "civilization.png"), 0, 1)


def _render_stages(ws, out_dir, stages):
    """阶段图（01_…22_ 命名，Golden Seeds 可对比）。"""
    sdir = os.path.join(out_dir, "stages")
    os.makedirs(sdir, exist_ok=True)
    idx = 0
    for name in stages:
        idx += 1
        tag = f"{idx:02d}_{name}.png"
        if name == "plates_raw" and ws.plates is not None:
            R.map.render_plates(ws.plates.raw, os.path.join(sdir, tag))
        elif name == "plates" and ws.plates is not None:
            R.map.render_plates(ws.plates.plates, os.path.join(sdir, tag))
        elif name == "boundaries" and ws.boundaries is not None:
            R.map.render_boundary(ws.boundaries["boundary_type"], os.path.join(sdir, tag))
        elif name == "crust" and ws.crust is not None:
            R.map.render_crust(ws.crust.type, os.path.join(sdir, tag))
        elif name == "land_sea":
            R.map.render_land_sea(ws.ocean["land_mask"], os.path.join(sdir, tag))
        elif name == "terrain_types" and ws.terrain is not None:
            R.map.render_terrain_types(ws.terrain["region"], os.path.join(sdir, tag))
        elif name == "elevation" and ws.terrain is not None:
            R.map.render_terrain(ws.terrain["elevation"], None, os.path.join(sdir, tag))
        elif name == "rivers" and ws.hydrology is not None:
            R.map.render_hydrology(ws.hydrology["river_mask"], os.path.join(sdir, tag))
        elif name == "temperature" and ws.climate is not None:
            R.map.render_float_field(ws.climate["temperature"], os.path.join(sdir, tag), -40, 40)
        elif name == "precipitation" and ws.climate is not None:
            R.map.render_float_field(ws.climate["precipitation"], os.path.join(sdir, tag), 0, 3000)
        elif name == "climate" and ws.climate is not None:
            R.map.save_png(R.colors.koppen_palette(ws.climate["koppen"]) * 255,
                           os.path.join(sdir, tag))
        elif name == "vegetation" and ws.vegetation is not None:
            R.map.render_satellite(ws.terrain["elevation"], ws.vegetation["biome"],
                                   ws.climate["ice"] if ws.climate else None,
                                   os.path.join(sdir, tag))
        elif name == "planet" and ws.vegetation is not None:
            R.map.render_planet(ws.terrain["elevation"], ws.vegetation["biome"],
                                ws.climate["ice"] if ws.climate else None,
                                ws.climate["koppen"] if ws.climate else None,
                                os.path.join(sdir, tag))
        elif name == "civilization" and ws.civilization is not None:
            R.map.render_float_field(ws.civilization["score"], os.path.join(sdir, tag), 0, 1)
