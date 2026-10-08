"""NPZ 字段存档 (30.1 / 37)：全部 (H,W) 场，二进制存档（替代 hdf5）。"""

from __future__ import annotations

import numpy as np


def fields_dict(world_state) -> dict:
    """把所有可存 (H,W) 场收集成 {name: ndarray}。"""
    d = {}
    if world_state.plates is not None:
        d["plates.id"] = world_state.plates.plates.astype(np.int16)
    if world_state.boundaries is not None and world_state.boundaries.get("boundary_type") is not None:
        d["boundaries.type"] = world_state.boundaries["boundary_type"].astype(np.int8)
    if world_state.crust is not None:
        d["crust.type"] = world_state.crust.type.astype(np.int8)
        d["crust.age"] = world_state.crust.age.astype(np.float32)
    if world_state.ocean and world_state.ocean.get("land_mask") is not None:
        d["ocean.land_mask"] = world_state.ocean["land_mask"]
    if world_state.terrain:
        d["terrain.elevation_macro"] = world_state.terrain["elevation_macro"].astype(np.float32)
        d["terrain.region"] = world_state.terrain["region"].astype(np.int8)
        d["terrain.elevation"] = world_state.terrain["elevation"].astype(np.float32)
    if world_state.atmosphere:
        d["atmosphere.temperature"] = world_state.atmosphere["temperature"].astype(np.float32)
        d["atmosphere.pressure"] = world_state.atmosphere["pressure"].astype(np.float32)
    if world_state.climate:
        d["climate.precipitation"] = world_state.climate["precipitation"].astype(np.float32)
        d["climate.sst"] = world_state.climate["sst"].astype(np.float32)
        d["climate.ice"] = world_state.climate["ice"]
    if world_state.hydrology:
        d["hydrology.flow_dir"] = world_state.hydrology["flow_dir"].astype(np.uint8)
        d["hydrology.flow_acc"] = np.log1p(world_state.hydrology["flow_acc"]).astype(np.float32)
        d["hydrology.river_mask"] = world_state.hydrology["river_mask"]
        d["hydrology.lakes"] = world_state.hydrology["lakes"]
    if world_state.vegetation:
        d["vegetation.biome"] = world_state.vegetation["biome_index"].astype(np.int8)
    if world_state.civilization:
        d["civilization.score"] = world_state.civilization["score"].astype(np.float32)
    if world_state.geology:
        d["geology.faults"] = world_state.geology["faults"].astype(np.float32)
        d["geology.rock"] = world_state.geology["rock"].astype(np.float32)
    return d


def save_npz(path, world_state):
    np.savez_compressed(path, **fields_dict(world_state))
    return path
