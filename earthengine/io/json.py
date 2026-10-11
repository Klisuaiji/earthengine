"""JSON 元数据读写（替代 protobuf / hdf5，49.2 依赖剥离）。"""

from __future__ import annotations

import json
import numpy as np


def world_meta(world_state, extra=None) -> dict:
    """把 WorldState 的元信息序列化为 JSON 可存 dict。"""
    meta = {
        "seed": world_state.seed,
        "engine_version": world_state.engine_version,
        "shape": [world_state.h, world_state.w],
        "planet": world_state.planet.to_dict() if world_state.planet else None,
    }
    if world_state.validation:
        meta["validation"] = {
            "passed": world_state.validation.get("passed"),
            "errors": world_state.validation.get("errors"),
            "warnings": world_state.validation.get("warnings"),
            "meta": world_state.validation.get("meta"),
        }
    if world_state.ocean and world_state.ocean.get("land_mask") is not None:
        meta["land_fraction"] = float(world_state.ocean["land_mask"].mean())
    if world_state.hydrology:
        meta["hydrology"] = {"uphill_segments": world_state.hydrology.get("uphill_segments")}
    if extra:
        meta.update(extra)
    return meta


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)
    return path


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
