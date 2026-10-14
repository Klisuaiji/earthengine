"""地质时间演化（阶段 2）：运动学驱动的球面大陆漂移。

不重写板块系统，也不堆噪声。复用已生成的欧拉角速度（每板块一个 omega 矢量，
rad/Myr），按时间步在球面上对所属板块做 Rodrigues 旋转，让大陆地壳、海陆掩膜、
大陆标签与板块边界一起随板块漂移。每个时间截面从同一初始世界出发，确定性、
可复现（相同 seed + dt 序列得到相同演化）。

设计（对齐 GPlates 思想的最小运动学版）：
- 每个板块保存可积分欧拉参数（轴 = omega/|omega|，角速率 = |omega|）。
- 大陆/地理特征关联到所属板块（plate id map）。
- 时间步前进时：对**目标网格**每个点，按其所属板块反旋转回 dt 前的位置，
  采样该位置的属性（海陆/大陆/边界）——即"未来网格点 = dt 前该板块上某点旋过来"。
  这样目标网格无空缺、属性无丢失（无分裂/碰撞时形状不变）。
"""

from __future__ import annotations

import math

import numpy as np

TAU = 2.0 * math.pi


def _rodrigues(P: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """对单位向量 P(...,3) 绕单位轴 axis(3,) 旋转 angle_rad 弧度。"""
    k = axis / (np.linalg.norm(axis) + 1e-12)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    dot = P @ k
    return (P * c + np.cross(P, k) * s + np.outer(dot, k) * (1.0 - c)).astype(np.float32)


def _unit_to_grid(P: np.ndarray, h: int, w: int):
    """单位向量 -> 最近邻网格行/列索引（与 LatLonGrid 行优先经纬网格一致）。"""
    z = np.clip(P[:, 2], -1.0, 1.0)
    lat = np.arcsin(z)
    lon = np.arctan2(P[:, 1], P[:, 0])
    yi = np.round((0.5 - lat / math.pi) * h).astype(int)
    xi = np.round(((lon / TAU) % 1.0) * w).astype(int)
    return np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)


# 角速率标定（rad/Myr）。板块生成的 omega 只取方向，速率统一到物理合理量级，
# 否则随机大小会让 30 Myr 转好几圈、最近邻采样把大陆旋丢。
_EFF_RATE = 0.025


def advance_step(P_grid: np.ndarray, plate_map: np.ndarray, velocities: dict,
                 fields: dict, dt_myr: float, h: int, w: int) -> dict:
    """前进一个时间步：返回漂移后各属性场（dict 同形）。

    fields: 每个 (h,w) 场（land / continent / boundary / crust_type ...）。
    P_grid: (h*w,3) 目标网格的单位向量。
    plate_map: (h*w,) 每像素所属板块 id。
    """
    out = {}
    flat_map = plate_map.ravel()
    for pid, omega in velocities.items():
        omega = np.asarray(omega, dtype=np.float64)
        rate0 = float(np.linalg.norm(omega))
        if rate0 < 1e-9:
            mask = flat_map == pid
            for name, f in fields.items():
                out.setdefault(name, f.copy())[mask.reshape(h, w)] = f[mask.reshape(h, w)]
            continue
        axis = omega / rate0                    # 只取旋转方向
        P_future = P_grid[flat_map == pid]      # 目标点单位向量
        P_past = _rodrigues(P_future, axis, -_EFF_RATE * dt_myr)   # 反旋回 dt 前
        yi, xi = _unit_to_grid(P_past, h, w)
        for name, f in fields.items():
            tgt = out.setdefault(name, np.full_like(f, f.ravel()[0] if f.size else 0))
            src = f.reshape(-1)
            tgt.reshape(-1)[flat_map == pid] = src[yi * w + xi]
    return out


def evolve_series(world, times_myr, dt_max=10.0):
    """从当前 WorldState 生成多个地质时间截面。

    times_myr: 目标时间（Myr，0 = 现在，正值=向未来演化）。
    返回 [(time, {land, continent, boundary, plate, crust_type})]，按时间顺序。
    用固定步长累计积分，保证可复现。
    """
    plates = world.plates
    P_grid = np.asarray(world.sphere.xyz()).reshape(-1, 3).astype(np.float64)
    h, w = world.h, world.w
    plate_map = plates.plates
    velocities = plates.velocities
    fields = {
        "land": world.ocean["land_mask"].astype(np.int8),
        "continent": world.ocean.get("continent_mask",
                                     np.zeros((h, w), np.int32)).astype(np.int32),
        "boundary": plates.boundary_type,
        "plate": plate_map.astype(np.int32),
    }
    out = []
    cur = 0.0
    cur_fields = dict(fields)
    for t in times_myr:
        # 从 cur 积分到 t，分固定小步（单步角速率*dt 可控）
        remaining = t - cur
        while abs(remaining) > 1e-6:
            step = math.copysign(min(dt_max, abs(remaining)), remaining)
            cur_fields = advance_step(P_grid, plate_map, velocities, cur_fields,
                                      step, h, w)
            cur += step
            remaining = t - cur
        out.append((t, {k: v.copy() for k, v in cur_fields.items()}))
    return out
