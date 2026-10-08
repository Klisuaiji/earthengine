# 地形（terrain）

`earthengine.terrain` 从“大陆地理”生成“地形规划”，再交给细节/增强层。

## 分层职责（明确写死）

| 层 | 职责 |
| --- | --- |
| `noise.py` | 值噪声 FBM（水平可平铺，numpy/scipy 双路径） |
| `macro.py` | 宏观高程：`LAND_BASE=3500 / MOUNTAIN_BONUS=1500 / OCEAN_BASE=-6000 / TRENCH=-8000 / RIDGE=-4500` |
| `planner.py` | **Terrain Region Planner**：12 类地形区域图（深海/大陆架/沿海/平原/丘陵/高原/山脉/盆地/洋中脊/海沟/火山/裂谷）+ `build_conditioning` |
| `mountains.py` | 沿汇聚边界窄高山带 |
| `plains.py` | 沿海–丘陵–高原–盆地 |
| `detail.py` | 分形海岸（可锁定 target_land）、山脊纹理、程序化高程 |

## 地形区域（terrain_regions）

```
P = Plains, H = Highlands, M = Mountains, D = Desert, B = Basin, I = Ice …
```

先给大陆“着色”（区域图），再转成 Terrain Diffusion 能理解的 conditioning。
山脉主要沿板块边界（汇聚/碰撞/俯冲/裂谷/热点），高原靠近大型山脉与大陆内部，
沙漠受纬度/降水/雨影/大陆性影响。

## 关键常量（早期管线已调优）

- `CELL=256`（历元 M 单元）
- 上述 `LAND_BASE/MOUNTAIN_BONUS/OCEAN_BASE/TRENCH/RIDGE`

## 确定性陆比

`detail.fractalize_coast(land_mask, seed, target_land=…)` 在分形海岸场取
`(1-target_land)` 分位数阈值，使最终陆地占比精确 == 目标（确定性强制）。
