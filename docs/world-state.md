# WorldState（世界状态）

`earthengine.pipeline.world.WorldState` 是模块之间**唯一**的交换对象 (30)。

- 所有场为 (H,W) 数组、等经纬投影、H:W = 1:2、缺省 1024×512；
- 字段形状与 dtype 见 `field_summary()`；
- 各阶段写入自己的子对象：`plates / boundaries / crust / ocean / terrain /
  atmosphere / climate / hydrology / geology / vegetation / civilization /
  terrain_conditioning / validation`。

## 地形类别（12 类，19 节）

`TT_DEEP, TT_SHELF, TT_COASTAL, TT_PLAIN, TT_HILL, TT_PLATEAU, TT_MOUNTAIN,
TT_BASIN, TT_RIDGE, TT_TRENCH, TT_VOLCANIC, TT_RIFT`

（v0 7 类 → v2.0 补 ridge/trench/volcanic/rift/coastal）

## 边界类型

`INTERIOR=0, CONVERGENT=1, DIVERGENT=2, TRANSFORM=3, OBLIQUE=4`

## 地壳类型

`CRUST_CONTINENTAL=0, CRUST_OCEANIC=1, CRUST_TRANSITIONAL=2`

## 只读快捷

`elevation / land_mask / temperature / precipitation / humidity / biome / h / w`

## 序列化

- `io/json.py`：`world.json`（元数据 + 验证）
- `io/npz.py`：`fields.npz`（全部 (H,W) 场）
- `io/export.py`：完整 world 包（world.json + fields.npz + 各图层 png + stages）

替代旧 hdf5 / protobuf / pypng（依赖剥离）。

## 字段契约（第 4 节：名称 / shape / 单位 / dtype / NoData / 生产·消费阶段）

坐标：等经纬投影，(H,W)，H:W = 1:2，经度环绕。高程基准统一为**海平面 0 m**：
陆地 > 0，海洋 < 0（垂直基准唯一，禁止混用归一化值）。dtype 统一 float32。

| 字段 | shape | 单位 | dtype | NoData | 生产阶段 | 消费阶段 |
| --- | --- | --- | --- | --- | --- | --- |
| `plates.id` | (H,W) | — | int16 | 无 | tectonics | crust/ocean/terrain/渲染 |
| `boundaries.type` | (H,W) | 码 | int8 | 0=内部 | tectonics | terrain/渲染 |
| `boundaries.convergent_dist` | (H,W) | ° | float32 | 远值 | tectonics | terrain(山脉) |
| `boundaries.divergent_dist` | (H,W) | ° | float32 | 远值 | tectonics | terrain(裂谷/洋脊) |
| `crust.type` | (H,W) | 码 | int8 | 0=陆壳 | crust | ocean/terrain |
| `crust.age` | (H,W) | Myr | float32 | 0 | crust(evolution) | 海深 |
| `ocean.land_mask` | (H,W) | — | bool | — | ocean/terrain | 全部下游 |
| `ocean.sea_level_m` | — | m | float32 | — | terrain | 渲染 |
| `terrain.elevation_macro` | (H,W) | m | float32 | 0 基准 | terrain | 下游 |
| `terrain.region` | (H,W) | 码 | int8 | 0=深海 | terrain | 渲染/AI |
| `terrain.elevation` | (H,W) | m | float32 | 0 基准 | enhance | 渲染/导出 |
| `atmosphere.temperature` | (H,W) | ℃ | float32 | — | atmosphere | climate/植被/文明 |
| `atmosphere.pressure` | (H,W) | hPa | float32 | — | atmosphere | 环流 |
| `climate.precipitation` | (H,W) | mm | float32 | 非负 | climate | 水文/植被 |
| `climate.koppen` | (H,W) | 码 | object | — | climate | 植被/渲染 |
| `climate.ice` | (H,W) | — | bool | — | climate | 植被/渲染 |
| `climate.sst` | (H,W) | ℃ | float32 | — | climate | 洋流 |
| `hydrology.flow_dir` | (H,W) | D8 码 | uint8 | 8=sink | hydrology | 河流/湖泊 |
| `hydrology.flow_acc` | (H,W) | 像元 | float64 | — | hydrology | 河流 |
| `hydrology.river_mask` | (H,W) | — | bool | — | hydrology | 植被/文明/渲染 |
| `hydrology.lakes` | (H,W) | — | bool | — | hydrology | 渲染 |
| `geology.faults` | (H,W) | 0-1 | float32 | — | geology | 渲染 |
| `geology.rock` | (H,W) | 0-1 | float32 | — | geology | 侵蚀 |
| `vegetation.biome` | (H,W) | 名称 | object | — | vegetation | 文明/渲染 |
| `civilization.score` | (H,W) | 0-1 | float32 | — | civilization | 渲染 |

**统一规则**：所有场经 `io/npz.py:fields_dict` 存档；`WorldValidator._hard_checks`
校验缺字段、shape 一致、NaN/Inf、海陆-高程矛盾、非法 plate ID、无效流向。
地形/海陆/气候三者共享同一 `ocean.land_mask`（渲染、地形规划、导出只用最终
mask）。
