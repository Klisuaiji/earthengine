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
