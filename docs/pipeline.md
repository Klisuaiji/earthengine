# Pipeline（阶段编排）

`earthengine.pipeline.stages.WorldGenerator` 是完整世界的编排器 (31)。

## 内部顺序

```
initialize_planet → simulate_tectonics → generate_crust → generate_ocean
→ generate_macro_terrain → simulate_atmosphere → simulate_climate
→ simulate_hydrology → simulate_geology → build_terrain_conditioning
→ enhance_terrain → simulate_vegetation → score_civilization → render_maps
```

## 阶段选择

`WorldGenerator.generate(stages=…)` 允许只跑部分阶段：

```python
WorldGenerator({"seed": 42}).generate(stages=["tectonics","ocean","terrain"])
```

别名：`tectonic→tectonics, ocean, terrain, climate, hydrology`。
每个阶段用独立 `SeedManager` 子流，保证可复现 (38)。

## WorldValidator

`pipeline/validator.py`：生成后按 34.6 量化阈值校验，输出分项 + overall。
`overall < threshold` → reject（重试上限由 config 控制，被拒原因记录进
world.json）。见 `docs/validator`。

## 缓存

`pipeline/cache.py`：阶段缓存（进程内 + 可选磁盘 npz），跨阶段复用中间结果。

## 产物

`generate` 输出 `WorldState`；`io/export.py` 导出 world 包（world.json +
fields.npz + 图层 png）。
