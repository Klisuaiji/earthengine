# 板块（tectonics）

`earthengine.tectonics` 是 v2.0 的地质核心：**运动驱动板块生成器**。

## 核心概念：先有构造边界，再有板块

- Voronoi 只负责**初始球面邻接拓扑**（`voronoi.py`），不直接决定板块形状；
- 每个板块有**运动向量**（欧拉角速度），板块形状/边界由运动与碰撞演化而来；
- 边界类型（汇聚/张裂/转换/斜向）由相对运动判定，而非像素坐标 hash。

模块 | 职责
--- | ---
`core.py` | `PlanetaryPlates` / `generate_tectonic_plates` / `generate_validated_world`（含宏观地理重滚）
`voronoi.py` | 域扭曲球面 Voronoi（无 PIL 依赖）
`seeds.py` | 微板块计数
`growth.py` | 板块生长
`plates.py` | v2.0 配置面（板块数范围、尺寸幂律等），转发 core
`motion.py` | 欧拉速度 / 相对速度
`boundaries.py` | `classify_by_motion` → 边界类型
`crust.py` | `CrustState`（type/age/density/thickness），per-pixel 地壳映射
`evolution.py` | 洋壳俯冲 / 陆缘碰撞 / 洋中脊 / 陆裂谷演化；`ocean_depth_m=2500+350√age`

## 板块 ≠ 大陆

- 板块影响山脉、海沟、火山带、岛弧；
- 大陆布局由 `ocean/basins.py` 的**宏观地理生成器**决定（超大陆 + 独立大陆 +
  澳大利亚型大陆），板块只提供地壳类型与边界信息。

## 验证与重滚

`generate_validated_world`：生成 → 验证 → 失败 → 换布局参数重滚 → 重新生成，
最多 `max_attempts` 次；被拒原因记录进 report。不做“修补地图”。

## 常量化目标

- 少数巨型板块 + 中型板块 + 少量微板块（面积幂律）；
- 边界类型 ≥ 3 类（汇聚/张裂/转换），避免退化；
- 板块面积基尼 ≥ 0.35（WARN）。
