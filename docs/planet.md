# 行星参数（planet）

`earthengine.planet.parameters.Planet` 是一切生成的起点：一颗行星的所有物理
性质集中在这里，后续大气/气候/水文都从它读取。

## 字段

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `radius_km` | 6371.0 | 行星半径（km） |
| `rotation_period_h` | 24.0 | 自转周期（h）→ 角速度 `omega_s` |
| `axial_tilt_deg` | 23.4 | 轴倾角（决定季节/纬度温度带） |
| `star_distance_au` | 1.0 | 距恒星距离（AU） |
| `star_luminosity` | 1.0 | 恒星光度 |
| `sea_level_m` | 0.0 | 海平面 |
| `ocean_fraction` | 0.70 | 目标海洋占比 |
| `initial_temperature_c` | 15.0 | 基准温度 |

## 派生量

- `solar_constant_wm2 = L / d² × 1361`（星距/光度缩放）
- `omega_s = 2π / T`（自转角速度，供科里奥利力）
- `cell_area_m2`（单像素地表面积）

## 球面网格

`earthengine.planet.sphere` 提供：

- `SphereGrid` 抽象：球面几何接口（xyz、邻域、geodesic、经纬投影）
- `LatLonGrid` 首个实现：等经纬网格，`H:W = 1:2`，经度环绕

`_lat_deg(h)` 返回 (h,) 纬度带（供温度带/气候带使用）。
`earthengine.planet.projection` 负责球面→2D 的投影/展示。

## 复现性

`earthengine.seed.SeedManager`：随机只由 `seed + scope` 决定，与分辨率/像素
坐标无关。各阶段用独立子流派生，保证同一 seed 结果可复现且稳定。
