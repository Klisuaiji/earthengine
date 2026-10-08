# 水文（hydrology）

`earthengine.hydrology` 生成河流、湖泊与内流盆地。

## D8 汇流（flow.py）

- `d8_flow(elev, land_mask)`：最陡下降汇流，经度环绕；
- 平滑高程 + 噪声打破平地（河流沿宽谷）；
- 海洋是汇（sink）；
- 返回 `flow_dir`（D8 码）/ `flow_acc`（汇水面积）/ `uphill`（逆坡河段数）。

### 逆坡断言（34.5）

流向只赋给 `drop > 0` 的单元（`has_out`），因此**每条河段严格下坡，
逆坡河段数恒为 0**——这是构造保证，不是事后清理。

## 河流（rivers.py）

`rivers_from_acc`：汇水面积 ≥ 阈值（`max(120, H·W/8000)`）的陆地像素为河。
`ocean_outlets`：河流入海口（陆地流向海洋的边）。

## 湖泊 / 内流盆地（lakes.py）

- `lakes`：陆地局部低点（流向自身）+ 足够汇水面积 → 湖；
- `closed_basins`：内流盆地近似。

## 河道渲染

河流宽度 ∝ `log2(flow_acc)`，高斯平滑后叠加到地形图（今怀古风格蓝河）。
