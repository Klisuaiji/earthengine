# 气候（climate / atmosphere）

`earthengine.atmosphere` + `earthengine.climate` 由行星参数驱动物理气候链。

## 大气

- `solar.py`：太阳辐射（含轴倾角、星距/光度缩放）
- `temperature.py`：`latitude_temperature`（纬度带 + 恒星参数缩放）；
  `solar_temperature`（纬度带 + 大陆性 + 垂直递减率 0.0065℃/m）；
  `continental_pressure`（热力气压 + 高地低压）
- `circulation.py`：`wind_field`（引入科里奥利 f=2Ωsinφ）

## 气候

- `precipitation.py`：`precipitation`（风场 + 高程 → 降水）
- `climate.py`：`koppen`（Köppen 分类）、`ice_layer`（冰盖）、`ocean_currents`（洋流）、`KOPPEN_RGB`

## 纬度–温度规律（34.6 硬断言）

以**气候学纬向平均**（每纬度带平均温度 vs |纬度|）做 Pearson 检验，
`|r| ≥ 0.70` 才合格；失败即 ERROR。温度带是地球物理因果的核心产物，
海洋/大陆/高程只做次级调制。

## 地理因果链

```
行星参数 → 太阳辐射 → 纬度温度带 → 海陆温差/垂直递减 → 气压/风场
        → 降水 → Köppen 气候 → 冰盖/洋流 → 生物群系 → 河流 → 文明
```
