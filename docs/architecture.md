# EarthEngine v2.0 — 架构总览

EarthEngine 是一个**架空行星自然地理生成引擎**：以 3D 球面作为真实计算空间、
以 2D 等经纬地图作为展示与数据交换形式，通过行星参数、板块运动、地壳演化、
海洋、大尺度地形、大气、气候、水文和地质过程，生成符合地球自然规律特征的
随机架空行星。

## 核心理念

| 原则 | 含义 |
| --- | --- |
| 球面计算 | 板块、边界、大陆都在球面几何上生成；2D 地图只是投影展示 |
| 地理因果 | 板块→大陆→山脉→气候→河流→植被→文明，逐层因果，不做贴图 |
| 模块化 | 每个子系统独立，只通过 `WorldState` 交换 |
| 可复现 | 随机只由 seed+scope 决定，与分辨率/像素坐标无关（`SeedManager`） |
| AI 可插拔 | 核心只依赖 `TerrainEnhancer` 抽象，AI 只是实现之一 |
| 发布零负担 | 核心依赖仅 numpy，剥离 noise/protobuf/hdf5/pypng（Pillow/scipy 可选） |

## 分层管线

```
初始化行星参数 → 板块运动 → 地壳演化 → 海洋/大陆 → 宏观地形
→ 大气 → 气候 → 水文 → 地质 → 地形增强(AI可插拔) → 植被 → 文明 → 渲染
```

每层职责单一，层间只通过 `WorldState` 传递；`WorldValidator` 在生成后按量化
阈值（见 `docs/validator`）校验地理规律，失败即重滚 seed 而非修补地图。

## 目录结构

```
earthengine/
  planet/       # Planet 参数 + SphereGrid + 投影
  tectonics/    # 运动驱动板块生成器 + 地壳演化
  ocean/        # 海盆/大陆布局 + 大陆架 + 海平面
  terrain/      # 噪声 + 宏观地形 + 地形区域规划 + 细节
  atmosphere/   # 太阳辐射 + 温度 + 环流
  climate/      # 降水 + Köppen 气候 + 洋流 + 冰盖
  hydrology/    # D8 汇流 + 河流 + 湖泊
  geology/      # 侵蚀 + 断层
  biology/      # 植被/生物群系
  civilization/ # 文明评分
  ai/           # TerrainEnhancer 抽象 + 程序化实现 + TerrainDiffusion 适配
  rendering/    # 全部图层渲染
  pipeline/     # WorldState + 阶段编排 + 验证 + 缓存
  io/           # png/json/npz/export
  cli/          # earthengine 命令
  web/          # 一键 Web
legacy/         # v1.x worldengine 历史代码（保留，不再演进）
tests/          # 地理规律量化测试
docs/           # 本套文档
```

## 如何开始

```bash
python3 -m pip install -e .            # 核心仅需 numpy
python3 -m earthengine generate --seed 12345 --out world/
python3 -m earthengine validate world/
python3 -m earthengine render world/ --style ancient
python3 -m earthengine web --port 8899  # 需要 flask（可选）
```

详细见 `docs/cli` 与 `docs/api`。
