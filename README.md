# EarthEngine v2.0

**架空行星自然地理生成引擎** —— 以 3D 球面为真实计算空间、以 2D 等经纬地图为
展示形式，通过行星参数、板块运动、地壳演化、海洋、大尺度地形、大气、气候、
水文、地质过程，生成符合地球自然规律特征的随机架空行星。

> 从 v1.x `worldengine` 重构而来。旧代码保留在 `legacy/`（不删除、不再演进），
> 新核心在 `earthengine/` 包中。

## 核心理念

- **球面计算**：板块/边界/大陆都在球面几何上生成，2D 地图只是投影展示；
- **地理因果**：板块 → 大陆 → 山脉 → 气候 → 河流 → 植被 → 文明，逐层因果；
- **模块化**：子系统独立，只通过 `WorldState` 交换；
- **可复现**：随机只由 seed+scope 决定，与分辨率无关；
- **AI 可插拔**：核心只依赖 `TerrainEnhancer` 抽象，无 AI 也能跑完整世界；
- **发布零负担**：核心依赖仅 numpy（Pillow/scipy/flask 可选）。

## 快速开始

```bash
pip install -e .                  # 核心仅需 numpy
python3 -m earthengine generate --seed 12345 --out world/
python3 -m earthengine validate world/
python3 -m earthengine render world/ --style ancient
python3 -m earthengine web --port 8899   # 需 flask（可选）
```

### CLI（见 `docs/cli`）

| 命令 | 说明 | 旧命令映射 |
| --- | --- | --- |
| `generate` | 生成一颗行星 | world |
| `validate` | 按量化阈值校验 | — |
| `render --style` | 渲染地图 | ancient_map |
| `export --format` | 导出 | info / export |
| `web` | 一键 Web | — |

## 目录

```
earthengine/   新核心（planet/tectonics/ocean/terrain/atmosphere/climate/
               hydrology/geology/biology/civilization/ai/rendering/pipeline/io/cli/web）
legacy/        v1.x worldengine + 旧 tools/tests/assets（保留）
tests/         地理规律量化测试（python3 tests/run_checks.py）
docs/          本套文档
```

## 验证

`earthengine validate` 与 `tests/run_checks.py` 按 34.6 量化阈值断言：
逆坡河段=0、山脉-汇聚边界相关 ≥0.60、纬度-温度 zonal |r|≥0.70、陆地占比、
最大大陆、板块面积基尼等。失败即重滚 seed，而非修补地图。

## 文档

`docs/`：architecture、planet、tectonics、ocean、terrain、climate、hydrology、
geology、terrain-diffusion、world-state、pipeline、validator、cli、api、development。

## 依赖

核心仅 `numpy`；`scipy`（可选，细化/验证加速）、`Pillow`（可选，PNG 出图）、
`flask`（可选，Web）、`torch + terrain_diffusion`（可选，AI 增强，缺失自动回落
程序化实现）。

License: MIT（见 LICENSE.txt）。
