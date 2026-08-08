# WorldEngine 球面 Voronoi 世界生成器

> A world generator built on WorldEngine, replacing the platec tectonic
> simulation with a **domain-warped spherical Voronoi** plate system, refined
> by a **Terrain-Diffusion** elevation model, and driven through a unified
> **25-step planet pipeline** with an interactive web UI.

本项目基于 [WorldEngine](https://github.com/Mindwerks/worldengine)，用**域扭曲球面 Voronoi** 取代传统的 platec 板块构造仿真，生成微板块并归并为 6 大板块；并以 **Terrain Diffusion** 扩散模型把构造高程精修为真实地形起伏，再依次运行构造 → 气候/渲染 → 生态/文明三大阶段，输出一套完整的行星阶段图。附带一个交互式 Web 生成器（端口 **8899**）与板块边界（含三联点）拓扑分析。

This project is built on [WorldEngine](https://github.com/Mindwerks/worldengine). It replaces the platec tectonic simulation with a **domain-warped spherical Voronoi** plate system, refines the tectonic elevation into real relief with a **Terrain-Diffusion** model, then runs a unified **25-step pipeline** (tectonics → climate/rendering → ecology/civilization) and ships an interactive web generator (port **8899**) plus plate-boundary (incl. triple-junction) topology analysis.

---

## 生成管线 / Generation Pipeline

管线由 `tools/planet_pipeline.py` 统一编排，分为三大阶段：

| 阶段 | 步骤 | 说明 | Stage |
|------|------|------|-------|
| **Phase 1 构造** | 1–3 | 板块模拟 / 合并 / 边界分类（生长·消亡·转换） | Tectonic |
| | 4–6 | 大陆模拟 / 海岸线细分 / 群岛拆分 | Continents & coast |
| | 7–8 | 初始着色 / 初始灰度高程 | Colouring |
| | 9 | **Terrain Diffusion** 精修真实地形 | Diffusion relief |
| **Phase 2 气候与渲染** | 13 | 太阳辐射 / 温度带 | Temperature |
| | 14–15 | 大陆气压 / 行星风带（含季风、地形偏转） | Pressure & wind |
| | 16–18 | 降水（迎风坡·雨影·内陆干旱）/ 简化 Köppen / 冰盖 | Precip & ice |
| | 19 | 风驱洋流（与冰盖耦合） | Ocean currents |
| | 20–22 | 图层合成 / 卫星图 / 行星图 | Composite & maps |
| **Phase 3 生态与文明** | 23 | 生物群系（气候 + 高程 + 洋流） | Biome |
| | 24–25 | 文明发展指数（宜居性）/ 文明起源点（隔离度筛选） | Civilization |

> 完整 25 步说明见 `tools/planet_pipeline.py` 文件头注释。

### 阶段图预览 / Stage Previews

构造阶段图（`stage_images/`，已纳入版本库）展示了球面 Voronoi 与扩散地形：

| 高程 / Elevation | 微板块 / Raw Plates | 大板块 / Merged Plates |
|------------------|---------------------|------------------------|
| ![elevation](stage_images/01_sv_elevation.png) | ![raw](stage_images/02_sv_raw_plates.png) | ![merged](stage_images/02b_sv_merged.png) |

| 温度 / Temperature | 降水 / Precipitation | 湿度 / Humidity |
|--------------------|----------------------|-----------------|
| ![temperature](stage_images/09_sv_temperature.png) | ![precipitation](stage_images/10_sv_precipitation.png) | ![humidity](stage_images/16_sv_humidity.png) |

| 生物群系 / Biome | 海洋深度 / Sea Depth | 冰盖 / Icecap |
|------------------|----------------------|---------------|
| ![biome](stage_images/18_sv_biome.png) | ![sea depth](stage_images/08_sv_sea_depth.png) | ![icecap](stage_images/19_sv_icecap.png) |

`planet_out/`（默认输出目录，未纳入版本库）包含完整的 22 张行星阶段图（`01_plates_raw.png` … `22_planet.png`）。

---

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Python 3.13（本项目在 WorkBuddy 托管运行时下测试）
- NumPy、Pillow、SciPy、Flask、h5py
- **numba**（扩散加载器依赖；装在 `envs/default` venv）
- **torch** + **terrain_diffusion**（Terrain Diffusion 真实地形精修所必需）
- 预训练的 Terrain Diffusion 检查点（扩散模型权重）

推荐在本机用虚拟环境运行：

```bash
# 在 managed 运行时下创建 venv 并安装依赖
python -m venv envs/default
envs/default/Scripts/pip install numpy pillow scipy flask h5py numba torch
# terrain_diffusion 按其官方说明安装（含预训练权重）
```

### 离线生成行星 / Generate a planet offline

```bash
python tools/generate_planet.py --seed 1234567 --width 1024 --height 512
```

输出写入 `planet_out/`，包含全部阶段图与中间 npy。

### 交互式 Web 生成器 / Interactive web generator

提供一键启动脚本（已写死使用 `envs/default` 解释器，避免误用无 numba/torch 的运行时）：

```bash
run_server.bat            # Windows：启动 Flask 服务（端口 8899）
# 或手动：
python tools/web_server.py --port 8899
```

打开 <http://localhost:8899> 即可通过左侧参数面板（种子、尺寸、板块数、域扭曲幅度等）实时生成世界。前端 `tools/earth_app.html` 提供缩放/拖拽画布、比例尺、经纬度、右侧图层叠加栏与“查看生成过程”。首次请求会加载 Terrain Diffusion 模型。

> ⚠️ 必须通过 HTTP 访问（`http://localhost:8899`），**不要直接双击打开** `earth_app.html`（file:// 会因 CORS 无法调用 API）。

### 仅生成构造阶段图 / Tectonic stage images only

```bash
python tools/generate_stages_voronoi.py        # 默认 1024×512，种子 1
```

输出写入 `stage_images/`（已纳入版本库，作为管线可视化样例）。

---

## 核心算法 / Core Algorithm

### 1. 球面 Voronoi（`worldengine/spherical_voronoi.py`）

`generate_spherical_world(seed, w, h, n_raw, n_big)` 生成 `(raw, merged, land_mask, continent_mask)`：

1. **种子采样**：球面最远点采样，首个种子远离球面中心，保证板块不汇聚于两极。
2. **域扭曲**：低分辨率 FBM 噪声对像素方向做微扰（默认 `DOMAIN_AMP=0.22`），产生平滑自然的弯曲边界——**噪声只作用于 Voronoi 输入坐标，不参与边界输出**。
3. **压力松弛**：按面积均衡迭代调整各板块压力，保证无空板块、面积大致均匀；距离按分块重算，**内存占用恒定**，支持超大分辨率。
4. **大陆生长**：取海洋种子 + 距其最远的大陆核，沿邻接图做面积均衡多源 BFS，生成 2 海洋 + 4 大陆共 6 大板块（`DEFAULT_N_OCEAN=2`）。
5. **碎片/飞地清理**：仅保留最大连通分量，极小孤立块归并到相邻板块。

性能优化：`scipy.ndimage` 向量化组件标记与距离变换，板块面积/质心用 `numpy.bincount` 一次扫描完成。

### 2. 板块边界与三联点（`worldengine/plate_boundaries.py`）

`classify_boundaries(raw, merged, ...)` 返回每个边界像素的类型：

- **convergent（消亡）**：两板块相向汇聚；
- **divergent（生长）**：两板块背离分离；
- **transform（转换）**：两板块水平错动。

**三联点检测（jarcs）**：每个边界像素按其所分隔的**板块对 (A, B)** 编号（`arc_of`）；遍历每个 2×2 角块，若其中 ≥2 个像素为边界像素，则收集其 `arc_of` 集合，集合大小 ≥3 即记为一个三联点。返回 `junctions`（cell 索引 → 相邻 arc id 列表）与 `junction_count`，可直接用于生长/消亡边界的拓扑分析。

### 3. Terrain Diffusion 真实地形（`tools/diffusion_world.py` + `tools/_load_st_chunked.py`）

- Voronoi `land_mask` → 构造粗条件高程（陆地 +4000m / 海洋 −8000m）→ `WorldPipeline.set_custom_conditioning_import` 注入；
- 分块（256×256）生成真实高程，再用 `clamp_land_sea` 按 Voronoi 符号钳制海陆；
- 后处理 `refine_coastline_and_islands` 补充分形海岸线、群岛/半岛与陆内基底起伏；
- 扩散模型原始气候通道依赖 WorldClim 统计校准（当前环境缺失 rasterio 时走 identity 回退），因此温度/降水由自包含的物理气候模型计算（地形驱动，非扩散模型原生）。

---

## 项目结构 / Project Layout

```
tools/
  generate_planet.py          # 25 步行星管线入口（CLI）
  planet_pipeline.py          # 三阶段编排（构造/气候/生态文明）
  web_server.py               # Flask 交互式生成器（API + 前端）
  earth_app.html              # 三栏交互前端（控制面板/画布/图层栏）
  diffusion_world.py          # Terrain Diffusion 地形精修
  _load_st_chunked.py         # 分块加载扩散模型权重（适配本机显存）
  generate_stages_voronoi.py  # 仅构造阶段图（写入 stage_images/）
worldengine/
  spherical_voronoi.py        # 域扭曲球面 Voronoi 核心
  plate_boundaries.py         # 边界分类 + 三联点检测（jarcs）
  plates.py                   # 板块生成入口（无 PyPlatec C 扩展）
  ...                         # WorldEngine 其余底层库（气候、绘制、序列化等）
stage_images/                 # 构造阶段图样例（纳入版本库）
planet_out/                   # 完整行星阶段图（默认输出，未纳入版本库）
run_server.bat                # Windows 一键启动 Web 服务
```

---

## 已知限制 / Known Limitations

- 扩散地形当前依赖 `envs/default` venv 中的 torch / numba；使用其它无该依赖的解释器会报 `ModuleNotFoundError: numba`。
- 扩散模型的原生气候通道需要 rasterio + WorldClim 统计文件；缺失时温度/降水由物理模型替代。
- 高分辨率（≥ 2048×1024）的多瓦片扩散尚未充分验证，当前稳定配置为 1024×512。

---

## License

WorldEngine is available under the MIT License — see `LICENSE` in the project root.
