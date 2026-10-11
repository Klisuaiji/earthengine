# EarthEngine v2.0 继续开发 · 阶段报告（A–E）

本文件记录《EarthEngine v2.0 继续开发文档》阶段 A–E 的完成情况。
状态标签：**已实现** / **测试通过** / **仅静态检查** / **尚未实现**。

## 阶段 A：核心可靠性（依赖闭包 + 验证器分级 + WorldState 契约）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `pipeline/dependencies.py`（DAG 依赖闭包） | 已实现 + 测试通过 | `_STAGE_DEPS`、`dependency_closure`、`detect_cycle`、`missing_inputs`、`StageDependencyError` |
| `pipeline/stages.py` 用闭包生成 `_STAGE_ORDER` | 已实现 + 测试通过 | `generate()` 执行前置闭包、记录 `_executed_stages`；未知阶段抛可读错误 |
| 验证器分级重构（ERROR/WARNING/METRIC） | 已实现 + 测试通过 | `passed` 只由 ERROR 决定，不被平均分抵消；report 含 errors/warnings/metrics/meta |
| 硬校验（缺字段/shape/NaN/海陆矛盾/非法 plate/流向） | 已实现 + 测试通过 | 注入海陆矛盾 → `passed=False` |
| WorldState 字段契约文档 | 已实现 | `docs/world-state.md`：名称/shape/单位/dtype/NoData/生产·消费阶段；高程基准海平面、陆地>0 海洋<0、float32、H:W=1:2 |
| CLI / json 消费点适配 | 已实现 + 测试通过 | `cli/main.py`、`io/json.py` |
| 测试 `tests/test_pipeline.py` | 测试通过 | 7/7 |

## 阶段 B：地理因果一致性

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 地壳独立于 plate ID | 已实现 + 测试通过 | 同板块多种地壳、不同板块共享地壳类型 |
| 海陆-水深一致 | 已实现 + 测试通过 | 陆地高程>0、海洋<0，矛盾 ≤0.02 |
| 山脉落汇聚边界（≥0.60） | 已实现 + 测试通过 | 5° 缓冲区内 |
| 海沟落汇聚边界（≥0.30） | 已实现 + 测试通过 | 5° 缓冲区内 |
| 测试 `tests/test_geocausality.py` | 测试通过 | 5/5 |
| 多 seed 质量报告 | 已实现 + 测试通过 | `tests/quality_report.py` → `reports/quality_report.json`，7/7 seed 全 OK、0 ERROR、0 WARNING |

## 阶段 C：AI 接口（Terrain Enhancer 契约 + 可回落）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| `TerrainEnhancer.generate` 契约（float32、量纲米、海陆符号） | 已实现 + 测试通过 | 陆地>0、海洋<0 |
| ProceduralEnhancer 默认 fallback | 已实现 + 测试通过 | 无 AI 可跑、不破坏海陆拓扑 |
| TerrainDiffusionAdapter 无 torch 自动回落 | 已实现 + 测试通过 | 与 procedural 一致（np.allclose） |
| registry 注册 / resolve_device | 已实现 + 测试通过 | mock enhancer 可调用 |
| 测试 `tests/test_ai.py` | 测试通过 | 5/5 |

## 阶段 D：Web 地图工作台（绿白浅色主题）

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 后端统一 JSON envelope | 已实现 + 真实 HTTP 测试通过 | `{success, data|error}`；error.type ∈ input/generate/missing_dep |
| 参数校验（seed/分辨率/W=2H/detail） | 已实现 + 真实 HTTP 测试通过 | 非法→400 input |
| 三工作模式 + 图层图例 + 工具清单 | 已实现 + 真实 HTTP 测试通过 | `/api/meta` |
| `/api/generate`（14 阶段 + 6 图层 base64 + 验证） | 已实现 + 真实 HTTP 测试通过 | 实测 land=0.300、stages=14、passed=True |
| 前端绿白工作台（三栏布局） | 已实现 + 真实 HTTP 测试通过 | 浅色底/白面板/绿强调 #3d8b4e/细边框/低对比度经纬网 |
| Canvas 缩放(滚轮锚点)+拖拽 | 已实现（前端） | 静态检查：通过真实服务器加载 HTML |
| 缩略图视口同步 | 已实现（前端） | — |
| 底部真实状态栏（Lat/Lon/Zoom/Scale/Res/Seed/Stage） | 已实现（前端） | 真实值，非写死 |
| 坐标定位（如 30N 120E） | 已实现（前端） | — |
| 未迁移工具禁用/标注「规划中」 | 已实现 | select/inspect/plates/terrain/rivers/measure 标规划中 |
| 测试 `tests/test_web.py` | 测试通过 | 8/8（envelope/校验/错误分类/产图层） |

> 说明：前端交互（缩放拖拽/缩略图/坐标定位）为**已实现 + 仅静态检查**——已验证
> 真实服务器返回该 HTML、/api/meta 与 /api/generate 契约一致，未做浏览器自动化
> 交互断言。

## 阶段 E：发布验收

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 全测试套件回归 | 测试通过 | pipeline 7 + geocausality 5 + ai 5 + web 8 + run_checks 8 = 33 项 |
| 核心依赖最小化 | 已实现 | 核心仅 numpy；scipy/image/web/ai/dev 可选分组 |
| CLI 无模型生成/验证 | 已实现 + 测试通过 | `WorldGenerator` + `WorldValidator` 全链路 |
| 产物清单核对 | 已实现（部分） | export 产出 15 png + world.json + fields.npz；`bathymetry.png`/`validation_report.json` 未单独产出（见下） |
| 文档更新 | 已实现 | README 回归表、本报告、world-state 契约 |
| git 提交 | 已实现 | 提交本报告对应 commit（见提交说明） |
| EXE 打包 | 尚未实现 | 按文档属「核心稳定之后」，标注未实现 |

### 未完成项（如实标注）

- **`bathymetry.png`、`validation_report.json`**：`io/export.py` 当前产出
  15 张 png + `world.json` + `fields.npz`，未单独导出 bathymetry 图层与
  validation_report 文件（验证报告在 `world.json` 内）。属产物清单差异。
- **EXE 打包**：文档定位「核心稳定之后」，本阶段不打包。
- **浏览器自动化交互断言**：前端交互为静态检查，未做 GUI 自动化点击断言。
