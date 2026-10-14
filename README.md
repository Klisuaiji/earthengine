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

## 地质时间演化

以**运动学驱动的球面欧拉旋转**为核心（参考 GPlates 的刚性板块重建思路）：每个
板块携带一个欧拉角速度矢量（轴 + 角速率），按时间步在球面上旋转所属大陆地壳、
海陆掩膜、大陆标签与板块边界，输出多个地质时间截面。同一 seed + 时间序列可复现
相同演化；无分裂/碰撞时大陆形状与陆面积保持不变。

![地质时间演化：0 / 30 / 60 Ma 时间截面（seed 123456）](docs/evolution_timesteps.png)

```bash
python3 examples/evolve_timesteps.py --seed 123456 --times 0,30,60
# -> reports/evolution_timesteps.png
```

- **阶段 2 大陆漂移**：`earthengine/tectonics/geological_evolution.py`，
  Rodrigues 球面旋转 + 最近邻反投影采样，确定性时间积分。
- **阶段 3 构造事件**：地壳演化（俯冲海沟 / 陆缘碰撞造山 / 洋脊 / 裂谷）由
  `tectonics/evolution.py` 更新厚度/age/elevation_bias，并经 `stage_terrain`
  叠加进最终高程——构造事件改变地形，而非只改边界颜色。
- **阶段 4 时间截面接回**：每个时间截面的海陆 + 边界交回 `terrain.detail`
  重新生成对应高程，地形与地壳同属一个时间截面。

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

### 回归测试（python3 tests/*.py，无 pytest 依赖）

| 脚本 | 覆盖 | 状态 |
| --- | --- | --- |
| `tests/run_checks.py` | 阶段依赖闭包、海陆-高程、山脉/海沟/纬度因果 | 8/8 |
| `tests/test_pipeline.py` | 依赖闭包 DAG、验证器分级（ERROR 决定通过） | 7/7 |
| `tests/test_geocausality.py` | 地壳独立于 plate ID、海陆-水深一致、山脉/海沟落汇聚边界 | 6/6 |
| `tests/test_ai.py` | AI 接口契约、procedural fallback、registry、device | 5/5 |
| `tests/test_web.py` | Web envelope/参数校验/错误分类/生成产图层 | 8/8 |
| `tests/quality_report.py` | 多 seed 质量报告 → `reports/quality_report.json` | 16/16 seed |

断言合计 34 项 + 多种子稳定性 16/16 seed。

## 文档

`docs/`：architecture、planet、tectonics、ocean、terrain、climate、hydrology、
geology、terrain-diffusion、world-state、pipeline、validator、cli、api、
development、development-status（阶段 A–E 完成报告）。

## 依赖

核心运行时如实声明为 **numpy + scipy**：scipy 被核心生成/验证硬依赖
（tectonics/terrain/ocean/hydrology/geology/validator/geography_validator 内联
import），缺失时 `validator` 显式报错而非静默跳过关键检查。
`Pillow`（可选，PNG 出图）、`flask`（可选，Web，缺失时 `run_server` 打印提示
并返回码 1）、`torch + terrain_diffusion`（可选，AI 增强，缺失自动回落程序化实现）。

## 许可证与商业授权

EarthEngine 原创部分采用项目自定义的“源码可见、商业使用需事先书面授权、修改版本需按相同条款公开对应源代码”许可，详见 [LICENSE.txt](LICENSE.txt)。

- 个人、教育、研究等非商业用途：可按许可证条款使用和修改。
- 商业使用、商业部署、付费产品或营收服务：必须事先取得版权持有者的书面授权。
- 发布修改版或向用户提供修改后的托管服务：必须按许可证公开完整对应源代码；这不替代商业授权要求。
- `legacy/` 中的 WorldEngine 代码及其他第三方材料继续遵守各自原有许可证；MIT 文本保存在 [LICENSE-MIT.txt](LICENSE-MIT.txt)，来源说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**注意：** 这是项目自定义的源码可见许可证，不是 OSI 认可的开源许可证。当前尚未完成逐文件来源审计，因此不能将所有文件都视为仅受自定义许可证约束。
