# Terrain Diffusion（AI 增强，可选）

`earthengine.ai` 是 AI 可插拔的地形增强层。**核心只依赖抽象，无 AI 也能跑**。

## 职责边界（明确写死）

Terrain Diffusion **永不决定**：

- 大陆位置 / 海洋位置 / 大陆数量 / 超大陆结构 / 澳大利亚式隔离

它只接收：

```
Global mask + Terrain region map + 必要 conditioning
```

然后负责：山脉真实起伏、丘陵、峡谷、侵蚀感、局部高度变化、地形连续性。

## 架构

```
ai/base.py              TerrainEnhancer 抽象 + ProceduralEnhancer(默认)
ai/registry.py          backend 注册表（procedural / terrain_diffusion）
ai/terrain_diffusion.py TerrainDiffusionAdapter（torch 缺失自动回落 procedural）
ai/terrain_diffusion_impl.py   迁移自 tools/diffusion_world.py（torch 路径）
ai/_load_st_chunked.py  迁移自 tools/（chunked GPU 加载，懒 import）
```

## 契约（24.3）

`ProceduralEnhancer` 与 `TerrainDiffusionAdapter` 必须返回
**同 dtype、同量纲（米）、同海陆符号**的数组，使下游 Hydrology / Vegetation /
Rendering 无分支。Adapter 无 torch / 无 conditioning 时回落 procedural，
结果与 procedural 一致（已验证 `np.allclose`）。

## 分块与连续性

全球地图负责宏观一致（coarse 条件网格），TD 负责区域细节（tile 分块），
拼接为无缝世界（Seamless）。`snr0` 控制扩散强度，`tile` 控制分块尺寸。

## 全局 vs 分块

```
Global 4096²/8192² → coarse 条件网格 → 512²/1024² terrain chunks → TD → 拼接
```
