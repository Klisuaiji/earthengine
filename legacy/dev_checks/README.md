# dev_checks/ — 开发验证脚本

这些脚本不属于上游 WorldEngine 测试套件，是本地优化/修复时使用的验证工具。
运行方式（默认 `.venv`；CUDA 相关的用 `.venv-torch`）：

| 脚本 | 用途 | 解释器 |
|------|------|--------|
| `test_pb_equivalence.py` | 向量化 `plate_boundaries.classify_boundaries` 与原版逐位对比（129 用例） | `.venv` |
| `test_sv_equivalence.py` | 优化后 `spherical_voronoi` 与原版逐位对比 + 性能对比 | `.venv` |
| `test_pipeline_smoke.py` | 25 步行星管线 CPU 链路冒烟（不含扩散步） | `.venv` |
| `test_cuda.py` | CUDA 可用性 + 分块加载器把 base U-Net 权重加载进 GPU | `.venv-torch` |
| `test_cuda_e2e.py` | 端到端 CUDA 扩散推理（128×128，保守内存参数） | `.venv-torch` |

原版参考实现在 `tests/reference_impls/`（来自 GitHub master 的优化前快照）。

示例：

```bash
.venv\Scripts\python.exe dev_checks\test_pb_equivalence.py
.venv-torch\Scripts\python.exe dev_checks\test_cuda_e2e.py
```
