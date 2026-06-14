---
abstract_name: DeepCubeA 24-Puzzle 适配计划
source_contents:
  - "wiki/request/huarongdao-n5-official-model.md"
  - "wiki/plan/deepcubea-puzzle24-adaptation.md"
  - "config.py"
  - "deepcubea_official_network.py"
  - "deepcubea_official_search.py"
  - "DeepCubeA/saved_models/puzzle24/"
  - "DeepCubeA/data/puzzle24/test/data_0.pkl"
dependencies:
  - "wiki/abstract/deepcubea-official-model-evaluation.md"
  - "wiki/abstract/deepcubea-official-repo.md"
  - "wiki/abstract/docs/deepcubea-research.md"
created_at: 2026-06-14 20:00:00
updated_at: 2026-06-14 20:30:00
---
# 摘要：DeepCubeA 24-Puzzle 适配计划

## 核心结论与关键信息

- **变更规模极小**：仅 2 个文件需要修改代码（`config.py` + `deepcubea_official_network.py`），其余 7 个文件已 N 泛型自动适配
- **config.py**：`HUARONGDAO_N=4→5`，`DEEPCUBEA_OFFICIAL_MODEL_PATH` 指向 puzzle24，`DEEPCUBEA_OFFICIAL_DATA_DIR` 改为 `data/puzzle24/`
- **deepcubea_official_network.py**：`ResnetModel` 参数 `state_dim=N², one_hot_depth=N²` 从 `HUARONGDAO_N` 推导（替换硬编码 16），`predict_j`/`predict_j_batch` docstring 中 `(16,)` → `(N2,)`
- **ResnetModel 调用点确认**：唯一调用在 `load_official_model()` L104，不传参完全依赖默认值，改默认值自动覆盖
- **参数不变**：T_MAX/MAX_EXPAND_NODES/LAMBDA 等沿用 15-puzzle 值
- **仅评估不做训练**
- **数据预处理**：`data_0.pkl`（496 状态，tiles uint8，解路径 64-110 mean 89.4）→ `data/puzzle24/tiles.npy (int64)` + `solution_lengths.npy (int64)`。dtype 选 int64 避免 `_GOAL_BYTES` 比较失败（历史 Bug）
- **模型**：62MB，puzzle24 官方 ResnetModel，DataParallel 训练（`module.` 前缀自动剥离）
- **核心风险**：`MAX_EXPAND_NODES=10000` 对 puzzle24 可能太低（15-puzzle 官方展开 196-466 万节点）；先用 10000 跑，胜率 0% 则调大
- **S3 冒烟通过标准**：取 `tiles[0]` 和 `tiles[-1]` 各测一次，`25 ≤ J(s) ≤ 140` 通过，负值/NaN 失败，`>200` 可疑不阻断

## 内容概述

> 计划文档定义了 6 个步骤（S0-S5）：数据预处理 → config 修改 → 网络适配 → 冒烟测试 → Bellman MSE 评估 → 加权 A* 评估。需求文档记录了约束条件和关键决策。计划已通过用户确认，补充了 S3 冒烟通过标准和代码审查确认。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-official-model-evaluation.md`（三项评估框架复用）、`wiki/abstract/deepcubea-official-repo.md`（ResnetModel 架构知识）
- **下游被依赖**：无（评估性质，不产生被依赖的产出）
- **变更扩散评估**：低（2 文件代码修改 + 1 数据预处理，不影响其余模块）
