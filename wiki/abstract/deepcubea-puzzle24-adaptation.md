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
  - "data/tiles.npy"
  - "data/solution_lengths.npy"
  - "data/model_state_dict.pt"
dependencies:
  - "wiki/abstract/deepcubea-official-model-evaluation.md"
  - "wiki/abstract/deepcubea-official-repo.md"
  - "wiki/abstract/docs/deepcubea-research.md"
created_at: 2026-06-14 20:00:00
updated_at: 2026-06-14 22:22:00
---
# 摘要：DeepCubeA 24-Puzzle 适配计划

## 核心结论与关键信息

- **变更规模极小**：仅 2 个文件需要修改代码（`config.py` + `deepcubea_official_network.py`），其余 7 个文件已 N 泛型自动适配
- **config.py**：`HUARONGDAO_N=4→5`，`USE_OFFICIAL_DATA=False→True`；`MODEL_PATH`/`DATA_DIR` 不变（分支专属，data/ 下文件直接覆盖）
- **模型文件**：从 `DeepCubeA/saved_models/puzzle24/current/model_state_dict.pt` 拷贝到 `data/model_state_dict.pt`（62MB）
- **deepcubea_official_network.py**（Option B）：`load_official_model()` L107 显式传参 `state_dim=N2, one_hot_depth=N2`；`predict_j`/`predict_j_batch` docstring 更新；ResnetModel 默认值不变
- **参数不变**：T_MAX/MAX_EXPAND_NODES/LAMBDA 等沿用 15-puzzle 值
- **仅评估不做训练**
- **数据预处理**：`data_0.pkl`（496 状态，tiles uint8，解路径 64-110 mean 89.4）→ `data/tiles.npy (496,25) int64` + `data/solution_lengths.npy (496,) int64`。dtype 选 int64 避免 `_GOAL_BYTES` 比较失败（历史 Bug 复盘教训）
- **模型**：62MB，puzzle24 官方 ResnetModel，DataParallel 训练（`module.` 前缀自动剥离），fc1.weight [5000,625]=N²×N² 确认匹配
- **核心风险**：`MAX_EXPAND_NODES=10000` 对 puzzle24 极低（官方展开 85M~15B 节点）；先 10000 冒烟验证 pipeline 通畅，后续由用户调整
- **S3 冒烟通过**：J(s₀)=104.11, J(sₙ)=89.16，均在 [25, 140]，PASS

## 内容概述

> S0-S3 已完成，S4-S5 待执行。修正：数据/模型文件直接放 `data/` 根目录，不创建 `puzzle24/` 子目录（分支专属）。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-official-model-evaluation.md`（三项评估框架复用）、`wiki/abstract/deepcubea-official-repo.md`（ResnetModel 架构知识）
- **下游被依赖**：无（评估性质，不产生被依赖的产出）
- **变更扩散评估**：低（2 文件代码修改 + 1 数据预处理，不影响其余模块）
