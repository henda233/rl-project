---
abstract_name: DeepCubeA 官方仓库探索笔记
source_contents:
  - "docs/deepcubea/official-repo-notes.md"
  - "DeepCubeA/README.md"
  - "DeepCubeA/utils/pytorch_models.py"
  - "DeepCubeA/environments/n_puzzle.py"
  - "DeepCubeA/search_methods/astar.py"
  - "DeepCubeA/ctg_approx/avi.py"
  - "DeepCubeA/saved_models/puzzle15/"
  - "DeepCubeA/results/puzzle15/"
dependencies:
  - "wiki/abstract/docs/deepcubea-research.md"
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/abstract/deepcubea-search.md"
created_at: 2026-06-13 23:50
updated_at: 2026-06-13 23:50
---
# 摘要：DeepCubeA 官方仓库探索笔记

## 核心结论与关键信息

- **仓库**：DeepCubeA 官方 PyTorch 实现，支持 7 种谜题（含 15-puzzle）
- **网络架构不兼容**：官方 `ResnetModel`（5000-1000 隐藏层 + BN + 内部 one_hot，约 15M 参数）与当前项目 `DeepCubeANetwork`（256/512 隐藏层 + LN + 外部预编码，约 1.3M 参数）完全不同，state_dict 无法直接加载
- **puzzle15 预训练模型位置**：`DeepCubeA/saved_models/puzzle15/current/model_state_dict.pt`（可能含 DataParallel 的 `module.` 前缀）
- **官方评估结果**：500 状态 100% 求解（36-69 步，avg 52），展开 196-466 万节点，求解时间 5.3-12.9s/状态
- **puzzle15 训练参数**：50M states/update, batch_size=10000, max_itrs=1M, loss_thresh=0.1, back_max=500, 30 并行进程
- **状态表示差异**：官方使用 NPuzzleState 对象（含 tiles + hash），当前使用裸 numpy + bytes hash
- **A* 搜索差异**：官方为 class-based 多实例批处理设计，支持 C++ 后端；当前为 function-based 单实例
- **NPuzzleState 可拷贝**：轻量类（仅 tiles 数组 + hash），可脱离官方仓库的其余依赖独立使用

## 内容概述

> `docs/deepcubea/official-repo-notes.md` 记录了官方仓库的完整目录结构、ResnetModel / NPuzzleState / A* 搜索 / AVI 训练四大核心组件的详细分析，以及与当前项目实现的关键差异对比。是连接官方模型到当前项目评估框架的适配工作基础。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/deepcubea-research.md`（论文原理）
- **下游被依赖**：`wiki/plan/deepcubea-official-model-evaluation.md`（适配评估计划）、`wiki/abstract/deepcubea-bwas-analysis.md`（BWAS 分析）
- **变更扩散评估**：低（纯笔记，不修改代码）
