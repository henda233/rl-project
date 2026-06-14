---
abstract_name: DeepCubeA 官方 BWAS 实现分析
source_contents:
  - "DeepCubeA/search_methods/astar.py"
  - "DeepCubeA/cpp/parallel_weighted_astar.cpp"
  - "DeepCubeA/utils/nnet_utils.py"
  - "deepcubea_search.py"
dependencies:
  - "wiki/abstract/deepcubea-official-repo.md"
  - "wiki/abstract/deepcubea-search.md"
created_at: 2026-06-14 19:00
updated_at: 2026-06-14 19:00
---
# 摘要：DeepCubeA 官方 BWAS 实现分析

## 核心结论与关键信息

- **官方实现了 BWAS**：Python 层 `AStar` 类 + `Instance` 类 + C++ 层 `parallel_weighted_astar`，但核心优化（子节点合并 batch GPU 推理）我们已通过 `predict_j_batch` 等价实现
- **官方多实例设计未被实际使用**：`AStar` 架构支持多个起始状态同时搜索 + 跨实例共享 GPU 推理，但 `bwas_python()` 和 `bwas_cpp()` 实际只传单状态 `AStar([state], ...)`，多实例能力是闲置代码
- **不建议拷贝**：(1) 无功能缺口，批量 GPU 推理已有；(2) class-based（AStar/Instance/Node）+ NPuzzleState 与当前 function-based + bytes/numpy 风格冲突，适配成本高；(3) 投入产出比低
- **单 GPU 完全可运行**：`nnet_utils.get_device()` 只用 `cuda:0`，DataParallel 可选，官方 puzzle15 评估（5.3-12.9s/状态）即在单 GPU 完成
- **官方 BWAS 两层架构**：Python `AStar`（多实例调度）+ C++ `parallel_weighted_astar`（OpenMP 并行展开 + Unix socket 通信 GPU 推理）
- **值得关注而非拷贝的点**：batch_size=20000（每次 pop 大量节点）、weight=0.8、clip_zero（负 J(s) 裁剪为 0）

## 内容概述

> 深入阅读官方 `astar.py`（642 行）、`parallel_weighted_astar.cpp`（397 行）、`nnet_utils.py`（312 行）后，与当前项目 `deepcubea_search.py` 逐项对比。官方 BWAS 的"batch"本质是每次迭代将所有子节点合并为一个 batch 送 GPU 推理 —— 这正是我们 `predict_j_batch` 的功能。多实例架构虽存在但官方自身未启用，拷贝改造投入产出比低。

## 官方 BWAS 架构详解

### Python 层 (`astar.py`)

- **`Node`**：搜索树节点，包含 state / path_cost / heuristic / cost / parent / children / bellman
- **`Instance`**：单状态搜索实例，持有独立 open_set（heapq）+ closed_dict（`Dict[State, float]`）
- **`AStar`**：多实例调度器，`step()` 每步依次执行 pop → expand → heuristic（batch GPU）→ check_closed → add_to_open
- **`bwas_python()`**：实际入口，逐状态 `AStar([state], env, heuristic_fn, [args.weight])` 循环，并未启用多实例

### C++ 层 (`parallel_weighted_astar.cpp`)

- 单状态搜索，`numParallel` 控制每次 pop 的节点数
- OpenMP `#pragma omp parallel for` 并行展开子节点
- 通过 Unix socket 将子节点批量发送给 Python 进程 → GPU 推理 → 接收 J(s) 值回传
- `bwas_cpp()` 逐状态 `Popen` 启动 C++ 进程，逐状态求解

## 与我们实现的逐项对比

| 维度 | 官方 BWAS | 我们 weighted_astar |
|---|---|---|
| 结构 | class AStar + Instance + Node | function |
| 状态 | NPuzzleState 对象 | bytes hash + numpy grid |
| 展开 | env.expand(states) | get_children(grid, blank_idx) |
| 批量 GPU 推理 | ✅ heuristic_fn(states) 合并 batch | ✅ predict_j_batch 同样做法 |
| 多实例 | 架构支持但实际未用 | 不支持 |
| closed set | dict[State, path_cost]（支持更短路径覆盖） | set[bytes] |
| 路径回溯 | Node parent 指针链 | g_score 字典存 parent bytes |
| 权重公式 | f = weight × path_cost + heuristic | f = λ × g + h（等价） |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-official-repo.md`（官方仓库结构）、`wiki/abstract/deepcubea-search.md`（当前搜索实现）
- **下游被依赖**：无（纯分析结论，不产生代码变更）
- **变更扩散评估**：低（决策是不拷贝，不影响现有代码）
