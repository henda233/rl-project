---
abstract_name: DeepCubeA 加权 A* 搜索
source_contents:
  - "deepcubea_search.py"
  - "config.py"
dependencies:
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/abstract/docs/deepcubea-research.md"
  - "wiki/abstract/huarongdao-env.md"
created_at: 2026-06-11 19:30:00
updated_at: 2026-06-11 20:00:00
---
# 摘要：DeepCubeA 加权 A* 搜索

## 核心结论与关键信息

- **加权公式**：f(s) = λ·g(s) + h(s)，h(s) = J(s) 由训练好的 DeepCubeANetwork 提供
- **搜索算法**：基于 heapq 优先队列的标准 A*，每次展开 f 最小节点，通过 came_from 字典回溯路径
- **批量预测优化**：每次节点展开将所有子节点一次性送入 `predict_j_batch`，N 次前向 → 1 次，理论提速 ~4 倍
- **状态表示**：tuple(int) 用于 hash，numpy 数组用于 transition 计算
- **三档评估**：`[T_MIN, T_MAX]` 均分为 Short/Medium/Long 三档，每档 `NUM_TEST_STATES/3` 个状态，42 固定种子随机游走生成
- **失败判定**：展开节点数超过 `MAX_EXPAND_NODES` 或队列为空 → 判为求解失败
- **模型加载**：`DEEPCUBEA_MODEL_PATH` 默认空字符串，运行时必须显式指定，防止静默加载错误模型

## 首次评估结果（训练参数：T~U(1,100), 5k states, 1k iter, best loss=0.0138）

| 指标 | Short (1-34步, 34状态) | 分析 |
|---|---|---|
| 胜率 | 11/34 (32.35%) | 仅能解极近距离状态 |
| 平均解路径长度 | 0.9 | 解出的几乎都是距目标 0-2 步的状态 |
| 平均展开节点数 | 6,765 | 逼近上限 10,000，启发放函数带偏搜索 |

**根因**：J(s) 训练严重不足。5k 状态 × T~U(1,100) 的随机游走效率极低（分支因子 2-3，兜圈子严重），实际覆盖的仅是距目标 ~15-20 最优步内的球。4131 去重状态不足以学到可泛化的值函数。λ=0.3 进一步放大启发函数误差。

**反馈驱动参数调整**：见 `deepcubea-network.md` 训练参数更新（T~U(10,500), 50k states, LR=1e-4, 2k iter）。

## 内容概述

> `deepcubea_search.py` 包含 `load_model`、`weighted_astar`（批量预测优化）、`evaluate`（进度指示 + 分档统计）三个核心函数。测试脚本 `deepcubea_search_smoke_test.py`（正向随机游走构造单元验证）和 `deepcubea_search_test.py`（完整评估）供用户手动执行。

## 依赖与影响链

- **上游依赖**：`config.py`（DEEPCUBEA_MODEL_PATH/LAMBDA/NUM_TEST_STATES/MAX_EXPAND_NODES + T_MIN/T_MAX）、`deepcubea_network.py`（DeepCubeANetwork/transition/encode）
- **下游被依赖**：`wiki/plan/deepcubea-search.md`（重新评估待新模型训练完成）
- **变更扩散评估**：低（搜索模块独立，不修改环境和网络代码）
