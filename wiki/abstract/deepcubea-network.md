---
abstract_name: DeepCubeA 神经网络模块
source_contents:
  - "deepcubea_network.py"
  - "deepcubea_train.py"
  - "config.py"
dependencies:
  - "wiki/abstract/huarongdao-env.md"
  - "wiki/abstract/docs/deepcubea-research.md"
  - "wiki/abstract/deepcubea-search.md"
created_at: 2026-06-11 19:00:00
updated_at: 2026-06-11 20:00:00
---
# 摘要：DeepCubeA 神经网络模块

## 核心结论与关键信息

- **one-hot 编码**：16 位置 × 16 值 = 256 维，位置 i 值 v → bit `i*16+v`，空格(0)作为普通值参与编码
- **纯函数状态转移**：`transition(grid_flat, action)` 无环境副作用，非法动作返回 None，action 语义与环境一致（0=上,1=下,2=左,3=右）
- **网络结构**：Input(256) → FC(256)+ReLU → FC(256)+ReLU → 4×ResBlock → FC(1)，输出标量 J(s)
- **残差块**：FC(256)+ReLU → FC(256) → +input → ReLU
- **训练方式**：近似值迭代 (AVI)，每 epoch 全量计算 J'(s) = min_a (1+J(A(s,a))) 作为 target（目标状态 J'=0），MSE loss + Adam 做 batch 训练
- **AVI 发散**：函数逼近器 Bellman 备份误差累积的典型模式，无 target network 时 epoch 527 后 loss 从 0.014 升至 0.26

## 首次训练（v1）

| 参数 | 值 | 说明 |
|---|---|---|
| T | U(1, 100) | 随机游走步数范围 |
| 训练集 | 4,131 unique（5,000 raw） | 去重后 |
| LR | 1e-3 | Adam |
| Iter | 1,000 | 最佳 loss=0.0138 @ epoch 527 |
| 模型 | `results/train_20260611_183307/models/deepcubea_heuristic_best.pt` | |

**评估反馈**：A* 搜索 Short 档（1-34 步）胜率仅 32%，仅能解极近距离状态。根因——(1) 随机游走 100 步实际最优距离仅 ~15-20，训练集集中在一个小半径球内；(2) 4k 去重状态不足以泛化；(3) LR=1e-3 震荡大。

## 第二次训练（v2，待执行）

| 参数 | 旧值 | 新值 | 调整依据 |
|---|---|---|---|
| T_MIN | 1 | 10 | 跳过琐碎近距状态，容量分配给有意义的 |
| T_MAX | 100 | 500 | 抵达更远状态，覆盖中远距离区域 |
| TRAIN_SET_SIZE | 5,000 | 50,000 | 10 倍覆盖，适配 RTX 3050 4GB 显存 |
| LR | 1e-3 | 1e-4 | 压低 Bellman 备份震荡 |
| ITERATIONS | 1,000 | 2,000 | 更多迭代补偿更大训练集 |
| BATCH_SIZE | 256 | 256（不变） | |
| HIDDEN_DIM | 256 | 256（不变） | |

**预估**：RTX 3050 峰值显存 ~450 MB，训练时间 40-60 分钟。

## 内容概述

> `deepcubea_network.py` 包含编码模块和网络类，提供 encode/encode_batch/decode/transition/DeepCubeANetwork。`deepcubea_train.py` 包含 generate_training_data（反向随机游走生成训练状态）、compute_targets（全量 Bellman 备份计算 J'(s)）、train（训练主循环 + checkpoint + loss 曲线）。训练数据从目标状态反向生成，去重在训练开始时执行。

## 依赖与影响链

- **上游依赖**：`config.py`（HUARONGDAO_N=4 → N2=16；DEEPCUBEA_* 训练参数）
- **下游被依赖**：`wiki/abstract/deepcubea-search.md`（加权 A* 搜索，使用 best.pt 作为启发函数）
- **变更扩散评估**：中（训练参数修改影响训练脚本和下游搜索评估）
