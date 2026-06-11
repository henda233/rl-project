---
abstract_name: DeepCubeA 神经网络模块
source_contents:
  - "deepcubea_network.py"
  - "deepcubea_train.py"
  - "config.py"
dependencies:
  - "wiki/abstract/huarongdao-env.md"
  - "wiki/abstract/docs/deepcubea-research.md"
created_at: 2026-06-11 19:00:00
updated_at: 2026-06-11 18:50:00
---
# 摘要：DeepCubeA 神经网络模块

## 核心结论与关键信息

- **one-hot 编码**：16 位置 × 16 值 = 256 维，位置 i 值 v → bit `i*16+v`，空格(0)作为普通值参与编码
- **纯函数状态转移**：`transition(grid_flat, action)` 无环境副作用，非法动作返回 None，action 语义与环境一致（0=上,1=下,2=左,3=右）
- **网络结构**：Input(256) → FC(256)+ReLU → FC(256)+ReLU → 4×ResBlock → FC(1)，输出标量 J(s)
- **残差块**：FC(256)+ReLU → FC(256) → +input → ReLU
- **训练方式**：近似值迭代 (AVI)，每 epoch 全量计算 J'(s) = min_a (1+J(A(s,a))) 作为 target（目标状态 J'=0），MSE loss + Adam 做 batch 训练
- **训练结果**：4131 去重状态（5000 raw, T~U(1,100)），1000  epoch，最佳 loss=0.0138（epoch 527），模型 `results/train_20260611_183307/deepcubea_heuristic_best.pt`
- **AVI 发散**：epoch 527 后 loss 从 0.014 升至 0.26 —— 函数逼近器 Bellman 备份误差累积的典型模式

## 内容概述

> `deepcubea_network.py` 包含编码模块和网络类，提供 encode/encode_batch/decode/transition/DeepCubeANetwork。`deepcubea_train.py` 包含 generate_training_data（反向随机游走生成训练状态）、compute_targets（全量 Bellman 备份计算 J'(s)）、train（训练主循环 + checkpoint + loss 曲线）。训练数据从目标状态反向生成，去重在训练开始时执行。

## 依赖与影响链

- **上游依赖**：`config.py`（HUARONGDAO_N=4 → N2=16；DEEPCUBEA_* 训练参数）
- **下游被依赖**：`wiki/plan/deepcubea-search.md`（加权 A* 搜索，使用 best.pt 作为启发函数）
- **变更扩散评估**：中（网络结构修改会影响训练脚本，编码格式修改会影响下游 A* 搜索）
