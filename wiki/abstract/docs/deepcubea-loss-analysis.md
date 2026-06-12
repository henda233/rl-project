---
abstract_name: DeepCubeA 训练 Loss 曲线分析
source_contents:
  - "docs/DeepCubeA训练loss曲线分析.md"
dependencies:
  - "wiki/abstract/deepcubea-target-network.md"
  - "wiki/abstract/deepcubea-online-validation.md"
created_at: 2026-06-12 20:00:00
updated_at: 2026-06-12 20:00:00
---
# 摘要：DeepCubeA 训练 Loss 曲线分析

## 核心结论与关键信息

- **内层收敛、外层发散**：内层训练迅速将 loss 驱动至 ~0（过拟合当前目标），但外层 saw-tooth 峰值从 0.03 持续攀升至 0.35，表明 Bellman 目标在逐轮恶化
- **根因是 Bellman 备份正反馈雪崩**：目标值剧烈漂移 + 误差累积 + 输出层无界 → J(s) 系统性膨胀
- **诊断**："过拟合于当前错误目标"——每轮学得很好，但下一轮发现上轮学到的是错的
- **改进建议**：引入目标网络（滞后更新）、检查数据生成难度分布、梯度裁剪、学习率衰减
- **与 wiki 笔记的关联**：该分析与 2026-06-11 记录的"AVI 训练发散实证"描述同一机制——Bellman 备份正反馈雪崩，仅时间尺度不同（每 epoch 爆炸 vs 每外层迭代爬升）

## 内容概述

> 针对当前训练配置（外层50/内层1000/早停100/在线采样35000+离线5000）的 loss 曲线分析。从内层和外层两个角度评估收敛性，诊断为目标值剧烈漂移和误差累积导致的价值估计发散，给出四项改进建议。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-target-network.md`（AVI 训练架构）、`wiki/abstract/deepcubea-online-validation.md`（在线采样设计）
- **下游被依赖**：`wiki/plan/deepcubea-target-threshold.md`（本分析为计划的核心动机和背景依据）
- **变更扩散评估**：低（纯分析文档，不涉及代码变更）
