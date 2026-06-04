---
abstract_name: Actor-Critic算法理论
source_contents:
  - "docs/actor-critic算法.md"
dependencies:
  - "wiki/abstract/docs/project-overview.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：Actor-Critic 算法

## 核心结论与关键信息

- **本质**：基于策略的算法，同时学习策略函数（Actor）和价值函数（Critic）
- **核心公式**：TD 残差 `δ = r + γ·V(s') - V(s)` 作为策略梯度更新的权重
- **Actor 更新**：策略梯度 `θ += α_θ · Σ δ · ∇_θ log π(a|s)`
- **Critic 更新**：MSE 损失 `L = 1/2 · (r + γ·V(s') - V(s))²`，梯度下降更新
- **关键优势**：可在每一步后更新（不依赖蒙特卡洛完整轨迹），不限制任务步数
- **与 REINFORCE 对比**：方差更小、更鲁棒，无需等待 episode 结束

## 内容概述

> 算法流程：初始化策略网络参数 θ 和价值网络参数 ω → 每个 episode 用当前策略采样轨迹 → 每步计算 δ → 用 δ 更新价值网络（MSE）和策略网络（加权梯度）→ 循环。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/project-overview.md`
- **下游被依赖**：`wiki/request/initial-requirements.md`（需求实现的核心算法依据）
- **变更扩散评估**：中（算法参数调整影响训练代码）
