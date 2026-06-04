---
abstract_name: 智能体训练指南
source_contents:
  - "docs/gymnasium/智能体训练.md"
dependencies:
  - "wiki/abstract/docs/project-overview.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：智能体训练指南

## 核心结论与关键信息

- **训练循环模式**：`env.reset()` → 循环 `agent.select_action()` → `env.step()` → `agent.update()` → 直到 done → `decay_epsilon()`
- **ε-greedy 探索**：以概率 ε 随机探索，以概率 1-ε 选择最优动作，ε 逐步衰减
- **gymnasium API 关键差异**（对比旧 gym）：
  - `env.step(action)` 返回 5 个值：`(obs, reward, terminated, truncated, info)`
  - `done = terminated or truncated`
  - `env.reset(seed=...)` 替代 `env.seed()`
- **可视化**：`matplotlib` 绘制 rewards/episode length/training error 的移动平均曲线
- **超参数**：learning_rate (0.001-0.1)、discount_factor (0.9-0.99)、episodes 数量

## 内容概述

> 以 Blackjack 的 Q-learning 为例讲解训练流程：Q 表/网络的值迭代更新（Bellman 方程）、探索策略选择、训练循环实现、结果可视化和常见问题排查。项目需将其中的模式适配到 Actor-Critic 算法。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/project-overview.md`
- **下游被依赖**：`wiki/request/initial-requirements.md`（训练循环实现参考）
- **变更扩散评估**：中（训练循环模式影响代码结构）
