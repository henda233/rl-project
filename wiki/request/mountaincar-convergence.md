---
request_name: MountainCar 稀疏奖励收敛
source_contents:
  - "agent.py"
  - "config.py"
status: pending
created_at: 2026-06-04 20:30:00
---
# 需求：解决 MountainCar 稀疏奖励导致的 Actor-Critic 不收敛问题

## 需求描述

当前 on-policy Actor-Critic 在 MountainCar-v0 上 1500 episode return 持续为 -200，智能体无法习得摆动策略。需要在不破坏 Actor-Critic 框架的前提下，通过最小改动解决收敛问题。

### 约束

- 必须保持 Actor-Critic 框架
- 可以修改奖励函数（reward shaping）
- 目标：return 稳定在 -120~-100，solve rate > 95%（接近最优解）

### 方案

奖励塑形（potential-based shaping，势函数 Φ(s)=|velocity|，缩放 C=10）+ ε-greedy 探索（0.3→0.01，衰减 0.999/每 episode）。

## 关联摘要

- `wiki/abstract/actor-critic-impl.md`
- `wiki/abstract/gymnasium/mountain-car.md`
- `wiki/abstract/docs/actor-critic-algorithm.md`
- `wiki/abstract/plan-actor-critic.md`
