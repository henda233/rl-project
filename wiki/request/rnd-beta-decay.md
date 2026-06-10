---
request_name: RND_BETA 线性衰减
status: completed
created_at: 2026-06-05 21:30:00
---
# 需求：RND_BETA 参数线性衰减

## 背景

当前 PPO-RND 算法中 `RND_BETA = 100` 为常数，内在奖励权重在整个训练过程中不变。在 MountainCar 2D 状态空间中，训练后期应逐步降低探索权重，让外部任务奖励主导策略优化。

## 需求描述

将 `RND_BETA` 改为随时间线性衰减：每 episode 减去固定衰减量，直到触及终止值地板。

## 实现规格

| 参数 | 值 | 说明 |
|---|---|---|
| `RND_BETA` | 100（不变） | β 起始值 |
| `RND_BETA_END` | 1.0 | β 终止值（地板） |
| `RND_BETA_DECAY` | 0.05 | 每 episode 线性衰减量 |

衰减公式：`β_episode = max(RND_BETA_END, β_last - RND_BETA_DECAY)`

推算：`(100 - 1) / 0.05 = 1980` episode 后 β 到达地板，之后保持 1.0 直到训练结束。
