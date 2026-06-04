---
abstract_name: MountainCar 环境规范
source_contents:
  - "docs/gymnasium/Mountain Car.md"
dependencies:
  - "wiki/abstract/docs/project-overview.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：MountainCar 环境

## 核心结论与关键信息

- **动作空间**：`Discrete(3)` — 0: 向左加速, 1: 不加速, 2: 向右加速
- **观测空间**：`Box([-1.2, -0.07], [0.6, 0.07], (2,), float32)` — [位置, 速度]
- **动力学**：
  - `速度_{t+1} = 速度_t + (动作-1)×0.001 - cos(3×位置_t)×0.0025`
  - `位置_{t+1} = 位置_t + 速度_{t+1}`
  - 撞墙（位置 ≤ -1.2 或 ≥ 0.6）时速度归零
- **初始状态**：位置 ∈ [-0.6, -0.4] 随机，速度 = 0
- **奖励**：每步 -1（鼓励尽快到达目标）
- **终止条件**：位置 ≥ 0.5（成功）或 200 步（截断）
- **创建方式**：`gym.make("MountainCar-v0")`，可选参数 `render_mode`、`goal_velocity`

## 内容概述

> MountainCar 是经典控制 MDP，车辆停在正弦波谷底，需要策略性加速冲上右侧山顶。环境提供位置和速度两个连续观测值，3 个离散动作。奖励稀疏（每步 -1），需要智能体学会"先向左后向右"的摆动策略。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/project-overview.md`
- **下游被依赖**：`wiki/request/initial-requirements.md`（环境搭建的直接参考）
- **变更扩散评估**：低（预设环境，参数固定）
