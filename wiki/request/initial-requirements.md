---
request_name: 初始项目需求
source_contents:
  - "docs/项目文档.md"
status: pending
created_at: 2026-06-04 18:36:00
---
# 需求：MountainCar + Actor-Critic 强化学习项目

## 需求描述

基于 gymnasium 的预设环境 MountainCar-v0，实现 Actor-Critic 算法，训练智能体取得高分（最优策略）。

### 需求 1：搭建强化学习任务环境

使用 `gymnasium.make("MountainCar-v0")` 创建环境。环境特性：
- 动作空间：Discrete(3) — 左加速、不加速、右加速
- 观测空间：Box(2,) — [位置 ∈ [-1.2, 0.6], 速度 ∈ [-0.07, 0.07]]
- 奖励：每步 -1
- 终止：位置 ≥ 0.5（成功）或 200 步（截断）

### 需求 2：实现 Actor-Critic 算法

- 参考 `examples/actor_critic_example.py` 的算法逻辑
- 必须使用 gymnasium API（非旧 gym）
- 策略网络（Actor）输出动作概率分布
- 价值网络（Critic）输出状态价值
- 使用 TD 残差驱动双网络更新
- 仅使用 numpy、pytorch 基础组件，禁止使用现成 RL 库

### 需求 3：训练与可视化

- 运行训练，智能体逐步学习最优策略
- 训练过程中显示环境画面（render）
- 训练结束后绘制迭代数-累积回报曲线图

### 需求 4：预留扩展

- 后续会增加新需求

## 关联摘要

- `wiki/abstract/docs/project-overview.md`
- `wiki/abstract/docs/actor-critic-algorithm.md`
- `wiki/abstract/gymnasium/mountain-car.md`
- `wiki/abstract/gymnasium/agent-training.md`
- `wiki/abstract/gymnasium/recording-agent.md`
- `wiki/abstract/examples/actor-critic-example.md`
- `wiki/abstract/examples/rl-utils.md`
