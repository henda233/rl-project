---
request_name: PPO 算法实现需求
status: pending
created_at: 2026-06-04 23:30:00
---
# 需求：引入 PPO 算法解决 MountainCar 收敛问题

## 背景

Actor-Critic 在 MountainCar-v0 上经历三轮奖励塑形（velocity-based → energy-based → episode-relative progress），5000 episode 后 return 仍为 -200 不收敛。推测根因为 AC 算法局限性（TD(0) 高方差、单次数据利用、无约束策略更新），而非奖励塑形设计问题。

## 需求描述

参考 `examples/ppo_example.py`，实现 PPO（Proximal Policy Optimization）算法，目标是在 MountainCar-v0 上达成收敛。

## 关键决策

1. **奖励塑形**：保留当前 episode-relative progress 塑形方案
2. **代码组织**：新建 `ppo_agent.py`，与 `agent.py` 并存，各自独立
3. **训练循环**：PPO 使用独立训练循环
4. **ε-greedy**：PPO 中移除 ε-greedy（Categorical 采样 + clip 约束自带探索）
5. **超参数**：GAMMA 从 0.5 提升至 0.98（PPO 标准值，支持长程信用分配）

## 关联计划

- [implement-ppo](../plan/implement-ppo.md)
