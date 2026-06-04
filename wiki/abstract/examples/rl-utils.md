---
abstract_name: RL 工具函数
source_contents:
  - "examples/rl_utils.py"
dependencies:
  - "wiki/abstract/examples/actor-critic-example.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：RL 工具函数

## 核心结论与关键信息

- **`ReplayBuffer`**：固定容量 deque 存储 transitions `(s, a, r, ns, done)`，支持 `add`、`sample(batch_size)`、`size`
- **`moving_average(a, window_size)`**：计算序列的滑动平均，首尾用递减窗口
- **`train_on_policy_agent(env, agent, num_episodes)`**：on-policy 训练循环，每个 episode 收集完整轨迹后调用 `agent.update(transition_dict)`，返回 `return_list`
- **`train_off_policy_agent(env, agent, num_episodes, replay_buffer, min_size, batch_size)`**：off-policy 训练，每步可能从 buffer 采样更新
- **`compute_advantage(gamma, lmbda, td_delta)`**：GAE 优势函数计算
- **注意**：这些函数同样使用旧 gym API，需适配 gymnasium

## 内容概述

> 提供 RL 训练中常用的工具：经验回放缓冲区、滑动平均、on-policy / off-policy 训练循环、GAE 计算。Actor-Critic 项目主要需要 `train_on_policy_agent` 和 `moving_average` 的逻辑（需适配 gymnasium API）。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/examples/actor-critic-example.md`
- **下游被依赖**：`wiki/request/initial-requirements.md`（训练流程工具参考）
- **变更扩散评估**：中（训练循环工具，非核心算法逻辑）
