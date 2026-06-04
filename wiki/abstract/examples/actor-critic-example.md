---
abstract_name: Actor-Critic 参考实现
source_contents:
  - "examples/actor_critic_example.py"
dependencies:
  - "wiki/abstract/docs/actor-critic-algorithm.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：Actor-Critic 参考实现

## 核心结论与关键信息

- **PolicyNet**：`Linear(state_dim, hidden_dim) → ReLU → Linear(hidden_dim, action_dim) → Softmax`，输出动作概率分布
- **ValueNet**：`Linear(state_dim, hidden_dim) → ReLU → Linear(hidden_dim, 1)`，输出状态价值标量
- **ActorCritic 类**：
  - `take_action(state)`：通过 Categorical 分布采样动作
  - `update(transition_dict)`：批量更新，核心逻辑：
    - TD target = `rewards + γ · critic(next_states) · (1 - dones)`
    - TD delta = `td_target - critic(states)`
    - Actor loss = `mean(-log_prob · td_delta.detach())`
    - Critic loss = `MSE(critic(states), td_target.detach())`
- **关键注意**：该示例使用旧版 `gym` API（CartPole-v0、4 返回值 step、env.seed），需改写为 gymnasium API

## 内容概述

> 一个完整的 Actor-Critic on-policy 实现：双网络结构（策略网络 + 价值网络），各自独立 Adam 优化器，on-policy 训练循环（采样一个完整 episode → 打包为 transition_dict → 一次性更新）。算法逻辑可直接参考，但需适配 gymnasium API 和 MountainCar-v0 环境。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/actor-critic-algorithm.md`（算法理论）
- **下游被依赖**：`wiki/request/initial-requirements.md`（代码实现的直接参考）
- **变更扩散评估**：高（核心实现模板，几乎所有代码以此为蓝本）
