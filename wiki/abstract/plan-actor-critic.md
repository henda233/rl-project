---
abstract_name: Actor-Critic算法实现计划
source_contents:
  - "wiki/plan/implement-actor-critic.md"
dependencies:
  - "wiki/abstract/env-setup.md"
  - "wiki/abstract/docs/actor-critic-algorithm.md"
  - "wiki/abstract/examples/actor-critic-example.md"
created_at: 2026-06-04 19:00:00
updated_at: 2026-06-04 19:00:00
---
# 摘要：Actor-Critic 算法实现计划

## 核心结论与关键信息

- **文件组织**：单文件 `agent.py` 包含 PolicyNet、ValueNet、ActorCritic、训练循环、绘图全部逻辑
- **超参数追加到 `config.py`**：HIDDEN_DIM=128、ACTOR_LR=1e-3、CRITIC_LR=1e-2、GAMMA=0.98、NUM_EPISODES=500
- **探索策略**：仅用 Categorical 采样，不加 ε-greedy（softmax 随机性 + 3 离散动作足够探索）
- **渲染策略**：`RENDER_MODE` 默认 `None`，`make_env(render_mode=...)` 支持覆盖，训练不渲染
- **收敛预期**：500 episode 不求完美学会，回报从 -200 → -150~-120 趋势即通过自检
- **必须适配 gymnasium API**：`reset(seed=...)` → `(obs, info)`，`step` → 5 元组，`done = terminated or truncated`

## 内容概述

> 计划分为 5 步：S1 修改 config.py/env.py 新增超参数和 render_mode 参数 → S2 创建 agent.py（网络 + Agent 类）→ S3 集成训练循环 + 绘图 → S4 自检验证收敛趋势 → S5 编写摘要更新 index。算法逻辑参考 examples/actor_critic_example.py，TD 残差驱动双网络更新，on-policy 每个 episode 收集完整轨迹后一次性更新。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/env-setup.md`（环境模块）、`wiki/abstract/docs/actor-critic-algorithm.md`（算法理论）、`wiki/abstract/examples/actor-critic-example.md`（参考实现）
- **下游被依赖**：后续需求 3（训练与可视化）将基于 `agent.py` 进行调参优化和可视化增强
- **变更扩散评估**：高（agent.py 是项目核心，config.py 超参数被所有后续模块使用）
