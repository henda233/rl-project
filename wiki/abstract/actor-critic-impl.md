---
abstract_name: Actor-Critic 实现模块
source_contents:
  - "config.py"
  - "env.py"
  - "agent.py"
dependencies:
  - "wiki/abstract/env-setup.md"
  - "wiki/abstract/docs/actor-critic-algorithm.md"
  - "wiki/abstract/examples/actor-critic-example.md"
  - "wiki/abstract/plan-actor-critic.md"
created_at: 2026-06-04 20:00:00
updated_at: 2026-06-04 20:00:00
---
# 摘要：Actor-Critic 实现模块

## 核心结论与关键信息

- **config.py**：RENDER_MODE=None（训练不渲染）、HIDDEN_DIM=128、ACTOR_LR=1e-3、CRITIC_LR=1e-2、GAMMA=0.98、NUM_EPISODES=5000、EVAL_INTERVAL=50
- **env.py**：`make_env(render_mode=None)` 支持覆盖默认 render_mode，通过 `gym.make(ENV_NAME, render_mode=mode)` 创建环境
- **agent.py**（单文件，~145 行）：
  - `PolicyNet`：Linear(2,128)→ReLU→Linear(128,3)→Softmax
  - `ValueNet`：Linear(2,128)→ReLU→Linear(128,1)
  - `ActorCritic`：双网络 + 各自 Adam 优化器，Categorical 采样动作，TD 残差批量更新（td_delta 加权 Actor、td_target MSE 更新 Critic）
  - `train_on_policy_agent`：gymnasium API 适配，tqdm 进度条，每 10 episode 打印近 10 轮平均回报，每 50 episode 渲染评估
  - `moving_average` + `plot_return`：滑动平均平滑 + matplotlib 回报曲线绘制
- **探索策略**：仅 Categorical 采样，无 ε-greedy
- **已知问题**：MountainCar 奖励稀疏（每步 -1），on-policy Actor-Critic 在 1500 episode 内 return 持续 -200 不收敛，后续需通过奖励塑形或改进探索解决

## 内容概述

> config.py 集中管理环境和算法超参数（8 个配置项）。env.py 封装 gymnasium 环境创建，支持 render_mode 参数覆盖。agent.py 包含完整的 Actor-Critic 实现：双网络（PolicyNet/ValueNet）定义、ActorCritic 类（take_action + update）、gymnasium 适配的 on-policy 训练循环、评估函数、滑动平均与绘图工具、main 入口。代码不含第三方 RL 库，仅依赖 numpy + pytorch + gymnasium + tqdm + matplotlib。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/env-setup.md`（环境模块）、`wiki/abstract/docs/actor-critic-algorithm.md`（算法理论）、`wiki/abstract/examples/actor-critic-example.md`（参考实现）、`wiki/abstract/plan-actor-critic.md`（执行计划）
- **下游被依赖**：后续需求 3（训练与可视化优化）将基于 agent.py 进行收敛性改进
- **变更扩散评估**：高（agent.py 为核心模块，config.py 超参数被所有模块使用）
