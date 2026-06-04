# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-04 23:15:00

## 模块总览

| 摘要名称 | 摘要路径 | 关键摘要内容 | 依赖健康度 | 最后更新 |
|---|---|---|---|---|
| `项目概述` | [🔗](../wiki/abstract/docs/project-overview.md) | MountainCar-v0 + Actor-Critic 项目总览，技术栈与约束 | ✅ 无循环依赖 | 06-04 |
| `Actor-Critic算法` | [🔗](../wiki/abstract/docs/actor-critic-algorithm.md) | TD残差驱动的双网络（Actor/Critic）更新公式与流程 | ✅ 无循环依赖 | 06-04 |
| `MountainCar环境` | [🔗](../wiki/abstract/gymnasium/mountain-car.md) | Discrete(3) 动作，Box(2) 观测，每步-1奖励，200步截断 | ✅ 无循环依赖 | 06-04 |
| `智能体训练` | [🔗](../wiki/abstract/gymnasium/agent-training.md) | ε-greedy、训练循环模式、gym/gymnasium API差异 | ✅ 无循环依赖 | 06-04 |
| `自定义环境` | [🔗](../wiki/abstract/gymnasium/custom-env.md) | gymnasium.Env 继承规范、reset/step 实现模板（当前阶段不直接使用） | ✅ 无循环依赖 | 06-04 |
| `智能体记录` | [🔗](../wiki/abstract/gymnasium/recording-agent.md) | RecordEpisodeStatistics/RecordVideo wrapper 用法 | ✅ 无循环依赖 | 06-04 |
| `AC参考实现` | [🔗](../wiki/abstract/examples/actor-critic-example.md) | PolicyNet/ValueNet 网络结构、ActorCritic.update 核心逻辑 | ✅ 无循环依赖 | 06-04 |
| `RL工具函数` | [🔗](../wiki/abstract/examples/rl-utils.md) | ReplayBuffer、moving_average、on-policy训练循环 | ✅ 无循环依赖 | 06-04 |
| `环境搭建模块` | [🔗](../wiki/abstract/env-setup.md) | config.py + make_env() 工厂函数，MountainCar-v0 环境封装 | ✅ 无循环依赖 | 06-04 |
| `AC算法实现计划` | [🔗](../wiki/abstract/plan-actor-critic.md) | agent.py 单文件实现，超参数与渲染策略决策 | ✅ 无循环依赖 | 06-04 |
| `AC算法实现模块` | [🔗](../wiki/abstract/actor-critic-impl.md) | agent.py 完整实现，gymnasium 适配，已知稀疏奖励不收敛问题 | ✅ 无循环依赖 | 06-04 |
| `MountainCar收敛方案` | [🔗](../wiki/plan/mountaincar-convergence.md) | 奖励塑形(Φ=|v|,C=10) + ε-greedy(0.3→0.01,decay=0.999)，已执行但不收敛 | ✅ 无循环依赖 | 06-04 |
| `Energy-Based 塑形计划` | [🔗](../wiki/plan/energy-based-shaping.md) | Φ=sin(3·pos)+v²/(2g)，C=10，替换 velocity-based 势函数，物理直觉驱动 | ✅ 无循环依赖 | 06-04 |
| `Energy-Based 塑形实现` | [🔗](../wiki/abstract/energy-based-shaping.md) | 势函数公式、最终超参数、改动范围与不改动部分 | ✅ 无循环依赖 | 06-04 |
| `回合相对进度塑形` | [🔗](../wiki/abstract/episode-relative-progress.md) | 追踪 max_x/min_x 打破记录奖励 + 速度信号 + 近终点惩罚，替换 energy-based | ✅ 无循环依赖 | 06-04 |

## 需求列表

| 需求名称 | 需求路径 | 状态 | 关联计划 |
|---|---|---|---|
| `初始项目需求` | [🔗](../wiki/request/initial-requirements.md) | pending | `实现强化学习算法`（已完成） |
| `MountainCar收敛需求` | [🔗](../wiki/request/mountaincar-convergence.md) | completed | `MountainCar稀疏奖励收敛`→`Energy-Based 奖励塑形`（已完成） |
| `Energy-Based 塑形需求` | [🔗](../wiki/request/energy-based-shaping.md) | completed | `Energy-Based 奖励塑形`（已完成） |
| `回合相对进度塑形需求` | [🔗](../wiki/request/episode-relative-progress.md) | completed | `回合相对进度奖励塑形`（已完成，但不收敛） |

## 计划列表

| 计划名称 | 计划路径 | 状态 |
|---|---|---|
| `搭建强化学习任务环境` | [🔗](../wiki/plan/setup-environment.md) | completed |
| `实现强化学习算法` | [🔗](../wiki/plan/implement-actor-critic.md) | completed |
| `MountainCar稀疏奖励收敛` | [🔗](../wiki/plan/mountaincar-convergence.md) | completed |
| `Energy-Based 奖励塑形` | [🔗](../wiki/plan/energy-based-shaping.md) | completed |
| `回合相对进度奖励塑形` | [🔗](../wiki/plan/episode-relative-progress.md) | completed |

## TODO列表

- [ ] 解决 AC 算法不收敛问题（三种奖励塑形方案均失败，推测根因为 AC 算法本身局限性：网络结构、on-policy 样本效率、超参数等）

## 笔记

- **项目文档纠正**：`docs/项目文档.md` 中"自定义2D物理环境"的描述有误，实际项目使用 gymnasium 预设的 MountainCar-v0 环境（2026-06-04 确认）。
- **API 适配要点**：参考代码 `examples/` 使用旧版 gym API，所有代码需使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。
- **MountainCar 稀疏奖励收敛问题**（2026-06-04）：on-policy Actor-Critic 在 MountainCar-v0 上 1500 episode return 仍为 -200。每步奖励恒为 -1，智能体无法获得正向反馈引导探索。
- **三次奖励塑形尝试均不收敛**（2026-06-04）：
  - Velocity-based (Φ=|v|,C=10)：速度评价是位置盲的，撞墙蓄力阶段速度突降会惩罚关键行为，γ<1 泄漏使净贡献为负。
  - Energy-Based (Φ=sin(3·pos)+v²/(2g),C=10)：sin(3·pos) 在关键路径非单调（-1.2→-0.5 段从 +0.44 跌至 -1.0），势函数设计过于复杂。
  - 回合相对进度（记录打破 + 速度 + 近终点惩罚）：代码可运行，5000 episode 仍不收敛。
  - **结论**：根因不是奖励塑形，而是 AC 算法本身局限性。后续方向：调整网络结构（HIDDEN_DIM、层数）、尝试 DQN+经验回放、调整 GAMMA/LR 等超参数。

## 全局更新日志（近5条）

- `06-04 23:15`: 回合相对进度塑形完成——创建 request/plan/abstract，代码可运行但不收敛，推测 AC 算法局限。
- `06-04 22:40`: Energy-Based 塑形执行完毕——创建 abstract、更新 index，config.py 新增 G/PE_COEFFICIENT，agent.py 替换公式。
- `06-04 21:00`: MountainCar 收敛计划执行完毕——奖励塑形(Φ=|v|,C=10) + ε-greedy(0.3→0.01,decay=0.999)，训练增加塑性奖励显示。
- `06-04 20:00`: AC 算法实现完成——config.py 新增超参数、agent.py 创建（PolicyNet/ValueNet/ActorCritic/训练循环/绘图），稀疏奖励不收敛已知。
- `06-04 18:36`: WIKI 记忆库初始化——创建 8 个摘要、1 个需求文档、index.md、readme.md。
