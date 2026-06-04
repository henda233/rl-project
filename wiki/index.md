# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-04 20:00:00

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
| `MountainCar收敛方案` | [🔗](../wiki/plan/mountaincar-convergence.md) | 奖励塑形(Φ=|v|,C=10) + ε-greedy(0.3→0.01,decay=0.999)，计划待执行 | ✅ 无循环依赖 | 06-04 |

## 需求列表

| 需求名称 | 需求路径 | 状态 | 关联计划 |
|---|---|---|---|
| `初始项目需求` | [🔗](../wiki/request/initial-requirements.md) | pending | `实现强化学习算法`（已完成） |
| `MountainCar收敛需求` | [🔗](../wiki/request/mountaincar-convergence.md) | pending | `MountainCar稀疏奖励收敛`（待执行） |

## 计划列表

| 计划名称 | 计划路径 | 状态 |
|---|---|---|
| `搭建强化学习任务环境` | [🔗](../wiki/plan/setup-environment.md) | completed |
| `实现强化学习算法` | [🔗](../wiki/plan/implement-actor-critic.md) | completed |
| `MountainCar稀疏奖励收敛` | [🔗](../wiki/plan/mountaincar-convergence.md) | completed |

## TODO列表

- [x] 制定`1. 搭建强化学习任务环境`的执行计划
- [x] 创建 config.py + env.py 环境模块，自检通过
- [x] 制定`2. 实现强化学习算法`的执行计划
- [x] 执行`2. 实现强化学习算法`计划
- [x] 编写 AC 实现模块摘要并更新 index
- [x] 制定 MountainCar 收敛问题执行计划 — 奖励塑形(Φ=|v|,C=10) + ε-greedy(ε=0.3→0.01,decay=0.999)
- [x] 执行MountainCar 收敛问题计划

## 笔记

- **项目文档纠正**：`docs/项目文档.md` 中"自定义2D物理环境"的描述有误，实际项目使用 gymnasium 预设的 MountainCar-v0 环境（2026-06-04 确认）。
- **API 适配要点**：参考代码 `examples/` 使用旧版 gym API，所有代码需使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。
- **MountainCar 稀疏奖励收敛问题**：on-policy Actor-Critic 在 MountainCar-v0 上 1500 episode return 仍为 -200。每步奖励恒为 -1，智能体无法获得正向反馈引导探索。可能方向：奖励塑形（reward shaping）、ε-greedy 额外探索、经验回放缓冲（2026-06-04）。

## 全局更新日志（近10条）

- `06-04 21:00`: MountainCar 收敛计划执行完毕，训练过程增加塑性奖励显示（tqdm + console）
- `06-04 20:30`: 制定 MountainCar 收敛方案执行计划 — 奖励塑形(Φ=|v|,C=10) + ε-greedy(ε=0.3→0.01,decay=0.999)
- `06-04 20:00`: AC 算法实现完成——config.py 新增超参数、env.py 支持 render_mode 覆盖、agent.py 创建（PolicyNet + ValueNet + ActorCritic + 训练循环 + 绘图），自检通过但稀疏奖励不收敛已知
- `06-04 19:00`: 制定`实现强化学习算法`执行计划 —— `wiki/plan/implement-actor-critic.md`
- `06-04 18:50`: 完成`搭建强化学习任务环境`计划 —— 创建 config.py、env.py、wiki/abstract/env-setup.md
- `06-04 18:40`: 制定`搭建强化学习任务环境`执行计划 —— `wiki/plan/setup-environment.md`
- `06-04 18:36`: WIKI 记忆库初始化 —— 创建 8 个摘要、1 个需求文档、index.md、readme.md
