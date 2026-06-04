# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-04 18:36:00

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

## 需求列表

| 需求名称 | 需求路径 | 状态 | 关联计划 |
|---|---|---|---|
| `初始项目需求` | [🔗](../wiki/request/initial-requirements.md) | pending | 待制定 |

## 计划列表

| 计划名称 | 计划路径 | 状态 |
|---|---|---|
| （暂无） | — | — |

## TODO列表

- [ ] 制定`1. 搭建强化学习任务环境`的执行计划
- [ ] 实现 MountainCar-v0 环境 + Actor-Critic 算法
- [ ] 完成训练与可视化

## 笔记

- **项目文档纠正**：`docs/项目文档.md` 中"自定义2D物理环境"的描述有误，实际项目使用 gymnasium 预设的 MountainCar-v0 环境（2026-06-04 确认）。
- **API 适配要点**：参考代码 `examples/` 使用旧版 gym API，所有代码需使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。

## 全局更新日志（近10条）

- `06-04 18:36`: WIKI 记忆库初始化 —— 创建 8 个摘要、1 个需求文档、index.md、readme.md
