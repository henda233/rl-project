---
abstract_name: 自定义环境创建
source_contents:
  - "docs/gymnasium/创建自定义环境.md"
dependencies:
  - "wiki/abstract/docs/project-overview.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：自定义环境创建

## 核心结论与关键信息

- **继承 `gymnasium.Env`**，必须定义 `observation_space` 和 `action_space`
- **必须实现的方法**：
  - `__init__`：定义空间
  - `reset(seed, options)`：必须调用 `super().reset(seed=seed)`，返回 `(obs, info)`
  - `step(action)`：返回 `(obs, reward, terminated, truncated, info)`
  - `render()`：可选，根据 `render_mode` 处理
- **环境注册**：`gym.register(id="...", entry_point=Class)`，之后可用 `gym.make()` 创建
- **验证**：`gymnasium.utils.env_checker.check_env(env)` 检查合规性
- **常用 Wrapper**：`FlattenObservation`、`RecordEpisodeStatistics`、`RecordVideo`

## 内容概述

> 以 GridWorld 为例讲解自定义环境的完整流程：设计阶段（5 个关键问题）→ 代码实现（`__init__` / `_get_obs` / `reset` / `step`）→ 注册 → 调试。同时列举常见陷阱（奖励设计、状态表示、动作空间、边界处理）。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/project-overview.md`
- **下游被依赖**：当前无直接依赖（后续自定义环境需求会用到）
- **变更扩散评估**：低（当前阶段使用预设 MountainCar，不直接使用）
