---
abstract_name: 项目概述
source_contents:
  - "docs/项目文档.md"
dependencies: []
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：项目概述

## 核心结论与关键信息

- **项目目标**：使用 gymnasium 的预设环境 MountainCar-v0，实现 Actor-Critic 算法，训练获得高分的智能体（最优策略）
- **环境**：MountainCar-v0（NOT 自定义环境，项目文档中"自定义2D物理环境"的表述已纠正）
- **算法**：Actor-Critic（策略网络 + 价值网络，TD 残差驱动更新）
- **技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium
- **禁止使用**：现成的强化学习模型/算法库
- **参考资料**：仅限 `docs/` 目录下文件，禁止访问网络

## 内容概述

> 项目文档定义了 4 项需求：搭建 MountainCar-v0 环境、实现 Actor-Critic 算法、训练智能体并可视化、预留后续需求扩展空间。技术栈限定在 5 个依赖库内，强调从零实现算法。

## 依赖与影响链

- **上游依赖**：无
- **下游被依赖**：`wiki/abstract/docs/actor-critic-algorithm.md`、`wiki/abstract/gymnasium/mountain-car.md`
- **变更扩散评估**：高（需求变更会影响所有下游摘要和计划）
