---
abstract_name: 项目概述
source_contents:
  - "env_digital_huarongdao.py"
  - "ppo_agent.py"
  - "ppo_rnd_agent.py"
  - "config.py"
dependencies: []
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-11 17:40:00
---
# 摘要：项目概述

## 核心结论与关键信息

- **项目目标**：实现 n×n 数字华容道（滑块拼图）强化学习环境，使用 PPO + RND 算法训练智能体求解
- **环境**：`DigitalHuarongdaoEnv`（gymnasium.Env 子类），Discrete(4) 动作空间（上/下/左/右），正向打乱保证可解
- **算法**：PPO（Clipped Surrogate Objective + GAE + Advantage Norm + Entropy Bonus）+ RND 内在奖励探索
- **技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium
- **环境渲染**：支持 ansi（控制台）、rgb_array（点阵数字）、human（pygame 窗口）三种模式
- **非法动作处理**：环境区分合法/非法奖励（-1 / -2），通过差异化惩罚引导智能体避开非法动作

## 内容概述

> 项目聚焦数字华容道问题：从 n×n 滑块拼图的 shuffled 状态出发，智能体通过上/下/左/右移动空格，将棋盘恢复为目标顺序。训练采用 PPO on-policy 算法，可选 RND 内在奖励增强探索。

## 依赖与影响链

- **上游依赖**：无
- **下游被依赖**：`wiki/abstract/ppo-impl.md`、`wiki/abstract/rnd-impl.md`、`wiki/abstract/huarongdao-env.md`
- **变更扩散评估**：中（核心环境/算法变更影响所有下游模块）
