---
abstract_name: 环境搭建模块
source_contents:
  - "config.py"
  - "env.py"
dependencies:
  - "wiki/abstract/gymnasium/mountain-car.md"
created_at: 2026-06-04 18:40:00
updated_at: 2026-06-04 18:40:00
---
# 摘要：环境搭建模块

## 核心结论与关键信息

- `config.py` 提供 `ENV_NAME`、`RENDER_MODE`、`MAX_EPISODE_STEPS` 三个环境常量
- `env.py` 提供 `make_env()` 工厂函数，调用 `gym.make(ENV_NAME, render_mode=RENDER_MODE)` 创建环境实例
- `render_mode="human"` 弹出渲染窗口（后续训练阶段可能调整）
- `env.step(action)` 返回 5 元组 `(obs, reward, terminated, truncated, info)`
- `done = terminated or truncated`
- 自检块验证 5 个随机动作 episode 正常运行，已通过

## 内容概述

> `config.py` 集中管理环境相关配置常量；`env.py` 封装环境创建工作，模块顶层不创建实例（无副作用），通过 `make_env()` 按需创建。自检块位于 `if __name__ == "__main__"`，不会被 import 触发。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/gymnasium/mountain-car.md`（环境规范）、`config.py`
- **下游被依赖**：后续 Actor-Critic 算法实现会通过 `make_env()` 获取环境
- **变更扩散评估**：中（RENDER_MODE 变更会影响所有使用 make_env() 的模块）
