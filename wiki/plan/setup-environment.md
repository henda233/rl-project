---
plan_name: 搭建强化学习任务环境
related_request: "wiki/request/initial-requirements.md"
status: completed
created_at: 2026-06-04 18:40:00
---
# 执行计划：搭建强化学习任务环境（需求 1）

## 背景

基于 gymnasium 预设环境 MountainCar-v0，创建项目的基础环境模块和配置模块。为后续需求 2（Actor-Critic）和需求 3（训练与可视化）提供可直接导入使用的 `env` 对象和配置常量。

## 内容

### 交付物 1：`config.py`

存放环境相关配置：

| 配置项 | 值 | 说明 |
|---|---|---|
| `ENV_NAME` | `"MountainCar-v0"` | 环境 ID |
| `RENDER_MODE` | `"human"` | 渲染模式，弹出窗口 |
| `MAX_EPISODE_STEPS` | `200` | MountainCar 截断步数上限 |

### 交付物 2：`env.py`

模块级导出的 gymnasium 环境实例，关键要点：

- 从 `config` 导入 `ENV_NAME` 和 `RENDER_MODE`
- 模块级创建：`env = gym.make(ENV_NAME, render_mode=RENDER_MODE)`
- 包含 `if __name__ == "__main__"` 自检验证块（随机动作运行 5 个 episode，打印每步信息）

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | 创建 `config.py` 环境配置 | 无 | `config.py` | 已完成 |
| `S2` | 创建 `env.py` 环境模块（含自检） | `S1` | `env.py` | 已完成 |
| `S3` | 运行自检验证环境正常 | `S2` | 无（验证步骤） | 已完成 |
| `S4` | 编写摘要并更新 index | `S3` | `wiki/abstract/env-setup.md`、`wiki/index.md` | 已完成 |

## 风险与约束声明

- **约束**：仅使用 gymnasium API，不使用旧版 gym
- **约束**：`render_mode="human"` 在无图形界面的服务器环境无法运行（本地开发环境不受影响）
- **风险**：低 —— MountainCar-v0 是 gymnasium 内置的确定性环境，无兼容性问题

## 测试验证方案

1. `python env.py` 执行自检块，验证：
   - 环境可以成功 `reset()`，返回 `(obs, info)`
   - `env.step(action)` 返回 5 元组 `(obs, reward, terminated, truncated, info)`
   - `done = terminated or truncated` 逻辑正确
   - 5 个 episode 均正常完成
2. 确认弹出 MountainCar 渲染窗口

## 📝 执行记录

- `2026-06-04 18:50`: S1-S4 全部完成，自检通过，摘要和 index 已更新
- `2026-06-04 18:40`: 计划已生成，待用户确认后执行
