---
plan_name: 实现强化学习算法
related_request: "wiki/request/initial-requirements.md"
status: waiting
created_at: 2026-06-04 19:00:00
---
# 执行计划：实现 Actor-Critic 算法（需求 2）

## 背景

基于已完成的 `config.py` + `env.py` 环境模块，从零实现 Actor-Critic 算法核心。算法采用 TD 残差驱动的 on-policy 双网络架构，适配 gymnasium API 和 MountainCar-v0 环境。参考蓝本为 `examples/actor_critic_example.py`（需改写旧 gym API 调用）。

## 内容

### 交付物 1：修改 `config.py`

新增算法超参数，调整渲染策略：

| 配置项 | 值 | 说明 |
|---|---|---|
| `RENDER_MODE` | `None` | 默认不渲染（训练用）；评估时通过 `make_env(render_mode="human")` 覆盖 |
| `HIDDEN_DIM` | `128` | 策略/价值网络隐藏层维度 |
| `ACTOR_LR` | `1e-3` | Actor 学习率 |
| `CRITIC_LR` | `1e-2` | Critic 学习率（通常比 Actor 大） |
| `GAMMA` | `0.98` | 折扣因子 |
| `NUM_EPISODES` | `500` | 训练轮数 |

`make_env()` 增加可选 `render_mode` 参数，允许调用方覆盖默认值。

### 交付物 2：`agent.py`

单文件包含所有算法核心：

- **PolicyNet**(nn.Module)：`Linear(state_dim, hidden_dim) → ReLU → Linear(hidden_dim, action_dim) → Softmax`，输出动作概率分布
- **ValueNet**(nn.Module)：`Linear(state_dim, hidden_dim) → ReLU → Linear(hidden_dim, 1)`，输出状态价值标量
- **ActorCritic**：
  - `__init__`：双网络 + 各自 Adam 优化器 + gamma + device
  - `take_action(state)`：Categorical 分布采样（天然随机探索，不额外加 ε-greedy）
  - `update(transition_dict)`：批量 TD 残差更新（detach 阻断梯度，MSE 更新 Critic，加权 log_prob 更新 Actor）

### 交付物 3：训练循环与可视化（集成于 `agent.py`）

- `moving_average(a, window_size)`：滑动平均（复制自 `examples/rl_utils.py`）
- `train_on_policy_agent(env, agent, num_episodes)`：gymnasium 适配的 on-policy 训练循环
  - `env.reset()` → `(obs, info)`（适配 gymnasium）
  - `env.step(action)` → 5 元组 + `done = terminated or truncated`
  - tqdm 进度条
- `main` 入口：创建环境（`render_mode=None`）→ 实例化 Agent → 训练 → `plot_return` 绘制迭代数-累积回报曲线

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | 修改 `config.py`：新增算法超参数，`make_env` 支持 `render_mode` 参数 | 无 | `config.py`、`env.py` | 待完成 |
| `S2` | 创建 `agent.py`：PolicyNet + ValueNet + ActorCritic + 工具函数 | `S1` | `agent.py` | 待完成 |
| `S3` | 在 `agent.py` 中实现训练循环 + 绘图 + main 入口 | `S2` | `agent.py` | 待完成 |
| `S4` | 运行训练自检：500 episode，验证回报收敛趋势 | `S3` | 无（验证步骤） | 待完成 |
| `S5` | 编写摘要并更新 index | `S4` | `wiki/abstract/actor-critic-impl.md`、`wiki/index.md` | 待完成 |

## 风险与约束声明

- **约束**：仅使用 numpy + pytorch 基础组件，禁止引入现成 RL 库
- **约束**：所有 gym 调用必须使用 gymnasium API（`reset(seed=...)` 返回 `(obs, info)`，`step` 返回 5 元组）
- **约束**：训练默认 `render_mode=None`，不弹出渲染窗口
- **风险**：MountainCar 奖励稀疏（每步 -1），500 episode 可能不足以学会摆动策略。可接受的首个里程碑是回报从 -200 向 -150~-120 收敛趋势，后续需求 3 再调参优化

## 测试验证方案

1. `uv run python agent.py` 执行完整训练流程，验证：
   - 无 import 错误，无 gymnasium API 兼容性问题
   - 500 episode 训练完成，tqdm 进度条正常显示
   - 终端输出每 10 episode 的平均回报
   - 训练结束后弹出 matplotlib 回报曲线图（窗口需手动关闭）
2. 收敛性验证：回报曲线整体呈上升趋势（不要求完全学到最优策略）

## 📝 执行记录

- `2026-06-04 19:00`: 计划已生成，待用户确认
