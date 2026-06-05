---
abstract_name: PPO 算法实现
source_contents:
  - "ppo_agent.py"
  - "config.py"
dependencies:
  - "wiki/abstract/episode-relative-progress.md"
  - "wiki/request/implement-ppo.md"
  - "wiki/plan/implement-ppo.md"
created_at: 2026-06-04 23:50:00
updated_at: 2026-06-05 18:45:00
---
# 摘要：PPO 算法实现

## 核心结论与关键信息

- PPO 通过 Clipped Surrogate Objective + GAE + 多轮更新替代 AC 的 TD(0) 单步更新
- 网络结构与 AC 一致：单隐藏层 256（PolicyNet + ValueNet），公平对比
- GAE 正确实现终端状态截断：`A_t = δ_t + γλ(1-done_t) · A_{t+1}`
- 添加 Advantage Normalization + Entropy Bonus（coef=0.01），替代 ε-greedy 探索
- AC/PPO 训练参数完全分离（config.py 中独立前缀），奖励塑形参数共享
- ppo_agent.py 自包含，与 agent.py 零耦合
- `PPO_USE_GPU=False` 控制训练设备（True=cuda, False=cpu），默认 CPU，移除自动检测
- **PPO 再训练**：通过 config.py 中 `PPO_ACTOR_MODEL_PATH` / `PPO_CRITIC_MODEL_PATH` 指定已有权重文件进行再训练，仅恢复 state_dict（optimizer 重新初始化），输出到时间戳目录 `results/retrain_YYYYMMDD_HHMMSS/`。再训练 episode 数由 `PPO_RETRAIN_NUM_EPISODES` 独立控制。仅 `ppo_agent.py` 支持，不涉及 `ppo_parallel.py`。

## 内容概述

> `ppo_agent.py` 包含完整的 PPO 实现：PolicyNet/ValueNet、compute_gae（含 dones 截断）、PPO 类（clipped objective + advantage norm + entropy bonus）、on-policy 训练循环（复用 episode-relative progress 奖励塑形）、moving_average/evaluate/plot_return 工具函数、main 入口。
>
> `config.py` 新增 PPO_ 前缀参数组（PPO_HIDDEN_DIM=256, PPO_GAMMA=0.98, PPO_LMBDA=0.95, PPO_EPOCHS=10, PPO_EPS=0.2, PPO_NUM_EPISODES=5000, PPO_EVAL_INTERVAL=200, PPO_ENTROPY_COEF=0.01, PPO_USE_GPU=False, PPO_RETRAIN_NUM_EPISODES=5000），与 AC 参数分节管理。再训练配置项：PPO_ACTOR_MODEL_PATH、PPO_CRITIC_MODEL_PATH（空字符串=从头训练，非空=加载权重再训练），两个路径必须同时为空或同时非空。

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| GAE 实现 | 标准公式 + dones 截断 | 正确性优先于简化 |
| 网络深度 | 单隐藏层 256 | 与 AC 一致，公平对比 |
| Advantage Norm | 启用 | 稳定训练 |
| Entropy Bonus | 启用（0.01） | 替代被移除的 ε-greedy |
| ε-greedy | 移除 | ratio = π_new/π_old 需要干净的概率比 |
| 参数分离 | AC/PPO 独立前缀 | 互不干扰 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/episode-relative-progress.md`（奖励塑形方案）、`wiki/request/implement-ppo.md`（需求）、`wiki/plan/implement-ppo.md`（执行计划）
- **下游被依赖**：无（独立模块）
- **变更扩散评估**：低（ppo_agent.py 自包含，config.py 仅追加 PPO 参数段，AC 代码不受影响）
