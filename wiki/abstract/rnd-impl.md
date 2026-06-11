---
abstract_name: RND 探索算法实现
source_contents:
  - "ppo_rnd_agent.py"
  - "config.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/request/rnd-implementation.md"
  - "wiki/plan/implement-rnd.md"
created_at: 2026-06-05 20:00:00
updated_at: 2026-06-11 17:40:00
---
# 摘要：RND 探索算法实现

## 核心结论与关键信息

- RND (Random Network Distillation) 通过状态新颖性产生内在奖励，增强华容道环境探索
- 奖励公式：`r_total = r_ext + β · r_int_norm`，其中 `r_int_norm = r_int / (running_std + 1e-8)`（仅除标准差，不减均值）。β 线性衰减：`β_episode = max(RND_BETA_END, β_last - RND_BETA_DECAY)`，从 RND_BETA 衰减到 RND_BETA_END
- 滑动缓冲区保留最近 `RND_BUFFER_SIZE` 个 episode 的状态，用于训练预测网络
- ppo_agent.py 和 ppo_rnd_agent.py 完全独立，零耦合
- 最优模型按原始return（`episode_original_return`）保存，通关检测：`max_placed >= board_size**2 - 1`
- RND_HIDDEN_DIM=256（华容道适配）

## 内容概述

> `ppo_rnd_agent.py` 自包含完整 RND+PPO 实现：RNDTargetNet（固定随机投影）、RNDPredictorNet（可训练）、RNDModule（归一化+训练）、滑动状态缓冲区、PPO 训练循环（PolicyNet/ValueNet/GAE/clipped objective）。
>
> `config.py` RND 配置段：RND_HIDDEN_DIM=256、RND_OUTPUT_DIM=256、RND_LR=1e-3、RND_BETA=100、RND_BETA_END=1.0、RND_BETA_DECAY=0.05、RND_EPOCHS=5、RND_NUM_EPISODES=10000、RND_BUFFER_SIZE=5。

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 归一化 | 仅除标准差 | 保留原始 MSE 量级信息 |
| 缓冲区 | 滑动窗口（5 episode） | 控制训练成本，状态覆盖近期探索 |
| RND 更新时机 | PPO 更新后 | 先利用当前奖励信号优化策略，再更新新颖性估计 |
| 数据范围 | 缓冲区全量状态 | 避免单 episode 过拟合 |
| 运行标准差 | 增量平均更新 | `α=1/(update_count+1)` 保证初期快速适应 |
| β 衰减 | 线性衰减到 RND_BETA_END | 训练后期减少探索，聚焦利用 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO 核心实现）、`docs/RND算法.md`（算法理论）
- **下游被依赖**：无
- **变更扩散评估**：低（ppo_rnd_agent.py 自包含，config.py 仅追加 RND 参数段）
