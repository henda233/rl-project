---
abstract_name: RND 探索算法实现
source_contents:
  - "ppo_rnd_agent.py"
  - "config.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/potential-based-shaping.md"
  - "wiki/request/rnd-implementation.md"
  - "wiki/plan/implement-rnd.md"
created_at: 2026-06-05 20:00:00
updated_at: 2026-06-05 21:00:00
---
# 摘要：RND 探索算法实现

## 核心结论与关键信息

- RND (Random Network Distillation) 通过状态新颖性产生内在奖励，替代势能塑形解决 MountainCar 稀疏奖励问题
- 奖励公式：`r_total = r_ext + β · r_int_norm`，其中 `r_int_norm = r_int / (running_std + 1e-8)`（仅除标准差，不减均值）
- 滑动缓冲区保留最近 `RND_BUFFER_SIZE` 个 episode 的状态，用于训练预测网络
- ppo_agent.py 和 ppo_rnd_agent.py 完全独立，零耦合；ppo_agent.py 从头训练也改为时间戳目录
- 最优模型按原始return（`episode_original_return`）保存，通关检测沿用 `max_position >= 0.5`

## 内容概述

> `ppo_rnd_agent.py` 自包含完整 RND+PPO 实现：RNDTargetNet（固定随机投影）、RNDPredictorNet（可训练）、RNDModule（归一化+训练）、滑动状态缓冲区、PPO 训练循环（PolicyNet/ValueNet/GAE/clipped objective）。
>
> `config.py` RND 配置段新增：RND_HIDDEN_DIM=128、RND_OUTPUT_DIM=256、RND_LR=1e-3、RND_BETA=100、RND_EPOCHS=5、RND_NUM_EPISODES=10000、RND_BUFFER_SIZE=5。

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 归一化 | 仅除标准差 | 保留原始 MSE 量级信息 |
| 缓冲区 | 滑动窗口（5 episode） | 控制训练成本，状态覆盖近期探索 |
| RND 更新时机 | PPO 更新后 | 先利用当前奖励信号优化策略，再更新新颖性估计 |
| 数据范围 | 缓冲区全量状态 | 避免单 episode 过拟合 |
| 运行标准差 | 增量平均更新 | `α=1/(update_count+1)` 保证初期快速适应 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO 核心实现）、`wiki/abstract/potential-based-shaping.md`（被替代方案）、`docs/RND算法.md`（算法理论）
- **下游被依赖**：无
- **变更扩散评估**：低（ppo_rnd_agent.py 自包含，config.py 仅追加 RND 参数段，ppo_agent.py 仅修改输出目录）
