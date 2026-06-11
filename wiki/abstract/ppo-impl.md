---
abstract_name: PPO 算法实现
source_contents:
  - "ppo_agent.py"
  - "config.py"
dependencies:
  - "wiki/request/implement-ppo.md"
  - "wiki/plan/implement-ppo.md"
  - "wiki/abstract/huarongdao-env.md"
created_at: 2026-06-04 23:50:00
updated_at: 2026-06-11 23:59:00
---
# 摘要：PPO 算法实现

## 核心结论与关键信息

- PPO 通过 Clipped Surrogate Objective + GAE + 多轮更新实现稳定 on-policy 训练
- 网络结构：PolicyNet（输出动作 logits）+ ValueNet（输出状态价值），隐藏层 256
- GAE 正确实现终端状态截断：`A_t = δ_t + γλ(1-done_t) · A_{t+1}`
- 添加 Advantage Normalization + Entropy Bonus（coef=0.01）替代 ε-greedy 探索
- `PPO_USE_GPU=False` 控制训练设备（True=cuda, False=cpu），默认 CPU
- 最优模型按**原始 return**保存（不含塑形/好奇心奖励）
- 通关检测：`max_placed >= board_size**2 - 1`（所有滑块归位）
- 训练输出到时间戳目录 `results/train_YYYYMMDD_HHMMSS/`

## 内容概述

> `ppo_agent.py` 包含完整 PPO 实现：PolicyNet/ValueNet、compute_gae（含 dones 截断）、PPO 类（clipped objective + advantage norm + entropy bonus）、on-policy 训练循环、moving_average/evaluate/plot_return 工具函数、main 入口。
>
> `config.py` 包含 PPO_ 前缀参数组（PPO_HIDDEN_DIM=256, PPO_GAMMA=0.98, PPO_LMBDA=0.95, PPO_EPOCHS=10, PPO_EPS=0.2, PPO_NUM_EPISODES=10000, PPO_EVAL_INTERVAL=1000, PPO_ENTROPY_COEF=0.05, PPO_USE_GPU=False）和 HUARONGDAO_ 前缀环境参数。

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| GAE 实现 | 标准公式 + dones 截断 | 正确性优先于简化 |
| 网络深度 | 单隐藏层 256 | 平衡容量与训练效率 |
| Advantage Norm | 启用 | 稳定训练 |
| Entropy Bonus | 启用（0.01） | 替代被移除的 ε-greedy |
| 模型保存 | 原始 return | 直接衡量任务效率，不被内在奖励污染 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/huarongdao-env.md`（环境规范）、`wiki/request/implement-ppo.md`（需求）、`wiki/plan/implement-ppo.md`（执行计划）
- **下游被依赖**：`wiki/abstract/rnd-impl.md`（RND 基于 PPO 扩展）
- **变更扩散评估**：低（ppo_agent.py 自包含，config.py PPO 参数段独立）
