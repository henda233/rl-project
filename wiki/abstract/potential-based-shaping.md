---
abstract_name: Potential-Based 奖励塑形
source_contents:
  - "config.py"
  - "ppo_agent.py"
  - "agent.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/actor-critic-impl.md"
  - "wiki/abstract/episode-relative-progress.md"
created_at: 2026-06-05 00:00:00
updated_at: 2026-06-05 00:00:00
---
# 摘要：Potential-Based 奖励塑形

## 核心结论与关键信息

- **势函数**：`Φ(s) = k · position`，仅依赖位置，k 默认为 1（`config.py` 中 `POTENTIAL_K` 可调）。
- **塑形公式**：`shaped_reward = original_reward + (γ·Φ(s') - Φ(s))`，标准 Ng et al. potential-based shaping，不改变最优策略。
- **直观解释**：向右移动（pos' > pos）时 Φ 增大，F > 0，鼓励向目标方向探索；向左移动时 F < 0。
- **position 范围** [-1.2, 0.6]，目标在 0.5。k=1 时 F 量级与原始每步 -1 奖励相当。
- **彻底移除**：回合相对进度塑形（记录打破、速度信号、停滞惩罚、近终点惩罚）全部删除，`config.py` 中 7 个进度参数清除。
- **ppo_agent.py 使用 PPO_GAMMA（0.98），agent.py 使用 GAMMA（0.5）**，各自使用对应的折扣因子。

## 内容概述

> 将 `ppo_agent.py` 和 `agent.py` 中复杂的回合相对进度奖励塑形替换为极简的 potential-based shaping：`Φ = k·pos`。`config.py` 移除所有进度塑形参数，新增 `POTENTIAL_K=1`。两个算法的训练循环各仅一行塑形代码。

## 与之前塑形方案的对比

| 方案 | 势函数/机制 | 问题 |
|---|---|---|
| Velocity-based | Φ = \|v\| | 速度盲，撞墙蓄力阶段惩罚关键行为 |
| Energy-Based | Φ = sin(3·pos) + v²/(2g) | sin 非单调，设计过于复杂 |
| 回合相对进度 | 记录打破 + 速度 + 惩罚 | 非势能，不保证策略不变性，仍不收敛 |
| **Potential-Based** | **Φ = k·pos** | **极简、单调、Ng et al. 定理保证** |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO 实现）、`wiki/abstract/actor-critic-impl.md`（AC 实现）、`wiki/abstract/episode-relative-progress.md`（被替换的方案）
- **下游被依赖**：无
- **变更扩散评估**：低（config.py 参数替换，两个 agent 各一行公式替换）
