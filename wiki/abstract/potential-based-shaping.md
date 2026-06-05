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
updated_at: 2026-06-05 19:00:00
---
# 摘要：Potential-Based 奖励塑形

## 核心结论与关键信息

- **势函数**：`Φ(s) = k · (position + 10 · |v|)`，k 默认为 2（`config.py` 中 `POTENTIAL_K=2` 可调）。速度项代表动能，|v| 越大势能越高，引导智能体加速脱离山谷。
- **塑形公式**：`shaped_reward = original_reward + (γ·Φ(s') - Φ(s))`，标准 Ng et al. potential-based shaping，不改变最优策略。
- **直观解释**：向右移动（pos' > pos）时 Φ 增大，F > 0；加速（|v'| 增大）时同样 F > 0。速度与位置量级对齐（10×），动能主导势能增量。
- **量级分析**：position ∈ [-1.2, 0.6]，|v| ∈ [0, 0.07]，10·|v| ∈ [0, 0.7]，速度项贡献约为位置项的 40%。k=2 时 F 量级显著超过原始每步 -1 奖励。
- **彻底移除**：回合相对进度塑形（记录打破、速度信号、停滞惩罚、近终点惩罚）全部删除，`config.py` 中 7 个进度参数清除。
- **ppo_agent.py 使用 PPO_GAMMA（0.98），agent.py 使用 GAMMA（0.5）**，各自使用对应的折扣因子。

## 内容概述

> 将 `ppo_agent.py` 和 `agent.py` 中复杂的回合相对进度奖励塑形替换为极简的 potential-based shaping。当前版本 Φ = k·(pos + 10·|v|)（仅 PPO），v1 Φ = k·pos（AC 沿用）。config.py 中 POTENTIAL_K=2。

## 与之前塑形方案的对比

| 方案 | 势函数/机制 | 问题 |
|---|---|---|
| Velocity-based | Φ = \|v\| | 速度盲，撞墙蓄力阶段惩罚关键行为 |
| Energy-Based | Φ = sin(3·pos) + v²/(2g) | sin 非单调，设计过于复杂 |
| 回合相对进度 | 记录打破 + 速度 + 惩罚 | 非势能，不保证策略不变性，仍不收敛 |
| **Potential-Based (v2)** | **Φ = k·(pos + 10·|v|)** | **位置+动能，加速收敛，Ng et al. 定理保证** |
| Potential-Based (v1) | Φ = k·pos | 极简，收敛速度慢且有随机性 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO 实现）、`wiki/abstract/actor-critic-impl.md`（AC 实现）、`wiki/abstract/episode-relative-progress.md`（被替换的方案）
- **下游被依赖**：无
- **变更扩散评估**：低（config.py 参数替换，两个 agent 各一行公式替换）
