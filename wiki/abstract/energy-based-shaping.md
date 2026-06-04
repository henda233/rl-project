---
abstract_name: Energy-Based 奖励塑形实现
source_contents:
  - "config.py"
  - "agent.py"
dependencies:
  - "wiki/abstract/actor-critic-impl.md"
  - "wiki/abstract/gymnasium/mountain-car.md"
  - "wiki/plan/mountaincar-convergence.md"
  - "wiki/plan/energy-based-shaping.md"
created_at: 2026-06-04 22:40:00
updated_at: 2026-06-04 22:40:00
---
# 摘要：Energy-Based 奖励塑形实现

## 核心结论与关键信息

- **势函数**：`Φ(s) = PE_COEFFICIENT × sin(3·pos) + v² / (2×G)`，基于 MountainCar 物理总机械能（势能+动能），天然从谷底指向目标。
- **关键优势**：起始 Φ₀ ≈ -1.0，目标 Φ_T ≈ 1.0，telescoping sum 净 shaping 为正值（velocity-based 为负值），不惩罚左墙蓄力阶段的速度突降。
- **最终超参数**：`REWARD_SHAPING_SCALE=10`（保守起点）、`G=0.0025`（MountainCar 重力常量）、`PE_COEFFICIENT=1.0`（可调 PE/KE 权重）、`EPSILON=0.3`、`EPSILON_DECAY=0.999`、`EPSILON_MIN=0.01`。
- **不改动部分**：ActorCritic 类、训练循环结构、ε-greedy 机制、评估函数、env.py。
- **Ng et al. 定理保证**：potential-based shaping 不改变最优策略。

## 内容概述

> 将 `agent.py` 中 `train_on_policy_agent` 的 shaped_reward 公式从 `γ·|v'| - |v|`（velocity-based）替换为 `γ·Φ(s') - Φ(s)`（energy-based）。`config.py` 新增 G、PE_COEFFICIENT 两个超参数，ε-greedy 调整为更低初始探索率。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/actor-critic-impl.md`（AC 实现），`wiki/plan/mountaincar-convergence.md`（velocity-based 失败根因诊断）
- **下游被依赖**：暂无
- **变更扩散评估**：低（仅 config.py 超参数和 agent.py 单行公式替换）
