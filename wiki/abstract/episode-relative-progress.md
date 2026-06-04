---
abstract_name: 回合相对进度奖励塑形实现
source_contents:
  - "config.py"
  - "agent.py"
  - "docs/关于阶梯式奖励设置.md"
dependencies:
  - "wiki/abstract/actor-critic-impl.md"
  - "wiki/abstract/energy-based-shaping.md"
created_at: 2026-06-04 23:00:00
updated_at: 2026-06-04 23:15:00
---
# 摘要：回合相对进度奖励塑形实现

## 核心结论与关键信息

- **方案来源**：`docs/关于阶梯式奖励设置.md` 最终方案（DQN 100% 成功率），改为适用于 on-policy Actor-Critic。
- **核心机制**：每个 episode 内追踪 `min_x`（最左）和 `max_x`（最右），打破记录时给予比例奖励：`(x-max_x)*10`（右向）或 `(min_x-x)*5`（左向）。辅以速度奖励 `abs(speed)*5`（speed>0.01）和停滞惩罚 `-1`。
- **近终点紧迫**：episode 被 truncated（超时）且 `x > 0.3` 时，对最后一步施加 `-20` 惩罚，防止"接近目标但步数耗尽"。
- **彻底移除 energy-based 势函数**：`REWARD_SHAPING_SCALE`、`G`、`PE_COEFFICIENT` 全部删除，`from math import sin` 移除。
- **最终超参数**：`PROGRESS_RIGHT_SCALE=10`、`PROGRESS_LEFT_SCALE=5`、`SPEED_REWARD_SCALE=5`、`SPEED_THRESHOLD=0.01`、`STAGNATION_PENALTY=1`、`NEAR_GOAL_THRESHOLD=0.3`、`NEAR_GOAL_PENALTY=20`。
- **训练结果**：5000 episode 仍不收敛。推测根因不是奖励塑形，而是 AC 算法本身局限性（网络结构、on-policy 样本效率）。

## 内容概述

> 将 energy-based potential shaping 替换为回合相对进度奖励。每个 episode 独立追踪位置记录，显式奖励双向探索（左借力、右冲坡），速度信号辅助。代码可运行但不收敛，问题转向上层算法结构。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/actor-critic-impl.md`（AC 实现），`wiki/abstract/energy-based-shaping.md`（被替换的方案），`docs/关于阶梯式奖励设置.md`（方案来源）
- **下游被依赖**：暂无
- **变更扩散评估**：低（仅 config.py 参数替换和 agent.py 塑形逻辑替换）
