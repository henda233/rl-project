---
request_name: Energy-Based 奖励塑形
source_contents:
  - "agent.py"
  - "config.py"
status: pending
created_at: 2026-06-04 22:00:00
---
# 需求：用 Energy-Based 势函数替换 Velocity-Based 势函数

## 需求描述

当前奖励塑形使用 Φ(s) = |velocity|，在 MountainCar 上无法引导收敛（根因诊断见讨论）。

需要将势函数替换为 total mechanical energy：

```
Φ(s) = sin(3·position) + velocity²/(2·g)    (g = 0.0025)
```

该势函数在物理上追踪小车的机械能守恒，天然指向高势能的目标，且不惩罚必要的"左墙蓄力"阶段。

### 约束

- 保持 potential-based shaping 框架（Ng et al. 定理）
- 保持 Actor-Critic 架构不变
- 保持 ε-greedy 探索机制（可作为安全网保留）
- 目标：return 稳定在 -120~-100，solve rate > 95%

### 方案

势函数替换 + 调整 REWARD_SHAPING_SCALE 适配能量值域 + 可能的 ε-greedy 参数微调。

## 关联摘要

- `wiki/abstract/actor-critic-impl.md`
- `wiki/abstract/docs/actor-critic-algorithm.md`
- `wiki/abstract/gymnasium/mountain-car.md`
- `wiki/plan/mountaincar-convergence.md`
