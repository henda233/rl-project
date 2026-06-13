---
abstract_name: PPO + DeepCubeA J(s) 势函数塑形分析
source_contents:
  - "wiki/request/ppo-deepcubea-shaping.md"
dependencies:
  - "wiki/abstract/docs/deepcubea-research.md"
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/rnd-impl.md"
  - "wiki/abstract/deepcubea-target-network.md"
  - "wiki/abstract/deepcubea-online-validation.md"
created_at: 2026-06-13 21:00:00
updated_at: 2026-06-13 21:00:00
---
# 摘要：PPO + DeepCubeA J(s) 势函数塑形分析

## 核心结论与关键信息

### 方案 A vs B

- **方案 A（势函数塑形）**：r_int = J(s) - J(s')，设 Φ = -J，γ=1。直接评估动作后果——最优动作为 +1，其他 <1。理论上保持最优策略不变（Ng et al., 1999）。总奖励 r_total = -1 + β·(J(s) - J(s'))。
- **方案 B（状态惩罚）**：r_int = -J(s)。信号粒度粗糙——只告诉 agent"你离目标多远"，不告诉"哪个动作更好"。方案 A 优于 B。

### 联合训练可行性

**三重风险**：

1. **冷启动陷阱**：J(s) 初始随机 → shaped reward 纯噪声 → PPO 学不到策略 → 数据差 → J(s) 无法改进 → 死锁
2. **双重正反馈雪崩（核心风险）**：当前项目已实证 Bellman 备份正反馈问题（需 θ_c 阈值更新硬阻断）。联合训练有两条反馈链路（J→reward→π→data→J），偏差放大风险更大。
3. **非平稳数据分布**：PPO on-policy 数据分布持续漂移，J(s) 训练无 i.i.d. 保证，不同于 DeepCubeA 离线随机游走的稳定分布。

### 推荐路径：阶段性联合训练

| 阶段 | 内容 | 特点 |
|---|---|---|
| Phase 1 | 随机游走预训练 J(s)，冻结 | 当前 DeepCubeA 流程 |
| Phase 2 | PPO + 冻结 J(s) shaping | 验证前提：J(s) 作为内在奖励是否有效 |
| Phase 3 | 解冻 J(s)，用 PPO rollout 数据 + θ_c 阈值更新微调，低频率更新（如每 50 episode 一次） | 松散联合，避免紧耦合 |
| Phase 4 | 更新后 J(s) 重新 shaping 训练 PPO | 迭代提升 |

**关键原则**：不要直接从联合训练开始，先验证冻结 J(s) shaping 的前提。

### PPO + J(s) Shaping vs DeepCubeA A*

| 维度 | PPO + J(s) Shaping | DeepCubeA A* |
|---|---|---|
| 推理计算 | 单次前向传播 | 每展开节点一次前向传播 |
| 单步耗时 | ~0.1ms | ~10-100ms |
| 错误恢复 | 无——策略误差累积 | 搜索可纠正单步启发式错误 |
| 短距离问题 | 可能更快 | 展开少，也快 |
| 长距离问题 | 误差累积，大概率失败 | 搜索深度增加但能纠错 |
| 对 J 精度依赖 | 间接——通过 reward 影响策略梯度 | 直接——A* 贪心选择依赖 h 排序 |

**核心判断**：PPO 无法替代搜索的"前瞻"纠错能力。短距 PPO 推理快，长距 A* 搜索纠错是关键优势。

## 内容概述

> 本摘要记录了对 DeepCubeA J(s) 作为 PPO 内在奖励的可行性分析与讨论。涵盖方案 A/B 的理论对比、联合训练的三重风险分析、阶段性联合训练的推荐路径、以及 PPO+Shaping vs A* 搜索的优劣对比。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/deepcubea-research.md`（DeepCubeA 算法）、`wiki/abstract/ppo-impl.md`（PPO 实现）、`wiki/abstract/deepcubea-target-network.md`（θ_c 阈值更新机制）
- **下游被依赖**：无（纯分析文档，待确认可行性后制定实现计划）
- **变更扩散评估**：低（纯分析摘要，不涉及代码变更）
