---
request_name: RND 探索算法实现
source_contents:
  - "docs/RND算法.md"
status: completed
created_at: 2026-06-05 19:30:00
---
# 需求：PPO + RND 探索算法

## 需求描述

参考 `docs/RND算法.md`，在现有 PPO 基础上实现 Random Network Distillation (RND) 算法，解决 MountainCar 稀疏奖励问题。

### 核心要求

1. **替代势能塑形**：移除现有 potential-based shaping (Φ = k·(pos + 10·|v|))，用 RND 内在奖励替代
2. **独立文件**：新建 `ppo_rnd_agent.py`，与 `ppo_agent.py` 解耦，训练设计结构相同
3. **参数配置**：RND 相关参数添加到 `config.py` 中，包括：隐藏层维度、输出维度、学习率、β 系数、训练轮数
4. **评估与保存**：沿用 PPO 的评估间隔、最优模型保存、通关检测、可视化

### 设计决策

- 奖励公式：`r_total = r_ext + β · r_int_norm`
- 归一化方式：仅除以运行标准差（不减均值）
- RND 预测网络每 episode 结束后更新 RND_EPOCHS 轮
- 网络架构与 PPO 保持一致的隐藏层维度

## 关联摘要

- `wiki/abstract/ppo-impl.md`（PPO 实现）
- `wiki/abstract/potential-based-shaping.md`（待替换方案）
- `wiki/abstract/docs/`（算法理论）
