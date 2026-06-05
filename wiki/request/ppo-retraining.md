---
request_name: PPO 再训练功能
status: pending
created_at: 2026-06-05
---
# 需求：PPO 再训练功能

## 需求描述

读取指定模型参数文件，进行再训练。

- 仅限于 PPO 算法（`ppo_agent.py`）
- 在 `config.py` 中新增模型文件路径参数
- 为空 → 从头训练
- 非空 → 读取模型参数进行再训练

## 设计决策（已与用户对齐）

| 决策项 | 选择 |
|---|---|
| 模型路径格式 | actor 和 critic 分开两个 `.pth` 文件路径 |
| 影响范围 | 仅 `ppo_agent.py`，不涉及 `ppo_parallel.py` |
| 恢复内容 | 仅恢复网络权重（state_dict），optimizer 重新初始化，episode 从 1 计数 |
| 输出目录 | 再训练时保存到时间戳目录 `results/retrain_YYYYMMDD_HHMMSS/` |

## 关联计划

- [PPO 再训练执行计划](../plan/ppo-retraining.md)

## 关联摘要

- [PPO 算法实现](../abstract/ppo-impl.md)
