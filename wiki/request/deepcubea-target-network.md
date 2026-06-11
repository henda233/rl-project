---
request_name: DeepCubeA 批量更新 + 固定目标 AVI
created_at: 2026-06-11 22:30
---
# 需求：DeepCubeA 批量更新 + 固定目标 AVI 训练

## 问题描述

当前 DeepCubeA 训练每 epoch 重算 Bellman 备份目标，目标随网络更新同步变化（"移动靶"），已实证导致 loss 在 epoch 2331 爆炸至 5.6×10^21。需改为文献原版的批量更新 + 固定目标设计。

## 需求内容

按 DeepCubeA 文献实现**批量更新 + 固定目标 AVI 训练**：

- 外层循环：用当前网络计算全量状态的 Bellman 备份 J'(s)（固定目标）
- 内层循环：以固定 J'(s) 为目标训练网络多 epoch，直至收敛（patience 早停）
- 收敛后重新计算 J'(s) 进入下一轮外层迭代

## 关键决策（已对齐）

- **方案**：文献原版批量更新 + 固定目标（废弃原 DQN 式 target network 方案）
- **选型理由**：离线全量数据 + 确定性 Bellman 备份，target network 属过度设计
- **超参**：`OUTER_ITER=20`、`INNER_EPOCHS=100`、`INNER_PATIENCE=10`
- **模型选择**：取最后一轮外层迭代的模型（最多 Bellman 备份步数）
- **早停机制**：内层 loss 连续不改善即提前退出
- **不引入 LR scheduler**，保持简单
- 基于当前 v2 训练参数叠加（200k states, lr=1e-4, hidden=512, batch=1024）

## 关联计划

- `wiki/plan/deepcubea-target-network.md`
