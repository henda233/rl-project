---
request_name: DeepCubeA Checkpoint 与 Loss 图保存优化
created_at: 2026-06-13 21:30
status: pending
---

# 需求：DeepCubeA Checkpoint 与 Loss 图保存优化

## 问题描述

1. **Checkpoint 文件膨胀**：外层+内层训练时，checkpoint 按外层迭代次数命名保存，外层迭代数量大时产生大量模型参数文件。改为只维护最新外层迭代的一个 checkpoint 文件。
2. **Loss 图缺失**：当前仅在训练全部结束后保存 loss 曲线图。需改为每次 checkpoint 时同步保存 loss 图，维护逻辑与 checkpoint 相同。

## 具体需求

| 项目 | 变更内容 |
|---|---|
| Checkpoint 文件 | `deepcubea_heuristic_epoch_{outer_iter}.pt` → `deepcubea_heuristic_latest.pt`（覆盖式保存） |
| `final.pt` | **保持不变**，训练结束后仍保存 |
| Loss 图 | 每次外层迭代结束时保存累积 loss 曲线为 `loss_curve_latest.png` |
| Loss 数据 | 同步保存累积 loss 原始数据为 `loss_curve_latest.csv` |
