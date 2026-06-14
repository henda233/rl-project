---
request_name: 适配 24-Puzzle 官方预训练模型
created_at: 2026-06-14 20:00:00
---
# 需求：适配 24-Puzzle 官方预训练模型

## 需求描述

将当前项目从 15-puzzle（N=4）切换为 24-puzzle（N=5），加载官方预训练 ResnetModel 并运行三项评估。

## 约束条件

- 仅评估官方预训练模型，不做训练
- 参数与 15-puzzle 完全相同（T_MAX=500, MAX_EXPAND_NODES=10000, LAMBDA=1.0 等）
- ResnetModel 维度从 `HUARONGDAO_N` 推导（一个分支一个问题）
- git 分支由用户自行创建和维护
- 不做 35/48-puzzle（计算量太大）

## 关键决策

- ResnetModel 参数化方式：从 `HUARONGDAO_N` 推导 `state_dim=N², one_hot_depth=N²`
- 测试数据：官方 `DeepCubeA/data/puzzle24/test/data_0.pkl`（496 状态，解路径 64-110）
- 数据转换 dtype：`int64`（与 `_GOAL_GRID` dtype 一致，避免 bytes 比较 bug）
