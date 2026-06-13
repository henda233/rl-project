---
plan_name: DeepCubeA Checkpoint 与 Loss 图保存优化
related_request: "wiki/request/deepcubea-checkpoint-optimization.md"
status: completed
created_at: 2026-06-13 21:30
---
# 执行计划：DeepCubeA Checkpoint 与 Loss 图保存优化

## 背景

当前训练代码在外层迭代循环中按 `outer_iter` 编号保存 checkpoint，外层迭代数多时产生大量 `.pt` 文件。同时 loss 曲线图仅在训练全部结束后保存一次，无法在训练中途查看进度。

## 内容

修改 `deepcubea_train.py` 中的 checkpoint 保存逻辑和 loss 图生成逻辑，改为覆盖式保存（只维护最新一份），并在每次外层迭代结束时同步保存累积 loss 曲线图和 CSV 数据。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | checkpoint 文件名改为 `deepcubea_heuristic_latest.pt`（覆盖式保存），删除旧的按 `outer_iter` 编号命名逻辑 | 无 | `deepcubea_train.py:247-250` | 已完成 |
| `S2` | 每次外层迭代结束后，保存累积 loss 的 CSV 文件 `loss_curve_latest.csv` | `S1` | `deepcubea_train.py`（S1 同位置之后） | 已完成 |
| `S3` | 每次外层迭代结束后，绘制并保存累积 loss 曲线图 `loss_curve_latest.png` | `S2` | `deepcubea_train.py`（S2 同位置之后） | 已完成 |
| `S4` | 验证：运行冒烟测试确认 checkpoint 覆盖式保存 + loss CSV + loss 图均正常生成 | `S3` | 冒烟测试输出 | 已完成 |
| `S5` | 更新 wiki 摘要 + index | `S4` | `wiki/abstract/` + `wiki/index.md` | 已完成 |

## 风险与约束声明

- **低风险**：仅改动 `deepcubea_train.py` 一个文件，不影响网络结构、训练逻辑、搜索模块
- `final.pt` 保持不变，确保训练结束后有明确命名的最终模型
- loss 图的绘制逻辑与现有代码完全相同（`outer_boundaries` + `theta_update_points` 标记），只是将绘制时机从"训练结束后"提前到"每次外层迭代结束后"

## 测试验证方案

1. 运行 `deepcubea_online_smoke_test.py`（或等效的短训练），确认：
   - `models/deepcubea_heuristic_latest.pt` 存在且为最后一次外层迭代的模型
   - `models/deepcubea_heuristic_final.pt` 存在（训练结束后）
   - `imgs/loss_curve_latest.png` 存在且包含累积 loss 曲线
   - `imgs/loss_curve_latest.csv` 存在且数据行数与 `all_losses` 一致
   - 旧命名的 `deepcubea_heuristic_epoch_*.pt` 不再生成

## 📝 执行记录

- `2026-06-13 21:30`: 计划已生成
- `2026-06-13 22:30`: S1-S5 全部完成 —— checkpoint 改为 latest.pt 覆盖式保存，CSV（epoch/loss/is_outer_start/theta_updated）和 loss 曲线图每轮外层迭代后同步更新，末尾冗余绘图代码已删除
