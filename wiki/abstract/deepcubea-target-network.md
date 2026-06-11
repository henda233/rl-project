---
abstract_name: DeepCubeA 批量更新 + 固定目标 AVI
source_contents:
  - "deepcubea_train.py"
  - "deepcubea_network.py"
  - "config.py"
dependencies:
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/plan/deepcubea-target-network.md"
  - "wiki/request/deepcubea-target-network.md"
created_at: 2026-06-11 21:30
updated_at: 2026-06-11 23:16
---
# 摘要：DeepCubeA 批量更新 + 固定目标 AVI 训练

## 核心结论与关键信息

- **根因**：当前训练每 epoch 重算 Bellman 备份目标 → 目标随网络更新同步变化（"移动靶"）→ 正反馈雪崩（实证：epoch 2331 loss 爆炸至 10^21）
- **方案变更（2026-06-11）**：放弃 DQN 式 target network，采用 DeepCubeA 文献原版**批量更新 + 固定目标**
  - 原因：离线全量数据 + 确定性 Bellman 备份场景，target network 是过度设计；批量更新计算量更低、代码更简单、理论对齐更清晰
- **训练结构**：外层 Bellman 备份（20 轮）→ 内层固定目标监督学习（max 100 epoch，patience 10 早停）
- **不引入额外网络实例**：仅修改训练循环结构，网络和搜索模块不变
- **模型保存**：最后一轮外层迭代的模型作为 final（拥有最多 Bellman 备份步数）
- **Config 变更**：移除 `DEEPCUBEA_ITERATIONS`，新增 `DEEPCUBEA_OUTER_ITER=20` / `DEEPCUBEA_INNER_EPOCHS=100` / `DEEPCUBEA_INNER_PATIENCE=10`

## 内容概述

> 已实现。外层 Bellman 备份（20 轮）固定 J'(s)，内层早停监督学习（max 100 epoch, patience=10），双层 tqdm 显示进度，loss 曲线用竖线标记外层边界可视化 saw-tooth 模式。compute_targets 改用 inference_mode()。冒烟测试通过（100 states/3 outer×10 inner/patience=3，早停 25/30 epoch）。总 epoch 上限 2000，目标计算从 2000 次降至 ≤20 次。

## 依赖与影响链

- **上游依赖**：`deepcubea_network.py`（网络类）、`config.py`（训练参数）、`deepcubea_train.py`（训练脚本）
- **下游被依赖**：无（不影响搜索模块）
- **变更扩散评估**：低（仅 config + train 两文件）
