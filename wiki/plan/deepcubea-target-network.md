---
plan_name: DeepCubeA 批量更新 + 固定目标 AVI
related_request: "wiki/request/deepcubea-target-network.md"
status: completed
created_at: 2026-06-11 22:30
---
# 执行计划：DeepCubeA 批量更新 + 固定目标 AVI 训练

## 背景与目的

当前训练循环每 epoch 重算 Bellman 备份目标，目标随网络更新同步变化（"移动靶"），已实证导致 loss 在 epoch 2331 爆炸至 10^21。

DeepCubeA 文献采用**批量更新 + 固定目标**：外层计算一次 Bellman 备份得到 J'(s)，内层多 epoch 以固定目标做监督学习，收敛后重新备份进入下一轮。此方案无需额外 target network 实例，且与 AVI 理论语义直接对应：
> 外层 = 值迭代 Bellman 备份，内层 = 函数逼近监督学习。

**决策记录（2026-06-11）**：
- 放弃 DQN 式 target network（hard/soft 双模式），采用文献原版批量更新 + 固定目标
- 选择理由：离线全量数据 + 确定性 Bellman 备份场景下，target network 是过度设计；批量更新计算量更低（目标计算从 2000 次降至 20 次）、代码更简单、理论对齐更清晰

## 内容

### 训练循环结构

```
外层迭代 (OUTER_ITER 轮):
  ├─ compute_targets: 全量状态 Bellman 备份 → 固定 J'(s)
  └─ 内层训练 (max INNER_EPOCHS, patience 早停):
       ├─ mini-batch SGD, MSE(J(s;θ), J'(s))
       ├─ 目标固定不动
       └─ 收敛或达到上限后退出内层，进入下一轮外层
```

### Config 参数变更

| 操作 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 移除 | `DEEPCUBEA_ITERATIONS` | — | 替换为外层×内层 |
| 移除 | `DEEPCUBEA_CHECKPOINT_INTERVAL` | — | 替换为外层每轮保存 |
| 新增 | `DEEPCUBEA_OUTER_ITER` | `20` | Bellman 备份总轮数 |
| 新增 | `DEEPCUBEA_INNER_EPOCHS` | `100` | 每轮最大训练 epoch |
| 新增 | `DEEPCUBEA_INNER_PATIENCE` | `10` | 内层 loss 连续不改善 epoch 数触发早停 |

总 epoch 上限 = 20 × 100 = 2000（与当前 ITERATIONS=2000 持平），目标计算从 2000 次降至 ≤20 次。

### 内层早停逻辑

```python
best_inner_loss = inf
patience_left = INNER_PATIENCE
for inner_epoch in range(INNER_EPOCHS):
    avg_loss = train_one_epoch(fixed_targets)
    if avg_loss < best_inner_loss:
        best_inner_loss = avg_loss
        patience_left = INNER_PATIENCE
    else:
        patience_left -= 1
        if patience_left == 0:
            break  # 收敛，进入下一轮 Bellman 备份
```

### 模型保存

- 每轮外层迭代完成后保存 checkpoint（`epoch_N.pt`，N 为外层编号）
- 训练结束后，最终 checkpoint 即为最终交付物（最后一轮 = 拥有最多 Bellman 备份步数）
- 同时保存 final checkpoint（`deepcubea_heuristic_final.pt`）

### Loss 曲线

绘制全部 epoch 的 loss 曲线，用竖线标记外层迭代边界以可视化 saw-tooth 模式（新目标 → loss 跳升 → 内层收敛下降）。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物 | 状态 |
|---|---|---|---|---|
| `S1` | config.py：移除 ITERATIONS/CHECKPOINT_INTERVAL，新增 OUTER_ITER/INNER_EPOCHS/INNER_PATIENCE | 无 | `config.py` | 已完成 |
| `S2` | deepcubea_train.py：嵌套训练循环 + B1早停 + 外层模型保存 + saw-tooth loss曲线 | `S1` | `deepcubea_train.py` | 已完成 |
| `S3` | 冒烟测试：小数据集（100 states, 3 outer × 10 inner, patience=3）验证早停生效 | `S2` | 测试通过 | 已完成 |
| `S4` | 更新 WIKI 摘要与索引 | `S3` | abstract + index | 已完成 |

### 早停权重策略

选择 B1：内层早停时维持最终权重（不恢复到 best loss 时的权重）。理由：固定目标下 loss 收敛后漂移量可忽略，且下一外层 Bellman 备份会重新定标。

### 方案决策记录（2026-06-11 讨论）

- **文献对比**：论文 DAVI 用 θ_c 条件更新（loss < ε）、动态采样、少量梯度步。我们的场景不同：内层训练到收敛（充分拟合）后无条件重算，不会出现论文所述的"每次迭代更新 θ_c 导致不稳定"。
- **早停权重**：选 B1（取最终），简洁。
- **ε 阈值守卫**：不加，回退到无条件批量化目标重算。

## 风险与约束声明

- **兼容性**：不影响 `deepcubea_network.py`（网络结构不变），不影响 `deepcubea_search.py`（搜索推理不变）
- **早停阈值**：patience=10 是经验默认值，若内层始终未达到早停条件，最多跑满 INNER_EPOCHS 后切换
- **计算量**：目标计算从 2000 次降至 ≤20 次，训练瓶颈转移至内层 SGD，整体效率提升
- **诊断信号**：若 AVI 收敛，后期外层迭代的初始 loss（新目标下）应逐轮降低，最终趋近于 0
- **v2 参数保持不变**：TRAIN_SET_SIZE=200k, LR=1e-4, BATCH_SIZE=1024, HIDDEN_DIM=512

## 测试验证方案

1. 小数据集（100 states, 3 outer × 10 inner, patience=3）跑通
2. 确认早停在 loss 不下降时正确触发
3. 确认 loss 曲线 saw-tooth 模式正常、外层迭代边界标记正确
4. 确认 `deepcubea_heuristic_final.pt` 保存为最后一轮外层迭代的权重

## 📝 执行记录

- `2026-06-11 23:16`: S1-S4 全部完成 —— config 参数替换（OUTER_ITER=20/INNER_EPOCHS=100/INNER_PATIENCE=10）；训练循环重构为外层 Bellman 备份 + 内层早停监督学习（双层 tqdm + saw-tooth loss 曲线 + 外层边界竖线）；compute_targets 改用 inference_mode()；烟雾测试通过（100 states/3 outer×10 inner/patience=3，早停在 25/30 epoch 触发）
- `2026-06-11 22:30`: 方案决策 —— 放弃 target network（路径 B），采用批量更新 + 固定目标（路径 A）；原因详见背景章节
- `2026-06-11 22:30`: 实证确认计划必要性 —— v2 参数 epoch 2331 loss 爆炸至 5.6×10^21，根因为 Bellman 备份正反馈雪崩
- `2026-06-11 21:30`: 原版 Target Network 计划已生成（已废弃）
