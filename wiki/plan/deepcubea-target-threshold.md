---
plan_name: deepcubea-target-threshold
related_request: ""
references:
  - "wiki/abstract/docs/deepcubea-loss-analysis.md"
  - "docs/DeepCubeA文献重要内容.md"
status: completed
created_at: 2026-06-12 20:00:00
---
# 执行计划：DeepCubeA 论文阈值 θ_c 更新 + 验证集适配

## 背景与目的

当前训练的 Bellman 目标由当前网络 θ 直接计算（每轮外层无条件重算），导致正反馈雪崩——loss 峰值从 0.03 攀升至 0.35（详见 [DeepCubeA 训练 Loss 曲线分析](../abstract/docs/deepcubea-loss-analysis.md)）。这与 wiki 笔记中记录的 v2 epoch 2331 爆炸（10^21）属同一机制：Bellman 备份正反馈雪崩，仅时间尺度不同。

DeepCubeA 论文 DAVI 算法（算法 1，见 `docs/DeepCubeA文献重要内容.md`）使用独立的 target 参数 θ_c 计算目标，且仅在 loss < ε 时才更新 θ_c。论文作者明确测试了"每次迭代都更新 θ_c"并发现不稳定，阈值更新才能获得更好性能。

本计划将当前训练循环对齐论文的阈值更新机制，并适配当前"外层小迭代、内层大迭代"的训练结构（通过可选的验证集拆分防止过拟合绕过阈值）。

## 内容

### 一、Target 参数 θ_c

引入独立的 target 参数 θ_c（`DeepCubeANetwork` 实例），仅用于 Bellman 目标计算。训练网络 θ 正常更新；θ_c 在满足阈值条件时从 θ 复制。

- 初始化：θ_c ← θ（随机初始化一致）
- Bellman 备份：`compute_targets(states, target_network, device)` —— 用 θ_c 而非当前 θ
- 更新条件：若 `check_loss < ε`，则 `θ_c ← θ`

### 二、阈值检查（两种模式，由 VAL_SPLIT 控制）

**模式 A：VAL_SPLIT > 0（验证集模式，适配当前"外层小迭代、内层大迭代"）**
- 每轮外层生成的状态按比例拆分为训练集/验证集
- 内层训练仅在训练集上进行
- 内层结束后在验证集上计算 Bellman MSE
- 若验证集 loss < ε，更新 θ_c

**模式 B：VAL_SPLIT = 0（训练集模式，适配论文"外层大迭代、内层小迭代"）**
- 不拆分验证集，内层训练在所有状态上进行
- 内层结束后使用训练 loss（best_inner_loss）与 ε 比较
- 若训练 loss < ε，更新 θ_c

### 三、Config 新增参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEEPCUBEA_TARGET_EPSILON` | 0.05 | Bellman 误差阈值，低于此值更新 θ_c |
| `DEEPCUBEA_VAL_SPLIT` | 0.2 | 验证集比例，0 = 禁用验证集（用训练 loss 判断） |

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| S1 | config.py 新增 `DEEPCUBEA_TARGET_EPSILON` + `DEEPCUBEA_VAL_SPLIT` | 无 | `config.py` | 已完成 |
| S2 | deepcubea_train.py 引入 target_network，Bellman 备份改用 θ_c | S1 | `deepcubea_train.py` | 已完成 |
| S3 | deepcubea_train.py 实现验证集拆分 + 阈值检查 + θ_c 更新逻辑 | S1, S2 | `deepcubea_train.py` | 已完成 |
| S4 | 外层日志增强：tqdm postfix 显示 θ_c 更新状态 + val_loss + loss 曲线 θ_c 标记 | S3 | `deepcubea_train.py` | 已完成 |
| S5 | 更新 wiki 摘要 + index | S4 | `wiki/abstract/deepcubea-online-validation.md`, `wiki/index.md` | 已完成 |

### S2 详情 — θ_c 引入与 Bellman 备份切换

```python
# 初始化（train() 函数内，创建 network 后）
target_network = DeepCubeANetwork().to(device)
target_network.load_state_dict(
    {k: v.clone() for k, v in network.state_dict().items()}
)

# 外层循环内 —— Bellman 备份改用 target_network
targets = compute_targets(train_states, target_network, device).to(device)
```

### S3 详情 — 验证集拆分 + 阈值检查

```python
# 合并去重后、编码后，拆分 train/val
if DEEPCUBEA_VAL_SPLIT > 0:
    n_val = max(1, int(len(train_states) * DEEPCUBEA_VAL_SPLIT))
    perm = torch.randperm(len(train_states), device=device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    val_onehot = train_onehot[val_idx]
    val_targets = targets[val_idx]
    train_onehot = train_onehot[train_idx]
    targets = targets[train_idx]

# 内层训练（仅在 train 部分）...

# 内层结束后阈值检查
if DEEPCUBEA_VAL_SPLIT > 0:
    with torch.inference_mode():
        val_pred = network(val_onehot)
        check_loss = F.mse_loss(val_pred, val_targets).item()
else:
    check_loss = best_inner_loss

theta_updated = (check_loss < DEEPCUBEA_TARGET_EPSILON)
if theta_updated:
    target_network.load_state_dict(
        {k: v.clone() for k, v in network.state_dict().items()}
    )
```

### S4 详情 — 日志增强

outer_pbar postfix 新增字段：
- `θ_c`: `"updated"` 或 `"frozen"`（后者附 check_loss 值）
- `val_loss`: 验证集 loss（仅 VAL_SPLIT > 0 时显示）

## 风险与约束声明

- **ε 调参风险**：ε 过小 → θ_c 永不更新，训练停滞；ε 过大 → 退化为当前无条件更新。用户将 ε 作为可调参数，需根据 loss 曲线初期峰值（~0.03）和经验设定初始值
- **VAL_SPLIT 交互**：VAL_SPLIT=0 时依赖训练 loss 判断，当内层大迭代早停时 loss 可能总是远低于 ε（过拟合），导致阈值形同虚设。此模式适用于用户后续调整为"外层大迭代、内层小迭代"时使用
- **第一轮外层迭代**：θ_c = θ（随机初始化），Bellman 目标质量取决于随机网络输出。这是预期行为——后续轮次通过阈值机制自然过滤
- **target_network 显存**：额外维护一个 DeepCubeANetwork 实例，参数量约 1.2M（hidden=512），Float32 约 4.8MB，对 RTX 3050 4GB 影响极小
- **旧模型兼容性**：训练逻辑变更，不影响网络结构和已保存的 .pt 文件

## 测试验证方案

1. 小规模冒烟测试：OUTER_ITER=3, INNER_EPOCHS=10, ONLINE_BATCH=500, BASE_BATCH=200, VAL_SPLIT=0.2, TARGET_EPSILON=0.05
2. 验证 θ_c 更新/冻结日志正确输出
3. 验证 VAL_SPLIT=0 模式下回退为训练 loss 判断
4. 对比新训练 loss 曲线：观察 saw-tooth 峰值是否从递增变为稳定/递减

## 📝 执行记录

- `2026-06-12 20:00`: 计划已生成
- `2026-06-13 12:00`: S1-S5 全部执行完成 —— config 新增 TARGET_EPSILON(0.05)+VAL_SPLIT(0.2)；train 引入 target_network(θ_c) + Bellman 备份改用 θ_c + val 拆分 + 阈值检查 + θ_c 更新 + tqdm 增强 + loss 曲线 θ_c 标记；wiki 摘要+index 更新
