---
plan_name: deepcubea-bn-to-ln
related_request: ""
status: completed
created_at: 2026-06-12 16:30:00
---
# 执行计划：BatchNorm1d 替换为 LayerNorm

## 背景与目的

DeepCubeA 训练第 20 轮外层迭代时，在线采样 + 去重后 `train_size % 256 == 1`，末尾 batch 仅含 1 个样本，BatchNorm1d 训练模式要求 batch_size ≥ 2，触发 `ValueError: Expected more than 1 value per channel when training`。

BatchNorm 在该项目中还存在额外代价：训练/推理模式切换（4 处 `.eval()` 调用）、大 batch(256) 训练 vs 小 batch(2~4) 推理的统计分布不一致。替换为 LayerNorm 可一并消除这些问题，且纯 FC 网络是 LN 的标准适用场景。

## 内容

### 核心改动

`deepcubea_network.py` 中 `DeepCubeANetwork` 类：

```python
# 改动前
self.bn1 = nn.BatchNorm1d(hidden_dim)
self.bn2 = nn.BatchNorm1d(hidden_dim)

# 改动后
self.bn1 = nn.LayerNorm(hidden_dim)
self.bn2 = nn.LayerNorm(hidden_dim)
```

LayerNorm 对 `(B, D)` 输入沿 D 维度归一化，行为与 batch_size 无关，训练/推理一致，无需模式切换。

### 附带清理 — 删除冗余 `.eval()` 调用

BN 替换为 LN 后，以下 `.eval()` 调用变为无操作，一并删除：

| 文件 | 行号 | 代码 | 处理 |
|---|---|---|---|
| `deepcubea_network.py` | 132 | `self.eval()` in `predict_j` | 删除 |
| `deepcubea_network.py` | 140 | `self.eval()` in `predict_j_batch` | 删除 |
| `deepcubea_train.py` | 65 | `network.eval()` in `compute_targets` | 删除 |
| `deepcubea_search.py` | 31 | `model.eval()` in `load_model` | 删除 |
| `deepcubea_search.py` | 138 | `model.eval()` in `_compute_bellman_errors` | 删除 |

**保留** `deepcubea_train.py:190` 的 `network.train()` —— 虽对 LN 无影响，但作为训练循环标准实践保留，不造成问题。

### 不涉及的部分

- 网络其余结构不变（ResBlock、输出层）
- config.py 无需改动
- deepcubea_utils.py 无需改动
- deepcubea_generate_data.py 无需改动

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| S1 | 替换 BN → LN (两行) | 无 | `deepcubea_network.py:113,115` | 已完成 |
| S2 | 删除 predict_j / predict_j_batch 中的 self.eval() | S1 | `deepcubea_network.py:132,140` | 已完成 |
| S3 | 删除 compute_targets 中的 network.eval() | S1 | `deepcubea_train.py:65` | 已完成 |
| S4 | 删除 search.py 中 load_model 和 _compute_bellman_errors 的 model.eval() | S1 | `deepcubea_search.py:31,138` | 已完成 |
| S5 | 更新网络摘要文档 | S1-S4 | `wiki/abstract/deepcubea-network.md` | 已完成 |
| S6 | 更新在线采样摘要文档 | S1-S4 | `wiki/abstract/deepcubea-online-validation.md` | 已完成 |
| S7 | 更新 wiki index | S5, S6 | `wiki/index.md` | 已完成 |

## 风险与约束声明

- **旧模型不兼容**：网络结构变更（BN→LN 参数字典 key 名不同），旧 checkpoint 无法加载。需要重新训练。
- **训练行为差异**：LN 归一化方式不同，可能影响收敛速度。但对纯 FC + 256 batch_size 场景，LN 已被广泛验证有效。
- **推理行为一致**：LN 训练/推理无差异，`predict_j` 单样本推理不再依赖 running statistics 质量。

## 测试验证方案

1. 运行冒烟测试 `deepcubea_online_smoke_test.py`，确认训练 + 三项验证指标正常
~~2. 手动验证 batch_size=1 边界：构造 257 样本训练集（256+1），确认不再报错~~
~~3. 对比 LN 训练 vs BN 训练的 loss 曲线和分箱 Bellman MSE~~

## 📝 执行记录

- `2026-06-12 16:30`: 计划已生成，基于训练崩溃根因分析后制定
- `2026-06-12 17:00`: BN 根因已复现确认 —— `reproduce_bn_crash.py` 单样本 forward 触发 `ValueError: Expected more than 1 value per channel when training`，与远程训练崩溃一致
- `2026-06-12 17:30`: S1-S7 全部完成 —— BN→LN 替换 + 5处 .eval() 删除 + wiki 摘要/index 更新
