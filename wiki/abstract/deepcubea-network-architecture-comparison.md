---
abstract_name: DeepCubeA 官方 vs 自研网络架构对比
source_contents:
  - "deepcubea_official_network.py"
  - "deepcubea_network.py"
dependencies:
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/abstract/deepcubea-official-model-evaluation.md"
  - "wiki/abstract/deepcubea-official-repo.md"
created_at: 2026-06-15 19:00:00
updated_at: 2026-06-15 19:00:00
---
# 摘要：DeepCubeA 官方 vs 自研网络架构对比

## 核心结论与关键信息

### ResnetModel（官方）架构

```
Input: raw int tiles (B, 16) values 0~15
  ↓ 内部 F.one_hot → view (B, 256)
FC(256, 5000) → BatchNorm1d(5000) → ReLU
  ↓
FC(5000, 1000) → BatchNorm1d(1000) → ReLU
  ↓
4× ResBlock (each on 1000 dim):
  FC→BN→ReLU→FC→BN→+skip→ReLU
  ↓
Linear(1000, 1) → scalar J(s)
```

- 参数量约 15M，BN 在头层和残差块内均存在，ResBlock 内 FC 前后各一个 BN
- 模型本身接收原始整数 tiles，one-hot 在 forward 内部通过 `F.one_hot` 动态计算
- 外挂 `OfficialModelWrapper` 处理 numpy→tensor 转换，模型 forward 返回 (B, 1) 不做 squeeze

### DeepCubeANetwork（自研）架构

```
Input: pre-encoded one-hot (B, 256) float32（外部 numpy encode_batch 预处理）
  ↓
FC(256, H) → LayerNorm(H) → ReLU
  ↓
FC(H, H) → LayerNorm(H) → ReLU
  ↓
4× ResidualBlock (each on H dim):
  FC→ReLU→FC→+skip→ReLU（无归一化层）
  ↓
Linear(H, 1) → squeeze(-1) → scalar J(s)
```

- 参数量约 1.3M（H=512），仅头层有 LayerNorm，ResBlock 内无任何归一化
- one-hot 编码在模型外部通过 numpy 预处理完成，模型接收已编码的 (B, 256) float32
- `predict_j` / `predict_j_batch` 作为模型方法，forward 内部 squeeze

### 核心差异（除神经元数量外）

| 维度 | 官方 ResnetModel | 自研 DeepCubeANetwork |
|---|---|---|
| One-hot 编码位置 | 模型内部（`F.one_hot` 动态算） | 模型外部（numpy `encode_batch` 预处理） |
| 输入格式 | 原始整数 (B, 16) int64 | 已编码 one-hot (B, 256) float32 |
| 归一化类型 | BatchNorm1d（头层 + ResBlock 内） | LayerNorm（仅头层），ResBlock 内无归一化 |
| ResBlock 内归一化 | 有：FC→BN→ReLU→FC→BN | 无：FC→ReLU→FC（裸残差） |
| ResBlock 激活顺序 | FC→BN→ReLU→FC→BN→+skip→ReLU | FC→ReLU→FC→+skip→ReLU |
| 接口层 | 模型外挂 `OfficialModelWrapper` | `predict_j`/`predict_j_batch` 作为模型方法 |
| 输出维度 | (B, 1)，需外部 `.squeeze()` | (B,)，forward 内已 `squeeze(-1)` |
| 参数量 | ~15M | ~1.3M（H=512） |

三大核心架构差异：
1. **归一化层**：BN vs LN + ResBlock 内有无
2. **编码位置**：模型内部 vs 外部预处理
3. **ResBlock 内部结构**：激活顺序 + 是否含 BN

## 内容概述

> 逐层对比 `deepcubea_official_network.py` 的 `ResnetModel` 和 `deepcubea_network.py` 的 `DeepCubeANetwork` 的网络架构，覆盖输入处理、头层、残差块、输出层、接口层的完整差异。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-network.md`（自研网络 v1-v4 演进）、`wiki/abstract/deepcubea-official-model-evaluation.md`（官方模型评估）、`wiki/abstract/deepcubea-official-repo.md`（官方仓库分析）
- **下游被依赖**：PPO + DeepCubeA 势函数塑形中的网络选型参考
- **变更扩散评估**：低（纯分析文档，不修改代码）
