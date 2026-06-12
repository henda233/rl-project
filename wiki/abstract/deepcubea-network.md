---
abstract_name: DeepCubeA 神经网络模块
source_contents:
  - "deepcubea_network.py"
  - "deepcubea_train.py"
  - "deepcubea_generate_data.py"
  - "config.py"
dependencies:
  - "wiki/abstract/huarongdao-env.md"
  - "wiki/abstract/docs/deepcubea-research.md"
  - "wiki/abstract/deepcubea-search.md"
created_at: 2026-06-11 19:00:00
updated_at: 2026-06-12 17:30:00
---
# 摘要：DeepCubeA 神经网络模块

## 核心结论与关键信息

- **one-hot 编码**：16 位置 × 16 值 = 256 维，位置 i 值 v → bit `i*16+v`，空格(0)作为普通值参与编码
- **纯函数状态转移**：`transition(grid_flat, action)` 无环境副作用，非法动作返回 None，action 语义与环境一致（0=上,1=下,2=左,3=右）
- **网络结构**：Input(256) → FC(256) → LN → ReLU → FC(256) → LN → ReLU → 4×ResBlock → FC(1)，输出标量 J(s)（v4: BN → LayerNorm 替换，消除 batch_size ≥ 2 约束；仅 FC 隐藏层，残差块不变）
- **LayerNorm 优势**：对 `(B, D)` 输入沿 D 维度归一化，行为与 batch_size 无关，训练/推理一致无需模式切换。已删除所有 `.eval()` 调用（predict_j/predict_j_batch/compute_targets/load_model/_compute_bellman_errors）
- **旧模型不兼容**：BN → LN 后参数字典 key 名变更，旧 checkpoint 无法加载，需重新训练
- **残差块**：FC(256)+ReLU → FC(256) → +input → ReLU
- **训练方式**：批量更新 + 固定目标 AVI（外层 Bellman 备份固定 J'(s)=min_a(1+J(A(s,a))) → 内层固定目标监督学习 + patience 早停），双层 tqdm 显示进度，saw-tooth loss 曲线用竖线标记外层边界
- **AVI 发散**：原训练（每 epoch 重算目标）在 epoch 2331 出现 loss 爆炸至 10^21，根因为 Bellman 备份正反馈雪崩；改为批量更新 + 固定目标后解决
- **数据生成/训练分离**：`deepcubea_generate_data.py` 独立生成训练数据并保存 `.npy`，`deepcubea_train.py` 从文件加载训练；`DEEPCUBEA_TRAIN_DATA_PATH` 配置数据路径；`DEEPCUBEA_LOG_INTERVAL` 已移除（tqdm 替代）

## 首次训练（v1）

| 参数 | 值 | 说明 |
|---|---|---|
| T | U(1, 100) | 随机游走步数范围 |
| 训练集 | 4,131 unique（5,000 raw） | 去重后 |
| LR | 1e-3 | Adam |
| Iter | 1,000 | 最佳 loss=0.0138 @ epoch 527 |
| 模型 | `results/train_20260611_183307/models/deepcubea_heuristic_best.pt` | |

**评估反馈**：A* 搜索 Short 档（1-34 步）胜率仅 32%，仅能解极近距离状态。根因——(1) 随机游走 100 步实际最优距离仅 ~15-20，训练集集中在一个小半径球内；(2) 4k 去重状态不足以泛化；(3) LR=1e-3 震荡大。

## 第二次训练（v2，已执行：批量更新 + 固定目标 AVI）

| 参数 | 值 | 说明 |
|---|---|---|
| T_MIN | 10 | 跳过琐碎近距状态 |
| T_MAX | 500 | 覆盖中远距离区域 |
| TRAIN_SET_SIZE | 200,000 | 适配 RTX 3050 4GB 显存 |
| LR | 1e-4 | 压低 Bellman 备份震荡 |
| OUTER_ITER | 20 | Bellman 备份轮数 |
| INNER_EPOCHS | 100 | 每轮最大训练 epoch |
| INNER_PATIENCE | 10 | 内层早停 patence |
| BATCH_SIZE | 1,024 | |
| HIDDEN_DIM | 512 | |

**训练方式**：外层 Bellman 备份固定 J'(s) + 内层早停监督学习（详见 `deepcubea-target-network.md`）。总 epoch 上限 = 20 × 100 = 2000，目标计算从 2000 次降至 ≤20 次。

## 内容概述

> `deepcubea_network.py` 包含编码模块和网络类，提供 encode/encode_batch/decode/transition/get_children/DeepCubeANetwork。`deepcubea_generate_data.py` 独立生成训练数据（反向随机游走 → 去重 → `.npy`），`deepcubea_train.py` 从 `.npy` 加载数据执行批量更新 + 固定目标 AVI 训练（外层 Bellman 备份 + 内层早停监督学习）。训练数据默认路径 `results/train_data/train_states.npy`，训练进度由双层 tqdm 显示。

## 依赖与影响链

- **上游依赖**：`config.py`（HUARONGDAO_N=4 → N2=16；DEEPCUBEA_* 训练参数）
- **下游被依赖**：`wiki/abstract/deepcubea-search.md`（加权 A* 搜索，使用 best.pt 作为启发函数）
- **变更扩散评估**：中（训练参数修改影响训练脚本和下游搜索评估）
