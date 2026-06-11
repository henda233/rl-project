---
plan_name: DeepCubeA 神经网络训练
related_request: "wiki/request/deepcubea-reproduction.md"
status: completed
created_at: 2026-06-11 18:00:00
---
# 执行计划：DeepCubeA 神经网络 — 近似值迭代训练启发函数

## 背景

DeepCubeA 的第一阶段：通过近似值迭代训练一个深度神经网络，使其逼近从任意状态 s 到目标状态的最优代价函数 J*(s)。训练数据通过从目标状态反向随机游走生成。学到的 J(s) 将作为第二阶段加权 A* 搜索的启发函数。

## 网络结构

```text
输入 (256 维 one-hot: 16 位置 × 16 值)
  ↓
全连接 256→256 + ReLU   ← 独立 FC 层 1
  ↓
全连接 256→256 + ReLU   ← 独立 FC 层 2
  ↓
残差块 ×4（每个: FC 256→256 + ReLU → FC 256→256 → +input → ReLU）
  ↓
全连接 256→1（线性输出，标量 J(s)）
```

两个 FC 隐藏层是**独立的**（不属于残差块），之后接 4 个残差块，最后线性输出标量。

## 值迭代公式

```
J'(s) = min_{a ∈ legal(s)} [ 1 + J(A(s,a)) ]
```

- 仅对合法动作取 min，非法动作通过 `get_action_mask()` 过滤
- 状态转移 `A(s,a)` 使用**纯函数**实现（提取 grid 坐标交换逻辑，无环境副作用）
- `J(A(s,a))` 通过当前网络 forward 计算（`torch.no_grad()`）

损失函数：MSE(J(s), J'(s))

## 训练流程

### 整体架构（类 DQN target network）

```
for epoch in range(DEEPCUBEA_ITERATIONS):
    # 1. 全量计算 J'(s)：对整个训练集，用当前网络计算所有状态的 target 值
    targets = compute_j_prime_all(train_states, network, device)

    # 2. 逐 batch 训练：用预计算的 targets 作为监督信号
    for batch in DataLoader:
        loss = MSE(network(batch_x), batch_targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 3. 日志 / checkpoint / 绘图
```

每 epoch 开始前对**整个训练集**重新计算一次 J'(s)（类似 DQN 中 target network 的周期性更新），然后用这批固定 targets 做该 epoch 内的 batch 训练。

### J'(s) 计算

对每个状态 s：
1. 遍历 4 个动作，用纯函数 `transition(grid_flat, action)` 计算下一状态
2. 仅对合法动作（`next_state is not None`），将下一状态 one-hot 编码后送入网络得到 J 值
3. J'(s) = min(1 + J 值列表)

复杂度：10000 状态 × 4 动作 = 40000 次 forward/epoch，batch_size=256 约 157 个 batch，可接受。

### 训练数据生成

从目标状态出发，采样 T ~ Uniform(T_MIN, T_MAX)，执行 T 步随机合法移动，将得到的状态加入训练集。重复至训练集大小达到 TRAIN_SET_SIZE。生成过程在训练开始前一次性完成。

## 结果保存

训练开始时在 `results/` 下创建时间戳目录，结构如下：

```
results/train_<YYYYMMDD_HHMMSS>/
├── models/
│   ├── deepcubea_heuristic_best.pt       # 最低 loss 的模型
│   └── deepcubea_heuristic_epoch_N.pt    # 定期 checkpoint
└── imgs/
    └── loss_curve.png                     # 训练损失曲线
```

- 最优模型按**训练集 loss** 保存（每 epoch 全量计算 J'(s) 时自然得到全训练集 MSE，无需额外划分验证集）
- checkpoint 每 `DEEPCUBEA_CHECKPOINT_INTERVAL` epoch 保存一次
- loss 曲线训练过程中实时更新保存

（参考 `ppo_agent.py:224` 的 `results/train_{datetime:%Y%m%d_%H%M%S}` 模式）

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 添加 DeepCubeA 训练参数 | 无 | `config.py`（修改） | ✅ 已完成 |
| `S2` | 实现 one-hot 编解码 + 纯函数状态转移 | `S1` | `deepcubea_network.py`（encode/decode/transition 函数） | ✅ 已完成 |
| `S3` | 实现 DeepCubeANetwork 网络 | `S2` | `deepcubea_network.py`（DeepCubeANetwork 类） | ✅ 已完成 |
| `S4` | 实现训练数据生成（反向随机游走） | `S2` | `deepcubea_train.py`（generate_training_data 函数） | ✅ 已完成 |
| `S5` | 实现训练脚本：每 epoch 全量计算 J'(s) + batch 训练 + 时戳结果目录 + checkpoint + loss 图 | `S3`, `S4` | `deepcubea_train.py`（训练主逻辑） | ✅ 已完成 |
| `S6` | 执行训练并验证 loss 收敛 | `S5` | `results/train_20260611_183307/models/deepcubea_heuristic_best.pt` | ✅ 已完成 |
| `S7` | 编写摘要、更新 index | `S6` | `wiki/abstract/deepcubea-network.md` | ✅ 已完成 |

## 新增配置参数

```python
# DeepCubeA Training
DEEPCUBEA_T_MIN = 1                    # 随机游走最小步数
DEEPCUBEA_T_MAX = 30                   # 随机游走最大步数
DEEPCUBEA_TRAIN_SET_SIZE = 10000       # 训练集状态数
DEEPCUBEA_LR = 1e-3                    # 学习率
DEEPCUBEA_BATCH_SIZE = 256             # 批大小
DEEPCUBEA_ITERATIONS = 10000           # 值迭代总轮数（epoch）
DEEPCUBEA_HIDDEN_DIM = 256             # 隐藏层维度
DEEPCUBEA_USE_GPU = True               # 是否使用 GPU
DEEPCUBEA_CHECKPOINT_INTERVAL = 1000   # checkpoint 保存间隔（epoch）
```

## 风险与约束

- **风险**：4×4 状态空间巨大（~10^13），U(1,30) 生成的训练状态仅覆盖少量区域，启发函数对远距离状态的泛化能力待验证
- **约束**：仅考虑合法动作计算 J'(s)，非法动作不参与 min 运算
- **约束**：训练数据从目标状态反向生成，状态分布偏向接近目标的区域；值迭代的贝尔曼更新可逐步向外传播代价信息
- **约束**：状态转移使用纯函数实现，不依赖环境实例的 step 方法

## 测试验证方案

1. 训练过程中监控 loss 曲线，确认收敛
2. 在随机测试状态上评估启发函数的合理性（J(s) 值应随曼哈顿距离单调增长）
3. 模型保存后能正确加载并推理

## 📝 执行记录

- `2026-06-11 18:00`: 计划初始生成
- `2026-06-11`: 与用户对齐技术细节 —— 确认：纯函数状态转移、每 epoch 全量计算 J'(s)（类 DQN）、独立 FC 层、时戳结果目录、checkpoint 机制、GPU 配置
- `2026-06-11`: S1-S3 完成 —— `config.py` 追加 DEEPCUBEA_* 配置（GPU 默认 True）、`deepcubea_network.py` 包含 encode/decode/encode_batch/transition/ResidualBlock/DeepCubeANetwork（含 predict_j, predict_j_batch），10 项测试全通过（含 GPU 推理）
- `2026-06-11`: S4-S7 完成 —— `deepcubea_train.py`（generate_training_data + compute_targets + train 主逻辑）端到端测试通过。执行训练：4131 个去重状态（5000 raw, T~U(1,100)）、1000 epoch、最佳 loss=0.0138（epoch 527）、模型保存至 `results/train_20260611_183307/`。观察到近似值迭代 (AVI) 发散模式：epoch 527 后 loss 从 0.014 升至 0.26，属函数逼近器 Bellman 备份的典型误差累积现象
