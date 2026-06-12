---
plan_name: deepcubea-online-sampling-validation
related_request: "wiki/request/deepcubea-online-sampling-validation.md"
status: waitting
created_at: 2026-06-12 12:00:00
---
# 执行计划：DeepCubeA 在线采样 + 轻量验证指标

## 背景与目的

当前离线固定 20 万状态集存在分布固化问题：网络反复拟合同一批状态，无法泛化到分布外的中远距离状态，导致 Short 档 47% 而 Medium/Hard 档 0%。本计划将训练改为在线混合采样、向网络添加 Batch Normalization 以对齐原论文并提升训练稳定性，并在 `deepcubea_search.py` 中新增轻量验证指标，实现训练-评估解耦。

## 内容

### 一、Batch Normalization

对齐 DeepCubeA 原论文，在所有隐藏层和残差块中添加 Batch Normalization，在激活函数之前应用。论文明确指出 BN 是其网络结构的一部分，有助于在 DAVI 目标不断变化（外层重算 Bellman 备份）的场景下保持训练稳定。

**改动位置（`deepcubea_network.py` 中 `DeepCubeANetwork` 类）：**
- FC 隐藏层：`Linear → BatchNorm1d → ReLU`
- 残差块内部保持不变
- 输出层不加 BN（保持 Linear → scalar）

**注意**：BN 引入后，网络推理模式需区分 `train()` / `eval()`——训练时用 batch statistics，推理时（Bellman 目标计算、A* 搜索）用 running statistics。

### 二、在线混合采样

参考 DeepCubeA 原论文 DAVI 算法（算法 1），每轮外层迭代从目标状态随机打乱生成训练状态（而非使用离线固定数据集）。同时保留基础数据集作为混合成分以保持训练稳定性。

**采样流程：**
1. 每轮外层迭代：从基础离线数据集随机采样 B' 个状态 + 新生成 B 个状态（从目标状态随机打乱 K∈[T_MIN, T_MAX] 步）
2. 合并 B+B' 个状态作为本轮训练集，计算 Bellman 目标，执行内层训练
3. 控制随机种子（每轮外层 + seed_offset），相邻轮次设置种子使采样部分重叠

### 三、轻量验证指标

在 `deepcubea_search.py` 中新增两种轻量指标，训练结束后单独运行评估：

**方案 B — 分箱 Bellman MSE（默认开启）：**
- 动态生成验证集：N 个状态（K∈[1,500] 均匀打乱），验证集种子参数化
- 用 K（打乱步数）作为最优距离 proxy 分箱：1-20 / 21-100 / 101-300 / 301-500
- 每箱计算 Bellman MSE = (J(s) - min_a(1+J(A(s,a))))²
- 输出分箱报告表格

**方案 C — 贪心展开节点数（flag 开启）：**
- 对验证集状态执行加权 A*（λ 可配），max_expand 截断（默认 500）
- 统计平均展开节点数，作为搜索效率指标
- 比完整 A* 快一个数量级

**方案 D — 完整加权 A\* 评估（flag 开启）：**
- 保留现有的 Short/Medium/Hard 三档完整搜索评估
- 由独立的 `--full-astar` flag 控制

### 四、评估策略

| 频率 | 指标 | 目的 |
|---|---|---|
| 训练结束后 | 方案 B（分箱 Bellman MSE） | 诊断 J 在各距离区间的准确性 |
| 训练结束后 | 方案 C（贪心展开数） | 验证搜索效率改善 |
| 训练结束后 | 方案 D（完整 A\* 三档） | 最终成绩单 |

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| S1 | config.py 新增在线采样与验证参数 | 无 | `config.py` | 待完成 |
| S2 | deepcubea_network.py 添加 Batch Normalization | 无 | `deepcubea_network.py` | 待完成 |
| S3 | deepcubea_train.py 实现在线混合采样训练循环 | S1 | `deepcubea_train.py` | 待完成 |
| S4 | deepcubea_search.py 新增方案 B：分箱 Bellman MSE（默认模式） | S1 | `deepcubea_search.py` | 待完成 |
| S5 | deepcubea_search.py 新增方案 C：贪心展开节点数（flag 控制） | S4 | `deepcubea_search.py` | 待完成 |
| S6 | deepcubea_search.py 保留方案 D：完整 A\* 三档评估（flag 控制） | S4 | `deepcubea_search.py` | 待完成 |
| S7 | 冒烟测试：在线采样训练 + 三项指标评估 | S2, S3, S6 | 无新增文件 | 待完成 |

### S1 详情 — Config 参数新增

| 参数名 | 默认值 | 说明 |
|---|---|---|
| `DEEPCUBEA_ONLINE_BATCH` | 20000 | 每轮外层在线生成状态数 B |
| `DEEPCUBEA_BASE_BATCH` | 5000 | 每轮外层从基础数据集采样状态数 B'，设 0 为纯在线 |
| `DEEPCUBEA_OUTER_SEED` | 42 | 外层迭代随机种子基准值 |
| `DEEPCUBEA_SEED_OVERLAP` | 0.3 | 相邻轮次采样状态重叠比例 |
| `DEEPCUBEA_VAL_SIZE` | 3000 | 验证集状态数量 |
| `DEEPCUBEA_VAL_SEED` | 12345 | 验证集生成随机种子 |
| `DEEPCUBEA_VAL_KBINS` | [20, 100, 300, 500] | 分箱边界（K proxy） |
| `DEEPCUBEA_VAL_GREEDY_EXPAND` | 500 | 贪心展开最大节点数 |
| `DEEPCUBEA_VAL_ASTAR_FLAG` | False | 是否启用完整 A\* 评估 |
| `DEEPCUBEA_VAL_GREEDY_FLAG` | False | 是否启用贪心展开评估 |
| `DEEPCUBEA_VAL_LAMBDA` | 0.8 | 贪心展开 / 完整 A\* 的加权系数 |

### S2 详情 — 添加 Batch Normalization

在 `DeepCubeANetwork` 的所有隐藏层和残差块中添加 `BatchNorm1d`，对齐原论文网络结构。

```
# FC 隐藏层改动
Linear(in, out) → BatchNorm1d(out) → ReLU

# 残差块（不变）
Linear(dim, dim) → ReLU → Linear(dim, dim) → + residual → ReLU

# 输出层（不变）
Linear(dim, 1)
```

**train/eval 模式切换：**
- 训练循环中：`network.train()` —— 使用 batch statistics
- Bellman 目标计算（`compute_targets`）：`network.eval()` + `torch.inference_mode()`
- A\* 搜索推理：`network.eval()` + `torch.inference_mode()`

### S3 详情 — 在线混合采样训练循环

```
for outer_iter in range(OUTER_ITER):
    # 1. 在线生成 B 个状态（随机打乱）
    online_states = generate_scrambled_states(B, T_MIN, T_MAX, seed=OUTER_SEED + outer_iter)
    
    # 2. 从基础数据集采样 B' 个状态（考虑重叠）
    base_states = sample_with_overlap(base_dataset, B', seed=OUTER_SEED + outer_iter, overlap=SEED_OVERLAP)
    
    # 3. 合并 + 去重
    train_states = unique(concat(online_states, base_states))
    
    # 4. 计算 Bellman 目标（固定 θ_c）
    targets = compute_bellman_targets(train_states, network)
    
    # 5. 内层训练（早停）
    inner_train(train_states, targets, INNER_EPOCHS, INNER_PATIENCE)
    
    # 6. 更新 θ_c
    θ_c ← θ
```

### S4 详情 — 方案 B 分箱 Bellman MSE

```
输入: 模型路径, 验证集参数(N, seed, K bins)
流程:
  1. 加载模型
  2. 动态生成验证集状态 + 记录每个状态的 K 值
  3. 对每个状态 s 计算: Bellman_error = (J(s) - min_a(1 + J(A(s,a))))²
  4. 按 K 值分箱，每箱统计 mean/median/std MSE
输出: 分箱报告表格
```

### S5 详情 — 方案 C 贪心展开节点数

```
输入: 模型路径, 验证集, max_expand, lambda
流程:
  1. 加载模型
  2. 对每个验证集状态执行加权 A*，max_expand 截断
  3. 统计: 平均展开节点数、求解成功率（在截断内）、平均解长度
输出: 贪心展开统计报告
```

### S6 详情 — 方案 D 完整 A\* 评估

```
保留现有 Short/Medium/Hard 三档评估逻辑
由 --full-astar flag 控制是否执行
```

## 风险与约束声明

- **BN train/eval 模式风险**：BN 引入后必须严格区分 `network.train()` / `network.eval()`。若 Bellman 目标计算时未切换到 eval 模式，batch statistics 会污染目标值，导致训练不稳定。A\* 搜索推理时需同时使用 `eval() + inference_mode()`
- **旧模型兼容性**：添加 BN 后网络结构变更，旧 checkpoint（`.pt`）无法加载。训练完成后旧评估脚本如果直接加载新模型也会报错，需要一并更新
- **分布偏移验证**：混合采样能否提升泛化，需要对比训练前后方案 B 分箱报告（训练 vs 在线采样训练）。如果远距箱 MSE 持续不下降，说明需要更多外层迭代或调整 K 分布
- **内存/显存**：在线生成状态在 CPU 执行（`generate_scrambled_states` 无 GPU 依赖），不影响显存；BN 额外参数和 running statistics 内存开销极小；B 的规模仅影响 Bellman 目标计算时间和内层训练数据量
- **种子可控性**：`OUTER_SEED + outer_iter` 的线性种子策略保证可复现性；`SEED_OVERLAP` 控制相邻轮次部分 overlap 以避免目标震荡
- **K proxy 局限性**：K 是打乱步数而非真实最优距离，分箱报告仅作为相对趋势参考，不做精确诊断

## 测试验证方案

1. 小规模冒烟测试（S7）：OUTER_ITER=3, INNER_EPOCHS=10, B=500, B'=200, VAL_SIZE=100
2. 验证 BN 的 train/eval 切换正确（Bellman 目标计算时 eval 模式不报错）
3. 验证三项指标运行不报错、分箱报告格式正确
4. 对比在线采样 vs 离线训练的分箱 Bellman MSE 报告，确认远距箱有改善
5. 完整 A\* 三档评估验证 Medium/Hard 胜率从 0 提升

## 📝 执行记录

- `2026-06-12 12:00`: 计划已生成，基于 DeepCubeA 文献与优化方向文档讨论后制定
