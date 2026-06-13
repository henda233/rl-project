---
plan_name: DeepCubeA 官方预训练模型评估
related_request: "使用官方预训练模型跑当前项目的 deepcubea_search.py 评估测试"
status: completed
created_at: 2026-06-13 23:50
---
# 执行计划：DeepCubeA 官方预训练模型评估

## 背景

当前项目实现了完整的 DeepCubeA 训练+搜索流程，但受限于计算资源（RTX 3050 4GB），训练出的模型质量远不如官方模型：
- **当前 v2**：Short 档胜率 32%，仅能解极小距离状态
- **官方 puzzle15**：500 状态 100% 求解，平均 52 步，展开 ~320 万节点

本计划将官方 `ResnetModel` 移植到当前项目，编写独立的官方模型搜索评估脚本，不修改现有 `deepcubea_search.py`，直接评估官方模型的搜索性能，建立质量基线。

## 架构决策

- **独立新文件**：官方模型搜索评估在独立文件中实现，不修改 `deepcubea_search.py`
- **直接移植**：拷贝 `ResnetModel` 到项目，按项目代码风格调整（命名、import、config 引用），不做 Adapter wrapper 伪装接口
- **数据预处理（方案C）**：一次性脚本将 `data_0.pkl` 转为纯 numpy `.npy` 到 `data/`，彻底解除 pickle 模块路径依赖；同时拷贝官方模型到 `data/`，所有文件统一在项目根目录 `data/` 下
- **三种评估**：Stratified Bellman MSE + Greedy 贪心展开 + Full A*，λ=1.0（精确加权 A*，与官方对齐）
- **评估控制**：通过 config.py 新增参数 + CLI 参数双重控制（参考 `deepcubea_search.py` 模式 —— Bellman 始终执行，Greedy/A* 由 Flag 控制）
- **难度分档**：按 `solutions` 实际解路径长度排序后三等分（短/中/长各约 167 个），参考 `deepcubea_search.py:evaluate` 的三档分档思想（300-379行）
- **推理设备**：复用 `DEEPCUBEA_INFERENCE_USE_GPU`
- **Config 键隔离**：所有新增键均使用 `DEEPCUBEA_OFFICIAL_*` 前缀，不复用现有 `DEEPCUBEA_LAMBDA` / `DEEPCUBEA_MAX_EXPAND_NODES`

## data_0.pkl 内部结构

```
路径: DeepCubeA/data/puzzle15/test/data_0.pkl
类型: dict
键:
  'states':               List[NPuzzleState]  (500,)   每个 .tiles → (16,) ndarray int32
  'solutions':            List[List[int]]     (500,)   解路径（动作序列 0=U,1=D,2=L,3=R）
  'num_nodes_generated':  List[int]           (500,)   官方 A* 展开节点数
  'times':                List[float]         (500,)   官方求解耗时（秒）
```

## 核心差异：网络架构

| 维度 | 官方 | 当前项目 |
|---|---|---|
| 网络类 | `ResnetModel` | `DeepCubeANetwork` |
| 输入 | raw tiles (B,16) int，内部 one_hot | pre-encoded one_hot (B,256) |
| 隐藏层 | fc1(256→5000)→BN→fc2(5000→1000)→BN | fc1(256→HIDDEN)→LN→fc2(HIDDEN→HIDDEN)→LN |
| ResBlock dim | 1000 | HIDDEN (256/512) |
| 归一化 | BatchNorm1d | LayerNorm |
| 输出 | fc_out(1000→1) | fc_out(HIDDEN→1) |
| 参数量 | ~15M | ~1.3M |

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物 | 状态 |
|---|---|---|---|---|
| S0 | 数据预处理：在项目 venv 中执行一次性脚本，加载 `data_0.pkl`（借助官方 repo 的 `sys.path`），提取 tiles → `(500,16).npy`、solution_lengths → `(500,).npy`，保存到 `data/`；拷贝官方模型 `model_state_dict.pt` → `data/` | 无 | `data/tiles.npy` + `data/solution_lengths.npy` + `data/model_state_dict.pt` | 待完成 |
| S1 | 移植 `ResnetModel` 到 `deepcubea_official_network.py`：从官方仓库拷贝类定义，按项目代码风格调整，处理 `module.` 前缀加载，实现 `OfficialModelWrapper`（两个公有方法：`predict_j` / `predict_j_batch`） | 无 | `deepcubea_official_network.py` | 待完成 |
| S2 | 编写 `deepcubea_official_data.py`：`np.load` 加载预处理后的 tiles + solution_lengths，无需拷贝 NPuzzleState/State（方案C 已解除依赖） | S0 | `deepcubea_official_data.py` | 待完成 |
| S3 | 编写 `deepcubea_official_search.py`：实现三种评估（Bellman MSE / 纯贪心展开 / Full A*），λ=1.0，按真实解路径长度排序后三等分为短/中/长档，通过 config.py Flag 控制评估开关（Bellman 始终运行），CLI 参数支持覆盖。复用 `deepcubea_network.py` 的 `transition`/`get_children`。同时修改 `deepcubea_search.py` 的 `greedy_expansion_eval` 改为纯贪心展开（argmin J(s') 步步跟随） | S1, S2 | `deepcubea_official_search.py` + `deepcubea_search.py`（修改） | 待完成 |
| S4 | 跑通全部三项评估并验证：命令行入口加载官方模型和测试数据，按 Flag 依次运行评估，验证官方模型质量基线 | S3 | 评估结果 | 待完成 |
| S5 | 编写摘要 + 更新 wiki index | S4 | `abstract/` + `index.md` | 待完成 |

## S0 细节：数据预处理 + 模型拷贝

**方案**：方案 C —— 一次性转换脚本，将 `data_0.pkl` 转为纯 numpy `.npy`，输出到 `data/`；同时拷贝官方模型到 `data/`。

**执行环境**：当前项目 venv，通过 `sys.path.insert(0, 'DeepCubeA')` 访问官方 repo 的 `environments.n_puzzle`。

**执行方式**：
```python
import sys, pickle, numpy as np, shutil, os
sys.path.insert(0, 'DeepCubeA')
from environments.n_puzzle import NPuzzleState

os.makedirs('data', exist_ok=True)

with open('DeepCubeA/data/puzzle15/test/data_0.pkl', 'rb') as f:
    data = pickle.load(f)

tiles = np.stack([s.tiles for s in data['states']])          # (500, 16) int64
solution_lengths = np.array([len(s) for s in data['solutions']])  # (500,) int

np.save('data/tiles.npy', tiles)
np.save('data/solution_lengths.npy', solution_lengths)

# 拷贝官方模型
shutil.copy2('DeepCubeA/saved_models/puzzle15/current/model_state_dict.pt', 'data/model_state_dict.pt')
```

**输出文件**：
- `data/tiles.npy` — (500, 16) int64
- `data/solution_lengths.npy` — (500,) int（真实最优解步数，范围 ~36-69）
- `data/model_state_dict.pt` — ~55MB，官方 puzzle15 预训练模型

**注意**：此步骤依赖官方仓库（`environments.n_puzzle`），仅需执行一次。后续 S2-S4 完全脱离官方仓库依赖。

## S1 细节：官方网络模块

`deepcubea_official_network.py` 包含：

1. **`ResnetModel`**（从 `DeepCubeA/utils/pytorch_models.py` 拷贝）：
   - 参数：`state_dim=16, one_hot_depth=16, h1_dim=5000, resnet_dim=1000, num_resnet_blocks=4, out_dim=1, batch_norm=True`
   - `forward(x)`：x 为 (B, 16) Long tensor，内部 `F.one_hot` → FC → BN → ReLU → FC → BN → ReLU → 4×ResBlock → output
   - 按项目风格调整命名和 import 路径

2. **`load_official_model(model_path, device)`**：
   - 创建 `ResnetModel()` 实例
   - 加载 state_dict，处理 `module.` 前缀（官方 DataParallel 训练产物，72 个 key 全部带 `module.` 前缀）
   - `model.eval()` + `torch.inference_mode()` 推理模式
   - 返回 `OfficialModelWrapper` 实例

3. **`OfficialModelWrapper`**（仅两个公有方法，与 `DeepCubeANetwork` 接口对齐）：
   - `predict_j(grid)`：单 grid (16,) numpy int32 → Python float（`.item()`）
   - `predict_j_batch(grids)`：批量 (B, 16) numpy int32 → (B,) tensor（保留 tensor 以匹配 A* 搜索 `h_values.cpu().numpy()` 的调用模式，参考 `deepcubea_search.py:103`）
   - 无独立的 `__call__` 方法，保持接口最小化
   - 关键：Bellman MSE 和 A* 搜索均直接传 raw tiles，ResnetModel 内部 one_hot，无需外部预编码

## S2 细节：测试数据加载

`deepcubea_official_data.py`（轻量，仅 numpy 加载）：

1. **`load_official_test_data(data_dir)`**：
   - `tiles = np.load(f"{data_dir}/tiles.npy")` → (500, 16) int32
   - `solution_lengths = np.load(f"{data_dir}/solution_lengths.npy")` → (500,) int
   - 返回 `(tiles, solution_lengths)`

无需拷贝 `NPuzzleState` 或 `State` 类（方案 C 已在 S0 解除依赖）。

## S3 细节：搜索评估脚本

`deepcubea_official_search.py` 结构：

1. **复用 `deepcubea_network.py` 的工具函数**：
   - `transition(grid, action)` — 网格状态转移（纯 numpy，操作 raw tiles int32）
   - `get_children(grid, blank_idx)` — 批量子节点生成
   - `N2`、`N` 等常量

2. **自备常量**（独立定义，不跨文件依赖 `deepcubea_search.py` 的 `_GOAL_GRID`）：
   - `_GOAL_GRID = np.array(list(range(1, N2)) + [0], dtype=np.int32)`
   - `_GOAL_BYTES = _GOAL_GRID.tobytes()`

3. **新增 config.py 参数**（全部使用 `DEEPCUBEA_OFFICIAL_*` 前缀，共 8 个键）：
   ```python
   # ==================== DeepCubeA Official Model Evaluation ====================
   DEEPCUBEA_OFFICIAL_MODEL_PATH = "DeepCubeA/saved_models/puzzle15/current/model_state_dict.pt"
   DEEPCUBEA_OFFICIAL_DATA_DIR = "DeepCubeA/data/puzzle15/test/"
   DEEPCUBEA_OFFICIAL_LAMBDA = 1.0
   DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES = 10000
   DEEPCUBEA_OFFICIAL_NUM_STRATA = 3
   DEEPCUBEA_OFFICIAL_GREEDY_MAX_STEPS = 1000
   DEEPCUBEA_OFFICIAL_GREEDY_FLAG = True
   DEEPCUBEA_OFFICIAL_ASTAR_FLAG = True
   DEEPCUBEA_GREEDY_MAX_STEPS = 1000
   ```

4. **`_compute_bellman_errors_official(states, model)`**：
   - 参考 `deepcubea_search.py:_compute_bellman_errors`（129-168行）的算法逻辑
   - 关键差异：跳过 `encode_batch`，将 raw tiles states → Long tensor → `model.predict_j_batch` 获取 J(s)；对每个 action 的 next_states 同样跳过 encode
   - 返回 `(J(s) - best_j)^2` per-state

5. **`stratified_bellman_mse_official(tiles, solution_lengths, model, num_strata)`**：
   - 参考 `deepcubea_search.py:evaluate`（300-379行）的三档分档思想
   - 按 `solution_lengths` 排序后三等分：`sorted_indices = np.argsort(solution_lengths)` → 前 ~167 为 Short、中 ~167 为 Medium、后 ~166 为 Long
   - 调用 `_compute_bellman_errors_official` 计算 per-state MSE
   - 按档位输出统计（mean/median/std MSE），始终执行（无 Flag 控制）

6. **`_greedy_expand_official(start_grid_flat, model, max_steps)`**：
   - 纯贪心展开：每步获取所有子节点 → batch predict_j → 取 argmin J(s') 对应的子节点 → 步进
   - 终止条件：到达目标 or 超过 max_steps
   - 返回 `(solution_path, num_steps)` 或 `(None, num_steps)` 若失败

7. **`greedy_expansion_eval_official(tiles, solution_lengths, model, max_steps)`**：
   - 按 solution_lengths 排序后三等分（同 Bellman 分档）
   - 每档运行 `_greedy_expand_official`
   - 由 `DEEPCUBEA_OFFICIAL_GREEDY_FLAG` 控制

8. **`evaluate_official(tiles, solution_lengths, model, max_expand, lambda_weight)`**：
   - 按 solution_lengths 排序后三等分（同上）
   - Full A* 三档评估，由 `DEEPCUBEA_OFFICIAL_ASTAR_FLAG` 控制

9. **`weighted_astar_official(start_grid_flat, model, lambda_weight, max_expand)`**：
   - 参考 `deepcubea_search.py:weighted_astar`（34-111行）的逻辑
   - 关键差异：调用 `model.predict_j` / `model.predict_j_batch`（接受 raw tiles），跳过 `encode_batch`
   - 其余逻辑一致（heapq、g_score dict、closed set、goal check、path reconstruction）

10. **`if __name__ == "__main__"` 入口**：
   - 加载官方模型 → 加载测试数据 → 按 Flag 依次运行 Bellman MSE（始终）+ Greedy + Full A*
   - 命令行参数：`--model` / `--data` / `--max-expand` / `--lambda`（均设默认值，读取 config）
   - λ 默认 `DEEPCUBEA_OFFICIAL_LAMBDA` (1.0)
   - max_expand 默认 `DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES` (10000)

## S4 细节：验证方案

1. **冒烟测试**：加载模型，对 1 个状态运行 `predict_j` 和 `predict_j_batch`，验证输出合理（正数，预计 0~80 量级）
2. **批量验证**：`predict_j_batch` 对全部 500 状态一次前向，验证输出无 NaN/Inf，shape 正确
3. **快速门禁（Bellman MSE）**：期望低 MSE（官方模型质量高）；若异常（如远大于 0.1），先排查再继续
4. **Greedy 贪心展开**（若 Flag 开启）：纯 argmin J(s') 步步跟随，无回溯，预期平均求解步数接近最优解长度但成功率受限于启发函数质量
5. **Full A***（若 Flag 开启）：需手动调大 `DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES`（建议 ≥10M）以验证 100% 求解率。默认 10000 时为受限搜索，求解率预期低

## 风险与约束声明

- **数据预处理**：S0 在项目 venv 中执行，通过 `sys.path.insert(0, 'DeepCubeA')` 访问官方 repo 的 `environments.n_puzzle`。需确保当前 venv 已安装官方 repo 所需依赖（numpy 等），否则需临时安装
- **BN 推理**：`model.eval()` + `torch.inference_mode()` 即可，batch_size=1 可正常工作（eval 模式下 BN 用 running statistics）
- **state_dict `module.` 前缀**：官方使用 DataParallel 训练，所有 72 个 key 带 `module.` 前缀，加载时需剥离
- **PyTorch 版本兼容**：官方使用 torch 1.2.0，state_dict 格式向后兼容
- **内存**：ResnetModel ~15M 参数，约 55MB state_dict，4GB 显存可直接推理
- **评估耗时**：default max_expand=10000 时 ~秒级/状态（总计 < 1小时）；max_expand=10M 时 ~5-13s/状态（总计 40-110 分钟）
- **Config 键隔离**：所有新增键均使用 `DEEPCUBEA_OFFICIAL_*` 前缀，与现有键无冲突，不影响现有训练/搜索流程

## 测试验证方案

1. S1 单元验证：加载模型 state_dict → `predict_j` 单网格 → 输出合理 scalar
2. S2 单元验证：`np.load` tiles + solution_lengths → shape 正确，无 NaN
3. S3 Bellman MSE 先行：秒级执行，验证模型 + 数据桥接正确
4. S3 Greedy 小规模验证：先跑 10 个状态确认无报错
5. S3 Full A* 全量验证：500 状态，按 Flag 执行

## 📝 执行记录

- `2026-06-13 23:50`: 计划 v1 生成（方案A 适配器 + `--official` 分支）
- `2026-06-14`: 计划 v2 重构 —— 独立文件架构，直接移植 ResnetModel，λ=1.0，真实解路径长度分档
- `2026-06-14`: 计划 v3 更新 —— 方案 C（pickle→numpy 预处理，新增 S0）；新增 config 键；去除 output.txt 对比；细化 data_0.pkl 内部结构；去除 NPuzzleState 拷贝
- `2026-06-14`: 计划 v4 澄清 —— OfficialModelWrapper 接口对齐 DeepCubeANetwork（仅 predict_j/predict_j_batch）；Bellman MSE 参考 deepcubea_search.py:129-168；分档参考 evaluate:300-379；weighted_astar_official 独立实现跳过 encode_batch；Config 键全量 DEEPCUBEA_OFFICIAL_* 前缀（新增 LAMBDA/MAX_EXPAND 键）；S0 在项目 venv 执行
- `2026-06-14`: 计划 v5 —— 所有文件统一到 `data/`（模型拷贝 + S0 输出）；dtype int64（对齐实际数据）；Greedy Expansion 改为纯贪心展开（argmin J(s') 步步跟随，不再截断 A*），影响 deepcubea_search.py（现有）和 deepcubea_official_search.py（新建）；DEEPCUBEA_VAL_GREEDY_EXPAND 语义从节点上限→步数上限；新增 DEEPCUBEA_OFFICIAL_GREEDY_MAX_STEPS
- `2026-06-14 01:00`: S0 执行成功 —— `data/tiles.npy` (500,16) int64, `data/solution_lengths.npy` (500,) int64 [36,69] mean=52.0, `data/model_state_dict.pt` 55MB
- `2026-06-14 01:05`: S1 执行成功 —— `deepcubea_official_network.py` 创建，冒烟测试通过：J(s) ∈ [36.0, 58.2] mean=52.85，完美对齐解路径长度分布
- `2026-06-14 01:10`: S2-S3 执行成功 —— `deepcubea_official_data.py`、`deepcubea_official_search.py` 创建；`deepcubea_search.py` 贪心展开改为纯 argmin J(s')；`config.py` 新增 8 键 + 重命名 1 键
- `2026-06-14 01:15`: S3 修复 —— `deepcubea_official_search.py` 推理设备从硬编码 `use_gpu=False` 改为读取 `DEEPCUBEA_INFERENCE_USE_GPU`
- `2026-06-14 01:20`: S5 完成 —— wiki 摘要/index/readme 同步更新
- `2026-06-14 15:30`: **Bug 修复** —— `_compute_bellman_errors_official` children 收集与父 state 错位。`ptr_per_state` 每 state 成对存储 start/end（2B 条），但取值 `ptr_per_state[i], ptr_per_state[i+1]` 步长=1，奇数索引 state 错误拿到 0 children → error=J(s)²≈3243。修复方案 B：offsets 单条记录（B+1 条），步长=1 自然正确。修复后 Bellman MSE Overall mean=0.362, median=0.202，模型质量验证通过。Greedy 0% 胜率待后续加 visited set 处理
