---
plan_name: DeepCubeA 官方模型适配 24-Puzzle
related_request: "wiki/request/huarongdao-n5-official-model.md"
status: waiting
created_at: 2026-06-14 20:00:00
---
# 执行计划：DeepCubeA 官方模型适配 24-Puzzle

## 背景与目的

将当前项目从 15-puzzle（N=4）切换为 24-puzzle（N=5），加载官方预训练 ResnetModel（`DeepCubeA/saved_models/puzzle24/current/model_state_dict.pt`）并运行三项评估（Bellman MSE / 贪心展开 / 加权 A*）。

不做训练，仅评估。参数沿用 15-puzzle 的相同值。`ResnetModel` 的维度参数从 `HUARONGDAO_N` 推导。

## 核心原则

- **一个分支一个问题**：当前分支硬编码 N=5，不保留 N=4 兼容
- **最小变更**：`deepcubea_network.py` / `deepcubea_search.py` / `deepcubea_utils.py` / `env_digital_huarongdao.py` 等已是 N 泛型，无需修改
- **仅评估**：不涉及 `deepcubea_train.py` / `deepcubea_generate_data.py` 的修改

## 需修改的文件（共 2 个）

### 1. `config.py`

| 键 | 旧值 | 新值 | 说明 |
|---|---|---|---|
| `HUARONGDAO_N` | `4` | `5` | N=4→N=5 |
| `DEEPCUBEA_OFFICIAL_MODEL_PATH` | `"data/model_state_dict.pt"` | `"DeepCubeA/saved_models/puzzle24/current/model_state_dict.pt"` | 指向 puzzle24 预训练模型 |
| `DEEPCUBEA_OFFICIAL_DATA_DIR` | `"data/"` | `"data/puzzle24/"` | puzzle24 测试数据独立目录 |

其余参数不变（`T_MAX=500`, `MAX_EXPAND_NODES=10000`, `LAMBDA=1.0` 等）。

### 2. `deepcubea_official_network.py`

| 位置 | 变更内容 |
|---|---|
| `ResnetModel.__init__` 默认参数 | `state_dim=16, one_hot_depth=16` → 从 `config.HUARONGDAO_N` 推导：`N2 = HUARONGDAO_N * HUARONGDAO_N`，`state_dim=N2, one_hot_depth=N2` |
| `load_official_model()` | 创建 `ResnetModel` 时传入 `state_dim=N2, one_hot_depth=N2` |
| docstring 中的 `(16,)` / `(B, 16)` | 更新为 `(N2,)` / `(B, N2)` 描述 |

新增 import：
```python
from config import HUARONGDAO_N
N2 = HUARONGDAO_N * HUARONGDAO_N
```

## 数据预处理（S0）

puzzle24 测试数据：`DeepCubeA/data/puzzle24/test/data_0.pkl`

结构与 puzzle15 一致：`{'states': List[NPuzzleState](496), 'solutions': List[List[int]](496), 'num_nodes_generated': List[int], 'times': List[float]}`

- tiles dtype: `uint8`（值范围 0-24 适合 uint8）
- 解路径长度：64-110，mean 89.4
- **dtype 一致性**：必须转为 `int64`（非 `int32`），因为 `deepcubea_official_search.py` 中 `_GOAL_GRID` 为 `int64`，`tobytes()` 比较对 dtype 敏感（历史 Bug 教训）

转换脚本（一次性执行）：
```python
import sys; sys.path.insert(0, "DeepCubeA")
import pickle, numpy as np
from pathlib import Path

d = pickle.load(open("DeepCubeA/data/puzzle24/test/data_0.pkl", "rb"))
tiles = np.array([s.tiles.astype(np.int64) for s in d["states"]])
solution_lengths = np.array([len(s) for s in d["solutions"]], dtype=np.int64)

Path("data/puzzle24").mkdir(parents=True, exist_ok=True)
np.save("data/puzzle24/tiles.npy", tiles)
np.save("data/puzzle24/solution_lengths.npy", solution_lengths)
print(f"Saved: {tiles.shape}, lengths [{solution_lengths.min()}, {solution_lengths.max()}]")
```

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S0` | 数据预处理：转换 puzzle24 pickle → numpy | 无 | `data/puzzle24/tiles.npy`, `data/puzzle24/solution_lengths.npy` | 待完成 |
| `S1` | 修改 `config.py`：HUARONGDAO_N=5 + 官方路径 | 无 | `config.py` | 待完成 |
| `S2` | 修改 `deepcubea_official_network.py`：ResnetModel 维度从 config 推导 | 无 | `deepcubea_official_network.py` | 待完成 |
| `S3` | 冒烟测试：加载模型 + 单状态 J(s) 预测 | `S1`, `S2` | 终端输出验证 | 待完成 |
| `S4` | 方案 A — Bellman MSE 评估 | `S0`, `S3` | 终端输出 | 待完成 |
| `S5` | 方案 C — 加权 A* 评估 | `S0`, `S3` | 终端输出 | 待完成 |

> 方案 B（贪心展开）默认不启用（`DEEPCUBEA_OFFICIAL_GREEDY_FLAG = False`），如有需要可单独开启。

## 风险与约束声明

1. **模型载入**：puzzle24 模型 62MB（vs puzzle15 55MB），使用 DataParallel 训练（带 `module.` 前缀），`load_official_model` 已有剥离逻辑，应直接兼容
2. **dtype 一致性**：puzzle24 tiles 原始为 `uint8`，转换为 `int32`；`OfficialModelWrapper` 内部转为 `int64` 喂给 ResnetModel，沿途不会溢出
3. **A* 展开节点上限**：puzzle15 评估时展开量级为百万级且 100% 求解；puzzle24 解路径更长（64-110 vs 36-69），`MAX_EXPAND_NODES=10000` 可能不足。若胜率为 0%，需调大此值
4. **`_GOAL_BYTES` dtype**：已在 S0 中处理 —— tiles 转换为 `int64` 与 `_GOAL_GRID` dtype 一致，避免 bytes 比较失败（历史 Bug：int32 vs int64 tobytes() 永远 False）
5. **不涉及训练**：`HUARONGDAO_N=5` 后若误运行 `deepcubea_train.py`，网络结构正确但无意义（此分支不训练）。建议不改训练相关代码，仅在计划笔记中记录

## 测试验证方案

### S3 冒烟测试通过标准

取 `tiles[0]` 和 `tiles[-1]` 各测一次，避免单个状态异常。

| 条件 | 判定 |
|---|---|
| `25 ≤ J(s) ≤ 140` | 通过（覆盖解路径 64-110 的合理波动范围） |
| 负值或 NaN | 失败 —— 架构/数据 dtype 不匹配，立刻排查 |
| `> 200` | 标记为可疑，不阻断后续步骤 |
| 单次推理耗时 < 1s | 通过 |

### S4 Bellman MSE

全部 496 状态跑通，Overall MSE 数值合理（puzzle15 约 0.36）。

### S5 加权 A*

三档分层评估完成，观察胜率与展开节点数。若 `max_expand=10000` 胜率为 0%，逐步调大至可求解。

## 代码审查确认

- `ResnetModel()` 唯一调用点在 `load_official_model()` 第 104 行，不传参数完全依赖默认值，改默认参数自动覆盖，无其他调用者
- `OfficialModelWrapper.predict_j` / `predict_j_batch` 的 docstring 硬编码 `(16,)` / `(B, 16)`，需顺手更新为 `(N2,)` 避免误导

📝 执行记录

- `2026-06-14 20:00`: 计划已生成，待用户确认后执行
- `2026-06-14 20:30`: 用户确认计划，补充 S3 冒烟通过标准 + 代码审查确认（ResnetModel 调用点 + docstring 更新）
