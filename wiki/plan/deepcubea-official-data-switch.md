---
plan_name: 官方 DeepCubeA A* 搜索测试数据来源开关
related_request: "wiki/request/deepcubea-official-data-switch.md"
status: completed
created_at: 2026-06-14 17:00
---
# 执行计划：官方 DeepCubeA A* 搜索测试数据来源开关

为 `deepcubea_official_search.py` 增加测试数据来源切换能力：
- `DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA=True` → 加载官方 `tiles.npy` + `solution_lengths.npy`（保持现有行为）
- `DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA=False` → 随机游走生成测试数据 + K 代理分层

## 内容

### 1. config.py 新增 4 参数

追加到 DeepCubeA Official Model Evaluation section 末尾。

### 2. deepcubea_official_search.py 新增 `_generate_test_states`

从 `deepcubea_search.py` 移植，逻辑一致：从 `_GOAL_GRID` 出发，每状态独立随机 t ∈ [T_MIN, T_MAX]，执行 t 步随机动作生成可解状态。返回 `(states_array, k_values_array)`。

### 3. `__main__` 分支逻辑

- `True` 分支：`load_official_test_data()` → `tiles, solution_lengths`，`stratify_label = "Len"`
- `False` 分支：`_generate_test_states()` → `tiles, k_values`（作为 sol_lens 传入），`stratify_label = "K"`

### 4. 三个评估函数适配

函数签名 `solution_lengths` 参数语义不变（传入什么分什么），仅输出表头从硬编码 `"Len Range"` 改为参数 `stratify_label + " Range"`。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增 4 参数 | 无 | `config.py` | 已完成 |
| `S2` | deepcubea_official_search.py 新增 `_generate_test_states` + `__main__` 分支 + 表头参数化 | `S1` | `deepcubea_official_search.py` | 已完成 |
| `S3` | 冒烟测试：True 分支（官方数据）与 False 分支（生成数据）各运行一次 | `S2` | 终端输出验证 | 已完成 |
| `S4` | 更新 wiki 摘要 + index | `S3` | `wiki/abstract/deepcubea-official-model-evaluation.md`、`wiki/index.md` | 已完成 |

## 风险与约束声明

- **低风险**：改动局限在 `deepcubea_official_search.py` 和 `config.py`，不触碰模型加载和搜索核心逻辑
- **K 代理偏差**：生成模式下表头标注 "K Range" 提醒用户 K 是打乱步数而非真实最优距离
- **默认值不变**：`USE_OFFICIAL_DATA=True` 保证现有调用行为完全不变

## 测试验证方案

1. `True` 分支：`uv run python deepcubea_official_search.py` → 输出应与当前一致（"Len Range" 表头、500 状态）
2. `False` 分支：修改 config 或传参后运行 → 表头 "K Range"、生成状态数 = `DEEPCUBEA_OFFICIAL_NUM_TEST_STATES`
3. 确认 `False` 模式下 K 值 ∈ [T_MIN, T_MAX] 且状态可解（均为 goal 可达状态）

## 📝 执行记录

- `2026-06-14 17:00`: 计划已生成
- `2026-06-14 17:30`: 全部完成 —— S1 config 新增 4 参数；S2 新增 `_generate_test_states` + `__main__` 分支 + 表头参数化；S3 冒烟测试双分支通过；S4 wiki 摘要/计划/index 同步更新
