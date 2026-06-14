---
request_name: 官方 DeepCubeA A* 搜索测试数据来源开关
created_at: 2026-06-14 17:00
---

# 需求：官方 DeepCubeA A* 搜索测试数据来源开关

## 需求描述

当前 `deepcubea_official_search.py` 的 A* 搜索硬编码使用官方测试数据（`data/tiles.npy` + `data/solution_lengths.npy`）。
增加一个 `config.py` 布尔参数控制是否使用官方数据：
- `True`（默认）：使用官方测试数据（现有逻辑）
- `False`：按照 `deepcubea_search.py` 的 `_generate_test_states` 方式，随机 walk 从 goal 生成测试数据

## 具体规格

### Config 新增参数（4 个，追加到 DeepCubeA Official Model Evaluation section）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA` | `True` | 是否使用官方测试数据 |
| `DEEPCUBEA_OFFICIAL_T_MIN` | `3` | 随机游走最小步数（False 时生效） |
| `DEEPCUBEA_OFFICIAL_T_MAX` | `500` | 随机游走最大步数（False 时生效） |
| `DEEPCUBEA_OFFICIAL_NUM_TEST_STATES` | `500` | 生成测试状态数（False 时生效） |

### deepcubea_official_search.py 修改

1. 新增 `_generate_test_states` 函数（从 `deepcubea_search.py` 移植，方案 A：每个 state 单独随机 t）
2. `__main__` 入口分支：根据 `DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA` 选择数据来源
3. 生成数据时：`k_values` 作为代理分层值，表头 "Len Range" → "K Range"
4. 随机种子：复用 `DEEPCUBEA_VAL_SEED`（12345）
5. 三个评估函数适配：`solution_lengths` → 通用 `stratify_values`，标签动态输出
