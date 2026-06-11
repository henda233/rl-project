---
plan_name: DeepCubeA A* 搜索推理性能优化
related_request: "wiki/request/deepcubea-inference-device-config.md"
status: completed
created_at: 2026-06-11 21:00:00
---
# 执行计划：DeepCubeA A* 搜索推理性能优化

## 背景与目的

当前 `weighted_astar` 搜索每展开一个节点的热路径存在多处低效点：

1. `transition()` 每次调用 `np.where(grid_flat == 0)` 找空格 —— 每节点展开 4 次
2. 4 次独立 `transition()` 调用，重复找空格、各创建 grid copy
3. `closed` set + `came_from` dict 双重哈希查找，且遇到已有路径时丢弃更短路径
4. tuple ↔ numpy 来回转换（16 个 int 的 Python 迭代），每节点一次
5. 小 batch（≤4）GPU 推理 kernel launch 开销大于计算收益
6. `torch.no_grad()` 未使用更高效的 `torch.inference_mode()`

**目标**：从代码层面消除上述瓶颈，提升 A* 搜索吞吐量。

## 内容

涉及 3 个文件的修改：

| 文件 | 改动范围 |
|------|---------|
| `config.py` | 新增 `DEEPCUBEA_INFERENCE_USE_GPU` 参数 |
| `deepcubea_network.py` | `get_children()` 批量状态转移；`torch.inference_mode()` |
| `deepcubea_search.py` | `weighted_astar()` 重构数据结构和展开逻辑；`load_model()` 设备可选 |

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | `config.py` 新增 `DEEPCUBEA_INFERENCE_USE_GPU = False` | 无 | `config.py` | 已完成 |
| `S2` | `deepcubea_network.py` 新增 `get_children(grid, blank_idx)` 批量状态转移函数，一次调用返回所有合法子节点；`predict_j`/`predict_j_batch` 中 `torch.no_grad()` → `torch.inference_mode()` | 无 | `deepcubea_network.py` | 已完成 |
| `S3` | `deepcubea_search.py` 重构 `weighted_astar()`：<br>1. 用 `g_score` dict 替代 `came_from`（统一存储 g/parent/action/blank_idx）<br>2. 用 `get_children()` 替代 4 次 `transition()` 循环<br>3. 新路径 g 更优时覆盖旧路径（修复长路径覆盖短路径问题）<br>4. 保留 `closed` set 用于展开节点去重 | `S2` | `deepcubea_search.py` | 已完成 |
| `S4` | `deepcubea_search.py` `load_model()` 新增 `use_gpu` 参数，默认读取 `DEEPCUBEA_INFERENCE_USE_GPU` | `S1` | `deepcubea_search.py` | 已完成 |
| `S5` | `deepcubea_search.py` 状态表示改用 `bytes`：<br>1. `grid.tobytes()` 替代 `tuple(int(x) for x in child_grid)` 作为哈希键<br>2. `np.frombuffer(state_bytes, dtype=np.int32).copy()` 替代 `np.array(list(current), dtype=np.int32)` 还原数组<br>3. `GOAL_STATE` 改为 `_GOAL_BYTES` 常量；同步更新 `deepcubea_search_smoke_test.py` | `S3` | `deepcubea_search.py`、`deepcubea_search_smoke_test.py` | 已完成 |
| `S6` | 运行 `deepcubea_search_smoke_test.py` 验证正确性 | `S5` | 测试全部通过 | 已完成 |
| `S7` | 更新 WIKI（摘要 + index） | `S6` | `wiki/abstract/deepcubea-search.md`、`wiki/index.md` | 已完成 |

### 各步骤详细说明

#### S1: 推理 GPU 配置

```python
# config.py 追加
DEEPCUBEA_INFERENCE_USE_GPU = False  # A* 搜索时是否使用 GPU（小 batch 建议 CPU）
```

与训练参数 `DEEPCUBEA_USE_GPU` 完全解耦。

#### S2: 批量状态转移 + inference_mode

新增 `get_children(grid, blank_idx)` — 一次调用完成当前节点的所有合法子节点生成：

```python
def get_children(grid, blank_idx):
    """返回 [(child_grid, action, new_blank_idx), ...] 仅合法子节点。
    调用方保证 grid 不会被外部修改。
    """
    r, c = blank_idx // N, blank_idx % N
    children = []
    for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            new_idx = nr * N + nc
            child = grid.copy()
            child[blank_idx] = grid[new_idx]
            child[new_idx] = 0
            children.append((child, action, new_idx))
    return children
```

与原 `transition()` 对比：
- 消除 4 次 `np.where`（空格位置由调用方传入）
- 4 次函数调用 → 1 次
- `_DR/_DC` 查表开销在循环内消除

同时将 `predict_j`/`predict_j_batch` 中的 `torch.no_grad()` 替换为 `torch.inference_mode()`。

#### S3: 重构 weighted_astar 数据结构

**现状**：

```python
came_from = {state: (parent, action, g)}   # 记录所有访问节点
closed = set()                               # 记录已展开节点
# 每个 child 两次查找: child in closed or child in came_from
# 问题: child 已在 came_from 时直接跳过，不比较 g 值 → 可能丢失更短路径
```

**改为**：

```python
g_score = {state: (g, parent, action, blank_idx)}  # 最佳已知 g + 路径信息
closed = set()                                       # 已展开节点
# 每个 child: child in closed → skip
#            child in g_score and new_g >= g_score[child][0] → skip
#            否则 → 记录/更新 g_score，加入 open set
```

改进点：
- `came_from` + `closed` → `g_score` + `closed`，语义更清晰
- 支持更短路径覆盖（修复隐式 bug）
- `blank_idx` 随状态存储，展开时无需 `np.where`

#### S4: load_model 设备可选

```python
def load_model(model_path=None, use_gpu=None):
    if use_gpu is None:
        use_gpu = DEEPCUBEA_INFERENCE_USE_GPU
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    # ... 其余不变
```

#### S5: bytes 状态表示

**现状**（每节点展开一次，每个 child 一次）：

```
tuple → list → np.array     # 展开节点 L60: tuple(int(x) for x in child_grid) → L69 还原时 np.array(list(current))
numpy → generator → tuple   # 每个 child L69: tuple(int(x) for x in child_grid)
```

**改为**：

```python
# 哈希键: grid.tobytes() → bytes (64 bytes for 16×int32)
# 还原: np.frombuffer(state_bytes, dtype=np.int32).copy()
```

对比：
- `grid.tobytes()` 是内存直接拷贝，无 Python 迭代
- `np.frombuffer().copy()` 是指针+内存拷贝，无 list 中间对象
- bytes 哈希速度远快于 tuple（C 级别 vs Python 级别）

#### S6: 验证

- `deepcubea_search_smoke_test.py` 全部通过
- 可选：编写简单的计时对比（优化前后同模型同测试状态）

## 预估加速比

| 优化项 | 累积加速 | 说明 |
|--------|---------|------|
| S2: 批量 transition + 消除 np.where | ~30% | 每节点省 3 次 np.where + 3 次函数调用 |
| S3: g_score 统一 + blank 跟踪 | ~10% | 减少查找，支持短路径覆盖 |
| S5: bytes 状态表示 | ~15% | 消除 tuple↔numpy Python 迭代 |
| S1+S4: CPU 推理 + S2 inference_mode | ~10% | 消除小 batch GPU 开销 |
| **合计** | **~50-65%** | 端到端搜索时间约缩短一半 |

> 注：实际加速比取决于状态空间大小和模型复杂度。上述为基于代码分析的保守估计。

## 风险与约束声明

- **低风险**：不改网络结构、权重、A* 算法框架、评估逻辑
- S3 中 `g_score` 支持更短路径覆盖是**行为变更**——可能改变搜索结果（路径更优/更短），但这是正向修复
- S5 bytes 表示对 N≤6 的棋盘适用（int32 × N² 在 bytes 安全范围内）
- `deepcubea_train.py` 不受影响（训练使用独立的数据生成/加载链路，不依赖搜索模块）
- 每步完成后可独立验证，出问题易于定位

## 测试验证方案

1. **单元验证**：`deepcubea_search_smoke_test.py` 所有测试通过
2. **功能验证**：`deepcubea_search_test.py` 评估流程正常完成
3. **正确性验证**：随机抽取 10 个状态，新旧实现求解路径一致（或新实现路径更短）
4. **性能验证**：同模型同测试状态下，新旧实现搜索耗时对比

## 📝 执行记录

- `2026-06-11 21:00`: 计划已生成（覆盖全部 6 个优化点）
- `2026-06-11 21:15`: S1-S7 全部执行完毕。smoke test 全部通过。
