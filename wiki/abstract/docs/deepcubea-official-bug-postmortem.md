---
abstract_name: DeepCubeA 官方模型评估 Bug 复盘
source_contents:
  - "deepcubea_official_search.py"
  - "deepcubea_search.py"
  - "deepcubea_official_network.py"
  - "deepcubea_network.py"
  - "deepcubea_official_data.py"
  - "data/tiles.npy"
dependencies:
  - "wiki/abstract/deepcubea-official-model-evaluation.md"
  - "wiki/abstract/deepcubea-search.md"
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/abstract/deepcubea-official-repo.md"
created_at: 2026-06-14 18:45
updated_at: 2026-06-14 18:45
---
# 摘要：DeepCubeA 官方模型评估 Bug 复盘

## 核心结论与关键信息

- **总计发现 3 个 bug**，均位于 `deepcubea_official_search.py`。其中 2 个导致方案 C（A* 评估）胜率 0%，1 个导致方案 A（Bellman MSE）数值错误（MSE 1411 → 0.362）。
- **3 个 bug 属同一模式**：从 `deepcubea_search.py` 移植代码到官方模型时，索引/指针/类型参数未对齐新上下文。
- **最隐蔽的 bug**（`_GOAL_BYTES` dtype）：相同数值、不同 dtype → `tobytes()` 永远不等 → A* 永远无法识别目标状态。无报错、无异常值，只有 0% 胜率这个间接信号。
- **预防策略**：跨模块共享的常量（目标状态、配置键前缀）统一 dtype 和存储格式；搜索模块的 idx/ptr 参数传递应加防御性断言。

## 内容概述

> 对 `deepcubea_official_search.py` 中 3 个已修复 bug 的全面复盘：症状、根因、修复、为何难以检测、通用教训。

## Bug 总览

| # | Bug | 位置 | 症状 | 影响范围 | 严重度 |
|---|---|---|---|---|---|
| 1 | ptr_per_state 索引偏移 | `_compute_bellman_errors_official` | Bellman MSE=1411（合理值 ~0.36） | 方案 A | **高** |
| 2 | blank_idx 硬编码 0 | `weighted_astar_official` | A* 第一步生成错误子节点 | 方案 C + 方案 B | **高** |
| 3 | _GOAL_BYTES dtype 不匹配 | `weighted_astar_official` + `_greedy_expand_official` | A*/贪心永远无法识别目标 | 方案 C + 方案 B | **高** |

## Bug 1: ptr_per_state 索引偏移

### 症状

Bellman MSE Overall mean≈1411（>>0.36），Odd-index 状态的 children 数为 0 → error=J(s)²。

### 根因

```python
# 错误：成对存储 start/end 标记
ptr_per_state.append(len(all_next_states))  # 每次推 start
ptr_per_state.append(len(all_next_states))  # 每次推 end
# → 2×B 条记录

# 但取值用步长=1
start, end = ptr_per_state[i], ptr_per_state[i + 1]  # i=0 → (0, 0) OK
# i=1 → (0, 3) WRONG — 拿到的 start 实际是上一个状态的 end
```

| i | 预期 start/end | 实际取值 | children 数 |
|---|---|---|---|
| 0 | (0, 3) | (0, 0) | 0 ❌ |
| 1 | (3, 6) | (0, 3) | 3 ✅ |
| 2 | (6, 9) | (3, 6) | 0 ❌ |

模式：奇数正常，偶数拿到 0 children → error = J(s)²。

### 修复

改用单条 offset 记录（B+1 条），步长=1 自然正确：

```python
offsets = [0]
for ...:
    ...
    offsets.append(len(all_next_states))
# offsets = [0, 3, 6, 9, ...]  — B+1 条
start, end = offsets[i], offsets[i + 1]  # 步长=1 始终正确
```

### 教训

**成对存储必须成对取值（步长=2），单条 offset 模式天然防错。**

---

## Bug 2: blank_idx 硬编码 0

### 症状

方案 C 全部状态展开到 10,000 上限无求解，胜率 0%。

### 根因

```python
# weighted_astar_official (修复前)
g_score = {start_bytes: (0, None, None, 0)}
#                                              ↑
#                                    硬编码为 0，而非空格实际位置
```

对比正确实现 `weighted_astar`：
```python
start_blank_idx = int(np.where(start_grid_flat == 0)[0][0])
g_score = {start_bytes: (0, None, None, start_blank_idx)}
```

`get_children(grid, blank_idx)` 用 `blank_idx` 定位空格 → 计算行列 → 判断合法移动方向。若空格在位置 10（棋盘中部），但传入 0（左上角），则：
- 生成的子节点交换位置 0 的 tile 而非位置 10 的空格 → 无效状态
- 上/左方向被误判为越界（实际合法），下/右方向误判为合法（实际可能越界）

**子节点的 `blank_idx` 是正确的**（因为 `new_blank_idx` 来自正确计算），但子节点状态本身已经错了。

### 修复

```python
start_blank_idx = int(np.argmin(start_grid_flat))
g_score = {start_bytes: (0, None, None, start_blank_idx)}
```

同时补上 `closed` set + `tiebreaker` + goal 短路判断（对齐 `weighted_astar` 的结构完整性）。

### 教训

**"显而易见"的默认值（0 表示起始位置）在有实际语义的坐标系中极其危险。任何位置/索引参数必须从实际数据计算，禁止硬编码。**

---

## Bug 3: _GOAL_BYTES dtype 不匹配

### 症状

同 Bug 2：方案 C 胜率 0%，全部展开到上限。

但关键是：**即使 Bug 2 修复后症状完全不变**——因为 A* 即使正确遍历了搜索空间，到达目标状态时也无法识别。

### 根因

```python
# deepcubea_official_search.py (修复前)
_GOAL_GRID = np.array([1, 2, ..., 15, 0], dtype=np.int32)  # 68 bytes
_GOAL_BYTES = _GOAL_GRID.tobytes()

# 但所有状态都是 int64：
# - _generate_test_states → np.array(states, dtype=np.int64)  # 136 bytes
# - load_official_test_data → tiles.npy, dtype=int64            # 136 bytes
# - get_children → grid.copy() 保留 int64 dtype                 # 136 bytes
```

实测验证：
```
int32_goal.tobytes() == int64_goal.tobytes()  →  False
```

A* 搜索中所有目标判断均失败：
```python
if start_bytes == _GOAL_BYTES:   # 永远 False
if child_bytes == _GOAL_BYTES:   # 永远 False
if current_bytes == _GOAL_BYTES:  # 永远 False
```

A* 永远无法识别目标状态 → 无解退出。

### 修复

```python
_GOAL_GRID = np.array([1, 2, ..., 15, 0], dtype=np.int64)  # 136 bytes
```

### 为什么难以检测

- 没有报错、崩溃、异常数值
- bytes 比较静默失败
- `len(bytes_a) != len(bytes_b)` 时 Python 直接返回 False（不抛异常）
- 唯一的信号是 0% 胜率，被 Bug 2 同时掩盖

### 教训

**bytes 比较对 dtype 极其敏感。`ndarray.tobytes()` 返回的字节序列包含 dtype 信息（字节宽度），不同 dtype 的相同数值产生不同字节序列。跨模块共享目标状态常量时，必须保证 dtype 一致性。备选方案：`np.array_equal()` 用于关键判断（慢但 dtype 不敏感），或统一使用同一模块导出的常量。**

---

## 共性分析

### Bug 模式

三个 bug 都源于同一操作模式：**从 `deepcubea_search.py` 参考代码移植到 `deepcubea_official_search.py` 时，关键参数未根据新上下文调整**。

| Bug | 参考代码 (deepcubea_search.py) | 移植代码 (official) | 差异 |
|---|---|---|---|
| ptr_per_state | 无（参考代码用 vectorized 实现） | 手动循环 + 成对存储 | 数据结构不同 |
| blank_idx | `int(np.where(...))[0]` | `0` | 遗忘了初始化步骤 |
| _GOAL_BYTES | int32（与 int32 状态一致） | int32（但状态是 int64） | dtype 上下文变化 |

### 测试盲区

- **方案 A（Bellman MSE）** 是正确的 → 暗示"模型 OK，搜索函数应该也 OK" → 实际上方案 A 只用数值计算，不涉及 bytes 比较和搜索循环
- **无单元测试**覆盖 `weighted_astar_official` 的单一已知路径 → 正向游走生成 → A* 求解的闭合验证
- **第一个 bug（ptr_per_state）有可见的数值异常**（MSE=1411 vs 预期 ~0.3），但后两个 bug 的症状完全重叠（都是 0% 胜率）

---

## 预防策略

1. **目标状态常量集中管理**：`_GOAL_GRID` / `_GOAL_BYTES` 应从 `deepcubea_network` 或专用 constants 模块统一导出，避免多份定义且 dtype 不一致。

2. **bytes 比较加防御断言**：
   ```python
   assert len(child_bytes) == len(_GOAL_BYTES), \
       f"dtype mismatch: {len(child_bytes)} vs {len(_GOAL_BYTES)}"
   ```
   一行断言即可在第一时间暴露 Bug 3。

3. **搜索模块的冒烟测试**：正向游走 N 步 → A* 求解 → 验证返回路径能回到目标状态。这是闭合验证，任何 seq 中的 bug 都会暴露。

4. **索引/指针参数禁止硬编码**：`blank_idx`、`offset`、`start/end` 等语义参数必须从实际数据计算或显式传递。

5. **复制粘贴移植时逐行对比 diff**：`weighted_astar` 与 `weighted_astar_official` 应在代码审查时逐行对账，差异点必须有注释说明原因。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-official-model-evaluation.md`（bug 发现与修复记录）、`wiki/abstract/deepcubea-search.md`（参考实现）
- **下游被依赖**：无
- **变更扩散评估**：低（复盘报告，不修改代码）
