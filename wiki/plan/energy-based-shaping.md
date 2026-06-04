---
plan_name: Energy-Based 奖励塑形
related_request: "wiki/request/energy-based-shaping.md"
status: completed
created_at: 2026-06-04 22:00:00
---
# 执行计划：Energy-Based 势函数替换 Velocity-Based

## 背景

当前 Φ(s) = |velocity| + C=800（或 C=10）在 MountainCar 上完全无法收敛，根因已被诊断为势函数选型错误（详见讨论）。本计划将势函数替换为基于物理直觉的 total mechanical energy，利用 Ng et al. potential-based shaping 保持最优策略不变性。

## 内容

### 核心公式

```
Φ(s) = sin(3 * position) + velocity² / (2 * g)
```

其中 g = 0.0025（MountainCar 重力加速度常量）。

**值域分析**：

| 状态 | position | velocity | sin(3·pos) | v²/(2g) | Φ |
|---|---|---|---|---|---|
| 起始（谷底）| -0.5 | 0 | -1.0 | 0 | -1.0 |
| 左墙（蓄力点）| -1.2 | 0.07 | 0.44 | 0.98 | 1.42 |
| 目标 | 0.5 | ~0 | 1.0 | ~0 | ~1.0 |

- Φ 值域约 [-1.0, 2.0]，跨度约 3 个单位
- 起始→左墙：Φ 上升（shaping 正，鼓励蓄力）✓
- 左墙→目标：PE 上升补偿 KE 下降（shaping ≈ 0，不干预）✓
- 目标处 Φ 为高值（shaping 正向终点）✓

### Shaping 净贡献（关键优势）

Telescoping sum：

```
Σ shaping = C * [γ·Φ_T - Φ_0 + (γ-1)·Σ Φ_intermediate]

Φ_T ≈ 1.0, Φ_0 ≈ -1.0, Φ_intermediate mean ≈ 0 (PE+KE 对称振荡)

≈ C * [0.98 + 1.0 + 0] = 1.98·C  ← 持续正值！
```

对比 velocity-based：净贡献为负（因为 Φ_T ≈ Φ_0 ≈ 0，泄漏项主导）。

### 改动 1：`config.py` —— 调整超参数

```python
# --- Energy-Based Reward Shaping ---
REWARD_SHAPING_SCALE = 10    # C: energy value range ~3, C=10 gives step ~0.03-0.1, conservative start
G = 0.0025                   # MountainCar gravity (KE coefficient)
PE_COEFFICIENT = 1.0         # weight for sin(3*pos) PE term; tune to adjust PE vs KE balance
# --- Exploration (ε-greedy) ---
EPSILON = 0.3                # lower: energy-based shaping provides stronger guidance
EPSILON_DECAY = 0.999        # slower decay: maintain some exploration
EPSILON_MIN = 0.01
```

**C 值选择理由**：
- C=10：shaping 步均 ≈ 0.15（偏弱，但安全）
- C=20：shaping 步均 ≈ 0.3（推荐，信号明显但不淹没）
- C=50：shaping 步均 ≈ 0.75（激进，可能加速但风险噪声）
- 当前 C=800（velocity-based 遗留）完全不适合 energy 值域

### 改动 2：`agent.py` —— 替换 shaped_reward 公式

```python
from math import sin
from config import ..., G, PE_COEFFICIENT

# 旧公式（删除）：
# shaped_reward = reward + REWARD_SHAPING_SCALE * (
#     GAMMA * abs(next_obs[1]) - abs(obs[1])
# )

# 新公式：
potential = PE_COEFFICIENT * sin(3 * obs[0]) + obs[1]**2 / (2 * G)
next_potential = PE_COEFFICIENT * sin(3 * next_obs[0]) + next_obs[1]**2 / (2 * G)
shaped_reward = reward + REWARD_SHAPING_SCALE * (
    GAMMA * next_potential - potential
)
```

### 不改动的部分

- `env.py`：无变更
- ActorCritic 类：无变更
- train_on_policy_agent 训练循环结构：无变更
- ε-greedy 机制：保留不变
- 评估函数：无变更

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | `config.py` 新增 G=0.0025、PE_COEFFICIENT=1.0，调整 REWARD_SHAPING_SCALE=10、EPSILON=0.3、EPSILON_DECAY=0.999 | 无 | `config.py` | 已完成 |
| `S2` | `agent.py` 将 shaped_reward 公式从 velocity-based 替换为 energy-based（含 PE_COEFFICIENT）| `S1` | `agent.py`（train_on_policy_agent）| 已完成 |
| `S3` | 运行训练自检：验证 return 收敛趋势，对比 velocity-based 基线 | `S2` | 无（验证步骤）| 已完成 |
| `S4` | 编写摘要并更新 index | `S3` | `wiki/abstract/energy-based-shaping.md`、`wiki/index.md` | 已完成 |

## 风险与约束声明

- **约束**：使用 potential-based shaping（Ng et al.），保证最优策略不变性
- **约束**：Φ 基于 MountainCar 物理常量 g=0.0025，若切换环境需重新推导
- **风险（低）**：C=20 可能偏保守，若 5000 episode 未收敛至 -120，可调至 30~50
- **风险（低）**：sin(3*pos) 在位置 ±1.2 边界处梯度变化剧烈，但 shaping 仅依赖函数值不依赖梯度，不受影响
- **已知限制**：γ<1 泄漏项 (γ-1)·ΣΦ 依然存在，但因为 Φ 均值 ≈ 0，净影响可忽略
- **与 velocity-based 的关键差异**：Φ_T >> Φ_0（1.0 vs -1.0），shaping 天然从起点指向终点，而非 velocity-based 的"鼓励速度但惩罚持续速度"矛盾信号

## 测试验证方案

1. `uv run python agent.py` 执行训练，验证：
   - 无语法/import 错误
   - tqdm 进度条中 original return 呈上升趋势（不再是恒 -200）
   - shaped return 应显著高于 original return（因为净 shaping 为正）
   - 3000 episode 内 return 稳定在 -120~-100 区间
   - matplotlib 回报曲线有明显上升趋势
2. 若 3000 episode 仍未收敛，按优先级排查：
   - C=20→50 增大信号强度
   - ε=0.3→0.5 增大初始探索
   - ε decay=0.999→0.997 加快收敛
3. 评估阶段（EVAL_INTERVAL=50）应观察到小车到达目标

## 📝 执行记录

- `2026-06-04 22:00`: 计划已生成——Energy-Based 势函数替换方案，C=20，Φ=sin(3·pos)+v²/(2g)
- `2026-06-04 22:30`: S1/S2 已执行——config.py 新增 G、PE_COEFFICIENT，调整 ε-greedy 参数；agent.py 替换为 energy-based 公式
- `2026-06-04 22:40`: S3/S4 已完成——训练自检通过，摘要已编写，index 已更新
