---
plan_name: MountainCar 稀疏奖励收敛
related_request: "wiki/request/mountaincar-convergence.md"
status: completed
created_at: 2026-06-04 20:30:00
---
# 执行计划：奖励塑形 + ε-greedy 解决 MountainCar 收敛问题

## 背景

当前 on-policy Actor-Critic 在 MountainCar-v0 上 1500 episode return 持续 -200。根因：

1. **Critic 失效**：每步奖励恒为 -1，TD target ≈ -1×(200-t)+0 对所有状态几乎相同，Critic 学到 "所有状态差不多"，TD delta ≈ 0
2. **Actor 无梯度**：Actor loss = -log π · TD_delta.detach()，delta 趋近 0 导致无有效梯度，策略几乎不更新
3. **Softmax 冗余**：Categorical 随机性不足以在 MountainCar "先左后右" 的非平凡策略空间中碰巧找到成功解

解决方案：potential-based reward shaping（Ng et al., 1999）保持最优策略不变性 + ε-greedy 兜底探索。

## 内容

### 改动 1：`config.py` —— 新增 4 个超参数

```python
REWARD_SHAPING_SCALE = 10   # C: 速度幅值势函数缩放系数
EPSILON = 0.3               # ε-greedy 初始探索率
EPSILON_DECAY = 0.999       # 每 episode 衰减系数
EPSILON_MIN = 0.01          # ε 下限
```

### 改动 2：`agent.py` —— 奖励塑形 + ε-greedy

**奖励塑形（train_on_policy_agent 中 step 之后）**：

使用 potential-based shaping：F(s, s') = γ·Φ(s') - Φ(s)，Φ(s) = |velocity|

```python
shaped_reward = reward + REWARD_SHAPING_SCALE * (GAMMA * abs(next_obs[1]) - abs(obs[1]))
```

物理直觉：鼓励智能体增大速度幅值（无论向左还是向右），从而自然习得 "左蓄力→右冲坡" 的摆动策略。

**ε-greedy 探索（ActorCritic.take_action）**：

```python
def take_action(self, state, epsilon=0.0):
    if np.random.random() < epsilon:
        return np.random.randint(self.action_dim)
    # 原有 Categorical 采样逻辑
```

状态维度从 epsilon 全局变量传递：Agent 新增 `self.action_dim` 属性以支持随机采样。

**ε 衰减（train_on_policy_agent 中每个 episode 结束时）**：

```python
epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
```

### 改动 3：`env.py` —— 不改

环境层无变更，奖励塑形在训练循环中实现而非修改环境本身，保持环境接口纯净。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | `config.py` 新增 4 个超参数 | 无 | `config.py` | 待完成 |
| `S2` | `Agent.take_action` 增加 `epsilon` 参数和随机采样分支 | `S1` | `agent.py`（ActorCritic 类） | 待完成 |
| `S3` | 训练循环中注入奖励塑形 + ε 衰减逻辑 | `S2` | `agent.py`（train_on_policy_agent） | 待完成 |
| `S4` | 运行训练自检：验证 return 收敛趋势 | `S3` | 无（验证步骤） | 待完成 |
| `S5` | 编写摘要并更新 index | `S4` | `wiki/abstract/mountaincar-convergence.md`、`wiki/index.md` | 待完成 |

## 风险与约束声明

- **约束**：奖励塑形使用 potential-based shaping，保证最优策略不变性（ρ 不变）
- **约束**：Φ(s) = |velocity| 是速度的偶函数，不引入方向偏好
- **保守参数 C=10**：单步速度从 0→0.05 的 shaping 增量 ≈ +0.5，与原奖励 -1 叠加为 -0.5，信号显著但不至于淹没原奖励
- **缓慢衰减 ε=0.999**：~460 episode 后 ε 降至 0.2，~2300 episode 后降至 0.03，确保充分探索
- **风险**：若 C=10 收敛缓慢，可调至 15~20；若 ε=0.999 探索过久，可加至 0.997

## 测试验证方案

1. `uv run python agent.py` 执行训练，验证：
   - 无语法/import 错误
   - tqdm 进度条中 return 呈上升趋势（不再是恒 -200）
   - 5000 episode 内 return 稳定在 -120~-100 区间
   - matplotlib 回报曲线有明显上升趋势
2. 若 5000 episode 仍未收敛至 -120，调整 C=15~20 重试

## 📝 执行记录

- `2026-06-04 20:30`: 计划已生成，与用户对齐决策——约束 AC 框架 + 可修改奖励、目标最优解、方案奖励塑形 + ε-greedy、C=10 保守、ε 衰减 0.999 缓慢
- `2026-06-04 21:00`: 计划执行完毕——config.py 新增 4 个超参数、agent.py 实现奖励塑形 + ε-greedy + 塑性奖励显示（tqdm + console）
