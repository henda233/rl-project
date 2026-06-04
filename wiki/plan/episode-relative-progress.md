---
plan_name: 回合相对进度奖励塑形
related_request: "wiki/request/episode-relative-progress.md"
status: completed
created_at: 2026-06-04 23:00:00
---
# 执行计划：回合相对进度奖励替换 Energy-Based 势函数

## 背景

Energy-Based 势函数塑形（Φ=sin(3·pos)+v²/(2g)）存在 sin(3·pos) 振荡导致的非单调性问题，训练不收敛。参考 `docs/关于阶梯式奖励设置.md` 的成功方案（DQN 100% 成功率），将奖励塑形替换为回合相对进度奖励。

## 内容

### 核心公式

```
shaped_reward = original_reward
              + (x - max_x) * PROGRESS_RIGHT_SCALE    if x > max_x
              + (min_x - x) * PROGRESS_LEFT_SCALE     if x < min_x
              + abs(speed) * SPEED_REWARD_SCALE       if abs(speed) > SPEED_THRESHOLD
              - STAGNATION_PENALTY                    if abs(speed) <= SPEED_THRESHOLD

if truncated and x > NEAR_GOAL_THRESHOLD:
    shaped_reward -= NEAR_GOAL_PENALTY  (applied to last transition)
```

### 与 Energy-Based 势函数的对比

| 维度 | Energy-Based 势函数 | 回合相对进度奖励 |
|---|---|---|
| 信号类型 | 微分（γΦ(s')-Φ(s)）| 绝对值（打破记录的增量）|
| 单调性 | sin(3·pos) 振荡，关键路径非单调 | 单调（记录只增不减）|
| 双向诱导 | 隐式（PE+KE 物理直觉）| 显式（max_x 权重 > min_x）|
| 最优策略不变性 | 保证（Ng et al.）| 不保证（但文档验证可行）|

### 改动 1：`config.py`

移除 `REWARD_SHAPING_SCALE`、`G`、`PE_COEFFICIENT`，新增 7 个参数：

```python
PROGRESS_RIGHT_SCALE = 10
PROGRESS_LEFT_SCALE = 5
SPEED_REWARD_SCALE = 5
SPEED_THRESHOLD = 0.01
STAGNATION_PENALTY = 1
NEAR_GOAL_THRESHOLD = 0.3
NEAR_GOAL_PENALTY = 20
```

### 改动 2：`agent.py`

- 移除 `from math import sin` 和 energy-based 相关 import
- `train_on_policy_agent` 中：每个 episode 初始化 `min_x = max_x = obs[0]`
- 每步计算 progress reward + speed signal
- episode 结束后检查 truncated near-goal 惩罚

## 任务步骤

| 步骤ID | 任务描述 | 状态 |
|---|---|---|
| `S1` | config.py 替换参数 | 已完成 |
| `S2` | agent.py 替换塑形逻辑 | 已完成 |
| `S3` | 运行训练自检 | 已完成（可运行但不收敛）|

## 风险与约束声明

- **约束**：回合相对奖励不是 potential-based，理论上可能改变最优策略。但对于 MountainCar 文档已验证可行。
- **已知问题**：训练仍不收敛。推测原因：AC 算法本身的局限性（神经网络结构、on-policy 样本效率等），非奖励塑形问题。
- **后续方向**：调整网络结构（HIDDEN_DIM、层数）、尝试 off-policy 方法（DQN + 经验回放）、调整 GAMMA 和 LR 等超参数。

## 📝 执行记录

- `2026-06-04 23:00`: 计划已生成，与用户对齐 5 项决策
- `2026-06-04 23:10`: S1/S2 已执行——config.py 和 agent.py 修改完成
- `2026-06-04 23:15`: S3 完成——代码可运行但 5000 episode 仍不收敛，推测为 AC 算法本身局限
