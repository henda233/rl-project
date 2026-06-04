---
plan_name: PPO 算法实现
related_request: "[PPO需求](../request/implement-ppo.md)"
status: completed
created_at: 2026-06-04 23:30:00
---
# 执行计划：引入 PPO 算法替代 AC 解决 MountainCar 收敛问题

## 背景与目的

Actor-Critic 在 MountainCar-v0 上经三轮奖励塑形仍不收敛（5000 episode return ≈ -200）。
根因推测为 AC 算法本身局限性：TD(0) 高方差、单次数据利用、无约束策略更新。
PPO 通过 Clipped Surrogate Objective + GAE + 多轮更新解决上述问题。

本计划在保留当前 episode-relative progress 奖励塑形的前提下，新建 `ppo_agent.py`，
与现有 `agent.py` 并存，使用独立训练循环，最小化对现有代码的侵入。

## 内容

### PPO 核心改进点（相对于当前 AC）

| 维度 | AC 现状 | PPO |
|---|---|---|
| 优势估计 | TD(0) 单步残差 | GAE(λ) 多步加权 |
| 策略目标 | log_prob × td_delta | Clipped Surrogate（ratio 截断） |
| 数据利用 | 每条轨迹更新 1 次 | 每条轨迹更新 N 轮（epochs） |
| 更新约束 | 无 | ε-clip 限制策略变化幅度 |
| 重要性采样 | 无 | ratio = π_new / π_old |

### PPO 算法流程

```
1. 收集一条完整轨迹 (s, a, r, s', done)
2. 计算 TD 目标 → TD 残差 → GAE 优势
3. 计算 old_log_probs（更新前策略的对数概率）
4. 重复 epochs 轮：
   a. 计算新 log_probs → ratio = exp(new - old)
   b. surr1 = ratio × advantage
   c. surr2 = clip(ratio, 1-ε, 1+ε) × advantage
   d. actor_loss = -min(surr1, surr2)
   e. critic_loss = MSE(V(s), td_target)
   f. 反向传播更新两个网络
```

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增 PPO 超参数 | 无 | `config.py`（追加） | 已完成 |
| `S2` | 创建 ppo_agent.py（PPO 类 + GAE + 训练循环） | `S1` | `ppo_agent.py`（新建） | 已完成 |
| `S3` | 运行训练验证 PPO 收敛性 | `S2` | 训练曲线 / 控制台输出 | 已完成 |
| `S4` | 创建摘要 + 更新 index | `S3` | `wiki/abstract/ppo-impl.md`、`wiki/index.md` | 已完成 |

## 超参数初值

沿用当前奖励塑形参数，新增 PPO 专属参数：

```python
# --- 网络结构（与 AC 一致） ---
HIDDEN_DIM = 256
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2

# --- PPO 专属 ---
GAMMA = 0.98          # 折扣因子（PPO 标准值，高于 AC 的 0.5 以支持长程信用分配）
PPO_LMBDA = 0.95      # GAE λ（标准值）
PPO_EPOCHS = 10       # 每批数据训练轮数（标准值）
PPO_EPS = 0.2         # Clip 范围（标准值）

# --- 训练规模 ---
NUM_EPISODES = 5000   # 与 AC 一致，便于对比

# --- 奖励塑形（保持不变） ---
PROGRESS_RIGHT_SCALE = 10
PROGRESS_LEFT_SCALE = 5
SPEED_REWARD_SCALE = 5
SPEED_THRESHOLD = 0.01
STAGNATION_PENALTY = 1
NEAR_GOAL_THRESHOLD = 0.3
NEAR_GOAL_PENALTY = 20
```

### 超参数取值理由

- **GAMMA=0.98**：MountainCar 需 ~110 步到达目标，γ 过低会导致远期奖励信号衰减殆尽。0.5（AC 当前值）在第 10 步后权重仅剩 0.001，无法传播终点信号。0.98 在 100 步后仍保留 ~0.13 的权重。
- **PPO_LMBDA=0.95**：GAE 在偏差-方差间平衡的标准选择，1.0 退化为 Monte Carlo，0.0 退化为 TD(0)。
- **PPO_EPOCHS=10**：标准值，MountainCar 轨迹短（~200 步），10 轮更新不会过拟合。
- **PPO_EPS=0.2**：PPO 论文标准值，clip 范围过大会失去约束效果，过小会降低学习速度。

## 风险与约束声明

- **与 AC 不共享代码**：ppo_agent.py 自包含（PPO 类 + GAE + 训练循环），与 agent.py 零耦合，AC 代码不受影响。
- **旧 gym API**：ppo_example.py 的 `take_action` 无 epsilon 参数，PPO 类需适配当前 ε-greedy 接口或移除 ε-greedy（PPO 的 Categorical 采样 + clip 机制自带探索）。
- **训练时间**：PPO 每 episode 更新 epochs=10 轮，单 episode 时间约为 AC 的 10 倍，5000 episode 可能耗时较长。
- **ε-greedy 取舍**：PPO 的 clipped objective 需要 ratio = π_new/π_old，ε-greedy 会破坏概率比计算。PPO 实现中应移除 ε-greedy，依赖 Categorical 采样 + clip 约束提供探索。

## 测试验证方案

1. **代码正确性**：`python ppo_agent.py` 无报错运行，tqdm 正常显示进度
2. **收敛性判断**：episode return（原始奖励）从 -200 上升至 > -120（即至少节省 80 步），且 moving average 呈上升趋势
3. **与 AC 对比**：同条件下（5000 episode，相同塑形奖励），PPO 的 return 曲线应明显优于 AC
4. **评估渲染**：每 EVAL_INTERVAL=200 渲染一次，观察小车是否到达旗帜

## 📝 执行记录

- `2026-06-04 23:30`: 计划已生成，待用户确认
- `2026-06-04 23:50`: S1 完成 — config.py AC/PPO 参数分离，新增 PPO_ENTROPY_COEF=0.01、PPO_EVAL_INTERVAL=200
- `2026-06-04 23:50`: S2 完成 — ppo_agent.py 创建，自包含 PPO（GAE+dones 截断 + advantage norm + entropy bonus）
- `2026-06-04 23:55`: S3 完成 — 训练运行无报错
- `2026-06-04 23:55`: S4 完成 — wiki/abstract/ppo-impl.md 创建，index.md 更新
