---
plan_name: PPO + DeepCubeA J(s) 势函数塑形实现
related_request: "wiki/request/ppo-deepcubea-agent.md"
status: completed
created_at: 2026-06-15 12:00:00
---
# 执行计划：PPO + DeepCubeA J(s) 势函数塑形

交代计划的背景、目的。

## 背景

已有可行性分析（`wiki/abstract/docs/ppo-deepcubea-shaping-analysis.md`）确认了方案 A（势函数塑形）优于方案 B（状态惩罚），推荐阶段性路径：先验证冻结 J(s) shaping 的前提（Phase 2）。

本计划执行 Phase 2：PPO + 冻结官方 DeepCubeA 模型 J(s) 作为势函数塑形，验证其有效性。

## 内容

### 总体方案

- 势函数 Φ(s) = -J(s)，塑形奖励 F(s,a,s') = γ·J(s') - J(s)（γ=0.98，与 PPO_DEEPCUBEA_GAMMA 一致）
- r_total = -1 + β·(J(s) - γ·J(s'))（环境步数惩罚 + β 加权塑形奖励）
- 仅 Actor 输入拼接 J(s)/j_max，Critic 保持原始 state
- 网络加深 + 残差连接（对齐 `deepcubea_official_network.py` ResBlock 结构，BN→LN），对抗 15-puzzle ~10^13 状态空间的梯度问题
- 所有代码在 `ppo_deepcubea_agent.py`，独立可运行

### 网络架构

对齐 `deepcubea_official_network.py` ResnetModel 的 ResBlock 结构，BN 替换为 LN：

**ResBlock**（hidden_dim → hidden_dim）：
```
┌─ FC(hidden_dim) → LN → ReLU → FC(hidden_dim) → LN ─┐
└────────────────── (+) ──────────────────────────────┘ → ReLU
```

**Actor**（输入 `state_dim + 1`，输出 `action_dim` logits）：
```
obs(n²) + j/j_max(1) → FC(hidden_dim) → LN → ReLU
  → [ResBlock × 2]（上述结构）
  → FC(action_dim)  # 无 softmax，Categorical(logits=...) 内部处理
```

**Critic**（输入 `state_dim`，输出 scalar）：
```
obs(n²) → FC(hidden_dim) → LN → ReLU
  → [ResBlock × 2]（同 Actor）
  → FC(1)
```

### 训练流程

```
1. 加载官方 DeepCubeA 模型（冻结，eval 模式）
2. 初始化 PPO Agent（Actor + Critic）
3. for each episode:
   a. reset env, 获取 obs
   b. obs_int = obs.astype(np.int64), j = model.predict_j(obs_int)
   c. for step in range(max_steps):
      - j_norm = j / j_max
      - action = agent.take_action(obs, j_norm)
      - next_obs, _, terminated, truncated, info = env.step(action)
      - done = terminated or truncated
      - next_obs_int = next_obs.astype(np.int64)
      - j_next = 0 if terminated else model.predict_j(next_obs_int)
      - r_int = j - γ * j_next  (γ = PPO_DEEPCUBEA_GAMMA，对齐 MDP 折扣因子)
      - r_total = -1 + β * r_int
      - 存储 transition: (obs原始, j_norm, action, r_total, next_obs原始, done)
      - obs = next_obs, j = j_next
      - if done: break
   d. PPO update（Actor 用 augmented state = concat(obs, j_norm)，Critic 用原始 obs）
```

### 过渡期存储

`transition_dict` 新增 `j_norms` 字段存储每步的 J(s)/j_max，update 时用于构造 Actor 的 augmented input。transition 中存储原始 obs（供 Critic），Actor forward 时从 j_norms 重建 augmented state。

### Config 参数

在 `config.py` 的 `DeepCubeA Official Model Evaluation` section 后新增：

```python
# ==================== PPO + DeepCubeA J(s) Shaping（ppo_deepcubea_agent.py 使用） ====================

PPO_DEEPCUBEA_HIDDEN_DIM = 256           # 隐藏层维度
PPO_DEEPCUBEA_ACTOR_LR = 1e-3            # Actor 学习率
PPO_DEEPCUBEA_CRITIC_LR = 1e-3           # Critic 学习率
PPO_DEEPCUBEA_GAMMA = 0.98               # 折扣因子（GAE + 势函数塑形共用）
PPO_DEEPCUBEA_LMBDA = 0.95               # GAE λ
PPO_DEEPCUBEA_EPOCHS = 10                # 每批数据训练轮数
PPO_DEEPCUBEA_EPS = 0.2                  # Clip 范围
PPO_DEEPCUBEA_NUM_EPISODES = 10000       # 训练总 episode 数
PPO_DEEPCUBEA_MAX_STEPS = 200            # 每 episode 最大步数（truncated 截断）
PPO_DEEPCUBEA_EVAL_INTERVAL = 1000       # 评估间隔（episode 数）
PPO_DEEPCUBEA_NUM_EVAL_EPISODES = 10     # 每次评估运行的 episode 数
PPO_DEEPCUBEA_ENTROPY_COEF = 0.05        # 熵正则化系数
PPO_DEEPCUBEA_BETA = 1.0                 # 塑形奖励权重（r_total = -1 + β·(J(s) - γ·J(s'))）
PPO_DEEPCUBEA_USE_GPU = False            # PPO 训练设备（True=cuda, False=cpu）
PPO_DEEPCUBEA_RESNET_BLOCKS = 2          # 残差块数量
PPO_DEEPCUBEA_J_MAX = 200.0              # J(s) 归一化上界（经验估计，S6 冒烟后调整）
PPO_DEEPCUBEA_MODEL_PATH = "data/model_state_dict.pt"  # DeepCubeA 模型路径
```

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| S1 | config.py 新增 `PPO_DEEPCUBEA_*` 参数段（16 参数） | 无 | `config.py`（追加） | 待完成 |
| S2 | 实现 ResBlock + PolicyNet + ValueNet（LayerNorm 残差网络） | S1 | `ppo_deepcubea_agent.py` | 待完成 |
| S3 | 实现 PPO 类（含 augmented state Actor input + GAE + update） | S2 | `ppo_deepcubea_agent.py` | 待完成 |
| S4 | 实现训练循环（DeepCubeA 加载 + rollout + J(s) 塑形奖励 + Critic 原始 state） | S3 | `ppo_deepcubea_agent.py` | 待完成 |
| S5 | 实现评估/绘图/main 入口 | S4 | `ppo_deepcubea_agent.py` | 待完成 |
| S6 | 冒烟测试：5 episode 运行不报错，J(s) ∈ 合理范围 | S5 | 终端输出 | 待完成 |
| S7 | WIKI 更新：摘要 + index | S6 | `wiki/abstract/ppo-deepcubea-agent.md` + `wiki/index.md` | 待完成 |

## 风险与约束声明

1. **j_max 未知**：先设 200.0（官方模型 J ∈ [36, 58] 在解附近，远距离状态可能更大），后续根据训练中实际 J(s) 分布调整
2. **逐 step 推理开销**：每个 transition 需 2 次 J(s) 推理（当前状态 + 下一状态），GPU 上 ~0.2ms/step，CPU 上可能更慢
3. **Critic 与 Actor 异构输入**：Critic 见原始 state，Actor 见 augmented state。两者共享 GAE advantage，理论上可行但需验证
4. **ResBlock 块内 LN**：对齐 `deepcubea_official_network.py`，每个 FC 后跟 LN（替换原 BN），残差连接包裹两个 FC+LN 对
5. **冻结 J(s)**：Phase 2 仅验证塑形有效性，不更新 DeepCubeA 模型
6. **γ 一致性**：塑形公式 `r_int = J(s) - γ·J(s')` 使用与 PPO_DEEPCUBEA_GAMMA 相同的 γ=0.98，保证 Ng et al. 最优策略不变性
7. **β 初始值**：β=1.0 意味着最优步（J 递减 1）时 r_total ≈ -1 + 1·(J(s) - 0.98*J(s')) ≈ 0，非最优步 < 0，目标步 r_total = -1 + β·J(s) > 0。后续根据训练情况调整
8. **max_steps=200**：复用环境已有的 MAX_STEPS 值，N=4 时 200 步足够覆盖探索；truncated 时正常计算 j_next，terminated 时 j_next=0

## 测试验证方案

1. **冒烟测试**：运行 `python ppo_deepcubea_agent.py`，5 episode 不崩溃
   - 验证 DeepCubeA 模型加载成功
   - 打印 J(s) 的 min / mean / max，确认在合理范围（期望 0~200，据此校验 j_max）
   - 验证 r_total 计算正确（r_total = -1 + β·(J(s) - γ·J(s'))，目标步 r_total > 0）
2. **短期训练**：100 episode 训练
   - return 曲线呈上升趋势（J(s) 递减 → r_total 递增）
   - Actor loss 收敛
3. **对比基线**：与 `ppo_agent.py`（无塑形）对比 return 曲线，验证塑形加速效果

## 📝 执行记录

- 2026-06-15 12:00: 计划已生成
- 2026-06-15 14:00: 二次讨论 —— ResBlock 去 LayerNorm（仅保留输入投影 LN）、r_total = -1 + β·(J-J')（γ=1）、新增 max_steps/β 参数、Actor/Critic LR 对齐 1e-3、S6 冒烟打印 J(s) 分布
- 2026-06-15 16:00: 三次讨论最终确认 —— ResBlock 对齐官方：FC→LN→ReLU→FC→LN→+skip→ReLU；塑形 γ=PPO_DEEPCUBEA_GAMMA(0.98)；Actor 输出无 softmax；块内 LN；评估指标 return+胜率+平均步数+NUM_EVAL_EPISODES
- 2026-06-15 17:00: 计划执行完成 —— S1-S7 全部完成，冒烟测试通过（J(s)∈[15,40]，j_max=200 保守覆盖），代码 `ppo_deepcubea_agent.py` + config 追加 17 参数
