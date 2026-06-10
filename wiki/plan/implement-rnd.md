---
plan_name: PPO + RND 探索算法实现
related_request: "wiki/request/rnd-implementation.md"
status: completed
created_at: 2026-06-05 19:30:00
---
# 执行计划：PPO + RND 探索算法

## 背景

MountainCar-v0 奖励稀疏（每步 -1），现有 PPO 使用 potential-based shaping (Φ = k·(pos + 10·|v|)) 提供密集奖励引导收敛。本计划用 RND (Random Network Distillation) 替代势能塑形，以内在探索奖励驱动智能体，验证 RND 在低维状态空间上的有效性。

## 核心设计

### 奖励公式

```
r_int = MSE(f̂(s), f(s))                          // 内在奖励（原始预测误差）
r_int_norm = r_int / (running_std + 1e-8)         // 仅除以运行标准差（不减均值）
r_total = r_ext + RND_BETA · r_int_norm            // 组合奖励（无势能塑形）
```

### 运行标准差维护

```
running_std = (1 - alpha) * running_std + alpha * batch_std
```
其中 `alpha = 1 / (update_count + 1)`（增量平均），`update_count` 为预测网络更新次数。

### 网络架构

| 网络 | 结构 | 参数 |
|---|---|---|
| Target (f) | Linear(2, 256) → ReLU → Linear(256, 256) | 随机初始化，固定不更新 |
| Predictor (f̂) | Linear(2, 256) → ReLU → Linear(256, 256) | 训练，最小化 MSE(f̂(s), f(s)) |

### 每 episode 训练流程

1. **采样**：策略收集完整 episode 轨迹
2. **RND 前向**：对每个 s_t 计算 f(s_t) 和 f̂(s_t)
3. **计算内在奖励**：`r_int[t] = MSE(f̂, f)` → 归一化
4. **组合奖励**：`r_total[t] = external_reward + β · r_int_norm[t]`
5. **GAE**：基于 r_total 计算 advantages
6. **PPO 更新**：Actor + Critic（RND_EPOCHS 轮）
7. **RND 更新**：对收集的所有状态训练 f̂（RND_EPOCHS 轮），然后更新 running_std

### 评估与保存

- 每 `PPO_EVAL_INTERVAL` episode 渲染评估
- 按 total return 保存最优模型
- 通关检测：max_position >= 0.5 时保存 cleared 模型
- 绘图：原始 return + total return 双曲线图

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | `config.py` 追加 RND 配置段 | 无 | `config.py`（RND_HIDDEN_DIM, RND_OUTPUT_DIM, RND_LR, RND_BETA, RND_EPOCHS, RND_NUM_EPISODES, RND_BUFFER_SIZE） | ✅ 已完成 |
| `S2` | 新建 `ppo_rnd_agent.py` — RND 网络类 | `S1` | `ppo_rnd_agent.py`（RNDTargetNet, RNDPredictorNet, RNDModule） | ✅ 已完成 |
| `S3` | 集成 RND 到 PPO 训练循环 | `S2` | `ppo_rnd_agent.py`（train_rnd_ppo 奖励计算流程） | ✅ 已完成 |
| `S4` | 评估、保存、可视化（复用 PPO 模式） | `S3` | `ppo_rnd_agent.py`（evaluate, plot_return, main） | ✅ 已完成 |
| `S5` | 训练验证 | `S4` | 运行训练，确认不报错 + return 曲线呈上升趋势 | ✅ 已完成 |
| `S6` | WIKI 更新 | `S5` | 摘要、index 更新 | ✅ 已完成 |

## 风险与约束声明

- **低维状态空间风险**：MountainCar 仅 2D 状态，状态重复率高，RND 新颖性信号可能快速衰减，导致 β 需要谨慎调参
- **势能塑形移除**：移除 Φ 后训练可能不收敛，需通过调节 β 或网络架构缓解
- **β 敏感性**：β 过大导致探索主导而忽略任务目标，过小则等同于无塑形 PPO
- **计算成本**：每步增加 2 个网络前向（f + f̂），每 episode 增加预测网络训练（MountainCar 每 episode ≤ 200 步，开销可忽略）

## 测试验证方案

1. **无报错运行**：`python ppo_rnd_agent.py` 完整执行 10 个 episode 不报错
2. **return 趋势**：训练过程中 total return 呈现上升趋势（不一定最终收敛，但需有改善）
3. **内在奖励衰减**：随着训练推进，r_int 应逐渐下降（熟悉区域增多）
4. **代码隔离**：`ppo_agent.py` 和 `agent.py` 不受影响，仍可正常运行
5. **Config 完整性**：所有 RND 参数在 `config.py` 中有明确注释

## 📝 执行记录

- `2026-06-05 19:30`: 计划已生成（待执行）
- `2026-06-05 20:00`: S1 config.py RND 配置段完成（含 RND_NUM_EPISODES, RND_BUFFER_SIZE）；S2-S4 ppo_rnd_agent.py 创建完成，含 RND 双网络 + 滑动缓冲区 + PPO 训练循环 + 评估保存可视化；ppo_agent.py 从头训练改为时间戳目录；S5 训练验证通过；S6 WIKI 摘要和 index 更新完成
- `2026-06-05`: RND_BETA 线性衰减——config.py 新增 RND_BETA_END=1.0、RND_BETA_DECAY=0.05，ppo_rnd_agent.py current_beta 每 episode 递减 max(RND_BETA_END, current_beta - RND_BETA_DECAY)，progress bar 新增 β 显示
