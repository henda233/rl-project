---
plan_name: PPO/RND Agent 切换数字华容道环境
related_request: "wiki/request/switch-to-huarongdao.md"
status: completed
created_at: 2026-06-10 15:00:00
---
# 执行计划：PPO/RND Agent 切换数字华容道环境

破坏性更新：将 `ppo_agent.py` 和 `ppo_rnd_agent.py` 从 MountainCar-v0 完全切换为数字华容道环境，删除所有 MountainCar 相关代码。

## 内容

核心思路：两个 agent 文件的结构（PolicyNet/ValueNet/GAE/PPO/RNDModule/训练循环）保持不变，仅修改环境导入、奖励获取方式、监控指标和通关检测逻辑。config.py 清理所有不再使用的配置段。

### 关键变更点

- **环境导入**：`from env import make_env` → `from env_digital_huarongdao import make_huarongdao_env`
- **奖励**：删除 POTENTIAL_K 塑形公式，直接使用 env 返回的原始 reward
- **RND**：`r_ext` 从 episode 中实际收集的 env reward 获取（而非硬编码 -1.0）
- **监控**：`max_position` → `placed_count` + `step_count`，进度条同步更新
- **通关**：`max_position >= 0.5` → `terminated` flag
- **RND_HIDDEN_DIM**：128 → 256

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 删除 MountainCar 配置段、调整 RND_HIDDEN_DIM | 无 | `config.py` | 已完成 |
| `S2` | env_digital_huarongdao.py — info 新增 `placed_count` | 无 | `env_digital_huarongdao.py` | 已完成 |
| `S3` | ppo_agent.py — 切换华容道环境、移除塑形/再训练、新监控指标 | `S1`, `S2` | `ppo_agent.py` | 已完成 |
| `S4` | ppo_rnd_agent.py — 切换华容道环境、修复 r_ext、新监控指标 | `S1`, `S2` | `ppo_rnd_agent.py` | 已完成 |
| `S5` | WIKI 维护 — 摘要 + index 更新 | `S1~S4` | `wiki/abstract/switch-to-huarongdao.md`、`wiki/index.md` | 已完成 |

## 详细变更说明

### S1: config.py

删除以下配置段：
- `# ==================== AC 配置 ====================` 全部内容（ENV_NAME, RENDER_MODE, MAX_EPISODE_STEPS, HIDDEN_DIM, ACTOR_LR, CRITIC_LR, GAMMA, NUM_EPISODES, EVAL_INTERVAL, EPSILON, EPSILON_DECAY, EPSILON_MIN）
- `# --- PPO 再训练 ---` 全部内容（PPO_ACTOR_MODEL_PATH, PPO_CRITIC_MODEL_PATH, PPO_RETRAIN_NUM_EPISODES）
- `# --- PPO 推理 ---` 全部内容（PPO_INFERENCE_ACTOR_PATH, PPO_INFERENCE_CRITIC_PATH）
- `# --- RND 推理 ---` 全部内容（RND_INFERENCE_ACTOR_PATH, RND_INFERENCE_CRITIC_PATH）
- `# ==================== 奖励塑形 ====================` 全部内容（POTENTIAL_K）
- `# ==================== PPO 并行训练配置 ====================` 全部内容（PPO_NUM_ENVS, POTENTIAL_K_START, POTENTIAL_K_INTERVAL）

修改：
- `RND_HIDDEN_DIM = 128` → `256`

### S2: env_digital_huarongdao.py

在 `_get_info()` 中新增 `placed_count` 字段：

```python
def _get_info(self):
    placed = int(np.sum((self._grid == self._goal_grid) & (self._grid != 0)))
    return {"step_count": self._step_count, "placed_count": placed}
```

### S3: ppo_agent.py

**导入变更**：
- 移除 `POTENTIAL_K` 导入
- 移除 `PPO_RETRAIN_NUM_EPISODES, PPO_ACTOR_MODEL_PATH, PPO_CRITIC_MODEL_PATH` 导入
- 移除 `from env import make_env`
- 新增 `from env_digital_huarongdao import make_huarongdao_env`

**`train_on_policy_agent` 变更**：
- 参数 `env` 改为华容道环境；所有 `env` 调用保持不变（接口兼容）
- 删除 `shaped_return_list`、`max_pos_list`
- 新增 `placed_list`、`step_list`
- 循环内删除 `shaped_reward` 和 `POTENTIAL_K` 塑形公式，直接使用 `reward`
- 新增 `placed = info.get('placed_count', 0)`
- 进度条显示 `placed` 和 `steps` 替代 `shaped` 和 `max_pos`
- 通关检测：`if terminated:`（替代 `max_position >= 0.5`）

**`plot_return` 变更**：
- 改为单图（仅原始 return），标题改为 Huarongdao
- 删除 shaped return 子图

**`main` 变更**：
- 删除 `has_actor_path` / `has_critic_path` 再训练分支
- 始终使用 `results_dir = f"results/train_{datetime.now():%Y%m%d_%H%M%S}"`
- 始终使用 `PPO_NUM_EPISODES`
- `make_env()` → `make_huarongdao_env()`

**`evaluate` 变更**：
- 无结构性变更（接口兼容）

### S4: ppo_rnd_agent.py

**导入变更**：
- 移除 `from env import make_env`
- 新增 `from env_digital_huarongdao import make_huarongdao_env`

**`train_rnd_ppo` 变更**：
- 删除 `max_pos_list`，新增 `placed_list`、`step_list`
- 循环内收集实际的 `reward` 和 `info['placed_count']`
- `r_ext` 从实际 env reward 获取（替代 `np.full(..., -1.0)`）
  实现：循环中 `episode_rewards.append(reward)`，循环后 `r_ext = np.array(episode_rewards)`
- 进度条显示 `placed` 和 `steps` 替代 `pos`
- 通关检测：`if terminated:`

**`plot_return` 变更**：
- 子图标题改为 Huarongdao

**`main` 变更**：
- `make_env()` → `make_huarongdao_env()`

## 风险与约束声明

- **agent.py / ppo_parallel.py / 推理脚本 损坏**：这些文件依赖被删除的配置项，后续运行会报 ImportError。用户已确认接受。
- **状态维度变化**：MountainCar state_dim=2 → 华容道 state_dim=n²。已加载的 MountainCar 模型权重与华容道不兼容。
- **超参数未经调优**：PPO 超参数（GAMMA=0.98, ENTROPY_COEF=0.05 等）沿用 MountainCar 的值，华容道可能需要不同设置。

## 测试验证方案

1. `python ppo_agent.py` — 确认能正常启动训练，无 import 错误，进度条正确显示 placed/steps
2. `python ppo_rnd_agent.py` — 同上，确认 r_ext 使用实际奖励、β 衰减正常
3. 观察前几个 episode 的 placed_count 和 step_count 是否合理
4. 检查 results 目录下是否正确保存模型和图表

## 📝 执行记录

- `2026-06-10 15:00`: 计划已生成，待用户 review
- `2026-06-10 15:45`: 用户确认计划，开始执行
- `2026-06-10 16:00`: S1 config.py 清理完成（删除AC/再训练/推理/塑形/并行训练段、RND_HIDDEN_DIM 128→256）
- `2026-06-10 16:00`: S2 env_digital_huarongdao.py _get_info() 新增 placed_count
- `2026-06-10 16:10`: S3 ppo_agent.py 切换完成（华容道环境、移除塑形/再训练、max_placed+steps监控、terminated通关检测、单图）
- `2026-06-10 16:15`: S4 ppo_rnd_agent.py 切换完成（华容道环境、r_ext从实际env reward获取、max_placed+steps监控、terminated通关检测）
- `2026-06-10 16:20`: S5 WIKI 维护
