# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-10

## 模块总览

| 摘要名称 | 摘要路径 | 核心要点 | 状态 |
|---|---|---|---|
| `项目概述` | [🔗](./abstract/docs/project-overview.md) | MountainCar-v0 + PPO/RND + 华容道，Python 3.12 + gymnasium + PyTorch | ✅ |
| `MountainCar 环境` | [🔗](./abstract/gymnasium/mountain-car.md) | Discrete(3) 动作，Box(2) 观测，-1/step，200 步截断 | ✅ |
| `Gymnasium 参考` | [🔗](./abstract/gymnasium/agent-training.md) / [custom-env](./abstract/gymnasium/custom-env.md) / [recording](./abstract/gymnasium/recording-agent.md) | ε-greedy 训练循环、Env 继承规范、Record wrapper | ✅ |
| `参考代码` | [🔗](./abstract/examples/actor-critic-example.md) / [rl-utils](./abstract/examples/rl-utils.md) | PolicyNet/ValueNet、ReplayBuffer、on-policy 循环 | ✅ |
| `AC + 奖励塑形` | [🔗](./abstract/actor-critic-impl.md) / [pot-based](./abstract/potential-based-shaping.md) | AC on-policy 稀疏奖励不收敛 → 四次塑形尝试 → Potential-Based Φ=k·pos 收敛 | ✅ |
| `PPO 生态` | [🔗](./abstract/ppo-impl.md) / [并行](./abstract/ppo-parallel-training.md) / [RND](./abstract/rnd-impl.md) / [推理](./abstract/inference-scripts.md) | GAE + advantage norm + entropy bonus，多 k 并行对比，RND 内在奖励探索，推理录制 mp4 | ✅ |
| `PPO 训练增强` | [🔗](./abstract/ppo-impl.md)（训练增强段） | 最优模型保存（原始 return）、通关检测、图保存、再训练 | ✅ |
| `数字华容道` | [🔗](./abstract/docs/digital-huarongdao-design.md) / [环境](./abstract/huarongdao-env.md) / [渲染测试](./abstract/huarongdao-render-test.md) | n×n 滑块拼图，Discrete(4)，正向打乱保证可解，ansi/rgb_array(点阵数字)/human 渲染 | ✅ |

## 计划列表

| 计划名称 | 计划路径 | 状态 |
|---|---|---|
| `搭建强化学习任务环境` | [🔗](./plan/setup-environment.md) | completed |
| `实现强化学习算法` | [🔗](./plan/implement-actor-critic.md) | completed |
| `MountainCar稀疏奖励收敛` | [🔗](./plan/mountaincar-convergence.md) | completed |
| `Energy-Based 奖励塑形` | [🔗](./plan/energy-based-shaping.md) | completed |
| `回合相对进度奖励塑形` | [🔗](./plan/episode-relative-progress.md) | completed |
| `PPO 算法实现` | [🔗](./plan/implement-ppo.md) | completed |
| `PPO 并行训练（多 k 对比）` | [🔗](./plan/ppo-parallel-training.md) | completed |
| `PPO 训练增强` | [🔗](./plan/ppo-training-enhancement.md) | completed |
| `PPO 再训练功能` | [🔗](./plan/ppo-retraining.md) | completed |
| `PPO + RND 探索算法` | [🔗](./plan/implement-rnd.md) | completed |
| `实现数字华容道游戏环境` | [🔗](./plan/implement-huarongdao-env.md) | completed |

## TODO列表

- [x] 数字华容道环境实现（env_digital_huarongdao.py + config 参数）

## 笔记

### 环境与工具

- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。
- **API 适配**：参考代码 `examples/` 使用旧版 gym API，所有代码需使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **项目文档纠正**：`docs/项目文档.md` 中"自定义2D物理环境"描述有误，实际项目使用 gymnasium 预设的 MountainCar-v0 环境。

### MountainCar 收敛历程（核心经验）

稀疏奖励（每步 -1）下 on-policy Actor-Critic 1500 episode return 仍为 -200。经历四次塑形尝试：
1. Velocity-based (Φ=|v|,C=10)：速度盲，撞墙蓄力阶段惩罚关键行为
2. Energy-Based (Φ=sin(3·pos)+v²/(2g),C=10)：sin(3·pos) 非单调，设计过于复杂
3. 回合相对进度（记录打破 + 速度 + 近终点惩罚）：非势能塑形，5000 episode 仍不收敛
4. **Potential-Based (Φ=k·pos)**（当前方案）：极简势函数，仅依赖位置，Ng et al. 定理保证策略不变性

**核心教训**：从"堆砌更多信号"转向"最简势函数"，让算法自身能力说话。v2 版 Φ=k·(pos+10·|v|) 加入速度项加速收敛（仅 PPO）。

### 模型保存标准

最优模型按**原始 return**（不含塑形/好奇心奖励）选择 —— 直接衡量任务效率，避免塑形奖励污染模型选择。

## 全局更新日志（近5条）

- `06-10 14:30`: 华容道渲染测试 —— `test_huarongdao_render.py` human 模式随机步可视化
- `06-10 14:00`: 华容道环境实现 —— `env_digital_huarongdao.py` + `config.py` HUARONGDAO 参数段，rgb_array 点阵数字渲染，env_checker 通过
- `06-10 12:00`: 华容道需求+计划 —— `wiki/request/digital-huarongdao-env.md`、`wiki/plan/implement-huarongdao-env.md`、设计摘要
- `06-05 21:30`: RND_BETA 线性衰减 —— `RND_BETA_END=1.0`，`RND_BETA_DECAY=0.05`，每 episode 递减
- `06-05 21:00`: 模型保存标准改为原始 return —— `best_shaped_return` → `best_original_return`
