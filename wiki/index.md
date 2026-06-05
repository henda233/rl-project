# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-05 20:00:00

## 模块总览

| 摘要名称 | 摘要路径 | 关键摘要内容 | 依赖健康度 | 最后更新 |
|---|---|---|---|---|
| `项目概述` | [🔗](./abstract/docs/project-overview.md) | MountainCar-v0 + Actor-Critic 项目总览，技术栈与约束 | ✅ | 06-04 |
| `Actor-Critic 算法` | [🔗](./abstract/docs/actor-critic-algorithm.md) | TD 残差驱动 Actor/Critic 双网络更新 | ✅ | 06-04 |
| `MountainCar 环境` | [🔗](./abstract/gymnasium/mountain-car.md) | Discrete(3) 动作，Box(2) 观测，-1/step，200 步截断 | ✅ | 06-04 |
| `Gymnasium 参考` | [🔗](./abstract/gymnasium/agent-training.md) / [custom-env](./abstract/gymnasium/custom-env.md) / [recording](./abstract/gymnasium/recording-agent.md) | ε-greedy 训练循环、Env 继承规范、Record wrapper 用法 | ✅ | 06-04 |
| `参考代码` | [🔗](./abstract/examples/actor-critic-example.md) / [rl-utils](./abstract/examples/rl-utils.md) | PolicyNet/ValueNet 结构、ReplayBuffer、on-policy 循环 | ✅ | 06-04 |
| `环境搭建` | [🔗](./abstract/env-setup.md) | config.py + make_env() 工厂函数 | ✅ | 06-04 |
| `AC 实现` | [🔗](./abstract/actor-critic-impl.md) / [计划](./plan/implement-actor-critic.md) | agent.py gymnasium 适配，已知稀疏奖励不收敛 | ✅ | 06-04 |
| `奖励塑形探索` | [🔗](./plan/mountaincar-convergence.md) / [energy](./plan/energy-based-shaping.md) / [progress](./plan/episode-relative-progress.md) | Velocity→Energy→Progress 三次尝试均不收敛，详见笔记 | ✅ | 06-04 |
| `Potential-Based 塑形` | [🔗](./abstract/potential-based-shaping.md) | Φ=k·(pos+10·|v|)（PPO v2），Φ=k·pos（AC v1），Ng et al. 定理保证 | ✅ | 06-05 |
| `PPO 算法` | [🔗](./abstract/ppo-impl.md) / [计划](./plan/implement-ppo.md) | GAE + advantage norm + entropy bonus，PPO_USE_GPU 控制设备 | ✅ | 06-05 |
| `PPO 并行训练` | [🔗](./abstract/ppo-parallel-training.md) / [计划](./plan/ppo-parallel-training.md) | multiprocessing 多 k 对比，自动选最优，双 subplot 可视化 | ✅ | 06-05 |
| `PPO 训练增强` | [计划](./plan/ppo-training-enhancement.md) / [需求](./request/ppo-training-enhancement.md) | 模型保存 + 通关检测(>=0.5) + 图保存 + 推理录制 mp4 | ✅ | 06-05 |
| `RND 探索算法` | [🔗](./abstract/rnd-impl.md) / [计划](./plan/implement-rnd.md) | PPO + RND 替代势能塑形，滑动缓冲区 + 内在奖励驱动探索 | ✅ | 06-05 |

## 需求列表

| 需求名称 | 需求路径 | 状态 | 关联计划 |
|---|---|---|---|
| `初始项目需求` | [🔗](./request/initial-requirements.md) | pending | `实现强化学习算法`（已完成） |
| `MountainCar收敛需求` | [🔗](./request/mountaincar-convergence.md) | completed | `MountainCar稀疏奖励收敛`→`Energy-Based 奖励塑形`（已完成） |
| `Energy-Based 塑形需求` | [🔗](./request/energy-based-shaping.md) | completed | `Energy-Based 奖励塑形`（已完成） |
| `回合相对进度塑形需求` | [🔗](./request/episode-relative-progress.md) | completed | `回合相对进度奖励塑形`（已完成，但不收敛） |
| `PPO算法实现需求` | [🔗](./request/implement-ppo.md) | completed | `PPO 算法实现`（已完成） |
| `PPO 并行训练需求` | [🔗](./request/ppo-parallel-training.md) | completed | `PPO 并行训练（多 k 对比）`（已完成） |
| `PPO 设备配置需求` | [🔗](./request/ppo-device-config.md) | completed | — |
| `PPO 训练增强需求` | [🔗](./request/ppo-training-enhancement.md) | completed | `PPO 训练增强`（已完成） |
| `PPO 再训练需求` | [🔗](./request/ppo-retraining.md) | completed | `PPO 再训练功能`（已完成） |
| `RND 探索算法需求` | [🔗](./request/rnd-implementation.md) | completed | `PPO + RND 探索算法`（已完成） |

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
| `PPO + RND 探索算法` | [🔗](./plan/implement-rnd.md) | ✅ completed |

## TODO列表

- [x] RND 探索算法实现（PPO + RND 替代势能塑形）

## 笔记

- **项目文档纠正**：`docs/项目文档.md` 中"自定义2D物理环境"的描述有误，实际项目使用 gymnasium 预设的 MountainCar-v0 环境（2026-06-04 确认）。
- **API 适配要点**：参考代码 `examples/` 使用旧版 gym API，所有代码需使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。
- **MountainCar 稀疏奖励收敛问题**（2026-06-04）：on-policy Actor-Critic 在 MountainCar-v0 上 1500 episode return 仍为 -200。每步奖励恒为 -1，智能体无法获得正向反馈引导探索。
- **四次奖励塑形尝试**（2026-06-05 更新）：
  - Velocity-based (Φ=|v|,C=10)：速度盲，撞墙蓄力阶段惩罚关键行为，γ<1 泄漏使净贡献为负。
  - Energy-Based (Φ=sin(3·pos)+v²/(2g),C=10)：sin(3·pos) 非单调，设计过于复杂。
  - 回合相对进度（记录打破 + 速度 + 近终点惩罚）：非势能塑形，5000 episode 仍不收敛。
  - **Potential-Based (Φ=k·pos)**（当前方案）：极简势函数，仅依赖位置，向右移动 ⇒ 正向塑形奖励，Ng et al. 定理保证策略不变性。k=1 默认，可通过 POTENTIAL_K 调整量级。
  - **设计理念转变**：从"堆砌更多信号"转向"最简势函数"，让算法自身能力说话。
  - **Potential-Based v2 (Φ=k·(pos+10·|v|))**（2026-06-05）：在 v1 基础上加入速度绝对值项。|v| 代表动能，乘 10 对齐量级。目标：加速收敛、降低收敛随机性。仅用于 PPO，AC 沿用 v1。

## 全局更新日志（近5条）

- `06-05 20:00`: RND 探索算法实现完成——ppo_rnd_agent.py 创建（RNDTargetNet/PredictorNet/RNDModule + 滑动缓冲区 + PPO 训练循环），config.py 追加 RND_NUM_EPISODES/RND_BUFFER_SIZE，ppo_agent.py 从头训练改为时间戳目录。训练验证通过。
- `06-05 19:30`: RND 探索算法计划制定——config.py 追加 RND 配置段，计划 `wiki/plan/implement-rnd.md` 创建，需求 `wiki/request/rnd-implementation.md` 创建。
- `06-05 19:00`: PPO 奖励塑形 v2——势函数改为 Φ=k·(pos+10·|v|)，速度项代表动能，乘 10 对齐量级。仅 ppo_agent.py 修改一行塑形公式。
- `06-05`: PPO 再训练计划已制定——config.py 新增双路径配置，ppo_agent.py 支持加载权重继续训练，时间戳输出目录。
- `06-05`: PPO 训练增强完成——最优模型保存(shaped return + cleared) + 通关检测 + savefig + run_ppo_agent.py 推理录制。
- `06-05`: PPO 训练增强计划制定——模型保存 + 通关检测(>=0.5) + 图保存到 results/imgs/ + 推理录制 mp4。
- `06-05`: PPO 设备配置——config.py 新增 PPO_USE_GPU，ppo_agent/ppo_parallel 统一读取配置。
- `06-05`: PPO 并行训练完成——ppo_parallel.py 多进程多 k 对比，自动选最优，双 subplot 可视化。
- `06-05`: Potential-Based 塑形——Φ=k·pos 极简势函数（当前方案），Ng et al. 定理保证策略不变性。
- `06-04`: PPO 算法实现——ppo_agent.py (GAE + advantage norm + entropy bonus)，替代 AC。
- `06-04`: 奖励塑形探索——Velocity→Energy→Progress 三次尝试均不收敛，详见笔记。
