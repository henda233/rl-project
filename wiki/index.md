# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-11 18:50

## 模块总览

| 摘要名称 | 摘要路径 | 核心要点 | 状态 |
|---|---|---|---|
| `项目概述` | [🔗](./abstract/docs/project-overview.md) | 数字华容道 + PPO/RND，Python 3.12 + gymnasium + PyTorch | ✅ |
| `数字华容道设计` | [🔗](./abstract/docs/digital-huarongdao-design.md) | n×n 滑块拼图游戏设计文档 | ✅ |
| `华容道环境` | [🔗](./abstract/huarongdao-env.md) | Discrete(4) 动作，正向打乱保证可解，ansi/rgb_array/human 三种渲染 | ✅ |
| `华容道渲染测试` | [🔗](./abstract/huarongdao-render-test.md) | human 模式随机步可视化测试 | ✅ |
| `PPO 实现` | [🔗](./abstract/ppo-impl.md) | GAE + advantage norm + entropy bonus，最优模型按原始 return 保存 | ✅ |
| `RND 实现` | [🔗](./abstract/rnd-impl.md) | 内在奖励探索，β 线性衰减，滑动缓冲区训练预测网络 | ✅ |
| `切换华容道` | [🔗](./abstract/switch-to-huarongdao.md) | PPO/RND Agent 从 MountainCar 切换为华容道的破坏性改造记录 | ✅ |
| `DeepCubeA 研究` | [🔗](./abstract/docs/deepcubea-research.md) | 启发式搜索优于RL，近似值迭代训练DNN逼近J(s)，加权A*求解 | ✅ |
| `DeepCubeA 网络模块` | [🔗](./abstract/deepcubea-network.md) | 256维one-hot编解码、纯函数transition、2FC+4ResBlock网络、predict_j接口 | ✅ |
| `Gym 参考` | [🔗](./abstract/gymnasium/agent-training.md) / [custom-env](./abstract/gymnasium/custom-env.md) / [recording](./abstract/gymnasium/recording-agent.md) | ε-greedy 训练循环、Env 继承规范、Record wrapper | ✅ |
| `参考代码` | [🔗](./abstract/examples/actor-critic-example.md) / [rl-utils](./abstract/examples/rl-utils.md) | PolicyNet/ValueNet、ReplayBuffer、on-policy 循环 | ✅ |

## 计划列表

| 计划名称 | 计划路径 | 状态 |
|---|---|---|
| `实现 PPO 算法` | [🔗](./plan/implement-ppo.md) | completed |
| `实现 RND 探索算法` | [🔗](./plan/implement-rnd.md) | completed |
| `实现数字华容道环境` | [🔗](./plan/implement-huarongdao-env.md) | completed |
| `PPO/RND Agent 切换华容道` | [🔗](./plan/switch-to-huarongdao.md) | completed |
| `DeepCubeA 神经网络训练` | [🔗](./plan/deepcubea-network.md) | completed（best loss=0.0138, epoch 527） |
| `DeepCubeA 加权 A* 搜索` | [🔗](./plan/deepcubea-search.md) | waiting |

## TODO列表

- [ ] 执行DeepCubeA 加权 A* 搜索（`wiki/plan/deepcubea-search.md`）

## 笔记

### AVI 训练发散（DeepCubeA）

近似值迭代 (AVI) 用函数逼近器做 Bellman 备份时，目标值每 epoch 变化导致误差累积、loss 发散。类似 DQN 无 target network 时的表现。缓解方案：target network、增大训练集、或用更慢的学习率。当前保留最佳 checkpoint（epoch 527, loss=0.0138）供 A* 搜索使用。

### 环境与工具

- **虚拟环境**：使用 `uv venv` / `uv pip` / `uv run python` 管理。
- **API 适配**：所有代码使用 gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。

### 模型保存标准

最优模型按**原始 return**（不含好奇心奖励）选择 —— 直接衡量任务效率，避免内在奖励污染模型选择。

### 非法动作惩罚（替代 Action Masking）

采样阶段不屏蔽非法动作，改为环境区分合法/非法奖励（`HUARONGDAO_LEGAL_STEP_REWARD=-1` / `HUARONGDAO_ILLEGAL_STEP_REWARD=-2`）。原因：Action Masking 导致采样分布（masked）与更新分布（unmasked）不一致，ratio 溢出产生 NaN。非法动作惩罚方案保持采样=更新分布，通过 -2 vs -1 差异化激励智能体学习避开非法动作。

### MountainCar 收敛历程（历史参考）

稀疏奖励（每步 -1）下 on-policy Actor-Critic 不收敛，经四次塑形尝试后 Potential-Based Φ=k·pos 收敛。核心教训：从"堆砌更多信号"转向"最简势函数"。此经验已应用于华容道奖励设计。

## 全局更新日志（近5条）

- `06-11 18:50`: DeepCubeA 训练完成 —— `deepcubea_train.py`（S4-S5）、执行训练（S6, 4131 状态/1000 epoch/best loss=0.0138）、WIKI 更新（S7）；观察到 AVI 发散，保留 best checkpoint，更新 index 笔记
- `06-11 19:00`: DeepCubeA S1-S3 完成 —— `config.py` 追加 DEEPCUBEA_* 参数（GPU 默认 True）、`deepcubea_network.py`（encode/decode/transition/DeepCubeANetwork），测试全通过；创建 `wiki/abstract/deepcubea-network.md`，更新 index 计划状态
- `06-11 18:00`: 制定 DeepCubeA 复现执行计划 —— 创建 `wiki/request/deepcubea-reproduction.md`、`wiki/plan/deepcubea-network.md`、`wiki/plan/deepcubea-search.md`、`wiki/abstract/docs/deepcubea-research.md`
- `06-11 17:40`: 清理 MountainCar 时代废弃代码（agent.py/env.py/ppo_parallel.py/run_*.py）及对应 WIKI 摘要/计划/需求；重写保留摘要和 index
- `06-10 17:30`: Action Masking 回退为非法动作惩罚 —— 因 NaN 问题移除采样 mask，env step 区分合法(-1)/非法(-2)奖励
- `06-10 16:20`: PPO/RND 切换华容道执行完毕 —— ppo_agent.py/ppo_rnd_agent.py 破坏性改造完成
- `06-10 14:30`: 华容道渲染测试 —— test_huarongdao_render.py human 模式随机步可视化
- `06-10 14:00`: 华容道环境实现 —— env_digital_huarongdao.py + config.py HUARONGDAO 参数段，env_checker 通过
