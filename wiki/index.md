# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-15 20:00

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
| `DeepCubeA 训练 Loss 曲线分析` | [🔗](./abstract/docs/deepcubea-loss-analysis.md) | 内层收敛/外层发散，saw-tooth 峰值 0.03→0.35，Bellman 备份正反馈雪崩诊断 | ✅ |
| `DeepCubeA 网络模块` | [🔗](./abstract/deepcubea-network.md) | v1-v3: BN; v4: BN→LayerNorm，消除 batch_size≥2 约束 + 删除所有 .eval() | ✅ |
| `DeepCubeA 加权 A* 搜索` | [🔗](./abstract/deepcubea-search.md) | 加权A*求解、三档评估、批量预测优化；推理设备解耦、bytes状态、可直接运行搜索评估 | ✅ |
| `DeepCubeA 批量更新 + 固定目标 AVI` | [🔗](./abstract/deepcubea-target-network.md) | 外层 Bellman 备份固定 J'(s)，内层早停监督学习，无额外 target network | ✅ |
| `DeepCubeA 在线采样 + 轻量验证` | [🔗](./abstract/deepcubea-online-validation.md) | 在线混合采样、LN、分层Bellman MSE+贪心展开+A*、θ_c阈值更新硬阻断Bellman正反馈雪崩 | ✅ |
| `PPO + J(s) 势函数塑形分析` | [🔗](./abstract/docs/ppo-deepcubea-shaping-analysis.md) | 方案A(势函数塑形) vs 方案B(状态惩罚)、联合训练三重风险、阶段性联合训练路径、vs A*搜索优劣对比 | ✅ |
| `PPO + J(s) 势函数塑形实现` | [🔗](./abstract/ppo-deepcubea-agent.md) | ppo_deepcubea_agent.py 单文件实现，ResBlock(LN) 对齐官方，γ=0.98 塑形，Actor/Critic 异构输入，冒烟通过 | ✅ |
| `DeepCubeA 官方仓库探索` | [🔗](./abstract/deepcubea-official-repo.md) | ResnetModel vs DeepCubeANetwork 架构对比、NPuzzleState、AVI 训练流程、puzzle15 预训练模型 | ✅ |
| `DeepCubeA 官方模型评估计划` | [🔗](./abstract/deepcubea-official-model-evaluation.md) | 方案C pickle→numpy、直接移植 ResnetModel、OfficialModelWrapper 对齐 DeepCubeANetwork 接口、解长度排序三等分、8 个 DEEPCUBEA_OFFICIAL_* config 键、参考 search.py 核心函数 | ✅ |
| `DeepCubeA 官方模型评估 Bug 复盘` | [🔗](./abstract/docs/deepcubea-official-bug-postmortem.md) | 3 个移植 bug（ptr_per_state 索引 + blank_idx 硬编码 + _GOAL_BYTES dtype）的症状/根因/修复/预防 | ✅ |
| `DeepCubeA 官方 BWAS 分析` | [🔗](./abstract/deepcubea-bwas-analysis.md) | 官方 BWAS 架构详解、与我们 predict_j_batch 的等价性论证、不建议拷贝的决策 | ✅ |
| `DeepCubeA 官方 vs 自研网络架构对比` | [🔗](./abstract/deepcubea-network-architecture-comparison.md) | 逐层对比：one-hot 内外、BN vs LN、ResBlock 归一化有无、激活顺序、参数量 ~15M vs ~1.3M | ✅ |
| `DeepCubeA 24-Puzzle 适配` | [🔗](./abstract/deepcubea-puzzle24-adaptation.md) | N=4→5，改 2 文件（config + official_network），仅评估不训练，计划已完成 | ✅ |
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
| `DeepCubeA 加权 A* 搜索` | [🔗](./plan/deepcubea-search.md) | completed（v1 评估完成，待 v2 重训后复评） |
| `DeepCubeA 数据生成/训练分离` | [🔗](./plan/deepcubea-data-training-separation.md) | completed |
| `DeepCubeA A* 搜索推理优化` | [🔗](./plan/deepcubea-inference-device-config.md) | completed（S1-S7 全部完成） |
| `DeepCubeA 批量更新 + 固定目标 AVI` | [🔗](./plan/deepcubea-target-network.md) | completed |
| `DeepCubeA 在线采样 + 轻量验证` | [🔗](./plan/deepcubea-online-sampling-validation.md) | completed |
| `DeepCubeA BN → LayerNorm 替换` | [🔗](./plan/deepcubea-bn-to-ln.md) | completed |
| `DeepCubeA 论文阈值 θ_c 更新` | [🔗](./plan/deepcubea-target-threshold.md) | completed |
| `DeepCubeA Checkpoint 与 Loss 图保存优化` | [🔗](./plan/deepcubea-checkpoint-optimization.md) | completed |
| `DeepCubeA 官方预训练模型评估` | [🔗](./plan/deepcubea-official-model-evaluation.md) | completed |
| `DeepCubeA 官方搜索测试数据来源开关` | [🔗](./plan/deepcubea-official-data-switch.md) | completed |
| `DeepCubeA 24-Puzzle 适配` | [🔗](./plan/deepcubea-puzzle24-adaptation.md) | completed |
| `PPO + DeepCubeA J(s) 势函数塑形实现` | [🔗](./plan/ppo-deepcubea-agent.md) | completed |

## TODO列表


## 笔记

### Bellman 正反馈雪崩与解决方案

网络输出层无界（Linear→scalar）导致系统性高估偏置在关键 hub 状态累积后逐 epoch 放大：J↑ → target=1+J↑ → 拟合更大 target → J↑↑。"固定目标 AVI"仅降低反馈频率，未消除正反馈机制本身。**θ_c 阈值更新**是治本方案：loss 不达标则目标完全冻结，硬阻断正反馈链路。关键实现：独立 target_network θ_c + 验证集拆分（VAL_SPLIT>0）+ loss 曲线红绿标记（外层边界/θ_c 更新点）。详见 `./abstract/deepcubea-target-network.md`、`./abstract/docs/deepcubea-loss-analysis.md`。

### 官方模型移植踩坑记录

- **ptr_per_state 索引 bug**：成对存储 start/end 但取值步长=1，奇数索引 state 拿到 0 children → error=J(s)²≈3243。修复：改用 offsets 单条记录（B+1 条）。教训：成对存储必须成对取值，单条 offset 更安全。
- **blank_idx=0 硬编码 bug**：g_score 初始化成 0 而非起始状态实际空格位置 → A* 第一步生成错误孩子 → 整个搜索树错位 → 胜率 0%。教训：idx/ptr 参数绝不能硬编码。
- **data_0.pkl 结构**：`dict{states, solutions, num_nodes_generated, times}`，方案 C 转 `tiles.npy` + `solution_lengths.npy` 解除 pickle 模块路径依赖。

### 其他设计决策

- **非法动作惩罚替代 Action Masking**：Action Masking 导致采样/更新分布不一致，ratio 溢出 NaN。
- **模型保存标准**：按原始 return（不含好奇心奖励）选择最优模型。
- **环境与工具**：`uv venv` / `uv pip` / `uv run python`；gymnasium API（`step` 返回 5 值，`done = terminated or truncated`）。
- **MountainCar 教训**：稀疏奖励下 on-policy Actor-Critic 不收敛，Potential-Based Φ 收敛。已应用于华容道奖励设计。

## 全局更新日志（近10条）

- `06-15 20:00`: wiki 记忆库压缩 —— index.md 笔记精简（4→3 节，~30→~15 行），更新日志 21→10 条；readme.md 导航 18→8 组，状态 19→7 条
- `06-15 19:30`: PPO + DeepCubeA 模型/图表保存机制对齐 DeepCubeA —— latest 间隔覆盖 + best + final 三策略，三面板累积图间隔更新，config 新增 `PPO_DEEPCUBEA_SAVE_INTERVAL`
- `06-15 19:00`: DeepCubeA 官方 vs 自研网络架构对比 —— 逐层对比 ResnetModel 与 DeepCubeANetwork（one-hot 位置、归一化类型、ResBlock 结构、参数量）
- `06-15 17:00`: PPO + DeepCubeA J(s) 势函数塑形实现完成 —— ppo_deepcubea_agent.py（ResBlock LN、γ=0.98 塑形、异构输入）+ config 17 参数；冒烟通过（J∈[15,40]，j_max=200 保守）
- `06-14 20:30`: DeepCubeA 24-Puzzle 适配计划用户确认 —— 补充 S3 冒烟通过标准（25≤J(s)≤140）、ResnetModel 调用点确认（仅 L104）、docstring 更新点备注；计划/摘要/索引已同步
- `06-14 20:00`: DeepCubeA 24-Puzzle 适配计划制定 —— N=4→5，仅改 2 文件（config + official_network），参数不变，仅评估不训练；wiki 需求/摘要/计划/索引已更新
- `06-14 19:00`: DeepCubeA 官方 BWAS 实现分析完成 —— 官方 AStar 类 + C++ parallel_weighted_astar 详解，核心结论：批量 GPU 推理我们已等价实现，多实例设计官方未实际启用，不建议拷贝；wiki 新增摘要/索引
- `06-14 18:45`: DeepCubeA 官方模型评估 Bug 复盘完成 —— 3 个移植 bug 全面分析（ptr_per_state 索引 + blank_idx 硬编码 + _GOAL_BYTES dtype），5 条预防策略；wiki 新增摘要/索引
- `06-14 18:00`: 修复 weighted_astar_official blank_idx 硬编码 0 bug —— A* 第一步生成错误孩子 → 方案 C 胜率 0%；补上 closed set + tiebreaker + goal 短路
- `06-14 17:30`: DeepCubeA 官方搜索测试数据来源开关完成 —— config 新增 4 参数，`deepcubea_official_search.py` 新增 `_generate_test_states` + 表头参数化
