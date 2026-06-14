# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-14 20:30

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
| `DeepCubeA 官方仓库探索` | [🔗](./abstract/deepcubea-official-repo.md) | ResnetModel vs DeepCubeANetwork 架构对比、NPuzzleState、AVI 训练流程、puzzle15 预训练模型 | ✅ |
| `DeepCubeA 官方模型评估计划` | [🔗](./abstract/deepcubea-official-model-evaluation.md) | 方案C pickle→numpy、直接移植 ResnetModel、OfficialModelWrapper 对齐 DeepCubeANetwork 接口、解长度排序三等分、8 个 DEEPCUBEA_OFFICIAL_* config 键、参考 search.py 核心函数 | ✅ |
| `DeepCubeA 官方模型评估 Bug 复盘` | [🔗](./abstract/docs/deepcubea-official-bug-postmortem.md) | 3 个移植 bug（ptr_per_state 索引 + blank_idx 硬编码 + _GOAL_BYTES dtype）的症状/根因/修复/预防 | ✅ |
| `DeepCubeA 官方 BWAS 分析` | [🔗](./abstract/deepcubea-bwas-analysis.md) | 官方 BWAS 架构详解、与我们 predict_j_batch 的等价性论证、不建议拷贝的决策 | ✅ |
| `DeepCubeA 24-Puzzle 适配` | [🔗](./abstract/deepcubea-puzzle24-adaptation.md) | N=4→5，改 2 文件（config + official_network），ResnetModel 维度从 HUARONGDAO_N 推导，仅评估不训练，S3 冒烟标准已定义，计划已确认 | ⏳ |
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
| `DeepCubeA 24-Puzzle 适配` | [🔗](./plan/deepcubea-puzzle24-adaptation.md) | waiting（用户已确认，补充 S3 冒烟标准 + 代码审查确认） |

## TODO列表


## 笔记

### Bellman 正反馈雪崩与解决方案演进

AVI 训练的核心问题：网络输出层无界（Linear→scalar），系统性高估偏置在关键 hub 状态累积越过临界点后，逐 epoch 放大：J↑ → target=1+J↑ → 拟合更大 target → J↑↑。该机制在不同时间尺度表现一致：
- v2 每 epoch 重算目标：epoch 2331 瞬间爆炸至 10^21
- 批量更新 + 固定目标 AVI：saw-tooth 峰值从 0.03 持续爬升至 0.35

"固定目标 AVI"仅降低了反馈频率（每外层 N epoch），未消除正反馈机制本身。**论文阈值 θ_c 更新**是治本方案：loss 不达标则目标完全冻结，硬阻断正反馈链路。软更新（DDQN τ 衰减）仅是减速带，θ 持续膨胀时 θ_c 最终仍会跟随。

### θ_c 阈值更新实现关键点（2026-06-13）

- 验证集拆分：外层次数少/内层多时内层总是过拟合到 loss→0，训练 loss 作为阈值形同虚设，必须用单独验证集检查（`VAL_SPLIT>0`）
- `target_network θ_c`：独立实例，Bellman 备份改用 θ_c，仅当 val_loss < ε 时 clone θ
- loss 曲线标记：红色虚竖线=外层边界，绿色虚竖线=θ_c 更新点

### data_0.pkl 内部结构与方案 C（2026-06-14）

官方测试数据 pickle 结构：`dict{'states': List[NPuzzleState](500), 'solutions': List[List[int]](500), 'num_nodes_generated': List[int](500), 'times': List[float](500)}`。每个 NPuzzleState 的 `.tiles` 为 `(16,) int32` ndarray。

方案 C 一次性将 pickle 转为 `tiles.npy (500,16)` + `solution_lengths.npy (500,)`，彻底解除 pickle 的 `environments.n_puzzle` 模块路径依赖。转换后不再需要拷贝 NPuzzleState/State 类。

### 核心设计决策

- **ptr_per_state 索引 bug（2026-06-14）**：`_compute_bellman_errors_official` 中 children 收集与父 state 错位。根因：每 state 成对存储 start/end（2B 条），但取值 `ptr_per_state[i], ptr_per_state[i+1]` 步长=1，奇数索引 state 拿到 0 children → error=J(s)²≈3243。修复：offsets 单条记录（B+1 条），步长=1 自然正确。教训：成对存储必须成对取值（步长=2），单条 offset 模式更安全。

- **weighted_astar_official blank_idx=0 bug（2026-06-14）**：`g_score` 初始化 `blank_idx=0` 而非起始状态实际空格位置，`get_children` 用错误位置判断合法移动 → A* 从第一步生成错误孩子 → 整个搜索树错位 → 方案 C 胜率 0%。教训：search 函数的 `blank_idx` 绝不能硬编码，必须从当前状态计算。这是官方搜索第二个 ptr/idx 类 bug，建议后续全面检查所有 idx/ptr 参数传递。

- **非法动作惩罚替代 Action Masking**：采样不屏蔽非法动作，环境区分合法(-1)/非法(-2)奖励。Action Masking 导致采样/更新分布不一致，ratio 溢出 NaN。
- **模型保存标准**：按原始 return（不含好奇心奖励）选择最优模型。
- **环境与工具**：`uv venv` / `uv pip` / `uv run python`；gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。
- **MountainCar 教训**：稀疏奖励下 on-policy Actor-Critic 不收敛，Potential-Based Φ=k·pos 收敛。已应用于华容道奖励设计。

## 全局更新日志（近10条）

- `06-14 20:30`: DeepCubeA 24-Puzzle 适配计划用户确认 —— 补充 S3 冒烟通过标准（25≤J(s)≤140）、ResnetModel 调用点确认（仅 L104）、docstring 更新点备注；计划/摘要/索引已同步
- `06-14 20:00`: DeepCubeA 24-Puzzle 适配计划制定 —— N=4→5，仅改 2 文件（config + official_network），参数不变，仅评估不训练；wiki 需求/摘要/计划/索引已更新
- `06-14 19:00`: DeepCubeA 官方 BWAS 实现分析完成 —— 官方 AStar 类 + C++ parallel_weighted_astar 详解，核心结论：批量 GPU 推理我们已等价实现，多实例设计官方未实际启用，不建议拷贝；wiki 新增摘要/索引
- `06-14 18:45`: DeepCubeA 官方模型评估 Bug 复盘完成 —— 3 个移植 bug 全面分析（ptr_per_state 索引 + blank_idx 硬编码 + _GOAL_BYTES dtype），5 条预防策略；wiki 新增摘要/索引
- `06-14 18:00`: 修复 weighted_astar_official blank_idx 硬编码 0 bug —— A* 第一步生成错误孩子 → 方案 C 胜率 0%；补上 closed set + tiebreaker + goal 短路
- `06-14 17:30`: DeepCubeA 官方搜索测试数据来源开关完成 —— config.py 新增 4 参数（USE_OFFICIAL_DATA/T_MIN/T_MAX/NUM_TEST_STATES），`deepcubea_official_search.py` 新增 `_generate_test_states` + `__main__` 分支 + 表头参数化（Len Range/K Range）；wiki 计划/摘要/index 同步更新
- `06-14 16:00`: config.py 二次整理 —— 删除孤儿参数 DEEPCUBEA_GREEDY_MAX_STEPS；拆分独立 DeepCubeA Data Generation section（T_MIN/T_MAX/TRAIN_SET_SIZE/TRAIN_DATA_PATH）；config 从 46 参数 7 section 精简为 45 参数 6 section
- `06-14 15:30`: 修复 `_compute_bellman_errors_official` ptr_per_state 索引 bug —— children 与父 state 错位，Bellman MSE 从 1411 降至 0.36；wiki 摘要/计划/index 同步更新
- `06-14 01:20`: DeepCubeA 官方预训练模型评估完成 —— S0-S5 全部执行：data/ 预处理 + ResnetModel 移植 + 贪心展开改造 + 三项评估脚本 + wiki 更新；冒烟验证 J(s) ∈ [36,58] mean=52.85 对齐解路径长度
- `06-14 15:00`: DeepCubeA 官方预训练模型评估计划 v4 澄清 —— OfficialModelWrapper 接口对齐 DeepCubeANetwork（仅 predict_j/predict_j_batch）；Bellman MSE 参考 search.py:129-168；分档参考 evaluate:300-379；weighted_astar_official 跳过 encode_batch；Config 键全量 DEEPCUBEA_OFFICIAL_* 前缀；S0 在项目 venv 执行；计划/摘要/index 同步更新
- `06-14`: DeepCubeA 官方预训练模型评估计划 v3 更新 —— 方案 C pickle→numpy，新增 S0 数据预处理，去除 NPuzzleState 拷贝，细化 data_0.pkl 结构，新增 config 键，去除 output.txt 对比，计划/摘要/index 同步更新
- `06-13 23:50`: DeepCubeA 官方仓库探索 + 官方预训练模型评估计划 —— 笔记、摘要、计划已写入 wiki
- `06-13 22:30`: DeepCubeA Checkpoint 与 Loss 图保存优化完成 —— latest.pt 覆盖式保存 + CSV + 累积 loss 曲线图每轮外层迭代同步更新
- `06-13 22:00`: wiki 记忆库压缩 —— index.md 笔记精简合并（Bellman 雪崩3→1），更新日志 18→10；readme.md 补充缺失摘要引用
- `06-13 21:30`: DeepCubeA Checkpoint 与 Loss 图保存优化计划制定
- `06-13 21:00`: PPO + DeepCubeA J(s) 势函数塑形分析 —— 方案A/B对比、联合训练风险、阶段性路径、vs A*搜索
- `06-13 12:00`: DeepCubeA 论文阈值 θ_c 更新完成 —— target_network、val_split、θ_c 硬阻断、loss 曲线绿色标记
- `06-12 20:00`: DeepCubeA 训练 loss 曲线分析 + θ_c 阈值更新计划
- `06-12 17:30`: BN → LayerNorm 替换完成 —— 消除 batch_size≥2 约束 + 删除所有 .eval()
- `06-12 15:00`: config.py 参数整理 —— 在线采样迁入 Training，Validation 合并入 Search
- `06-12 14:30`: DeepCubeA 在线采样 + 轻量验证完成 —— 在线混合采样 + LN + 三分层验证 + 冒烟测试
- `06-11 23:59`: 一致性检查 —— 删除冗余测试文件，修复 wiki 摘要与代码不一致
- `06-11 23:16`: DeepCubeA 批量更新 + 固定目标 AVI 完成 —— 外层 Bellman 备份 + 内层早停 + saw-tooth 曲线
- `06-11 21:15`: DeepCubeA A* 搜索推理优化完成 —— 批量转移 + bytes 状态 + GPU 可配
