# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-13 12:00

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

## TODO列表

（无待办事项）

## 笔记

### AVI 训练发散实证（2026-06-11）

v2 参数（T~U(10,500)/200k states/LR=1e-4/5000 epoch/hidden=512）在 epoch 2331 出现**突发性 loss 爆炸**：

- best=0.052390 → loss 瞬间跃迁至 ~5.6×10^21
- 根因：Bellman 备份正反馈雪崩。网络输出层无界（Linear→scalar），系统性高估偏置在关键 hub 状态累积越过临界点后，逐 epoch 放大：J↑ → target = 1+J↑ → 拟合更大 target → J↑↑
- 2331 epoch 属于"潜伏期"——网络维持近似 self-consistency（loss≈0.05），但偏置在缓慢累积。这与 v1 epoch 527 拐点属同种模式，只是 v2 更大的训练集延缓了崩溃
- **结论**：无 target network 时 AVI 训练不可持续，loss 低是暂时的、不可靠的 self-consistency 指标；target network 通过冻结 Bellman 备份目标打断正反馈循环

### Bellman 正反馈雪崩：同源异尺度（2026-06-12）

v2 每 epoch 重算目标时 epoch 2331 的瞬间爆炸（10^21）和当前每外层重算目标时 saw-tooth 峰值持续爬升（0.03→0.35），**本质是同一机制在不同时间尺度上的表现**：

| | v2 每 epoch 重算 | 当前每外层重算 |
|---|---|---|
| 反馈周期 | 1 epoch | ~N epoch（内层收敛） |
| 失效模式 | 单点瞬间爆炸 | 50 轮持续爬升 |
| 根因 | 同一 Bellman 正反馈循环 | 同一 Bellman 正反馈循环 |

"批量更新 + 固定目标 AVI"将重算频率从 2000 次（每 epoch）降至 20-50 次（每外层），**并未消除正反馈机制，只是降低了反馈频率**。爆炸从"秒炸"变成了"慢炸"。要彻底解决需要切断反馈链路（如论文的阈值 θ_c 更新）。

### DAVI 阈值更新 vs 软更新 Target Network（2026-06-12）

两种方案对比：

- **论文阈值更新（θ_c 仅在 loss < ε 时更新）**：硬阻断——loss 不达标则目标完全冻结。论文作者已验证"每次更新 θ_c → 不稳定，阈值更新 → 更好更稳定"。对正反馈雪崩是治本方案。
- **DDQN 风格软更新（θ_c ← (1-τ)θ_c + τθ）**：减速带——发散变慢但方向不变。θ 若持续膨胀，θ_c 最终会跟随。τ 的 sweet spot 是移动靶。

**结论**：对 AVI 训练，硬阈值阻断优于软更新平滑。论文有直接实证支撑。

### 验证集拆分的必要性条件（2026-06-12）

论文阈值更新的关键细节：**loss 检查必须在"训练未见过的数据"上进行**，否则内层过拟合会绕过阈值。

| 训练结构 | 验证集需求 |
|---|---|
| 外层多、内层少（论文：N epoch/轮） | 不需要 —— 每轮训练不充分，训练 loss 本身就有意义 |
| 外层少、内层多（当前：早停至 loss→0） | **需要** —— 内层总是过拟合到 loss→0，训练 loss 作为阈值形同虚设 |

`DEEPCUBEA_VAL_SPLIT` 开关即为适配两种场景：>0 启用验证集模式（当前结构），=0 回退训练 loss 模式（未来可能切换的论文结构）。

### DAVI θ_c 阈值更新实现（2026-06-13）

论文 DAVI 算法 1 的阈值更新机制已实现：
- **target_network θ_c**：独立 `DeepCubeANetwork` 实例，初始化时从 θ 深拷贝
- **Bellman 备份**：`compute_targets(states, target_network, device)`，用 θ_c 而非当前 θ
- **阈值检查**：每轮外层内层结束后检查 `check_loss < ε`（VAL_SPLIT>0 用 val_loss，=0 用 train_loss）
- **θ_c 更新**：满足条件时 `θ_c ← θ`（load_state_dict clone），硬阻断正反馈链路
- **loss 曲线**：红色虚竖线标记外层边界，绿色虚竖线标记 θ_c 更新点

### 核心设计决策
- **非法动作惩罚替代 Action Masking**：采样阶段不屏蔽非法动作，环境区分合法(-1)/非法(-2)奖励。Action Masking 导致采样/更新分布不一致，ratio 溢出 NaN。
- **模型保存标准**：最优模型按原始 return（不含好奇心奖励）选择，避免内在奖励污染模型选择。
- **环境与工具**：`uv venv` / `uv pip` / `uv run python` 管理；gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。

### MountainCar 收敛历程（历史参考）
稀疏奖励下 on-policy Actor-Critic 不收敛，Potential-Based Φ=k·pos 收敛。教训：从"堆砌更多信号"转向"最简势函数"。已应用于华容道奖励设计。

## 全局更新日志（近10条）

- `06-13 21:00`: PPO + DeepCubeA J(s) 势函数塑形分析 —— 方案A/B理论对比、联合训练三重风险（冷启动/双重正反馈/非平稳分布）、阶段性联合训练推荐路径、vs A*搜索优劣；新建 request+abstract+index更新
- `06-13 12:00`: DeepCubeA 论文阈值 θ_c 更新完成 —— config 新增 TARGET_EPSILON(0.05)+VAL_SPLIT(0.2)；train 引入 target_network θ_c，Bellman 备份改用 θ_c，val 拆分+阈值检查+θ_c 更新逻辑；tqdm 显示 θ_c 状态+val_loss；loss 曲线绿色标记 θ_c 更新点；更新 wiki 摘要+index
- `06-12 20:00`: DeepCubeA 论文阈值 θ_c 更新计划制定 —— 引入独立 target_network 计算 Bellman 目标，仅当 val_loss < ε 时更新 θ_c。新增 loss 曲线分析文档（docs/DeepCubeA训练loss曲线分析.md）及其摘要，作为计划核心依据
- `06-12 20:00`: 保存其他 AI 对 DeepCubeA 训练 loss 曲线的分析至 wiki —— 诊断内层收敛/外层发散、Bellman 备份正反馈雪崩。创建 docs 内容 + abstract 摘要 + index 索引
- `06-12 17:35`: wiki 记忆库一致性更新 —— readme.md/index.md/online-validation 摘要中 BN→LN 过期引用清理
- `06-12 17:30`: BN → LayerNorm 替换完成 —— deepcubea_network.py: BN→LN+注释保留变量名; 删除 predict_j/predict_j_batch/compute_targets/load_model/_compute_bellman_errors 共5处 .eval(); 更新 network/online-validation 摘要
- `06-12 17:00`: BN batch_size=1 崩溃复现确认 —— `reproduce_bn_crash.py` 单样本 forward 触发 ValueError，根因与远程训练一致；更新 network/online-validation 摘要 + BN→LN 计划执行记录
- `06-12 16:30`: BN→LayerNorm 替换计划制定 —— 训练 batch_size=1 崩溃根因分析；计划 7 步骤涵盖 network/train/search + wiki 摘要更新
- `06-12 15:00`: config.py 参数整理 —— 在线采样 4 参数迁入 Training 分组；Validation 分组合并入 Search 分组；命名/结构保持
- `06-12 14:30`: DeepCubeA 在线采样 + 轻量验证完成 —— S1-S9 全部执行；config 新增 10 个参数；deepcubea_utils.py 共享状态生成；network 添加 BN(仅 FC 隐藏层)+self.eval()；train 在线混合采样(B+B'+overlap)；search 三分层验证(Bellman MSE+贪心展开+完整A*)；generate_data 重构调用 utils；冒烟测试脚本交付
- `06-12 12:00`: DeepCubeA 在线采样 + 轻量验证计划制定
- `06-11 23:59`: 一致性检查 —— 删除冗余测试文件（test_deepcubea_network/test_deepcubea_training）；deepcubea_search_test 合并到搜索模块；修复 wiki 摘要与代码的不一致（PPO/RND/DeepCubeA 配置值、过时引用、v2参数表）
- `06-11 23:16`: DeepCubeA 批量更新 + 固定目标 AVI 完成 —— config 三参数替换；训练循环重构（外层 Bellman 备份 + 内层早停 + 双层 tqdm + saw-tooth 曲线）；烟雾测试通过（100 states, 早停 25/30 epoch）
- `06-11 22:30`: AVI 训练发散实证记录 —— v2 参数 epoch 2331 loss 爆炸至 10^21，分析 Bellman 备份正反馈雪崩机制
- `06-11 21:45`: DeepCubeA Target Network 计划制定（已废弃） —— hard/soft 双模式，freq=500/tau=0.005
- `06-11 21:15`: DeepCubeA A* 搜索推理优化 —— S1-S7 全部执行；批量转移+bytes状态+g_score字典+GPU可配+inference_mode；smoke test 通过
- `06-11 20:45`: DeepCubeA 训练进度改用 tqdm —— epoch 循环 tqdm，移除 `DEEPCUBEA_LOG_INTERVAL`
- `06-11 20:30`: DeepCubeA 数据生成/训练分离 —— S1 config 新增参数；S2 `deepcubea_generate_data.py`（tqdm+去重+.npy）；S3 `deepcubea_train.py` 从 .npy 加载
