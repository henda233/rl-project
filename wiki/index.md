# WIKI Index（全局摘要索引）

> 🔄 最后同步：2026-06-12 14:30

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
| `DeepCubeA 网络模块` | [🔗](./abstract/deepcubea-network.md) | v1: 4k states/T100/LR1e-3 → loss=0.014; v2: 200k states/T500/LR1e-4/hidden=512/batch=1024/批量更新+固定目标 | ✅ |
| `DeepCubeA 加权 A* 搜索` | [🔗](./abstract/deepcubea-search.md) | 加权A*求解、三档评估、批量预测优化；推理设备解耦、bytes状态、可直接运行搜索评估 | ✅ |
| `DeepCubeA 批量更新 + 固定目标 AVI` | [🔗](./abstract/deepcubea-target-network.md) | 外层 Bellman 备份固定 J'(s)，内层早停监督学习，无额外 target network | ✅ |
| `DeepCubeA 在线采样 + 轻量验证` | [🔗](./abstract/deepcubea-online-validation.md) | 在线混合采样(B+B'+overlap)、BN仅FC隐藏层、分层Bellman MSE+贪心展开+完整A*三方案验证 | ✅ |
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

## TODO列表

- [ ] 运行 `deepcubea_online_smoke_test.py` 验证在线采样 + 三项指标
- [ ] 重新训练模型（在线采样 + BN），对比新旧模型分箱 MSE 和 A* 胜率

## 笔记

### AVI 训练发散实证（2026-06-11）

v2 参数（T~U(10,500)/200k states/LR=1e-4/5000 epoch/hidden=512）在 epoch 2331 出现**突发性 loss 爆炸**：

- best=0.052390 → loss 瞬间跃迁至 ~5.6×10^21
- 根因：Bellman 备份正反馈雪崩。网络输出层无界（Linear→scalar），系统性高估偏置在关键 hub 状态累积越过临界点后，逐 epoch 放大：J↑ → target = 1+J↑ → 拟合更大 target → J↑↑
- 2331 epoch 属于"潜伏期"——网络维持近似 self-consistency（loss≈0.05），但偏置在缓慢累积。这与 v1 epoch 527 拐点属同种模式，只是 v2 更大的训练集延缓了崩溃
- **结论**：无 target network 时 AVI 训练不可持续，loss 低是暂时的、不可靠的 self-consistency 指标；target network 通过冻结 Bellman 备份目标打断正反馈循环

### DeepCubeA AVI 训练发散
近似值迭代用函数逼近器做 Bellman 备份，每 epoch 重算目标导致"移动靶"，已实证 loss 爆炸至 10^21。解决方案：批量更新 + 固定目标（文献原版），外层 Bellman 备份固定 J'(s)，内层早停监督学习。
→ **计划已制定**（`wiki/plan/deepcubea-target-network.md`），OUTER_ITER=20/INNER_EPOCHS=100/PATIENCE=10。

### 核心设计决策
- **非法动作惩罚替代 Action Masking**：采样阶段不屏蔽非法动作，环境区分合法(-1)/非法(-2)奖励。Action Masking 导致采样/更新分布不一致，ratio 溢出 NaN。
- **模型保存标准**：最优模型按原始 return（不含好奇心奖励）选择，避免内在奖励污染模型选择。
- **环境与工具**：`uv venv` / `uv pip` / `uv run python` 管理；gymnasium API（`step` 返回 5 值，`reset(seed=...)`，`done = terminated or truncated`）。

### MountainCar 收敛历程（历史参考）
稀疏奖励下 on-policy Actor-Critic 不收敛，Potential-Based Φ=k·pos 收敛。教训：从"堆砌更多信号"转向"最简势函数"。已应用于华容道奖励设计。

## 全局更新日志（近10条）

- `06-12 14:30`: DeepCubeA 在线采样 + 轻量验证完成 —— S1-S9 全部执行；config 新增 10 个参数；deepcubea_utils.py 共享状态生成；network 添加 BN(仅 FC 隐藏层)+self.eval()；train 在线混合采样(B+B'+overlap)；search 三分层验证(Bellman MSE+贪心展开+完整A*)；generate_data 重构调用 utils；冒烟测试脚本交付
- `06-12 12:00`: DeepCubeA 在线采样 + 轻量验证计划制定
- `06-11 23:59`: 一致性检查 —— 删除冗余测试文件（test_deepcubea_network/test_deepcubea_training）；deepcubea_search_test 合并到搜索模块；修复 wiki 摘要与代码的不一致（PPO/RND/DeepCubeA 配置值、过时引用、v2参数表）
- `06-11 23:16`: DeepCubeA 批量更新 + 固定目标 AVI 完成 —— config 三参数替换；训练循环重构（外层 Bellman 备份 + 内层早停 + 双层 tqdm + saw-tooth 曲线）；烟雾测试通过（100 states, 早停 25/30 epoch）
- `06-11 22:30`: AVI 训练发散实证记录 —— v2 参数 epoch 2331 loss 爆炸至 10^21，分析 Bellman 备份正反馈雪崩机制
- `06-11 21:45`: DeepCubeA Target Network 计划制定（已废弃） —— hard/soft 双模式，freq=500/tau=0.005
- `06-11 21:15`: DeepCubeA A* 搜索推理优化 —— S1-S7 全部执行；批量转移+bytes状态+g_score字典+GPU可配+inference_mode；smoke test 通过
- `06-11 20:45`: DeepCubeA 训练进度改用 tqdm —— epoch 循环 tqdm，移除 `DEEPCUBEA_LOG_INTERVAL`
- `06-11 20:30`: DeepCubeA 数据生成/训练分离 —— S1 config 新增参数；S2 `deepcubea_generate_data.py`（tqdm+去重+.npy）；S3 `deepcubea_train.py` 从 .npy 加载
- `06-11 20:00`: DeepCubeA v1 评估 + v2 参数调整 —— A* Short 档胜率仅 32%；根因：随机游走低效、状态覆盖不足、AVI 震荡；v2: T~U(10,500)/50k states/LR1e-4/2k iter
- `06-11 19:30`: DeepCubeA 加权 A* 搜索完成 —— `deepcubea_search.py` + 评估/测试脚本；WIKI 更新
- `06-11 18:50`: DeepCubeA 训练完成 —— 4131 状态/1000 epoch/best loss=0.0138；AVI 发散，保留 best checkpoint
- `06-11 18:00`: 制定 DeepCubeA 复现执行计划 —— 创建 request/plan/abstract；index 更新
- `06-11 17:40`: 清理 MountainCar 废弃代码及对应 WIKI 摘要/计划/需求；重写保留摘要
- `06-10 17:30`: Action Masking 回退为非法动作惩罚 —— 因 NaN 问题移除采样 mask
