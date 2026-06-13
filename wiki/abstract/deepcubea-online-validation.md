---
abstract_name: DeepCubeA 在线采样 + 轻量验证
source_contents:
  - "deepcubea_train.py"
  - "deepcubea_search.py"
  - "deepcubea_utils.py"
  - "deepcubea_generate_data.py"
  - "deepcubea_network.py"
  - "config.py"
  - "deepcubea_online_smoke_test.py"
dependencies:
  - "wiki/abstract/deepcubea-network.md"
  - "wiki/abstract/deepcubea-search.md"
  - "wiki/abstract/deepcubea-target-network.md"
  - "wiki/plan/deepcubea-online-sampling-validation.md"
  - "wiki/plan/deepcubea-target-threshold.md"
created_at: 2026-06-12 14:30:00
updated_at: 2026-06-13 22:30:00
---
# 摘要：DeepCubeA 在线采样 + 轻量验证

## 核心结论与关键信息

- **在线混合采样**：每轮外层迭代在线生成 B 个状态（从目标随机游走）+ 从基础数据集采样 B' 个（SEED_OVERLAP 控制跨轮保留比例），合并去重后训练，解除离线固定 20 万状态集的分布固化
- **LayerNorm**：仅 FC 隐藏层添加 LN（Linear→LN→ReLU），残差块不变；LN 训练/推理一致，已删除所有 `.eval()` 调用
- **分层 Bellman MSE（方案 B）**：验证集按 K（打乱步数）等宽分层（层数 `DEEPCUBEA_VAL_NUM_STRATA`），每层计算 Bellman MSE 统计（mean/median/std），默认运行
- **贪心展开（方案 C）**：truncated A*（max_expand 截断，默认 500），`DEEPCUBEA_VAL_GREEDY_FLAG` 控制，统计分箱求解率和平均展开节点数
- **完整 A* 三档（方案 D）**：保留现有 Short/Medium/Hard 评估，`DEEPCUBEA_VAL_ASTAR_FLAG` 控制
- **验证手动触发**：训练不自动触发验证，用户手动运行 `python deepcubea_search.py <model_path>`，配置 flags 控制方案 C/D
- **共享工具模块**：`deepcubea_utils.py` 提供 `generate_scrambled_states` 和 `generate_stratified_states`（支持 `start_state` 参数），被 train/search/generate_data 复用
- **旧模型不兼容**：BN → LN 后网络结构变更，旧 checkpoint 无法加载新网络
- **论文阈值 θ_c 更新**：引入独立 target_network 计算 Bellman 目标，仅当 check_loss < ε（默认 0.05）时 θ_c ← θ。硬阻断 Bellman 正反馈雪崩（每轮外层检查，内层迭代等价于论文 C 间隔）。支持 VAL_SPLIT 双模式：>0 验证集模式（适配外层少内层多），=0 训练 loss 模式（适配论文外层多内层少）
- **Checkpoint 与 Loss 图保存优化**：checkpoint 改为 `deepcubea_heuristic_latest.pt` 覆盖式保存（不再按 epoch 编号）；每次外层迭代后同步保存 `loss_curve_latest.csv`（epoch/loss/is_outer_start/theta_updated）和 `loss_curve_latest.png`（累积曲线+外层边界+θ_c 更新标记）；末尾冗余绘图代码已删除

## Config 新增参数（在线采样 + 轻量验证 + θ_c 阈值更新）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEEPCUBEA_ONLINE_BATCH` | 20000 | 每轮在线生成 B |
| `DEEPCUBEA_BASE_BATCH` | 5000 | 每轮基础数据集 B'，0=纯在线 |
| `DEEPCUBEA_OUTER_SEED` | 42 | 外层种子基准 |
| `DEEPCUBEA_SEED_OVERLAP` | 0.3 | base 采样跨轮重叠比例 |
| `DEEPCUBEA_TARGET_EPSILON` | 0.05 | Bellman 误差阈值 ε，低于此值更新 θ_c |
| `DEEPCUBEA_VAL_SPLIT` | 0.2 | 验证集比例，0 = 禁用验证集（用训练 loss 判断） |
| `DEEPCUBEA_VAL_SIZE` | 3000 | 验证集状态数 |
| `DEEPCUBEA_VAL_SEED` | 12345 | 验证集种子 |
| `DEEPCUBEA_VAL_NUM_STRATA` | 5 | 分层数 |
| `DEEPCUBEA_VAL_GREEDY_EXPAND` | 500 | 贪心展开截断 |
| `DEEPCUBEA_VAL_ASTAR_FLAG` | False | 启用完整 A* |
| `DEEPCUBEA_VAL_GREEDY_FLAG` | False | 启用贪心展开 |

## 内容概述

> 在线混合采样训练：每轮外层在线生成 B 状态 + 基础数据集采样 B'（SEED_OVERLAP 控制跨轮保留），合并去重后编码→Bellman 备份（用 θ_c 计算）→ 可选 val 拆分 → 内层早停训练（仅 train 部分）→ 阈值检查（val_loss 或 train_loss < ε → θ_c ← θ）。验证指标三方案：方案 B（分层 Bellman MSE，默认运行）、方案 C（贪心展开，flag 控制）、方案 D（完整 A* 三档，flag 控制）。deepcubea_utils.py 共享状态生成；deepcubea_generate_data.py 重构调用 utils；deepcubea_online_smoke_test.py 提供冒烟测试。

## 依赖与影响链

- **上游依赖**：`config.py`（DEEPCUBEA_* 参数）、`deepcubea_network.py`（LN 网络）、`deepcubea_utils.py`（状态生成）
- **下游被依赖**：无（训练/验证/生成模块均在本摘要覆盖范围内）
- **变更扩散评估**：高（config + network + train + search + generate_data + 新建 utils，6 个文件修改 + 1 个新建）
