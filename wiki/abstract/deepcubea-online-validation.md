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
created_at: 2026-06-12 14:30:00
updated_at: 2026-06-12 14:30:00
---
# 摘要：DeepCubeA 在线采样 + 轻量验证

## 核心结论与关键信息

- **在线混合采样**：每轮外层迭代在线生成 B 个状态（从目标随机游走）+ 从基础数据集采样 B' 个（SEED_OVERLAP 控制跨轮保留比例），合并去重后训练，解除离线固定 20 万状态集的分布固化
- **Batch Normalization**：仅 FC 隐藏层添加 BN（Linear→BN→ReLU），残差块不变；`predict_j`/`predict_j_batch` 内部显式 `self.eval()` 确保推理安全
- **分层 Bellman MSE（方案 B）**：验证集按 K（打乱步数）等宽分层（层数 `DEEPCUBEA_VAL_NUM_STRATA`），每层计算 Bellman MSE 统计（mean/median/std），默认运行
- **贪心展开（方案 C）**：truncated A*（max_expand 截断，默认 500），`DEEPCUBEA_VAL_GREEDY_FLAG` 控制，统计分箱求解率和平均展开节点数
- **完整 A* 三档（方案 D）**：保留现有 Short/Medium/Hard 评估，`DEEPCUBEA_VAL_ASTAR_FLAG` 控制
- **验证手动触发**：训练不自动触发验证，用户手动运行 `python deepcubea_search.py <model_path>`，配置 flags 控制方案 C/D
- **共享工具模块**：`deepcubea_utils.py` 提供 `generate_scrambled_states` 和 `generate_stratified_states`（支持 `start_state` 参数），被 train/search/generate_data 复用
- **旧模型不兼容**：添加 BN 后网络结构变更，旧 checkpoint 无法加载新网络

## Config 新增参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEEPCUBEA_ONLINE_BATCH` | 20000 | 每轮在线生成 B |
| `DEEPCUBEA_BASE_BATCH` | 5000 | 每轮基础数据集 B'，0=纯在线 |
| `DEEPCUBEA_OUTER_SEED` | 42 | 外层种子基准 |
| `DEEPCUBEA_SEED_OVERLAP` | 0.3 | base 采样跨轮重叠比例 |
| `DEEPCUBEA_VAL_SIZE` | 3000 | 验证集状态数 |
| `DEEPCUBEA_VAL_SEED` | 12345 | 验证集种子 |
| `DEEPCUBEA_VAL_NUM_STRATA` | 5 | 分层数 |
| `DEEPCUBEA_VAL_GREEDY_EXPAND` | 500 | 贪心展开截断 |
| `DEEPCUBEA_VAL_ASTAR_FLAG` | False | 启用完整 A* |
| `DEEPCUBEA_VAL_GREEDY_FLAG` | False | 启用贪心展开 |

## 内容概述

> 在线混合采样训练：每轮外层在线生成 B 状态 + 基础数据集采样 B'（SEED_OVERLAP 控制跨轮保留），合并去重后编码→Bellman 备份→内层早停训练。验证指标三方案：方案 B（分层 Bellman MSE，默认运行）、方案 C（贪心展开，flag 控制）、方案 D（完整 A* 三档，flag 控制）。deepcubea_utils.py 共享状态生成；deepcubea_generate_data.py 重构调用 utils；deepcubea_online_smoke_test.py 提供冒烟测试。

## 依赖与影响链

- **上游依赖**：`config.py`（DEEPCUBEA_* 参数）、`deepcubea_network.py`（BN 网络）、`deepcubea_utils.py`（状态生成）
- **下游被依赖**：无（训练/验证/生成模块均在本摘要覆盖范围内）
- **变更扩散评估**：高（config + network + train + search + generate_data + 新建 utils，6 个文件修改 + 1 个新建）
