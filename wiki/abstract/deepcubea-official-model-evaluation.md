---
abstract_name: DeepCubeA 官方预训练模型评估计划
source_contents:
  - "wiki/plan/deepcubea-official-model-evaluation.md"
  - "deepcubea_official_network.py"
  - "deepcubea_official_data.py"
  - "deepcubea_official_search.py"
  - "data/tiles.npy"
  - "data/solution_lengths.npy"
  - "data/model_state_dict.pt"
  - "deepcubea_search.py"
  - "config.py"
dependencies:
  - "wiki/abstract/deepcubea-official-repo.md"
  - "wiki/abstract/deepcubea-search.md"
  - "wiki/abstract/deepcubea-network.md"
created_at: 2026-06-13 23:50
updated_at: 2026-06-14 17:30
---
# 摘要：DeepCubeA 官方预训练模型评估计划

## 核心结论与关键信息

- **目标**：用官方预训练模型跑通三种评估，建立质量基线
- **架构**：独立新文件 `deepcubea_official_*.py`（network/data/search）；`deepcubea_search.py` 贪心展开从截断 A* 改为纯 argmin J(s')
- **数据预处理（方案 C）**：一次性脚本将 `data_0.pkl` 转为 `data/tiles.npy (500,16) int64` + `data/solution_lengths.npy (500,) int64`；同时拷贝模型到 `data/model_state_dict.pt`，所有文件统一在 `data/`
- **state_dict 加载**：官方 DataParallel 训练产物，72 个 key 全部带 `module.` 前缀，`load_official_model` 自动剥离
- **推理设备**：复用 `DEEPCUBEA_INFERENCE_USE_GPU`（当前 True），与 `deepcubea_search.py` 一致
- **冒烟验证通过**：J(s) ∈ [36.0, 58.2] mean=52.85，完美对齐解路径长度 [36, 69] mean=52.0
- **Bellman MSE 已验证**（修复后）：Overall mean=0.362, median=0.202, std=0.371；分档结果正常（Str1: 0.175, Str2: 0.209, Str3: 0.699），MSE 随解长度增大符合预期
- **三项评估**：Bellman MSE（始终执行）+ 纯贪心展开（argmin J(s') 步步跟随，max_steps 上限）+ Full A*（λ=1.0 加权 A*）
- **已知 Bug：ptr_per_state 索引错误**（2026-06-14）：`_compute_bellman_errors_official` 中 start/end 标记成对存储但取值步长=1 而非 2，导致奇数索引 state 拿到 0 children → error=J(s)²；修复方案 B（offsets 单条记录）已应用。根因分析：children 收集与父 state 错位，模型参数和 J(s) 预测本身正确
- **难度分档**：按 `solutions` 实际解路径长度排序后三等分（短/中/长各约 167 个）
- **Config 键**：新增 `DEEPCUBEA_OFFICIAL_*` 8 键 + `DEEPCUBEA_GREEDY_MAX_STEPS`；重命名 `DEEPCUBEA_VAL_GREEDY_EXPAND` → `DEEPCUBEA_VAL_GREEDY_MAX_STEPS`（语义：节点上限→步数上限）
- **测试数据来源开关**（2026-06-14）：新增 `DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA` 布尔开关（默认 True）；False 时随机游走生成测试数据 + K 代理分层，新增 3 配置键（T_MIN/T_MAX/NUM_TEST_STATES），种子复用 `DEEPCUBEA_VAL_SEED`

## 内容概述

> **交付物**（3 新文件 + 3 数据文件 + 2 修改文件）：
> - `deepcubea_official_network.py`：ResnetModel（15M 参数，BN，内部 one_hot）+ OfficialModelWrapper（predict_j / predict_j_batch）+ load_official_model（自动剥离 `module.` 前缀）
> - `deepcubea_official_data.py`：`load_official_test_data`（纯 np.load）
> - `deepcubea_official_search.py`：`stratified_bellman_mse_official`（始终执行）、`_greedy_expand_official`（纯贪心）、`greedy_expansion_eval_official`（Flag）、`weighted_astar_official`（λ=1.0）、`evaluate_official`（Flag）、`_generate_test_states`（随机游走生成测试数据，支持非官方数据模式）
> - `data/tiles.npy` + `data/solution_lengths.npy` + `data/model_state_dict.pt`（55MB）
> - 修改：`deepcubea_search.py`（贪心改为纯 argmin J(s')）、`config.py`（新增 9 键 + 重命名 1 键 → 新增 13 键 + 重命名 1 键）

## 依赖与影响链

- **上游依赖**：`wiki/abstract/deepcubea-official-repo.md`（仓库结构与官方实现细节）、`wiki/abstract/deepcubea-search.md`（搜索框架与评估模式）、`wiki/abstract/deepcubea-network.md`（`transition`/`get_children` 工具函数）
- **下游被依赖**：无
- **变更扩散评估**：低（新增 3 个文件 + 8 个 config 键，不修改任何现有文件）
