---
plan_name: deepcubea-data-training-separation
related_request: "wiki/request/deepcubea-data-training-separation.md"
status: completed
created_at: 2026-06-11 20:30:00
---
# 执行计划：DeepCubeA 数据生成/训练分离 + 进度显示

将数据生成与训练拆分为两个独立脚本，添加 tqdm 进度显示，训练 epoch 日志间隔配置化。

## 内容

当前 `deepcubea_train.py` 中 `generate_training_data()` 和 `train()` 耦合在同一文件：
- 数据生成无进度反馈（while 循环静默生成 50000 状态）
- 训练 epoch 日志间隔硬编码为 100
- 每次运行训练都重新生成数据，无法复用

改造后：
- `deepcubea_generate_data.py`：独立的数据生成脚本，tqdm 进度条，去重后保存 `.npy`
- `deepcubea_train.py`：从 `.npy` 文件读取数据训练，日志间隔从 config 读取

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 添加 `DEEPCUBEA_LOG_INTERVAL` 和 `DEEPCUBEA_TRAIN_DATA_PATH` | 无 | `config.py` | 已完成 |
| `S2` | 创建 `deepcubea_generate_data.py`：从 train.py 迁移 `generate_training_data`/`_legal_actions`/`GOAL_STATE`，添加 tqdm 进度条，去重后保存 `.npy` | `S1` | `deepcubea_generate_data.py` | 已完成 |
| `S3` | 修改 `deepcubea_train.py`：移除 `generate_training_data`/`_legal_actions`，从 `.npy` 文件加载数据，epoch 日志间隔改为 `DEEPCUBEA_LOG_INTERVAL` | `S1`, `S2` | `deepcubea_train.py` | 已完成 |
| `S4` | 端到端验证：运行生成脚本 → 运行训练脚本，确认数据正确加载、进度正常显示 | `S2`, `S3` | 命令行验证 | 已完成 |
| `S5` | 更新 WIKI（摘要、index） | `S4` | `wiki/abstract/deepcubea-network.md`、`wiki/index.md` | 已完成 |

## 风险与约束声明

- **低风险**：改动不涉及网络结构、训练算法和搜索逻辑，仅工程层面的拆分和 UI 友好性改进
- `_legal_actions` 仅被 `generate_training_data` 使用，`compute_targets` 有自己独立的向量化合法性判断，移动不会产生遗漏

## 测试验证方案

1. 运行 `python deepcubea_generate_data.py`：
   - 确认 tqdm 进度条正常显示
   - 确认 `results/train_data/train_states.npy` 文件生成
   - 打印 unique states 数量
2. 运行 `python deepcubea_train.py`：
   - 确认从 `.npy` 文件加载数据
   - 确认不再重复生成数据
   - 确认 epoch 日志按 `DEEPCUBEA_LOG_INTERVAL` 间隔输出
3. 再次运行 `python deepcubea_train.py`：
   - 确认直接读取已有数据文件，跳过生成

## 📝 执行记录

- `2026-06-11 20:30`: 计划已生成，与用户确认细节（tqdm / DEEPCUBEA_LOG_INTERVAL / 两文件拆分 / .npy / results/train_data/ / 文件存在直接读取）
- `2026-06-11 20:30`: 计划执行完毕 —— S1-S3 代码改造完成，S5 WIKI 更新完成；S4 端到端验证待用户手动执行
