# 需求：DeepCubeA 数据生成与训练分离 + 进度显示

**日期**：2026-06-11

## 需求描述

1. 训练模型前，数据生成时添加 tqdm 进度显示。
2. 训练 epoch 日志间隔改为可配置参数 `DEEPCUBEA_LOG_INTERVAL`（添加到 config.py），替代硬编码的 100。
3. 数据生成与训练模型分离：
   - 拆分为两个独立文件：`deepcubea_generate_data.py`（生成数据并保存）和 `deepcubea_train.py`（读取数据训练）。
   - 数据格式：`.npy`，保存去重后的 unique states。
   - 路径参数：`DEEPCUBEA_TRAIN_DATA_PATH`（保存路径 = 读取路径），默认 `results/train_data/train_states.npy`。
   - 训练时若文件已存在则直接读取，跳过生成。

## 确认信息

- 进度显示：tqdm 进度条
- 日志间隔参数名：`DEEPCUBEA_LOG_INTERVAL`，默认值 100
- 拆分方式：两个独立文件
- 数据格式：`.npy`
- 路径：保存路径 = 读取路径，`results/train_data/`
- 训练时行为：文件存在则直接读取
