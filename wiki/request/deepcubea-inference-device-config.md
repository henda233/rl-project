---
request_name: DeepCubeA 推理设备可配置
related_abstracts:
  - "wiki/abstract/deepcubea-search.md"
  - "wiki/abstract/deepcubea-network.md"
created_at: 2026-06-11 21:00:00
---
# 需求：DeepCubeA 推理时 GPU 使用可配置

## 背景

当前 `deepcubea_search.py` 的 `load_model()` 始终将模型加载到 GPU（若可用）。但 A* 搜索每次展开节点的 batch size 最多为 4（合法子节点数），GPU kernel launch 开销远超小 batch 计算收益。

## 需求描述

在 `config.py` 中新增参数 `DEEPCUBEA_INFERENCE_USE_GPU`，独立控制推理（A* 搜索）时是否使用 GPU，与训练参数 `DEEPCUBEA_USE_GPU` 解耦。

## 期望效果

- 默认推理使用 CPU（避免小 batch GPU 开销）
- 训练仍由 `DEEPCUBEA_USE_GPU` 独立控制
- `load_model()` 和 `predict_j`/`predict_j_batch` 根据配置决定设备
