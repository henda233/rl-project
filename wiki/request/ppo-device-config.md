---
request_name: PPO 设备配置需求
status: completed
created_at: 2026-06-05
---
# 需求：PPO 训练设备可配置化

## 背景

PPO 训练中 device 选择硬编码在代码中：`ppo_agent.py` 自动检测 CUDA 可用性，`ppo_parallel.py` 硬编码为 `"cpu"`。用户无法通过配置文件控制训练设备。

## 需求描述

向 `config.py` 的 PPO 配置段新增布尔参数 `PPO_USE_GPU`，默认 `False`（CPU），并让 `ppo_agent.py` 和 `ppo_parallel.py` 统一读取该参数。

## 关键决策

1. **参数类型**：布尔值 `PPO_USE_GPU = False`
2. **影响范围**：`ppo_agent.py` + `ppo_parallel.py`（PPO 算法及其并行训练）
3. **移除自动检测**：不再使用 `torch.cuda.is_available()` 自动判断，用户显式控制
