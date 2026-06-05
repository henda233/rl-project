---
request_name: PPO 训练增强（模型保存 + 通关检测 + 推理录制）
status: pending
created_at: 2026-06-05
---
# 需求：PPO 训练增强

## 需求描述

1. **最优模型保存**：PPO 训练过程中，按塑形 return 最高和通关成功两个标准自动保存最优模型参数到 `results/models/`
2. **通关胜利检测**：车辆位置 >= 0.5 判定为通关，记录日志并保存模型
3. **训练结果图保存**：训练完成后保存结果图到 `results/imgs/`
4. **推理录制脚本**：参考 `examples/run_ppo_agent.py`，实现加载模型参数、运行智能体、录制 mp4 视频

## 关联计划

- [PPO 训练增强执行计划](../plan/ppo-training-enhancement.md)

## 关联摘要

- [PPO 算法实现](../abstract/ppo-impl.md)
- [PPO 并行训练](../abstract/ppo-parallel-training.md)
