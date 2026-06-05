---
abstract_name: PPO 推理脚本
source_contents:
  - "run_ppo_agent.py"
  - "run_ppo_rnd_agent.py"
  - "config.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/rnd-impl.md"
created_at: 2026-06-05 20:30:00
updated_at: 2026-06-05 20:30:00
---
# 摘要：PPO 推理脚本

## 核心结论与关键信息

- `run_ppo_agent.py` 和 `run_ppo_rnd_agent.py` 分别为 PPO 和 RND+PPO 的推理录制脚本
- 两个脚本均加载 actor + critic 权重，视频输出到 `results/videos/<timestamp>/`
- 模型路径通过 config.py 配置：`PPO_INFERENCE_ACTOR_PATH/CRITIC_PATH`（PPO，默认 `results/models/ppo_actor_best.pth`）、`RND_INFERENCE_ACTOR_PATH/CRITIC_PATH`（RND，默认空，需手动指向 cleared 模型）
- RND 推理不需要加载 RND 模块（RND 仅用于训练的探索奖励）

## 内容概述

> `run_ppo_agent.py` 加载 PPO 训练的最优模型进行推理并录制视频。`run_ppo_rnd_agent.py` 加载 RND+PPO 训练的 cleared 模型（max_position >= 0.5）进行推理。两者结构一致：构造 agent → 加载权重 → RecordVideo wrapper → 运行 N 个 episode → 输出 return/length/max_pos。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO agent）、`wiki/abstract/rnd-impl.md`（RND agent）
- **下游被依赖**：无
- **变更扩散评估**：低（两个推理脚本独立，config.py 仅追加 4 个路径参数）
