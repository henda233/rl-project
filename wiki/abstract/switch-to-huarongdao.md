---
abstract_name: PPO/RND Agent 切换华容道环境
source_contents:
  - "wiki/request/switch-to-huarongdao.md"
  - "wiki/plan/switch-to-huarongdao.md"
  - "config.py"
  - "env_digital_huarongdao.py"
  - "ppo_agent.py"
  - "ppo_rnd_agent.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/rnd-impl.md"
  - "wiki/abstract/docs/digital-huarongdao-design.md"
created_at: 2026-06-10 15:00:00
updated_at: 2026-06-10 17:30:00
---
# 摘要：PPO/RND Agent 切换华容道环境

## 核心结论与关键信息

- **破坏性更新**：ppo_agent.py 和 ppo_rnd_agent.py 从 MountainCar-v0 完全切换为 DigitalHuarongdaoEnv，计划已执行完毕
- **删除奖励塑形**：POTENTIAL_K 势能塑形及塑形配置段彻底移除，直接使用华容道环境原始奖励（-1/步 + 首次归位+5）
- **RND r_ext 修复**：从硬编码 `np.full(..., -1.0)` 改为 `np.array(episode_rewards)`，使用 episode 实际外部奖励
- **监控指标替换**：max_position → max_placed（episode 内最高归位数）+ step_count（步数），进度条同步更新
- **通关检测**：`max_position >= 0.5` → `terminated` flag
- **RND_HIDDEN_DIM**：128 → 256
- **删除再训练**：ppo_agent.py 再训练分支及 config 中 PPO_ACTOR_MODEL_PATH / PPO_CRITIC_MODEL_PATH / PPO_RETRAIN_NUM_EPISODES / PPO/RND 推理路径全部移除
- **不改动**：env.py、agent.py、ppo_parallel.py、推理脚本、RND β 衰减参数（它们引用的被删除配置会导致 ImportError，用户已确认接受）
- **非法动作惩罚**（2026-06-10，替代 Action Masking）：采样阶段不再屏蔽非法动作，改为环境区分合法/非法奖励（`HUARONGDAO_LEGAL_STEP_REWARD=-1` / `HUARONGDAO_ILLEGAL_STEP_REWARD=-2`）。原因：Action Masking 导致采样分布（masked）与更新分布（unmasked）不一致，ratio 溢出产生 NaN。非法动作惩罚方案保持采样=更新分布，避免 NaN，同时通过 -2 vs -1 差异化激励智能体学习避开非法动作

## 内容概述

> 执行 5 步计划完成切换：S1 清理 config.py（删除AC/再训练/推理/塑形/并行训练段，RND_HIDDEN_DIM→256）→ S2 env_digital_huarongdao.py 增强 info → S3 ppo_agent.py 破坏性改造（华容道环境、删除塑形/再训练、max_placed+steps监控、terminated通关、单图）→ S4 ppo_rnd_agent.py 破坏性改造（华容道环境、r_ext从实际reward获取、max_placed+steps监控、terminated通关）→ S5 WIKI 维护。后追加 Action Masking（采样屏蔽非法动作），因 NaN 问题回退为非法动作惩罚方案（env 区分合法 -1 / 非法 -2 奖励）。

## 依赖与影响链

- **上游依赖**：`ppo-impl.md`（PPO 结构）、`rnd-impl.md`（RND 结构）、`digital-huarongdao-design.md`（华容道环境规格）
- **下游被依赖**：无
- **变更扩散评估**：高（config.py 删除多个配置段导致 agent.py / ppo_parallel.py / 推理脚本不可用）
