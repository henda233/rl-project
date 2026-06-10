# 需求：将 PPO 和 PPO+RND Agent 切换为数字华容道环境（破坏性更新）

## 需求描述

将 `ppo_agent.py` 和 `ppo_rnd_agent.py` 的训练环境从 MountainCar-v0 完全切换为数字华容道（DigitalHuarongdaoEnv），删除所有 MountainCar 相关代码和配置，不做向后兼容。

## 核心决策（已与用户对齐）

1. **彻底删除奖励塑形**：移除 POTENTIAL_K 势能塑形，直接使用华容道环境原始奖励（-1/步 + 首次归位+5）
2. **RND r_ext 从环境实际奖励获取**：不再硬编码 `-1.0`，改为从 episode 中收集的实际 env reward
3. **监控指标**：同时显示 placed_count（已归位方块数）和 step_count（步数）
4. **通关检测**：直接检查 `terminated` flag
5. **超参数调整**：RND_HIDDEN_DIM 128→256（状态空间 2→9 维）；其余不变
6. **env.py 不动**：agent 直接从 `env_digital_huarongdao` import `make_huarongdao_env`
7. **删除再训练**：ppo_agent.py 移除再训练分支及相关 config
8. **不添加 RND 再训练**
9. **保留 RND，β 衰减策略不变**

## 涉及文件

| 文件 | 操作 |
|---|---|
| `config.py` | 删除 AC 段、奖励塑形、PPO 再训练配置、PPO 并行训练配置；RND_HIDDEN_DIM 128→256 |
| `env_digital_huarongdao.py` | `_get_info()` 新增 `placed_count` 字段 |
| `ppo_agent.py` | 环境切换、删除塑形/再训练、替换监控指标 |
| `ppo_rnd_agent.py` | 环境切换、修复 r_ext 硬编码、替换监控指标 |

## 不做

- 不修改 `env.py`、`agent.py`、`ppo_parallel.py`、推理脚本（它们依赖的 config 被删除后会坏，已确认接受）
- 不添加 RND 再训练功能
- 不修改 RND β 衰减参数
