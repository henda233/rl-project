---
plan_name: PPO 训练增强（模型保存 + 通关检测 + 推理录制）
related_request: "wiki/request/ppo-training-enhancement.md"
status: completed
created_at: 2026-06-05
---
# 执行计划：PPO 训练增强

## 背景与目的

当前 `ppo_agent.py` 训练完成后仅 `plt.show()` 弹窗显示结果图，缺少模型持久化和通关检测能力。本计划为其增加：最优模型自动保存、通关胜利检测、训练结果图保存，以及独立的模型推理录制脚本。

## 内容

### 一、ppo_agent.py 改动

#### 1. 最优模型保存（`results/models/`）

- 追踪标准 A：**原始 return 最高** → `ppo_actor_best.pth` + `ppo_critic_best.pth`
- 追踪标准 B：**通关成功**（position >= 0.5）→ `ppo_actor_cleared.pth` + `ppo_critic_cleared.pth`
- 格式：分开保存 actor/critic 各自的 state_dict

#### 2. 通关胜利检测

- 每 episode 记录 `max_position`
- 当 `max_position >= 0.5`：打印 `[Cleared!]` 日志，保存通关模型

#### 3. 训练结果图（`results/imgs/`）

- `plot_return` 增加 `fig.savefig()`，同时保留 `plt.show()`
- 输出路径：`results/imgs/ppo_training_results.png`

### 二、config.py 改动

- 新增 `PPO_INFERENCE_EPISODES = 3`

### 三、run_ppo_agent.py（新建）

- 参考 `examples/run_ppo_agent.py`
- 加载 `results/models/ppo_actor_best.pth` + `ppo_critic_best.pth`
- `RecordVideo` wrapper → mp4 到 `results/videos/`
- 使用 `make_env(render_mode="rgb_array")` 保持项目一致性
- 环境：MountainCar-v0

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增 `PPO_INFERENCE_EPISODES = 3` | 无 | `config.py` | 已完成 |
| `S2` | ppo_agent.py `train_on_policy_agent` 增加最优模型追踪与保存 | `S1` | `ppo_agent.py` | 已完成 |
| `S3` | ppo_agent.py 增加通关胜利检测 | `S2` | `ppo_agent.py` | 已完成 |
| `S4` | ppo_agent.py `plot_return` 增加 `savefig` 到 `results/imgs/` | `S3` | `ppo_agent.py` | 已完成 |
| `S5` | 新建 `run_ppo_agent.py` 推理录制脚本 | `S1` | `run_ppo_agent.py` | 已完成 |

## 风险与约束声明

- `RecordVideo` 要求 `render_mode="rgb_array"`，需确保 `make_env` 支持传入 `render_mode`
- MountainCar-v0 通关条件 position >= 0.5 即为环境内置的 goal_position，到达后 episode 自动终止
- 模型保存和视频录制都需要磁盘空间，`results/` 下的子目录需自动创建

## 测试验证方案

1. 运行 `ppo_agent.py` 完成训练，检查：
   - `results/models/ppo_actor_best.pth` 和 `ppo_critic_best.pth` 存在
   - `results/imgs/ppo_training_results.png` 存在
   - 若有通关 episode，`results/models/ppo_actor_cleared.pth` 存在
   - 控制台有 `[Cleared!]` 日志
2. 运行 `run_ppo_agent.py`，检查：
   - `results/videos/` 下有 mp4 文件
   - 视频内容正常可播放

## 📝 执行记录

- `2026-06-05`: 计划已生成，需求与用户讨论对齐完毕
- `2026-06-05`: S1-S5 全部完成——config.py 新增 PPO_INFERENCE_EPISODES，ppo_agent.py 增加最优模型保存/通关检测/tqdm max_pos/savefig，根目录创建 run_ppo_agent.py 推理录制脚本
- `2026-06-05 21:00`: 最优模型保存标准从塑形return改为原始return
