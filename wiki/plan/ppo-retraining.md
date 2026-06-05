---
plan_name: PPO 再训练功能
related_request: "wiki/request/ppo-retraining.md"
status: completed
created_at: 2026-06-05
---
# 执行计划：PPO 再训练功能

## 背景与目的

当前 `ppo_agent.py` 只能从头训练模型。本计划为其增加再训练能力：通过在 `config.py` 中指定已有模型参数文件路径，加载权重后继续训练。仅影响 `ppo_agent.py`，不涉及 `ppo_parallel.py`。

## 用户确认的设计决策

| 决策项 | 选择 |
|---|---|
| 模型路径格式 | actor 和 critic 分开两个 `.pth` 路径 |
| 影响范围 | 仅 `ppo_agent.py` |
| 恢复内容 | 仅恢复网络权重（state_dict），optimizer 重新初始化，episode 从 1 开始 |
| episode 计数 | 从 1 重新计数 |
| 输出目录 | 再训练时保存到带时间戳的新目录 `results/retrain_YYYYMMDD_HHMMSS/` |

## 内容

### 一、config.py 改动

新增两个配置项：

```python
PPO_ACTOR_MODEL_PATH = ""   # 为空则从头训练；非空则加载指定路径的actor权重进行再训练
PPO_CRITIC_MODEL_PATH = ""  # 为空则从头训练；非空则加载指定路径的critic权重进行再训练
```

- 两个路径**同时为空** → 从头训练（保持现有行为）
- 两个路径**同时非空** → 再训练模式
- 仅一个非空 → 视为异常，`main()` 中打印警告并退出

### 二、ppo_agent.py 改动

#### 1. `main()` 函数

- 导入新配置项 `PPO_ACTOR_MODEL_PATH`, `PPO_CRITIC_MODEL_PATH`
- 判断再训练模式：
  - 若两个路径均非空 → `is_retrain = True`
  - 生成时间戳目录：`results/retrain_{datetime.now():%Y%m%d_%H%M%S}/`
  - 日志输出正在加载的模型路径
  - `torch.load` + `load_state_dict` 加载权重到 actor/critic
  - 若文件不存在 → `torch.load` 自然抛出 `FileNotFoundError`
- 将 `results_dir` 传入 `train_on_policy_agent`

```python
def main():
    device = torch.device("cuda" if PPO_USE_GPU else "cpu")
    env = make_env()
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(...)

    is_retrain = bool(PPO_ACTOR_MODEL_PATH and PPO_CRITIC_MODEL_PATH)
    if is_retrain:
        results_dir = f"results/retrain_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"[Retrain] Loading actor from: {PPO_ACTOR_MODEL_PATH}")
        agent.actor.load_state_dict(torch.load(PPO_ACTOR_MODEL_PATH, map_location=device))
        print(f"[Retrain] Loading critic from: {PPO_CRITIC_MODEL_PATH}")
        agent.critic.load_state_dict(torch.load(PPO_CRITIC_MODEL_PATH, map_location=device))
        print(f"[Retrain] Results will be saved to: {results_dir}/")
    else:
        results_dir = "results"

    return_list, shaped_return_list = train_on_policy_agent(
        env, agent, PPO_NUM_EPISODES, results_dir,
    )
    ...
```

#### 2. `train_on_policy_agent()` 函数

- 新增参数 `results_dir: str = "results"`
- 模型保存路径改为 `os.path.join(results_dir, "models", ...)`
- 其余逻辑不变

#### 3. `plot_return()` 函数

- 新增参数 `results_dir: str = "results"`
- 图片保存路径改为 `os.path.join(results_dir, "imgs", ...)`
- `main()` 调用时传入 `results_dir`

### 三、不涉及的文件

- `ppo_parallel.py` — 不修改
- `run_ppo_agent.py` — 不修改
- `env.py` — 不修改

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增 `PPO_ACTOR_MODEL_PATH` 和 `PPO_CRITIC_MODEL_PATH` | 无 | `config.py` | 待完成 |
| `S2` | ppo_agent.py `main()` 增加再训练模式判断、模型加载、时间戳目录生成 | `S1` | `ppo_agent.py` | 待完成 |
| `S3` | ppo_agent.py `train_on_policy_agent()` 增加 `results_dir` 参数 | `S2` | `ppo_agent.py` | 待完成 |
| `S4` | ppo_agent.py `plot_return()` 增加 `results_dir` 参数 | `S2` | `ppo_agent.py` | 待完成 |

## 风险与约束声明

- 再训练仅恢复权重，optimizer 状态（如 Adam 的动量）丢失，初期训练可能略有波动
- `torch.load` 默认使用 `weights_only=False`（PyTorch 2.0+ 需注意安全警告），加载自己保存的模型无安全风险
- 若用户指定了不匹配网络结构的 `.pth` 文件，`load_state_dict` 会抛出 `Missing key(s)` / `Unexpected key(s)` 错误
- 时间戳目录在每次再训练时创建新目录，不会覆盖历史结果

## 测试验证方案

1. **从头训练不变**：`PPO_ACTOR_MODEL_PATH = ""` 且 `PPO_CRITIC_MODEL_PATH = ""`，运行 `ppo_agent.py`，行为和输出与改动前完全一致
2. **再训练流程**：
   - 先完成一次短训练（如 500 episode），得到 `results/models/ppo_actor_best.pth`
   - 修改 config.py 填入路径，运行 `ppo_agent.py`
   - 验证：控制台打印 `[Retrain] Loading actor from: ...`
   - 验证：`results/retrain_YYYYMMDD_HHMMSS/` 目录被创建
   - 验证：其下 `models/` 和 `imgs/` 子目录有输出文件
3. **异常场景**：
   - 路径文件不存在 → `FileNotFoundError`
   - 仅填一个路径 → 打印警告并退出

## 📝 执行记录

- `2026-06-05`: 计划已生成，与用户对齐完毕
- `2026-06-05`: 执行完毕。config.py 新增 PPO_ACTOR_MODEL_PATH/PPO_CRITIC_MODEL_PATH/PPO_RETRAIN_NUM_EPISODES；ppo_agent.py 导入 sys/datetime，main() 增加再训练模式判断、权重加载、时间戳目录生成；train_on_policy_agent/plot_return 增加 results_dir 参数。单路径异常使用 sys.exit(1) 退出。
