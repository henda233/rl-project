---
plan_name: PPO 并行训练（多 k 值对比）
related_request: "wiki/request/ppo-parallel-training.md"
status: completed
created_at: 2026-06-05 00:00:00
---
# 执行计划：PPO 并行训练（多 k 值对比）

## 背景与目的

当前 PPO 使用单一 `POTENTIAL_K` 进行势能塑形训练。本计划实现多进程并行训练：同时开启 N 个独立环境，每个环境运行一个独立的 PPO agent，使用不同的势函数系数 k 值。训练结束后在同一张图上对比所有 k 值的训练曲线，自动选出最优 k，并保存最优模型参数。

## 内容

### 核心架构

```
main()
  ├── 计算 k_values = [k_start + i * interval for i in range(num_envs)]
  ├── 启动 N 个 multiprocessing.Process
  │     ├── Process 0: train_single_agent(k=k0, seed=0)  → Queue
  │     ├── Process 1: train_single_agent(k=k1, seed=1)  → Queue
  │     └── Process N: train_single_agent(k=kN, seed=N)  → Queue
  ├── 从 Queue 收集所有结果
  ├── plot_comparison() → 同一张图对比所有 k 值
  ├── 自动选出最优 k（最高原始 return 移动平均）
  └── 打印汇总报告
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 并行方式 | `multiprocessing.Process` | Python GIL 限制，多线程无法真正并行 |
| 模型保存 | `state_dict`（actor + critic） | 体积小，兼容性好 |
| 评估渲染 | 并行训练期间禁用 | 避免多窗口冲突 |
| 对比指标 | 原始 return + 塑形 return 双图 | 原始 return 公平对比，塑形 return 展示量级差异 |
| 最优 k 评判标准 | 最后 10% episode 的原始 return 移动平均 | 关注收敛后的最终性能，而非训练过程波动 |
| 单/多环境兼容 | `PPO_NUM_ENVS > 1` 并行，`== 1` 保持单进程 | 兼容原有调试流程 |

### 新增配置参数

```python
PPO_NUM_ENVS = 4            # 并行环境数量（默认 4）
POTENTIAL_K_START = 0.1     # k 起始值
POTENTIAL_K_INTERVAL = 2.0  # k 间隔 → k ∈ {0.1, 2.1, 4.1, 6.1}
```

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增并行训练参数 | 无 | `config.py` | 已完成 |
| `S2` | 新建 `ppo_parallel.py`：独立脚本，导入 ppo_agent 的 PPO/PolicyNet/ValueNet/moving_average | `S1` | `ppo_parallel.py` | 已完成 |
| `S3` | `train_single_agent` worker（无评估、跟踪最佳塑形 return 并保存模型） | `S2` | `ppo_parallel.py` | 已完成 |
| `S4` | `plot_comparison`（双图：原始 return + 塑形 return，N 条曲线叠加） | `S2` | `ppo_parallel.py` | 已完成 |
| `S5` | `main`：计算 k 值、spawn 多进程、Queue 收集结果、选最优 k、打印汇总报告 | `S3`, `S4` | `ppo_parallel.py` | 已完成 |
| `S6` | 更新 wiki 记忆库（计划状态、新增摘要、更新 index） | `S5` | `wiki/` | 已完成 |

## 风险与约束声明

- **Windows spawn 限制**：Windows 上 multiprocessing 使用 `spawn` 模式，worker 函数和参数必须可 pickle。`train_single_agent` 作为模块级函数可被 pickle。
- **内存占用**：N=4 时 4 个独立 PPO agent（每个约 2×256×2 + 256×1 ≈ 1K 参数），内存开销可忽略。
- **CPU 核心**：MountainCar 环境 step 极快（<1ms），瓶颈在网络前向/反向传播。4 进程在 4 核以上 CPU 可线性加速。
- **随机种子**：每个进程使用 `seed=i` 确保可复现。

## 测试验证方案

1. **冒烟测试**：`PPO_NUM_ENVS=1` 运行，结果应与当前单环境模式一致
2. **并行测试**：`PPO_NUM_ENVS=4` 运行，验证：
   - 4 个进程正常启动和退出
   - 4 条训练曲线在同一张图上展示
   - 最优 k 被正确选出
   - `results/models/` 下有对应最优 k 的模型文件
3. **模型可加载验证**：加载保存的 state_dict 到新 agent，evaluate 确认 return 与训练记录一致

## 执行记录

- `2026-06-05 00:00`: 计划已生成
- `2026-06-05`: 对齐决策——ppo_parallel.py 独立文件、模型保存标准=塑形 return 最高 episode、并行训练期间禁用评估
- `2026-06-05`: 计划执行完毕——config.py 新增 3 个参数、ppo_parallel.py 新建、wiki 记忆库更新
