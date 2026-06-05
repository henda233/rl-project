---
abstract_name: PPO 并行训练实现
source_contents:
  - "ppo_parallel.py"
  - "config.py"
dependencies:
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/potential-based-shaping.md"
  - "wiki/request/ppo-parallel-training.md"
  - "wiki/plan/ppo-parallel-training.md"
created_at: 2026-06-05
updated_at: 2026-06-05
---
# 摘要：PPO 并行训练实现

## 核心结论与关键信息

- 独立脚本 `ppo_parallel.py`，导入 `ppo_agent` 的 PolicyNet/ValueNet/PPO/moving_average，零耦合修改
- `multiprocessing.Process` + `spawn` 模式，跨平台一致
- 每个 worker 使用独立 k 值进行 potential-based shaping（Φ=k·pos），训练 500 episode
- 模型保存标准：训练过程中**塑形 return 最高**的单次 episode 对应的 state_dict
- 并行训练期间**完全禁用评估**（避免多窗口冲突）
- 对比指标：原始 return + 塑形 return 双 subplot，N 条曲线叠加，移动平均平滑
- 最优 k 评判标准：塑形 return 单次 episode 最高
- `PPO_USE_GPU` 控制 device 创建（`train_single_agent` 统一读取配置，默认 CPU）

## 内容概述

> `ppo_parallel.py` 包含：`train_single_agent(k, seed, queue)` worker 函数（POTENTIAL_K 参数化、跟踪最佳塑形 return、自动保存模型到 results/models/）、`plot_comparison(results)` 双 subplot 对比函数（viridis 色阶区分 N 条曲线）、`main()` 入口（计算 k 值、spawn N 进程、Queue 收集结果、排序输出汇总报告）。
>
> `config.py` 新增 `PPO_NUM_ENVS=4`、`POTENTIAL_K_START=0.1`、`POTENTIAL_K_INTERVAL=2.0` → k ∈ {0.1, 2.1, 4.1, 6.1}。

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 文件组织 | 独立 `ppo_parallel.py` | 不修改 ppo_agent.py，零耦合 |
| 并行方式 | `multiprocessing.Process` + spawn | Python GIL 限制，Windows/Linux 一致 |
| 模型保存标准 | 训练全程塑形 return 最高 episode | 用户指定：关注塑形奖励峰值 |
| 评估策略 | 并行训练期间完全禁用 | 避免多窗口渲染冲突 |
| 最优 k 标准 | 塑形 return 最高（best_shaped_return） | 与模型保存标准一致 |
| 单进程兼容 | 不保留 | ppo_parallel.py 专用于多 k 对比 |

## 依赖与影响链

- **上游依赖**：`wiki/abstract/ppo-impl.md`（PPO 类、网络结构）、`wiki/abstract/potential-based-shaping.md`（Φ=k·pos 公式）
- **下游被依赖**：无
- **变更扩散评估**：低（ppo_parallel.py 独立文件，config.py 仅追加 4 个参数，ppo_agent.py 同步读取 PPO_USE_GPU）
