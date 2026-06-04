---
abstract_name: 智能体记录与可视化
source_contents:
  - "docs/gymnasium/记录智能体.md"
dependencies:
  - "wiki/abstract/docs/project-overview.md"
created_at: 2026-06-04 18:36:00
updated_at: 2026-06-04 18:36:00
---
# 摘要：智能体记录与可视化

## 核心结论与关键信息

- **`RecordEpisodeStatistics`**：跟踪每 episode 的 reward、length、time，数据存入 `env.return_queue`、`env.length_queue`、`env.time_queue`
- **`RecordVideo`**：录制 MP4 视频，需 `render_mode="rgb_array"`，通过 `episode_trigger` 控制录制频率
- **训练期间**：定期录制（如每 250 episode），重点看统计数据
- **评估期间**：录制每个 episode，全面评估性能
- **可视化**：用 matplotlib 绘制 episode rewards 及移动平均曲线（滑动窗口）

## 内容概述

> 讲解如何使用 gymnasium wrappers 记录智能体行为。包含评估模式（全量录制 + 统计）和训练模式（定期录制）两种场景的完整代码示例，以及结果解读方法。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/project-overview.md`
- **下游被依赖**：`wiki/request/initial-requirements.md`（训练可视化需求的直接参考）
- **变更扩散评估**：低（wrapper 组合方式，非核心逻辑）
