---
abstract_name: 数字华容道游戏环境设计
source_contents:
  - "docs/数字华容道游戏环境设置.md"
dependencies:
  - "wiki/abstract/gymnasium/custom-env.md"
created_at: 2026-06-10 12:00:00
updated_at: 2026-06-10 14:00:00
---
# 摘要：数字华容道游戏环境设计

## 核心结论与关键信息

- **游戏规则**：n×n 方格面板，1..n²-1 数字 + 空格，点击空格相邻方块滑入空格
- **状态空间**：n×n 矩阵，展平为 (n²,) Box，空格用 0 表示
- **动作空间**：Discrete(4)，上/下/左/右（空格移动方向），无效动作返回原状态
- **奖励**：-1/步 + 首次归位 +5（每个数字每 episode 仅一次）
- **终止条件**：全部归位（terminated）或达到最大步数（truncated）
- **初始状态**：正向打乱（从目标状态执行 N 次随机合法移动，保证可解）
- **可配置参数**：`HUARONGDAO_N`(n)、`HUARONGDAO_MAX_STEPS`(200)、`HUARONGDAO_SHUFFLE_STEPS`(50)、`HUARONGDAO_FIRST_PLACEMENT_REWARD`(5)
- **渲染**：ansi + rgb_array（纯 numpy）+ human

## 内容概述

> 设计文档定义了数字华容道的游戏规则、状态空间、动作空间、奖励设置和终止条件。经过多轮讨论对齐后确认：标准相邻方块移动玩法、固定 4 方向动作编码、正向打乱保证可解性。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/gymnasium/custom-env.md`（Gymnasium Env 继承规范）
- **下游被依赖**：`wiki/plan/implement-huarongdao-env.md`（实现计划）
- **变更扩散评估**：中（新环境模块，独立于现有 MountainCar 相关代码）
