---
request_name: 复现DeepCubeA算法
source_contents:
  - "docs/关于解决数字华容道问题的研究.md"
dependencies:
  - "wiki/abstract/huarongdao-env.md"
created_at: 2026-06-11 18:00:00
---
# 需求：复现 DeepCubeA 算法求解数字华容道

## 需求描述

阅读参考文档 `docs/关于解决数字华容道问题的研究.md`，复现 DeepCubeA 算法，用于求解 4×4 数字华容道（15-puzzle）。

DeepCubeA 核心思想：使用近似值迭代训练神经网络逼近未来代价函数 J(s)，然后以 J(s) 作为启发式函数 h(x)，通过加权 A* 搜索求解谜题。

## 关键决策（已对齐）

| 决策项 | 决策结果 |
|---|---|
| 拼图规模 | 直接 4×4（15-puzzle） |
| 复现范围 | 分两阶段：Plan 1 神经网络训练 + Plan 2 加权 A* 搜索 |
| 环境 | 复用现有 `DigitalHuarongdaoEnv`，可按需修改 |
| 随机步数 T | 加入 config.py，默认 U(1, 30) |
| 网络结构 | 2 FC → 4 残差块 → 1 输出，hidden_dim=256 |
| 训练集 | ≥10000，默认 10000 |
| 加权 A* λ | 加入 config.py，默认 0.3 |
| 验收标准 | 胜率（主指标），求解步数（辅助） |
| 对比实验 | 暂不对比 PPO+RND |
