---
abstract_name: QUBE-15 量子 Hamiltonian 推导
source_contents:
  - "docs/qube15/derivation.md"
dependencies:
  - "wiki/abstract/docs/qube-paper.md"
  - "wiki/abstract/qube-code.md"
created_at: 2026-06-15 23:30
updated_at: 2026-06-15 23:30
---
# 摘要：QUBE-15 — 15-Puzzle 量子化密集奖励信号推导

## 核心结论与关键信息

- **量子态**：15 个瓦片，每个 $|k_x, k_y\rangle$（2D 位置量子数），空格不建模。全局态 = 15 粒子张量积。
- **位置算符**：U(1) 群表示，$\hat{K}_x|k_x,k_y\rangle = k_x|k_x,k_y\rangle$，本征值 = 当前位置 - 目标位置。
- **四阶段 Hamiltonian**（按行分组）：Phase 1→row1{tile 1-4}、Phase 2→row2{tile 5-8}、Phase 3→row3{tile 9-12}、Phase 4→row4{tile 13-15}。每阶段统一形式：局域项 $B\sum(k_x^2+k_y^2)$ + Ising 耦合项 $J\sum(k_{ix}^2 k_{jx}^2 + k_{iy}^2 k_{jy}^2)$。
- **约束处理**：软约束加权累积 $\hat{H}_k^{eff} = \hat{H}_k + \lambda\sum_{j<k}\hat{H}_j$，$\lambda=5$。全动作空间（上下左右），RL 智能体通过奖励信号自然学会复合操作。
- **参数**：B=1.0, J=0.1, λ=5.0, reward_scale=1/|S_k|（按阶段瓦片数归一化）。
- **RL 算法**：沿用现有 PPO 框架 + 单智能体按阶段切换 Hamiltonian（curriculum learning）。
- **期望值计算**：完全经典（本征态可直接代入），$\langle\hat{H}_k\rangle = B\sum(k_{ix}^2+k_{iy}^2) + J\sum(k_{ix}^2k_{jx}^2 + k_{iy}^2k_{jy}^2)$。
- **基态验证**：$\langle\hat{H}_{TOT}\rangle=0 \iff$ 全解（所有瓦片位于目标位置）。

## 内容概述

> 推导文档完整定义了 15-puzzle 的量子力学嵌入：从单粒子态 → 位置算符 → 多粒子张量积 → 四阶段 Ising Hamiltonian → 软约束奖励函数。核心创新在于将 QUBE 的 4-Hamiltonian 框架从魔方（3D 位置+取向，两种粒子类型）适配到 15-puzzle（2D 位置，单一瓦片类型），用行分组替代粒子类型分组。约束由硬约束（复合算符）降级为软约束（加权累积），保持实用性。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/qube-paper.md`（QUBE 量子框架）、`wiki/abstract/qube-code.md`（QUBE 代码实现参考）
- **下游被依赖**：`wiki/plan/qube15-implementation.md`（Phase 2 实现计划）
- **变更扩散评估**：低（独立推导文档，无代码依赖）
