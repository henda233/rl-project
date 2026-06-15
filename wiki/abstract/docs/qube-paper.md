---
abstract_name: QUBE 论文核心方法
source_contents:
  - "docs/QUBE/文献原文.md"
dependencies: []
created_at: 2026-06-15 23:00
updated_at: 2026-06-15 23:00
---
# 摘要：QUBE — 基于量子力学的魔方密集奖励信号

## 核心结论与关键信息

- **核心思想**：将魔方状态嵌入 Hilbert 空间，用 4 个 Ising-like Hamiltonian 为魔方构建密集奖励信号，解决状态空间极大（~4.3×10^19）下的稀疏奖励问题。
- **量子表示**：每个 cubie 状态由位置量子数（k_x, k_y, k_z）和取向量子数 s 描述。角块 = boson（spin-1，3 取向态），棱块 = fermion（spin-1/2，2 取向态）。
- **Hamiltonian 设计**：4 个 Hamiltonian —— H_o^c（角块取向）、H_o^e（棱块取向）、H_p^c（角块位置）、H_p^e（棱块位置），每个的基态（E=0）对应魔方该方面的已解状态。
- **四阶段求解**：逐阶段最小化单个 Hamiltonian，每进入下一阶段需保持前一阶段的基态（约束条件 [Ĥ_i, Â_{i+1}]|s_j⟩=0）。
- **奖励函数**：r = -⟨s|Ĥ|s⟩，即 Hamiltonian 期望值的负值，提供连续的密集奖励信号。
- **RL 算法**：使用 DDQN（Double DQN）+ Feed-Forward 网络，每阶段训练独立的网络。
- **结果**：4 个阶段成功率均在 90% 以上（Phase 1: 99.5%, Phase 2: 100%, Phase 3: 94.5%, Phase 4: 98.8%）。

## 内容概述

> 文献提出 QUBE (Quantum Unitary Representation for Combinatorial Problems)，将魔方组合优化问题嵌入量子力学框架。通过酉表示（unitary representation）描述 Rubik 群对魔方状态的作用，将已解状态定义为所有 Hamiltonian 的全局基态。核心创新在于用 Hamiltonian 能谱提供相空间中的连续度量（距离已解状态多远），从而将离散的组合问题转化为连续的能量最小化问题。

## 依赖与影响链

- **上游依赖**：无（独立文献）
- **下游被依赖**：`abstract/qube-code.md`（QUBE 代码探索）、Phase 2（15-puzzle 量子化推导）
- **变更扩散评估**：低
