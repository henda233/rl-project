---
abstract_name: QUBE-15 实现
source_contents:
  - "qube15/__init__.py"
  - "qube15/quantum_state.py"
  - "qube15/hamiltonian.py"
  - "qube15/test_hamiltonian.py"
  - "qube15/ppo_train.py"
  - "config.py"
dependencies:
  - "wiki/abstract/docs/qube15-derivation.md"
  - "wiki/abstract/docs/qube-paper.md"
  - "wiki/abstract/qube-code.md"
created_at: 2026-06-15 23:45
updated_at: 2026-06-15 23:45
---
# 摘要：QUBE-15 量子 Hamiltonian 密集奖励实现

## 核心结论与关键信息

- **量子态模块** (`qube15/quantum_state.py`)：纯函数式，grid → tile (k_x, k_y) 坐标映射，k_x=c-c_goal, k_y=r-r_goal，15 瓦片 × 4 Phase 分组（按行），空格不建模
- **Hamiltonian 模块** (`qube15/hamiltonian.py`)：纯 numpy 函数式，4 个 Ĥ_k（局域 BΣ(k_x²+k_y²) + Ising 耦合 JΣ(k_ix²k_jx²+k_iy²k_jy²)），软约束加权累积 Ĥ_k^eff = Ĥ_k + λΣ_{j<k}Ĥ_j，奖励 r_k = -α·⟨Ĥ_k^eff⟩/|S_k|
- **单元测试** (`qube15/test_hamiltonian.py`)：26 tests passed，覆盖已解零值、单瓦片位移、两瓦片耦合、软约束数值、Phase 4 归一化、连续过渡、随机/最坏值域有限性
- **PPO 训练** (`qube15/ppo_train.py`)：ResBlock+LN 网络（对齐官方架构，无 J(s) 输入）、HamiltonianRewardWrapper（外部 gym.Wrapper，不改动现有代码）、四阶段 curriculum（单调推进，滑动窗口切换）、gradient clipping max_norm=0.5 + reward clipping [-10, 0] 防梯度爆炸
- **Config** (`config.py`)：20 个 QUBE15_* 参数（B/J/λ/α/ε/window/shuffle_steps_P1~P4/max_steps_P1~P4/PPO 超参），独立于现有参数族
- **默认参数**：B=1.0, J=0.1, λ=5.0, α=0.1, ε=0.5, window=100, shuffle_steps=[25,50,70,100], max_steps=[80,150,200,300]

## 内容概述

> 将 QUBE 量子框架从魔方适配到 15-puzzle，实现四个模块：(1) quantum_state.py 提取瓦片位置量子数，(2) hamiltonian.py 计算四阶段 Ising Hamiltonian 期望值与密集奖励，(3) test_hamiltonian.py 26 项单元测试验证公式正确性，(4) ppo_train.py 集成 ResBlock+LN PPO + HamiltonianRewardWrapper + 四阶段 curriculum learning。100 episode 冒烟训练通过，Phase 1 已具备求解能力。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/qube15-derivation.md`（推导公式）、`wiki/abstract/qube-code.md`（QUBE 参考实现）、`wiki/abstract/docs/qube-paper.md`（QUBE 论文）
- **下游被依赖**：`wiki/plan/qube15-implementation.md`（本实现为其 S1-S4 交付物）
- **变更扩散评估**：低（qube15/ 独立目录，仅 config.py 新增参数）
