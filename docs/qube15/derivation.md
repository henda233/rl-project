# QUBE-15: 基于量子力学的 15-Puzzle 密集奖励信号推导

## 1. 量子态表示

### 单瓦片态

15-puzzle 有 15 个编号瓦片（空格不建模为量子粒子）。瓦片 i（i=1..15）的量子态：

$$|tile_i\rangle = |k_x^{(i)}, k_y^{(i)}\rangle$$

其中：
- $k_x^{(i)} = col_i - col_i^g$，$k_y^{(i)} = row_i - row_i^g$
- 目标位置（0-indexed）：$row_i^g = \lfloor(i-1)/4\rfloor$，$col_i^g = (i-1) \bmod 4$
- $k_x, k_y \in \{-3, -2, -1, 0, 1, 2, 3\}$

### 位置算符（U(1) 群表示）

$$\hat{K}_x^{(i)}|k_x,k_y\rangle = k_x^{(i)}|k_x,k_y\rangle, \quad \hat{K}_y^{(i)}|k_x,k_y\rangle = k_y^{(i)}|k_x,k_y\rangle$$

### 全局态（15 粒子张量积）

$$|Puzzle\rangle = \bigotimes_{i=1}^{15} |k_x^{(i)}, k_y^{(i)}\rangle$$

### 空格处理

空格不建模为量子粒子。理由：
- 空格无编号/身份，是「空洞」而非实体
- 空格位置由其余 15 瓦片完全确定
- 类比 QUBE 不建模中心块（centrals，仅 1 面、不可移动）

## 2. 四阶段 Hamiltonian 设计（方案 C）

### 阶段划分（按行分组）

| 阶段 | 瓦片集合 | 瓦片数 | 物理含义 |
|---|---|---|---|
| Phase 1 | {1, 2, 3, 4} | 4 | 第一行归位 |
| Phase 2 | {5, 6, 7, 8} | 4 | 第二行归位（保持第一行） |
| Phase 3 | {9, 10, 11, 12} | 4 | 第三行归位（保持前两行） |
| Phase 4 | {13, 14, 15} | 3 | 第四行归位（空格自动归位） |

### Hamiltonian 统一形式

$$\hat{H}_k = \sum_{i \in S_k} B \cdot \left([\hat{K}_x^{(i)}]^2 + [\hat{K}_y^{(i)}]^2\right) + \sum_{\substack{i<j \\ i,j \in S_k}} J \cdot \left([\hat{K}_x^{(i)}]^2[\hat{K}_x^{(j)}]^2 + [\hat{K}_y^{(i)}]^2[\hat{K}_y^{(j)}]^2\right)$$

- 第一项：局域场（local field）——单个瓦片偏离目标的惩罚
- 第二项：Ising 耦合——多个瓦片同时偏离的非线性惩罚

### 期望值计算（经典计算）

由于 $|Puzzle\rangle$ 是 $\hat{K}$ 的本征态，期望值退化为经典代数：

$$\langle \hat{H}_k \rangle = B\sum_{i \in S_k} (k_{ix}^2 + k_{iy}^2) + J\sum_{\substack{i<j \\ i,j \in S_k}} (k_{ix}^2 k_{jx}^2 + k_{iy}^2 k_{jy}^2)$$

### 基态验证

$$\langle \hat{H}_k \rangle = 0 \iff \forall i \in S_k: k_{ix}=k_{iy}=0$$

$$\langle \hat{H}_{TOT} \rangle = \langle \hat{H}_1 + \hat{H}_2 + \hat{H}_3 + \hat{H}_4 \rangle = 0 \iff \text{全解}$$

## 3. 约束处理（软约束 + 全动作空间）

QUBE 原文的硬约束 $[\hat{H}_i, \hat{A}_{i+1}]|s_j\rangle = 0$ 需复合算符（composed operators），枚举复杂且未开源。

采用**加权累积 Hamiltonian**作为软约束：

$$\hat{H}_k^{eff} = \hat{H}_k + \lambda \sum_{j<k} \hat{H}_j$$

各阶段奖励：
```
Phase 1: r₁ = -⟨Ĥ₁⟩
Phase 2: r₂ = -(⟨Ĥ₂⟩ + λ⟨Ĥ₁⟩)
Phase 3: r₃ = -(⟨Ĥ₃⟩ + λ⟨Ĥ₁⟩ + λ⟨Ĥ₂⟩)
Phase 4: r₄ = -(⟨Ĥ₄⟩ + λ⟨Ĥ₁⟩ + λ⟨Ĥ₂⟩ + λ⟨Ĥ₃⟩)
```

λ > 1 确保破坏已解行的惩罚大于推进当前行的收益。RL 智能体通过奖励信号自然学会复合操作（如在 Phase 2 中先 Down 再操作再 Up 以避开 row 1 瓦片）。

动作空间保持 4 个基础动作（上/下/左/右），不做显式限制。

## 4. 参数设定

| 参数 | 值 | 含义 |
|---|---|---|
| B | 1.0 | 局域场强度 |
| J | 0.1 | Ising 耦合常数 |
| λ | 5.0 | 跨阶段软约束权重 |
| reward_scale | 1/|S_k| | 奖励归一化（按阶段瓦片数平均） |

J=0.1 的理由：$k_x^2 k_y^2_{max} = 81$，J=1 时耦合项 $6 \times 2 \times 81 = 972$ 压倒局域项 $72$；J=0.1 保持两项同量级。

## 5. RL 算法选择

沿用现有 PPO 框架 + 单智能体切换 Hamiltonian（方案 B 决策），即：一个 PPO agent 在训练过程中按阶段切换奖励函数，类似 curriculum learning。

## 6. 与 QUBE 的对应关系

| 要素 | QUBE（魔方） | QUBE-15（15-puzzle） |
|---|---|---|
| 单粒子态 | $\|k_x,k_y,k_z,s\rangle$ | $\|k_x,k_y\rangle$ |
| 粒子类型 | boson(角块) + fermion(棱块) | 单一类型（瓦片） |
| 自由度 | 位置(3D) + 取向(spin) | 位置(2D) |
| Hamiltonian 数 | 4（2类型 × 2自由度） | 4（4行分组） |
| 约束 | [Ĥ_i, Â_{i+1}] = 0（复合算符） | 加权累积（软约束） |
| RL 算法 | DDQN × 4（每阶段独立网络） | PPO × 1（单智能体切换 Ĥ） |
| 依赖 | TensorFlow + numpy | PyTorch + numpy |
