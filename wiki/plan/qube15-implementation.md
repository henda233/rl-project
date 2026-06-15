---
plan_name: QUBE-15 量子 Hamiltonian 密集奖励实现
related_request: "Phase 2: 推导基于量子力学构建的适用于15-puzzle问题的密集奖励信号公式，编写代码实现"
status: executing
created_at: 2026-06-15 23:30
---
# 执行计划：QUBE-15 量子 Hamiltonian 密集奖励实现

基于 QUBE 量子框架，为 15-puzzle 构建四阶段 Ising Hamiltonian 密集奖励信号，并通过 PPO 训练求解。

## 设计决策（二次讨论确认）

| # | 决策项 | 决定 |
|---|--------|------|
| 1 | 奖励归一化 | `r = -α · ⟨Ĥ_k^eff⟩ / \|S_k\|`，α=0.1 全局缩放，对齐现有 PPO 奖励量级（~[-4, 0]）。α 放入 config 可调 |
| 2 | Phase 切换阈值 | per-tile 平均：`⟨Ĥ_k⟩ / \|S_k\| < ε`，默认 ε=0.5（e.g. Phase 1 阈值 ⟨Ĥ₁⟩ < 2.0，允许 ~2 瓦片各偏 1 步）。ε 放入 config |
| 3 | 权重继承 | 方案 A：warm start，Phase k+1 从 Phase k 训练完成的权重继续训练 |
| 4 | 打乱深度 | 各 Phase 独立参数，放入 config（`QUBE15_SHUFFLE_STEPS_P1~P4`） |
| 5 | Phase 4 的 J | 保持 J=0.1 统一不变；作为 config 参数可按 Phase 独立覆盖 |
| 6 | 对比基线 | 不做 |
| 7 | 坐标映射 | `k_y = r - r_goal`, `k_x = c - c_goal`，通过 `np.where(_grid == tile_id)` 查当前坐标 |
| 8 | 集成模式 | 方案 B：外部 wrapper，不修改 `env_digital_huarongdao.py` 或任何现有代码 |
| A | Critic 适应 | A3：不做特殊处理，依赖 PPO mini-batch 多 epoch 自然适应 |
| B | Phase 回退 | 单调推进（Phase 1→2→3→4），切换后不再回退 |
| C | 超参数差异化 | 所有 Phase 共用同一套 PPO 超参数，靠 warm start + curriculum 自然过渡 |

## 内容

参考 QUBE 算法（`baseline.py` + `Rubik.py` 的量子力学魔方环境），实现：
1. 量子态模块——15 瓦片的 Hilbert 空间嵌入
2. Hamiltonian 模块——4 个 Ising Hamiltonian + 期望值计算 + 阶段奖励
3. PPO 训练集成——四阶段 curriculum learning（外部 wrapper，不改动现有代码）
4. 单元测试——验证 Hamiltonian 正确性

**技术约束**：仅 numpy 依赖（对齐 QUBE 极简风格），PPO 训练部分可用 PyTorch（沿用项目栈）。

**PPO 训练模式**：复用现有 `ppo_agent.py` 的训练循环架构（非 `ppo_deepcubea_agent.py`），通过外部 Hamiltonian 奖励 wrapper 提供密集奖励。四阶段 curriculum 由 wrapper 控制阶段切换和奖励计算。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | 创建量子态模块 `qube15/quantum_state.py`：TileState（k_x, k_y），从 grid 构建多粒子态，坐标映射 `k_y=r-r_goal`, `k_x=c-c_goal` | 无 | `qube15/quantum_state.py` | ✅ 完成 |
| `S2` | 创建 Hamiltonian 模块 `qube15/hamiltonian.py`：4 个 Ĥ_k（局域项+Ising耦合项），期望值计算，阶段奖励函数（软约束累积 + α 缩放），参数 B/J/λ/α 可配置，纯 numpy | `S1` | `qube15/hamiltonian.py` | ✅ 完成 |
| `S3` | 创建单元测试 `qube15/test_hamiltonian.py`：已解状态 ⟨Ĥ⟩=0、单瓦片位移验证、耦合项正确性、软约束奖励数值验证、连续状态过渡测试 | `S2` | `qube15/test_hamiltonian.py` | ✅ 完成（26 pass） |
| `S4` | PPO 训练集成：Hamiltonian 奖励外部 wrapper（不改动 env）、四阶段 curriculum learning 训练循环、单调推进阶段切换（不修改现有代码）、Config 参数扩展（B/J/λ/α/ε/shuffle_steps_P1~P4/max_steps_P1~P4/episodes_P1~P4） | `S2` | `qube15/ppo_train.py`、`config.py`（仅新增参数） | ✅ 完成（冒烟通过） |
| `S5` | 运行训练并验证：Phase 1 冒烟测试（收敛到 ⟨Ĥ₁⟩/4 < ε）、完整四阶段训练、求解率评估 | `S3`, `S4` | 训练日志 + 模型文件 | 待完成 |

## 奖励公式

**Phase k 有效 Hamiltonian**：
$$\hat{H}_k^{eff} = \hat{H}_k + \lambda \sum_{j<k} \hat{H}_j$$

**奖励**：
$$r_k = -\alpha \cdot \frac{\langle \hat{H}_k^{eff} \rangle}{|S_k|}$$

**Phase 切换条件**（单调推进，不回退）：
$$\frac{\langle \hat{H}_k \rangle}{|S_k|} < \varepsilon \quad \text{（滑动平均）}$$

**默认参数**：B=1.0, J=0.1, λ=5.0, α=0.1, ε=0.5, PPO 超参数与现有 `PPO_*` 参数共用（`PPO_LR`, `PPO_GAMMA`, `PPO_EPOCHS` 等）。

## 风险与约束声明

1. **软约束 vs 硬约束**：采用加权累积而非复合算符硬约束，可能导致后期阶段智能体偶尔扰动前期已解瓦片。缓解：λ=5 提供足够惩罚力度。
2. **耦合项计算复杂度**：Ising 耦合需枚举 C(|S_k|, 2) 对，Phase 1/2/3 为 6 对，Phase 4 为 3 对，计算量可忽略。
3. **奖励尺度**：α=0.1 将归一化后的 Hamiltonian 缩放到 ~[-4, 0] 范围，与现有 PPO 每步 -1 奖励量级一致。PPO advantage normalization 进一步稳定。
4. **Phase 切换后 Critic 偏差**：切换 Phase 时 Value 函数输出会因奖励函数变化而突变，依赖 PPO mini-batch 多 epoch 更新自然适应（决策 A3）。前几个 episode 的 GAE advantage 估计可能不准，但不会导致训练崩溃。
5. **单调推进风险**：Phase 回退禁用后，若某 Phase 始终无法收敛（⟨Ĥ_k⟩ 不降），需人工介入。缓解：ε=0.5 提供合理容差。
6. **不修改现有代码**：所有新功能在 `qube15/` 目录下独立实现，仅 `config.py` 新增参数（不禁用或修改现有参数）。

## 测试验证方案

1. **S3 单元测试**（离线验证，无需 GPU）：
   - 已解 puzzle → ⟨Ĥ_k⟩ = 0 ∀k
   - 单瓦片位移 1 步 → ⟨Ĥ⟩ = B × 1（局域项） + 0（耦合项，仅一个瓦片偏离）
   - 两瓦片各位移 1 步 → ⟨Ĥ⟩ = B × 2 + J × 2（局域+耦合）
   - Phase 2 奖励 = -α × (⟨Ĥ₂⟩ + λ⟨Ĥ₁⟩) / 4，手动数值验证
   - α 缩放后奖励值域验证（确保在 ~[-4, 0]）

2. **S5 训练验证**（需 GPU）：
   - Phase 1 冒烟：100 步打乱，训练至 ⟨Ĥ₁⟩/4 < 0.5，验证 row 1 正确率 > 90%
   - 完整四阶段：从随机打乱状态出发，Phase 1→2→3→4 单调推进，验证最终求解率

## 📝 执行记录

- `2026-06-15 23:30`: 计划已生成，状态 → waiting
- `2026-06-16 00:00`: 二次讨论完成 —— 确认 8 项设计决策 + 3 个深入讨论点（A/A3, B/单调推进, C/统一参数），计划已更新
- `2026-06-16 00:30`: 三次讨论完成 —— 确认 7 组细节
- `2026-06-16 01:00`: S1 量子态模块完成（`qube15/quantum_state.py`）
- `2026-06-16 01:00`: S2 Hamiltonian 模块完成（`qube15/hamiltonian.py`）
- `2026-06-16 01:15`: S3 单元测试完成（`qube15/test_hamiltonian.py`）—— 26 tests passed
- `2026-06-16 01:30`: S4 PPO 训练集成完成（`qube15/ppo_train.py` + `config.py` 新增 20 参数）—— 冒烟通过
- `2026-06-16 01:45`: 修复 NaN 梯度爆炸（gradient clipping max_norm=0.5 + reward clipping -10）+ GBK 编码兼容；100 episode 冒烟训练通过，无 NaN，Phase 1 已有求解能力（~3-27 steps）
- `2026-06-16 01:30`: 状态 → executing：滑动窗口 100 episode 简单平均单次低于阈值切换、各 Phase 独立 shuffle/max_steps（P1:25/80 P2:50/150 P3:70/200 P4:100/300）、网络用 ResBlock+LN（无 J(s)）、全局 is_solved() 终止、奖励 r_k=-α·⟨Ĥ_k^eff⟩/|S_k|、纯函数式 Hamiltonian、Phase 4 无特殊处理
