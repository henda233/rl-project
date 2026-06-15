# 需求：PPO + DeepCubeA J(s) 势函数塑形实现

## 需求描述

将官方 DeepCubeA 启发函数 J(s) 作为 PPO 算法的奖励信号，实现为一个独立的 Agent 文件。

## 具体要求

1. **Actor 输入增强**：将 J(s)/j_max 拼接到环境状态特征后面，形成 Actor 输入 `[obs, j/j_max]`
2. **势函数塑形**：r_int = J(s) - γ·J(s')（Φ = -J，J 是代价函数），完全替代环境奖励
3. **网络加深**：添加残差连接（LayerNorm 版），防止梯度消失/爆炸
4. **单文件实现**：所有代码在 `ppo_deepcubea_agent.py`
5. **新 Config 参数**：在 `DeepCubeA Official Model Evaluation` 后面新增独立参数段
6. **复用华容道环境**：`env_digital_huarongdao.py`
7. **参考**：`ppo_agent.py`、`ppo_rnd_agent.py`
8. Critic 不加入 J(s)，仅 Actor 使用

## 交互记录

- 2026-06-15: 需求提出，方案讨论
- 2026-06-15: 确定 Actor 仅拼接 J(s)、Critic 保持原始 state
- 2026-06-15: 修正势函数公式为 F(s,a,s') = γ·Φ(s') - Φ(s)
- 2026-06-15: 确定 ResBlock 结构：LayerNorm 版、N=2、hidden_dim=256、逐 step 推理
