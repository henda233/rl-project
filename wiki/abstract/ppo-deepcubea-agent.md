---
abstract_name: PPO + DeepCubeA J(s) 势函数塑形实现
source_contents:
  - "ppo_deepcubea_agent.py"
  - "config.py"
dependencies:
  - "wiki/abstract/docs/ppo-deepcubea-shaping-analysis.md"
  - "wiki/abstract/ppo-impl.md"
  - "wiki/abstract/deepcubea-official-model-evaluation.md"
created_at: 2026-06-15 17:00:00
updated_at: 2026-06-15 19:30:00
---
# 摘要：PPO + DeepCubeA J(s) 势函数塑形实现

## 核心结论与关键信息

- **文件**：`ppo_deepcubea_agent.py`，独立可运行，复用 `env_digital_huarongdao.py` 环境和 `deepcubea_official_network.py` 的模型加载
- **势函数塑形**：Φ(s) = -J(s)，F(s,a,s') = γ·J(s') - J(s)，γ=0.98 对齐 PPO MDP 折扣因子（Ng et al. 最优策略不变性）
- **总奖励**：r_total = -1 + β·(J(s) - γ·J(s'))，β=1.0
- **Actor 输入增强**：concat(obs, J(s)/j_max)，j_max=200.0；Critic 保持原始 obs
- **网络架构**：对齐 `deepcubea_official_network.py` ResBlock（BN→LN）—— FC→LN→ReLU→FC→LN→+skip→ReLU，2 个残差块，hidden_dim=256
- **Actor 输出**：logits（无 softmax），Categorical 内部 log_softmax
- **Config**：18 个 `PPO_DEEPCUBEA_*` 参数，独立 section
- **模型保存**：三种策略 —— latest（每 `SAVE_INTERVAL` episode 覆盖）、best（return 新高）、final（训练结束）
- **图表保存**：三面板累积曲线（return / steps / win rate），每 `SAVE_INTERVAL` 覆盖式更新 `ppo_deepcubea_results_latest.png`，训练结束生成 `ppo_deepcubea_results.png`
- **评估指标**：return + 胜率（max_steps 内解决即胜）+ 平均步数（仅胜利 episode），NUM_EVAL_EPISODES=10
- **冒烟测试通过**：5 episode 不崩溃，J(s) ∈ [15, 40]，j_max=200 保守覆盖，r_total ∈ [-3.1, 2.1]

## 内容概述

> 实现 Phase 2（PPO + 冻结 DeepCubeA J(s) 势函数塑形），将官方 DeepCubeA 启发函数 J(s) 作为 PPO 的 potential-based shaping reward。Actor/Critic 异构输入（Actor 见 aug_state，Critic 见 raw_state），共享 GAE advantage。单文件实现，`python ppo_deepcubea_agent.py --smoke` 冒烟测试。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/ppo-deepcubea-shaping-analysis.md`（方案 A 可行性分析）、`wiki/abstract/deepcubea-official-model-evaluation.md`（官方模型加载）、`wiki/abstract/ppo-impl.md`（PPO 参考实现）
- **下游被依赖**：无（Phase 2 验证阶段，Phase 3/4 依赖本阶段结论）
- **变更扩散评估**：低（新增单文件 + config 追加，不影响现有模块）
