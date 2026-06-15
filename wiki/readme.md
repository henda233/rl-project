# WIKI 记忆库概述

## 项目简介

本项目是一个强化学习实践项目，包含两大模块：

1. **数字华容道 + PPO/RND**：实现 n×n 数字华容道（滑块拼图）环境，使用 PPO + RND 算法训练智能体求解。
2. **DeepCubeA 启发式搜索**：复现 DeepCubeA 论文方法——近似值迭代训练 DNN 逼近最优代价函数 J(s)，配合加权 A* 搜索求解华容道。

**技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium

**核心代码**：`env_digital_huarongdao.py`（环境）、`ppo_agent.py`（PPO）、`ppo_rnd_agent.py`（PPO+RND）、`config.py`（配置）、`deepcubea_network.py`（网络/编码/转移）、`deepcubea_train.py`（批量更新+固定目标AVI+θ_c阈值更新）、`deepcubea_search.py`（加权A*搜索+贪心展开+评估）、`deepcubea_generate_data.py`（训练数据生成）、`deepcubea_utils.py`（共享状态生成）、`deepcubea_official_network.py`（官方ResnetModel移植+OfficialModelWrapper）、`deepcubea_official_data.py`（官方测试数据加载）、`deepcubea_official_search.py`（官方模型三项评估）

## WIKI 结构

```
wiki/
├── readme.md          // 本文件 —— WIKI 概述
├── index.md           // 全局索引
├── abstract/          // 摘要目录
│   ├── docs/          // 设计与研究文档摘要
│   │   ├── project-overview.md
│   │   ├── digital-huarongdao-design.md
│   │   ├── deepcubea-research.md
│   │   ├── deepcubea-loss-analysis.md
│   │   ├── deepcubea-official-bug-postmortem.md
│   │   └── ppo-deepcubea-shaping-analysis.md
│   ├── gymnasium/     // gymnasium 参考文档摘要
│   │   ├── agent-training.md
│   │   ├── custom-env.md
│   │   └── recording-agent.md
│   ├── examples/      // 参考代码摘要
│   │   ├── actor-critic-example.md
│   │   └── rl-utils.md
│   ├── ppo-impl.md
│   ├── rnd-impl.md
│   ├── huarongdao-env.md
│   ├── huarongdao-render-test.md
│   ├── switch-to-huarongdao.md
│   ├── deepcubea-network.md
│   ├── deepcubea-search.md
│   ├── deepcubea-bwas-analysis.md
│   ├── deepcubea-target-network.md
│   ├── deepcubea-online-validation.md
│   ├── deepcubea-official-model-evaluation.md
│   ├── deepcubea-official-repo.md
│   ├── deepcubea-puzzle24-adaptation.md
│   ├── ppo-deepcubea-agent.md
│   └── deepcubea-network-architecture-comparison.md
├── request/           // 用户需求
└── plan/              // 执行计划
```

## 阅读导航

1. **了解项目全貌** → 从 `index.md` 开始
2. **华容道环境与设计** → `abstract/huarongdao-env.md`、`abstract/docs/digital-huarongdao-design.md`
3. **PPO + RND 算法** → `abstract/ppo-impl.md`、`abstract/rnd-impl.md`
4. **DeepCubeA 核心方法** → `abstract/docs/deepcubea-research.md`、`abstract/deepcubea-network.md`、`abstract/deepcubea-search.md`、`abstract/deepcubea-target-network.md`、`abstract/deepcubea-online-validation.md`、`abstract/docs/deepcubea-loss-analysis.md`
5. **PPO + DeepCubeA J(s) 势函数塑形** → `abstract/docs/ppo-deepcubea-shaping-analysis.md`、`abstract/ppo-deepcubea-agent.md`
6. **官方模型评估系列** → `abstract/deepcubea-official-model-evaluation.md`、`abstract/docs/deepcubea-official-bug-postmortem.md`、`abstract/deepcubea-bwas-analysis.md`、`abstract/deepcubea-network-architecture-comparison.md`
7. **24-Puzzle 适配** → `abstract/deepcubea-puzzle24-adaptation.md`
8. **参考文档** → `abstract/gymnasium/`、`abstract/examples/`

## 当前状态

- ✅ 数字华容道环境 + PPO + RND 算法实现
- ✅ DeepCubeA 核心链路：网络训练 → 加权 A* 搜索 → 推理优化 → 批量更新 AVI
- ✅ DeepCubeA 高级特性：在线采样 + LayerNorm + θ_c 阈值更新（硬阻断 Bellman 正反馈雪崩）
- ✅ 官方模型评估系列：移植 → 三项评估 → Bug 复盘 → BWAS 分析 → 架构对比
- ✅ PPO + DeepCubeA J(s) 势函数塑形（方案 A，γ=0.98）
- ✅ 工程优化：config 整理、Checkpoint/Loss 图保存优化
- ✅ 24-Puzzle 适配（仅评估，不训练）
