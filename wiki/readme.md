# WIKI 记忆库概述

## 项目简介

本项目是一个强化学习实践项目，包含两大模块：

1. **数字华容道 + PPO/RND**：实现 n×n 数字华容道（滑块拼图）环境，使用 PPO + RND 算法训练智能体求解。
2. **DeepCubeA 启发式搜索**：复现 DeepCubeA 论文方法——近似值迭代训练 DNN 逼近最优代价函数 J(s)，配合加权 A* 搜索求解华容道。

**技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium

**核心代码**：`env_digital_huarongdao.py`（环境）、`ppo_agent.py`（PPO）、`ppo_rnd_agent.py`（PPO+RND）、`config.py`（配置）、`deepcubea_network.py`（网络/编码/转移）、`deepcubea_train.py`（批量更新+固定目标AVI+θ_c阈值更新）、`deepcubea_search.py`（加权A*搜索+评估）、`deepcubea_generate_data.py`（训练数据生成）、`deepcubea_utils.py`（共享状态生成）

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
│   ├── deepcubea-target-network.md
│   └── deepcubea-online-validation.md
├── request/           // 用户需求
└── plan/              // 执行计划
```

## 阅读导航

1. **了解项目全貌** → 从 `index.md` 开始
2. **了解华容道环境** → `abstract/huarongdao-env.md`
3. **了解 PPO 算法** → `abstract/ppo-impl.md`
4. **了解 RND 算法** → `abstract/rnd-impl.md`
5. **了解 DeepCubeA 方法** → `abstract/docs/deepcubea-research.md`
6. **了解 DeepCubeA 网络** → `abstract/deepcubea-network.md`
7. **了解 DeepCubeA 搜索** → `abstract/deepcubea-search.md`
8. **了解 DeepCubeA 训练** → `abstract/deepcubea-target-network.md`
9. **了解 DeepCubeA 在线采样** → `abstract/deepcubea-online-validation.md`
10. **了解 DeepCubeA 训练 Loss 分析** → `abstract/docs/deepcubea-loss-analysis.md`
11. **了解 PPO + J(s) 势函数塑形分析** → `abstract/docs/ppo-deepcubea-shaping-analysis.md`
12. **查看设计文档** → `abstract/docs/digital-huarongdao-design.md`

## 当前状态

- ✅ 数字华容道环境实现
- ✅ PPO 算法实现
- ✅ PPO + RND 算法实现
- ✅ MountainCar 时代代码已清理
- ✅ DeepCubeA 神经网络训练完成（v1: best loss=0.0138）
- ✅ DeepCubeA 加权 A* 搜索完成（v1 Short 档胜率 32%，v2 待重训）
- ✅ DeepCubeA 数据生成/训练分离
- ✅ DeepCubeA A* 搜索推理优化（批量转移 + bytes 状态 + GPU 可配）
- ✅ DeepCubeA 批量更新 + 固定目标 AVI（外层 Bellman 备份 + 内层早停监督学习）
- ✅ DeepCubeA 在线采样 + 轻量验证（在线混合采样 + LN + 分层 Bellman MSE + 贪心展开 + 完整 A*）
- ✅ DeepCubeA BN → LayerNorm 替换（消除 batch_size≥2 约束 + 删除所有 .eval() 调用）
- ✅ DeepCubeA θ_c 阈值更新（独立 target_network + 验证集拆分 + 硬阻断 Bellman 正反馈雪崩）
- ✅ config.py 参数整理（在线采样迁入 Training；Validation 合并入 Search）
- ✅ 代码/测试清理（合并搜索测试到主模块，删除冗余旧测试）
- ✅ PPO + DeepCubeA J(s) 势函数塑形可行性分析
- ✅ DeepCubeA Checkpoint 与 Loss 图保存优化（latest.pt 覆盖式 + CSV + 累积曲线图每轮外层同步更新）
