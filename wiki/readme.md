# WIKI 记忆库概述

## 项目简介

本项目是一个强化学习实践项目，实现 n×n 数字华容道（滑块拼图）环境，使用 PPO + RND 算法训练智能体求解。

**技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium

**核心代码**：`env_digital_huarongdao.py`（环境）、`ppo_agent.py`（PPO）、`ppo_rnd_agent.py`（PPO+RND）、`config.py`（配置）

## WIKI 结构

```
wiki/
├── readme.md          // 本文件 —— WIKI 概述
├── index.md           // 全局索引
├── abstract/          // 摘要目录
│   ├── docs/          // 设计文档摘要
│   │   ├── project-overview.md
│   │   └── digital-huarongdao-design.md
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
│   └── switch-to-huarongdao.md
├── request/           // 用户需求
└── plan/              // 执行计划
```

## 阅读导航

1. **了解项目全貌** → 从 `index.md` 开始
2. **了解华容道环境** → `abstract/huarongdao-env.md`
3. **了解 PPO 算法** → `abstract/ppo-impl.md`
4. **了解 RND 算法** → `abstract/rnd-impl.md`
5. **查看设计文档** → `abstract/docs/digital-huarongdao-design.md`

## 当前状态

- ✅ 数字华容道环境实现
- ✅ PPO 算法实现
- ✅ PPO + RND 算法实现
- ✅ MountainCar 时代代码已清理
