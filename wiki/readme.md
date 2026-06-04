# WIKI 记忆库概述

## 项目简介

本项目是一个强化学习实践项目，使用 gymnasium 的 MountainCar-v0 环境，从零实现 Actor-Critic 算法，训练智能体学习最优控制策略。

**技术栈**：Python 3.12, numpy, pytorch, tqdm, matplotlib, gymnasium

## WIKI 结构

```
wiki/
├── readme.md          // 本文件 —— WIKI 概述
├── index.md           // 全局索引（所有摘要和需求的入口）
├── abstract/          // 摘要目录
│   ├── docs/          // 项目文档相关摘要
│   │   ├── project-overview.md
│   │   └── actor-critic-algorithm.md
│   ├── gymnasium/     // gymnasium 参考文档摘要
│   │   ├── mountain-car.md
│   │   ├── agent-training.md
│   │   ├── custom-env.md
│   │   └── recording-agent.md
│   └── examples/      // 参考代码摘要
│       ├── actor-critic-example.md
│       └── rl-utils.md
├── request/           // 用户需求
│   └── initial-requirements.md
└── plan/              // 执行计划（待创建）
```

## 阅读导航

1. **了解项目全貌** → 从 `index.md` 开始
2. **查看需求** → 阅读 `request/initial-requirements.md`
3. **了解算法理论** → `abstract/docs/actor-critic-algorithm.md`
4. **了解环境规范** → `abstract/gymnasium/mountain-car.md`
5. **参考实现** → `abstract/examples/actor-critic-example.md`

## 当前状态

- ✅ 虚拟环境和依赖已配置
- ✅ WIKI 记忆库已初始化
- ⬜ 待制定执行计划
- ⬜ 待实现代码
