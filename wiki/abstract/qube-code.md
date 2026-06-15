---
abstract_name: QUBE 代码探索
source_contents:
  - "QUBE/baseline.py"
  - "QUBE/Rubik.py"
  - "QUBE/Cube_unittest.py"
  - "QUBE/user_guide.md"
dependencies:
  - "wiki/abstract/docs/qube-paper.md"
created_at: 2026-06-15 23:00
updated_at: 2026-06-15 23:00
---
# 摘要：QUBE 代码仓库探索与环境跑通

## 核心结论与关键信息

- **代码范围**：仓库仅包含量子力学魔方环境（状态向量 + 群操作），不含论文中提到的 DDQN 训练循环。RL 部分是论文中实现但未开源。
- **第三方依赖**：仅 `numpy`（1.21.2），无 TensorFlow / PyTorch / 任何其他第三方库。
- **Python 版本**：代码标注 Python 3.9，当前项目 Python 3.12。仅需修复 1 处兼容性问题即可运行。
- **修复内容**：`Cube_unittest.py` 中 `TestPeriodicComposition` 类 6 处 `self.assert_()` → `self.assertTrue()`（Python 3.12 已移除 `assert_`）。
- **测试结果**：52 个单元测试全部通过。

## 内容概述

> 代码由 3 个文件组成：(1) `baseline.py` 定义基础类 —— Exponential（位置指数表示）、Cubie/Corner/Edge（cubie 状态向量）、Translation（平移算符）、Sigma（取向旋转算符，flip/clockwise/anticlockwise）、Permutations（置换算符）；(2) `Rubik.py` 定义 RubiksCube（魔方状态向量，solved/Cube 属性，reset/is_solved 方法）和 RubiksGroup（群操作，6 个生成器 U/D/L/R/F/B，@ 组合，* 作用于状态）；(3) `Cube_unittest.py` 包含 52 个单元测试（Exponential 属性、Sigma 矩阵乘、Edge/Corner 取向翻转、Permutations 组合、所有 6 个生成器的位置/周期性/组合正确性）。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/qube-paper.md`（QUBE 论文方法）
- **下游被依赖**：Phase 2（15-puzzle 量子化推导与实现）
- **变更扩散评估**：低（独立模块，仅 numpy 依赖）
