---
abstract_name: 数字华容道环境实现计划
source_contents:
  - "wiki/request/digital-huarongdao-env.md"
  - "wiki/plan/implement-huarongdao-env.md"
dependencies:
  - "wiki/abstract/docs/digital-huarongdao-design.md"
  - "wiki/abstract/gymnasium/custom-env.md"
created_at: 2026-06-10 12:00:00
updated_at: 2026-06-10 17:30:00
---
# 摘要：数字华容道环境实现计划

## 核心结论与关键信息

- **交付物**：`env_digital_huarongdao.py`（DigitalHuarongdaoEnv + make_huarongdao_env），`config.py`（6 个参数已追加）
- **观测**：`Box(shape=(n²,), dtype=float32)`，空格=0，数字保持原始值
- **动作**：`Discrete(4)`，0=上/1=下/2=左/3=右，无效动作不移动返回 reward=-2（区别于合法步 -1）
- **reset**：目标状态 + N 次随机合法移动打乱（保证可解），循环检查防止偶回目标
- **step**：空格交换 → 检测新归位（全网格扫描 + rewarded_mask）→ 合法：reward = -1 + FIRST_PLACEMENT_REWARD × 新归位数；非法：reward = -2 → 终止判断
- **render**：ansi（文本网格）、rgb_array（CELL_SIZE=80 钢蓝色块 + 5×3 点阵数字）、human（print ansi）
- **action_mask**：`get_action_mask()` 返回 `(4,)` bool 数组，根据空格位置标记合法移动方向（保留方法，当前训练流程未使用）
- **工厂函数**：`make_huarongdao_env(render_mode=None)`
- **自检**：9 个随机 episode 正常截断（200 步）+ gymnasium env_checker 通过

## 内容概述

> 实现计划涵盖 8 个步骤：config 参数 → 类骨架 → reset → step → render → 工厂函数+自检块 → 验证 → 摘要更新。计划已通过用户 review，默认值 HUARONGDAO_N=3、MAX_STEPS=200、SHUFFLE_STEPS=50。

## 依赖与影响链

- **上游依赖**：`wiki/abstract/docs/digital-huarongdao-design.md`（设计文档）、`wiki/abstract/gymnasium/custom-env.md`（Env 规范）
- **下游被依赖**：后续训练算法会通过 `make_huarongdao_env()` 调用
- **变更扩散评估**：低（独立新文件，config.py 仅追加参数）
