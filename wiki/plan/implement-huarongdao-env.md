---
plan_name: 实现数字华容道游戏环境
related_request: "wiki/request/digital-huarongdao-env.md"
status: completed
created_at: 2026-06-10 12:00:00
---
# 执行计划：实现数字华容道游戏环境

基于设计文档和用户确认的规格，实现数字华容道 Gymnasium 自定义环境。

## 内容

数字华容道是 n×n 滑块拼图游戏。玩家移动空格相邻的方块，目标是排列数字 1..n²-1 顺序、空格位于末尾。本计划实现一个标准的 Gymnasium Env 子类，供后续强化学习算法训练使用。

### 核心设计决策

- **观测空间**：`Box(low=0, high=n²-1, shape=(n²,), dtype=np.float32)`。将 n×n 网格展平为 1D 向量，空格用 0 表示，数字保持原始值。
- **动作空间**：`Discrete(4)`。0=上, 1=下, 2=左, 3=右，表示空格移动方向。
- **渲染**：支持 `ansi`（文本字符串）、`rgb_array`（numpy 数组图像）、`human`（控制台打印）。
- **正向打乱**：从目标状态出发执行 `HUARONGDAO_SHUFFLE_STEPS` 次随机合法移动，天然保证可解性。

### 与现有项目的关系

遵循现有项目模式：`config.py` 集中配置参数，环境文件提供工厂函数 `make_huarongdao_env()`。

## 任务步骤

| 步骤ID | 任务描述 | 前置依赖 | 交付物/修改路径 | 状态 |
|---|---|---|---|---|
| `S1` | config.py 新增数字华容道配置参数 | 无 | `config.py` | ✅ 已完成 |
| `S2` | 创建 `DigitalHuarongdaoEnv` 类骨架（`__init__`、observation_space、action_space） | `S1` | `env_digital_huarongdao.py` | ✅ 已完成 |
| `S3` | 实现 `reset()` —— 目标状态生成 + 正向打乱 + 状态初始化 | `S2` | `env_digital_huarongdao.py` | ✅ 已完成 |
| `S4` | 实现 `step()` —— 动作校验、状态转移、奖励计算、终止判断 | `S2` | `env_digital_huarongdao.py` | ✅ 已完成 |
| `S5` | 实现 `render()` —— ansi / rgb_array / human 三种模式 | `S2` | `env_digital_huarongdao.py` | ✅ 已完成 |
| `S6` | 添加 `make_huarongdao_env()` 工厂函数 + 自检块 `if __name__ == "__main__"` | `S3,S4,S5` | `env_digital_huarongdao.py` | ✅ 已完成 |
| `S7` | 自检验证 —— 运行随机动作 episode，检查合规性 | `S6` | — | ✅ 已完成 |
| `S8` | 编写摘要并更新 index | `S7` | `wiki/abstract/huarongdao-env.md`、`wiki/index.md` | ✅ 已完成 |

## 详细实现说明

### S1: config.py 参数

```python
HUARONGDAO_N = 3              # 棋盘大小 n
HUARONGDAO_MAX_STEPS = 200    # 最大步数
HUARONGDAO_SHUFFLE_STEPS = 50 # 打乱次数
```

### S2-S5: 环境类核心逻辑

**状态表示**：内部使用 `np.ndarray` shape=(n,n)，元素为整数（1..n²-1 或 0=空格）。观测返回展平的 float32 数组。

**目标状态**：
```
1    2    3
4    5    6
7    8    0     (n=3 示例)
```
即 `grid[i][j] = (i*n + j + 1) % (n*n)`。

**step(action) 流程**：
1. 找到空格位置 (er, ec)
2. 根据 action 计算目标位置 (tr, tc) = (er+dr, ec+dc)，其中 dr/dc 为 [(-1,0),(1,0),(0,-1),(0,1)]
3. 若目标越界 → 无效动作，返回原状态、reward=-1、不做状态更新（或可选地跳过）
4. 交换空格与目标方块
5. 检测新归位的数字（与目标状态比对，且 `rewarded_mask[digit]` 为 False）
6. 奖励 = -1 + 5 × (新归位数量)
7. 步数 +1，判断终止

**reset(seed, options) 流程**：
1. 调用 `super().reset(seed=seed)`
2. 构建目标状态网格
3. 复制目标状态，执行 N 次随机合法移动打乱
4. 重置 `rewarded_mask`、步数计数器
5. 返回 `(obs, info)`

**render() 行为**：
- `ansi`：返回多行字符串，如：
  ```
  +---+---+---+
  | 1 | 2 | 3 |
  +---+---+---+
  | 4 | 5 |   |
  +---+---+---+
  | 7 | 8 | 6 |
  +---+---+---+
  ```
- `rgb_array`：返回 (H, W, 3) uint8 的 numpy 数组，用色块 + 数字绘制面板
- `human`：`print(ansi)` 到控制台

### S6: 工厂函数

```python
def make_huarongdao_env(render_mode=None):
    return DigitalHuarongdaoEnv(render_mode=render_mode)
```

## 风险与约束声明

- **动作空间不变性**：Discrete(4) 在面板边缘时会有 1-2 个无效动作。`step()` 中无效动作返回原状态 + 负奖励，与 MountainCar 环境的行为一致（MountainCar 所有动作始终有效）。这需要 agent 学习避免无效移动。
- **奖励稀疏性**：-1/步 + 首次归位 +5 的奖励结构相对稀疏。对较大 n（如 4×4），归位可能需要几十步，agent 面临探索挑战——这是预期行为，后续可通过奖励塑形或好奇心机制改进。
- **打乱次数与难度**：`HUARONGDAO_SHUFFLE_STEPS` 决定初始混乱程度。过小 → 太简单，过大 → 可能极难。默认 50 步对 3×3 是合理的起始值。
- **rgb_array 渲染**：使用纯 numpy 绘制，不依赖 matplotlib，保持依赖简洁。

## 测试验证方案

1. **reset 验证**：reset 后观测 shape=(n²,)，值在 [0, n²-1] 范围内，空格（0）存在且唯一
2. **step 验证**：执行合法动作后空格位置移动正确；边缘越界动作返回原状态
3. **目标检测**：手动构造目标状态，reset 后 step count=0，检查是否正确识别
4. **奖励追踪**：同一数字多次进出正确位置，验证 +5 仅首次触发
5. **终止条件**：达到 max_steps 时 `truncated=True`；达成目标时 `terminated=True`
6. **正向打乱**：大量 shuffle 后状态 ≠ 目标状态（99.9%+ 概率）
7. **自检块**：运行 5 个随机动作 episode，无异常

## 📝 执行记录

- `2026-06-10 12:00`: 计划已生成，已通过用户 review，确认默认值 N=3 / MAX_STEPS=200 / SHUFFLE_STEPS=50
- `2026-06-10 12:00`: WIKI 记忆库已更新——设计摘要 + 计划摘要 + index 同步
- `2026-06-10 14:00`: 计划执行完成。env_digital_huarongdao.py 创建（DigitalHuarongdaoEnv + make_huarongdao_env），config.py 追加 4 个参数（含 FIRST_PLACEMENT_REWARD=5）。自检：9 随机 episode 正常截断 + gymnasium env_checker 通过。打乱后增加防偶回目标检查。rgb_array 渲染使用 CELL_SIZE=80 + 5×3 点阵数字字体。
