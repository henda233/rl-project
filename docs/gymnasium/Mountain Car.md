# Mountain Car (山地车)

此环境属于经典控制（ClassicControl）环境，其中包含有关该环境的通用信息。

| 属性 | 值 |
|------|-----|
| **动作空间** | Discrete(3) |
| **观测空间** | Box([-1.2 -0.07], [0.6 0.07], (2,)), float32 |
| **导入** | `gymnasium.make("MountainCar-v0")` |

## 描述

MountainCar MDP是一个确定性的马尔可夫决策过程，其背景是一辆车被随机放置在正弦波形山谷的底部。唯一的可用动作为向两个方向加速。该MDP的目标是策略性地加速车辆以到达右侧山顶的目标状态。Gymnasium中有两种MountainCar环境：一种是离散动作的，另一种是连续动作的。本版本为离散动作版本。

该MDP最早出现在Andrew Moore的博士论文（1990）中。

@TECHREPORT{Moore90efficientmemory-based,
  author = {Andrew William Moore},
  title = {Efficient Memory-based Learning for Robot Control},
  institution = {University of Cambridge},
  year = {1990}
}

## 观测空间

观测结果为一个ndarray，形状为(2,)，各元素对应如下：

| 编号 | 观测值 | 最小值 | 最大值 | 单位 |
|------|--------|--------|--------|------|
| 0 | 车辆在x轴上的位置 | -1.2 | 0.6 | 位置(m) |
| 1 | 车辆速度 | -0.07 | 0.07 | 速度(v) |

## 动作空间

共有3种离散确定性动作：

- 0：向左加速
- 1：不加速
- 2：向右加速

## 转换动力学

给定一个动作，山地车遵循以下转换动力学：

速度\(_{t+1}\) = 速度\(_t\) + (动作 - 1) × 力 - cos(3 × 位置\(_t\)) × 重力

位置\(_{t+1}\) = 位置\(_t\) + 速度\(_{t+1}\)

其中力 = 0.001，重力 = 0.0025。两端的碰撞是非弹性的，撞墙时速度设为0。位置被限制在[-1.2, 0.6]范围内，速度被限制在[-0.07, 0.07]范围内。

目标是尽可能快地到达右侧山顶的旗帜处。因此，智能体在每个时间步都会受到-1的奖励惩罚。

## 起始状态

车辆的初始位置在[-0.6, -0.4]之间随机均匀采样。车辆的起始速度始终设为0。

## 回合结束

如果发生以下任一情况，回合结束：

1. **终止**：车辆位置大于或等于0.5（右侧山顶的目标位置）。
2. **截断**：回合长度达到200个时间步。

## 参数

Mountain Car为`gymnasium.make`提供了两个参数：`render_mode`和`goal_velocity`。重置时，`options`参数允许用户更改用于确定新随机状态的边界。

```python
>>> import gymnasium as gym
>>> env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
>>> env
<TimeLimit<OrderEnforcing<PassiveEnvChecker<MountainCarEnv<MountainCar-v0>>>>
>>> env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})  # default x_init=np.pi, y_init=1.0
(array([-0.46352962, 0.], dtype=float32), {})
```

## 版本历史

- **v0**：初始版本发布