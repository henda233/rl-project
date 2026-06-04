# ==================== AC 配置（agent.py 使用） ====================

ENV_NAME = "MountainCar-v0"            # gymnasium 环境名称
RENDER_MODE = None                    # 渲染模式：None=不渲染, "human"=可视化渲染
MAX_EPISODE_STEPS = 200               # 单 episode 最大步数（MountainCar 截断上限）

HIDDEN_DIM = 256                      # 隐藏层神经元数（Actor/Critic 共用）
ACTOR_LR = 1e-3                       # Actor（策略网络）学习率
CRITIC_LR = 1e-2                      # Critic（价值网络）学习率
GAMMA = 0.5                           # 折扣因子 γ
NUM_EPISODES = 5000                   # 训练总 episode 数
EVAL_INTERVAL = 1000                  # 评估间隔（每 N 个 episode 渲染评估一次）

# --- ε-greedy 探索策略 ---
EPSILON = 0.5                         # 初始 ε（随机动作概率）
EPSILON_DECAY = 0.999                 # ε 衰减率（每 episode 乘以此系数）
EPSILON_MIN = 0.01                    # ε 下限

# ==================== PPO 配置（ppo_agent.py 使用） ====================

PPO_HIDDEN_DIM = 256                  # 隐藏层神经元数
PPO_ACTOR_LR = 1e-3                   # Actor 学习率
PPO_CRITIC_LR = 1e-2                  # Critic 学习率
PPO_GAMMA = 0.98                      # 折扣因子（支持长程信用分配）
PPO_LMBDA = 0.95                      # GAE λ
PPO_EPOCHS = 10                       # 每批数据训练轮数
PPO_EPS = 0.2                         # Clip 范围
PPO_NUM_EPISODES = 5000               # 训练总 episode 数
PPO_EVAL_INTERVAL = 200               # 评估间隔
PPO_ENTROPY_COEF = 0.01               # 熵正则化系数

# ==================== 奖励塑形（两个算法共用） ====================

# --- 回合相对进度奖励塑形（episode-relative progress reward） ---
# 每个 episode 内追踪最佳位置，打破记录时给予奖励。
# 参考 docs/关于阶梯式奖励设置.md 最终方案（100% 成功率）。
PROGRESS_RIGHT_SCALE = 10    # 刷新 max_x（最右位置）时的奖励系数
PROGRESS_LEFT_SCALE = 5      # 刷新 min_x（最左位置）时的奖励系数
SPEED_REWARD_SCALE = 5       # abs(speed) 奖励系数（speed > SPEED_THRESHOLD 时）
SPEED_THRESHOLD = 0.01       # 速度阈值：abs(speed) > 此值视为"有明显移动"
STAGNATION_PENALTY = 1       # 停滞惩罚（abs(speed) <= SPEED_THRESHOLD 时）
NEAR_GOAL_THRESHOLD = 0.3    # "接近目标"的位置阈值
NEAR_GOAL_PENALTY = 20       # 接近目标但超时截断（truncated）的惩罚
