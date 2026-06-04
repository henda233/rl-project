ENV_NAME = "MountainCar-v0"            # gymnasium 环境名称
RENDER_MODE = None                    # 渲染模式：None=不渲染, "human"=可视化渲染
MAX_EPISODE_STEPS = 200               # 单 episode 最大步数（MountainCar 截断上限）

HIDDEN_DIM = 128                      # 隐藏层神经元数（Actor/Critic 共用）
ACTOR_LR = 1e-3                       # Actor（策略网络）学习率
CRITIC_LR = 1e-2                      # Critic（价值网络）学习率
GAMMA = 0.98                          # 折扣因子 γ
NUM_EPISODES = 1000                   # 训练总 episode 数
EVAL_INTERVAL = 50                    # 评估间隔（每 N 个 episode 渲染评估一次）

# --- 基于势能的奖励塑形（potential-based shaping, Ng et al.） ---
# Φ(s) = PE_COEFFICIENT * sin(3*pos) + v² / (2*G)
# shaped_reward = original_reward + C * (γ·Φ(s') - Φ(s))
# Φ 值域约 [-1.0, 2.0]，跨度约 3。C=20 时每步 shaping 贡献约 0.1~0.3
# （原始 reward 每步 -1 的 10%~30%），保守但安全的起点。
REWARD_SHAPING_SCALE = 10             # C：塑形总体强度
G = 0.0025                            # MountainCar 重力加速度常量（引擎推力=0.001，重力=0.0025）
PE_COEFFICIENT = 0.5                  # sin(3*pos) 势能项权重，调整 PE 与 KE 的相对贡献

# --- ε-greedy 探索策略 ---
# energy-based shaping 提供更强的位置感知引导信号，降低初始探索率。
EPSILON = 0.3                         # 初始 ε（随机动作概率）
EPSILON_DECAY = 0.999                 # ε 衰减率（每 episode 乘以此系数）
EPSILON_MIN = 0.01                    # ε 下限
