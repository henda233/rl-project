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
PPO_NUM_EPISODES = 10000               # 训练总 episode 数
PPO_EVAL_INTERVAL = 500               # 评估间隔
PPO_ENTROPY_COEF = 0.05               # 熵正则化系数
PPO_USE_GPU = False                  # 是否使用 GPU 训练（True=cuda, False=cpu）
PPO_INFERENCE_EPISODES = 3           # 推理录制时运行的 episode 数

# --- PPO 再训练 ---
PPO_ACTOR_MODEL_PATH = ""            # 为空则从头训练；非空则加载指定路径的actor权重进行再训练
PPO_CRITIC_MODEL_PATH = ""           # 为空则从头训练；非空则加载指定路径的critic权重进行再训练
PPO_RETRAIN_NUM_EPISODES = 10000      # 再训练时的 episode 数

# ==================== RND 配置（ppo_rnd_agent.py 使用） ====================

RND_HIDDEN_DIM = 128                   # RND 隐藏层维度
RND_OUTPUT_DIM = 256                   # 目标/预测网络输出维度（随机投影维度）
RND_LR = 1e-3                          # RND 预测网络学习率
RND_BETA = 100                         # 内在奖励系数（β）
RND_EPOCHS = 5                        # 预测网络每批数据训练轮数
RND_NUM_EPISODES = 10000               # RND 训练总 episode 数
RND_BUFFER_SIZE = 5                    # 状态滑动缓冲区大小（保留最近 N 个 episode 的状态）

# ==================== 奖励塑形（两个算法共用） ====================

# --- Potential-Based 奖励塑形 ---
# Φ(position) = k * position
# shaped_reward = original_reward + (γ·Φ(s') - Φ(s))
# Ng et al. 定理保证：不改变最优策略。
POTENTIAL_K = 2  # 势函数系数 k

# ==================== PPO 并行训练配置（ppo_parallel.py 使用） ====================

PPO_NUM_ENVS = 5            # 并行环境数量
POTENTIAL_K_START = 1     # k 起始值
POTENTIAL_K_INTERVAL = 2  # k 间隔