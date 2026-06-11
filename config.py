# ==================== PPO 配置（ppo_agent.py 使用） ====================

PPO_HIDDEN_DIM = 256                  # 隐藏层神经元数
PPO_ACTOR_LR = 1e-3                   # Actor 学习率
PPO_CRITIC_LR = 1e-2                  # Critic 学习率
PPO_GAMMA = 0.98                      # 折扣因子（支持长程信用分配）
PPO_LMBDA = 0.95                      # GAE λ
PPO_EPOCHS = 10                       # 每批数据训练轮数
PPO_EPS = 0.2                         # Clip 范围
PPO_NUM_EPISODES = 10000               # 训练总 episode 数
PPO_EVAL_INTERVAL = 1000               # 评估间隔
PPO_ENTROPY_COEF = 0.05               # 熵正则化系数
PPO_USE_GPU = False                  # 是否使用 GPU 训练（True=cuda, False=cpu）
PPO_INFERENCE_EPISODES = 3           # 推理录制时运行的 episode 数

# ==================== RND 配置（ppo_rnd_agent.py 使用） ====================

RND_HIDDEN_DIM = 256                   # RND 隐藏层维度
RND_OUTPUT_DIM = 256                   # 目标/预测网络输出维度（随机投影维度）
RND_LR = 1e-3                          # RND 预测网络学习率
RND_BETA = 1                         # 内在奖励系数起始值（β）
RND_BETA_END = 1                      # 内在奖励系数终止值（线性衰减地板）
RND_BETA_DECAY = 0.005                  # 每 episode 线性衰减量
RND_EPOCHS = 5                        # 预测网络每批数据训练轮数
RND_NUM_EPISODES = 10000               # RND 训练总 episode 数
RND_BUFFER_SIZE = 1                    # 状态滑动缓冲区大小（保留最近 N 个 episode 的状态）

# ==================== 数字华容道配置（env_digital_huarongdao.py 使用） ====================

HUARONGDAO_N = 4                       # 棋盘大小 n×n
HUARONGDAO_MAX_STEPS = 200             # 最大步数
HUARONGDAO_SHUFFLE_STEPS = 50          # 正向打乱步数
HUARONGDAO_FIRST_PLACEMENT_REWARD = 2  # 首次归位奖励
HUARONGDAO_LEGAL_STEP_REWARD = -1     # 合法动作每步奖励
HUARONGDAO_ILLEGAL_STEP_REWARD = -2   # 非法动作每步奖励

# ==================== DeepCubeA Training（deepcubea_train.py 使用） ====================

DEEPCUBEA_T_MIN = 10                   # 随机游走最小步数（跳过琐碎近距状态）
DEEPCUBEA_T_MAX = 500                   # 随机游走最大步数（覆盖中远距离状态）
DEEPCUBEA_TRAIN_SET_SIZE = 50000      # 训练集状态数（适配 RTX 3050 4GB）
DEEPCUBEA_LR = 1e-4                    # 学习率（压低 Bellman 备份震荡）
DEEPCUBEA_BATCH_SIZE = 256             # 批大小
DEEPCUBEA_ITERATIONS = 2000           # 值迭代总轮数（epoch）
DEEPCUBEA_HIDDEN_DIM = 256             # 隐藏层维度
DEEPCUBEA_USE_GPU = True               # 是否使用 GPU
DEEPCUBEA_CHECKPOINT_INTERVAL = 500   # checkpoint 保存间隔（epoch）

# ==================== DeepCubeA Search（deepcubea_search.py 使用） ====================

DEEPCUBEA_MODEL_PATH = ""               # 训练好的网络权重路径（运行时必须显式指定）
DEEPCUBEA_LAMBDA = 1.0                  # 加权 A* 系数，λ·g(s) + h(s)
DEEPCUBEA_NUM_TEST_STATES = 100         # 评估测试状态总数（均分到短/中/长三档）
DEEPCUBEA_MAX_EXPAND_NODES = 10000    # 最大展开节点数，超限判失败