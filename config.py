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

# ==================== DeepCubeA Data Generation（deepcubea_generate_data.py 使用） ====================

DEEPCUBEA_T_MIN = 3                   # 随机游走最小步数（跳过琐碎近距状态）
DEEPCUBEA_T_MAX = 500                   # 随机游走最大步数（覆盖中远距离状态）
DEEPCUBEA_TRAIN_SET_SIZE = 200000      # 训练集状态数（适配 RTX 3050 4GB）
DEEPCUBEA_TRAIN_DATA_PATH = "results/train_data/train_states.npy"  # 训练数据保存/加载路径

# ==================== DeepCubeA Training（deepcubea_train.py 使用） ====================

DEEPCUBEA_LR = 1e-4                    # 学习率（压低 Bellman 备份震荡）
DEEPCUBEA_BATCH_SIZE = 256             # 批大小
DEEPCUBEA_OUTER_ITER = 200             # Bellman 备份轮数（外层迭代）
DEEPCUBEA_INNER_EPOCHS = 200          # 每轮最大训练 epoch（内层迭代）
DEEPCUBEA_INNER_PATIENCE = 15         # 内层早停 patience（loss 连续不改善 epoch 数）
DEEPCUBEA_HIDDEN_DIM = 512             # 隐藏层维度
DEEPCUBEA_USE_GPU = True               # 是否使用 GPU
DEEPCUBEA_ONLINE_BATCH = 25000        # 每轮外层在线生成状态数 B
DEEPCUBEA_BASE_BATCH = 5000           # 每轮外层从基础数据集采样状态数 B'，设 0 为纯在线
DEEPCUBEA_OUTER_SEED = 42             # 外层迭代随机种子基准值
DEEPCUBEA_SEED_OVERLAP = 0.3          # 相邻轮次 base 采样重叠比例
DEEPCUBEA_TARGET_EPSILON = 0.05       # Bellman 误差阈值，低于此值更新 θ_c（论文 ε）
DEEPCUBEA_VAL_SPLIT = 0.2             # 验证集比例，0 = 禁用验证集（用训练 loss 判断）

# ==================== DeepCubeA Search / Validation（deepcubea_search.py 使用） ====================

DEEPCUBEA_MODEL_PATH = ""               # 训练好的网络权重路径（运行时必须显式指定）
DEEPCUBEA_LAMBDA = 1.0                  # 加权 A* 系数，λ·g(s) + h(s)
DEEPCUBEA_NUM_TEST_STATES = 100         # 评估测试状态总数（均分到短/中/长三档）
DEEPCUBEA_MAX_EXPAND_NODES = 10000    # 最大展开节点数，超限判失败
DEEPCUBEA_INFERENCE_USE_GPU = True  # A* 搜索推理时是否使用 GPU（与训练 DEEPCUBEA_USE_GPU 解耦）
DEEPCUBEA_VAL_SIZE = 100             # 验证集状态数量
DEEPCUBEA_VAL_SEED = 12345            # 验证集生成随机种子
DEEPCUBEA_VAL_NUM_STRATA = 5          # 分层数（K 范围均分为等宽区间）
DEEPCUBEA_VAL_GREEDY_MAX_STEPS = 500   # 纯贪心展开最大步数（argmin J(s') 步步跟随，无回溯）
DEEPCUBEA_VAL_ASTAR_FLAG = True      # 是否启用完整 A* 三档评估
DEEPCUBEA_VAL_GREEDY_FLAG = False     # 是否启用贪心展开评估

# ==================== DeepCubeA Official Model Evaluation ====================
DEEPCUBEA_OFFICIAL_MODEL_PATH = "data/model_state_dict.pt"
DEEPCUBEA_OFFICIAL_DATA_DIR = "data/"
DEEPCUBEA_OFFICIAL_LAMBDA = 1.0
DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES = 10000
DEEPCUBEA_OFFICIAL_NUM_STRATA = 3
DEEPCUBEA_OFFICIAL_GREEDY_MAX_STEPS = 1000
DEEPCUBEA_OFFICIAL_GREEDY_FLAG = False
DEEPCUBEA_OFFICIAL_ASTAR_FLAG = True
DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA = False  # 是否使用官方测试数据，False 则随机游走生成
DEEPCUBEA_OFFICIAL_T_MIN = 3               # 随机游走最小步数（USE_OFFICIAL_DATA=False 时生效）
DEEPCUBEA_OFFICIAL_T_MAX = 500             # 随机游走最大步数（USE_OFFICIAL_DATA=False 时生效）
DEEPCUBEA_OFFICIAL_NUM_TEST_STATES = 500   # 生成测试状态数（USE_OFFICIAL_DATA=False 时生效）

# ==================== PPO + DeepCubeA J(s) Shaping（ppo_deepcubea_agent.py 使用） ====================

PPO_DEEPCUBEA_HIDDEN_DIM = 256           # 隐藏层维度
PPO_DEEPCUBEA_ACTOR_LR = 1e-3            # Actor 学习率
PPO_DEEPCUBEA_CRITIC_LR = 1e-3           # Critic 学习率
PPO_DEEPCUBEA_GAMMA = 0.98               # 折扣因子（GAE + 势函数塑形共用）
PPO_DEEPCUBEA_LMBDA = 0.95               # GAE λ
PPO_DEEPCUBEA_EPOCHS = 10                # 每批数据训练轮数
PPO_DEEPCUBEA_EPS = 0.2                  # Clip 范围
PPO_DEEPCUBEA_NUM_EPISODES = 10000       # 训练总 episode 数
PPO_DEEPCUBEA_MAX_STEPS = 200            # 每 episode 最大步数（truncated 截断）
PPO_DEEPCUBEA_EVAL_INTERVAL = 1000       # 评估间隔（episode 数）
PPO_DEEPCUBEA_NUM_EVAL_EPISODES = 10     # 每次评估运行的 episode 数
PPO_DEEPCUBEA_ENTROPY_COEF = 0.05        # 熵正则化系数
PPO_DEEPCUBEA_BETA = 1.0                 # 塑形奖励权重（r_total = -1 + β·(J(s) - γ·J(s'))）
PPO_DEEPCUBEA_USE_GPU = False            # PPO 训练设备（True=cuda, False=cpu）
PPO_DEEPCUBEA_RESNET_BLOCKS = 2          # 残差块数量
PPO_DEEPCUBEA_J_MAX = 50              # J(s) 归一化上界（经验估计，S6 冒烟后调整）
PPO_DEEPCUBEA_SAVE_INTERVAL = 10         # 模型/图表保存间隔（episode 数），0=禁用
PPO_DEEPCUBEA_MODEL_PATH = "data/model_state_dict.pt"  # DeepCubeA 模型路径

# ==================== QUBE-15 配置（qube15/ppo_train.py 使用） ====================

# Hamiltonian 参数
QUBE15_B = 1.0                          # 局域场强度
QUBE15_J = 0.1                          # Ising 耦合常数
QUBE15_LAMBDA = 5.0                     # 跨阶段软约束权重
QUBE15_ALPHA = 0.1                      # 奖励缩放系数（对齐 PPO 奖励量级 ~[-4, 0]）

# Phase 切换
QUBE15_EPSILON = 0.5                    # Phase 切换阈值（per-tile ⟨Ĥ_k⟩/|S_k| < ε）
QUBE15_SLIDING_WINDOW = 100             # 滑动平均窗口大小（episode 数）
QUBE15_PHASE_SWITCH_K = 3               # 连续 K 次检查通过才切换 Phase（防止噪声触发）

# 各 Phase 打乱步数
QUBE15_SHUFFLE_STEPS_P1 = 25
QUBE15_SHUFFLE_STEPS_P2 = 50
QUBE15_SHUFFLE_STEPS_P3 = 70
QUBE15_SHUFFLE_STEPS_P4 = 100

# 各 Phase 最大步数
QUBE15_MAX_STEPS_P1 = 80
QUBE15_MAX_STEPS_P2 = 150
QUBE15_MAX_STEPS_P3 = 200
QUBE15_MAX_STEPS_P4 = 300

# PPO 训练参数
QUBE15_HIDDEN_DIM = 256                 # 隐藏层维度（ResBlock+LN 网络）
QUBE15_ACTOR_LR = 1e-3                  # Actor 学习率
QUBE15_CRITIC_LR = 1e-3                 # Critic 学习率
QUBE15_GAMMA = 0.98                     # 折扣因子
QUBE15_LMBDA = 0.95                     # GAE λ
QUBE15_EPOCHS = 10                      # 每批数据训练轮数
QUBE15_EPS = 0.2                        # Clip 范围
QUBE15_NUM_EPISODES = 10000             # 训练总 episode 数（所有 phase 合计）
QUBE15_EVAL_INTERVAL = 500              # 评估间隔（episode 数）
QUBE15_NUM_EVAL_EPISODES = 10           # 每次评估运行的 episode 数
QUBE15_ENTROPY_COEF = 0.05              # 熵正则化系数
QUBE15_RESNET_BLOCKS = 2                # 残差块数量
QUBE15_USE_GPU = True                  # 是否使用 GPU 训练
QUBE15_SAVE_INTERVAL = 10               # 模型/图表保存间隔（episode 数），0=禁用
