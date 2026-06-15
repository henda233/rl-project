import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    PPO_DEEPCUBEA_HIDDEN_DIM, PPO_DEEPCUBEA_ACTOR_LR, PPO_DEEPCUBEA_CRITIC_LR,
    PPO_DEEPCUBEA_GAMMA, PPO_DEEPCUBEA_LMBDA, PPO_DEEPCUBEA_EPOCHS, PPO_DEEPCUBEA_EPS,
    PPO_DEEPCUBEA_NUM_EPISODES, PPO_DEEPCUBEA_MAX_STEPS, PPO_DEEPCUBEA_EVAL_INTERVAL,
    PPO_DEEPCUBEA_NUM_EVAL_EPISODES, PPO_DEEPCUBEA_ENTROPY_COEF, PPO_DEEPCUBEA_BETA,
    PPO_DEEPCUBEA_USE_GPU, PPO_DEEPCUBEA_RESNET_BLOCKS, PPO_DEEPCUBEA_J_MAX,
    PPO_DEEPCUBEA_SAVE_INTERVAL, PPO_DEEPCUBEA_MODEL_PATH,
)
from env_digital_huarongdao import make_huarongdao_env
from deepcubea_official_network import load_official_model


# ==================== Network Modules ====================

class ResBlock(nn.Module):
    """对齐 deepcubea_official_network.py ResnetModel 的残差块结构（BN→LN）。

    FC → LN → ReLU → FC → LN → +residual → ReLU
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        res_inp = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x + res_inp)
        return x


class PolicyNet(nn.Module):
    """Actor：输入 [obs, j/j_max]，输出 action logits。"""
    def __init__(self, input_dim, hidden_dim, action_dim, num_resnet_blocks):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(num_resnet_blocks)]
        )
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.ln_in(x)
        x = F.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        return x


class ValueNet(nn.Module):
    """Critic：输入原始 obs，输出 scalar 状态价值。"""
    def __init__(self, input_dim, hidden_dim, num_resnet_blocks):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(num_resnet_blocks)]
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.ln_in(x)
        x = F.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        return x


# ==================== GAE ====================

def compute_gae(gamma, lmbda, td_delta, dones):
    advantages = []
    advantage = 0.0
    for delta, done in zip(reversed(td_delta), reversed(dones)):
        advantage = delta + gamma * lmbda * (1 - done) * advantage
        advantages.append(advantage)
    advantages.reverse()
    return torch.tensor(advantages, dtype=torch.float)


# ==================== PPO Agent ====================

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, entropy_coef, device, num_resnet_blocks):
        self.actor = PolicyNet(state_dim + 1, hidden_dim, action_dim, num_resnet_blocks).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, num_resnet_blocks).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state, j_norm):
        state_t = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        j_t = torch.tensor([[j_norm]], dtype=torch.float).to(self.device)
        aug_state = torch.cat([state_t, j_t], dim=1)
        logits = self.actor(aug_state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        j_norms = torch.tensor(transition_dict['j_norms'], dtype=torch.float).view(-1, 1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        aug_states = torch.cat([states, j_norms], dim=1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = (td_target - self.critic(states)).detach().cpu().numpy()
        dones_np = dones.cpu().numpy()
        advantage = compute_gae(self.gamma, self.lmbda, td_delta.flatten(), dones_np.flatten())
        advantage = advantage.to(self.device).view(-1, 1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        old_log_probs = F.log_softmax(self.actor(aug_states), dim=1).gather(1, actions).detach()

        for _ in range(self.epochs):
            logits = self.actor(aug_states)
            log_probs = F.log_softmax(logits, dim=1).gather(1, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            action_dist = torch.distributions.Categorical(logits=logits)
            entropy = action_dist.entropy().mean()
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# ==================== Training ====================

def run_episode(env, agent, model, max_steps, beta, gamma, j_max):
    """执行单个 episode 的 rollout，返回 transition_dict 和 episode 统计。"""
    transition_dict = {
        'states': [], 'j_norms': [], 'actions': [], 'next_states': [],
        'rewards': [], 'dones': [],
    }
    obs, _info = env.reset()
    obs_int = obs.astype(np.int64)
    j = model.predict_j(obs_int)
    j_norm = j / j_max
    episode_return = 0.0
    step_count = 0
    terminated = False

    while step_count < max_steps:
        action = agent.take_action(obs, j_norm)
        next_obs, _, _terminated, truncated, info = env.step(action)
        done = _terminated or truncated
        step_count += 1

        next_obs_int = next_obs.astype(np.int64)
        if _terminated:
            j_next = 0.0
        else:
            j_next = model.predict_j(next_obs_int)

        r_int = j - gamma * j_next
        r_total = -1.0 + beta * r_int

        transition_dict['states'].append(obs)
        transition_dict['j_norms'].append(j_norm)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(r_total)
        transition_dict['next_states'].append(next_obs)
        transition_dict['dones'].append(done)

        episode_return += r_total
        obs = next_obs
        j = j_next
        j_norm = 0.0 if _terminated else j_next / j_max

        if done:
            terminated = _terminated
            break

    return transition_dict, episode_return, terminated, step_count


def evaluate(env, agent, model, num_eval_episodes, max_steps, beta, gamma, j_max):
    """多 episode 评估，返回 (avg_return, win_rate, avg_steps_solved)。"""
    total_return = 0.0
    solved_count = 0
    solved_steps = []

    for _ in range(num_eval_episodes):
        _transition_dict, ep_return, solved, steps = run_episode(
            env, agent, model, max_steps, beta, gamma, j_max)
        total_return += ep_return
        if solved:
            solved_count += 1
            solved_steps.append(steps)

    avg_return = total_return / num_eval_episodes
    win_rate = solved_count / num_eval_episodes
    avg_steps = float(np.mean(solved_steps)) if solved_steps else float('nan')
    return avg_return, win_rate, avg_steps


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, model, num_episodes, max_steps, beta, gamma,
                          j_max, eval_interval, num_eval_episodes, save_interval,
                          results_dir="results"):
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    return_list = []
    step_list = []
    solved_list = []
    best_original_return = -float('inf')
    j_values_all = []

    pbar = tqdm(range(1, num_episodes + 1), desc='Training PPO+J(s)')
    for i_episode in pbar:
        transition_dict, episode_return, solved, steps = run_episode(
            env, agent, model, max_steps, beta, gamma, j_max)

        # 收集 episode 中所有 J(s) 用于分布分析
        if transition_dict['j_norms']:
            j_vals = [jn * j_max for jn in transition_dict['j_norms']]
            j_values_all.extend(j_vals)

        return_list.append(episode_return)
        step_list.append(steps)
        solved_list.append(solved)
        agent.update(transition_dict)

        if episode_return > best_original_return:
            best_original_return = episode_return
            torch.save(agent.actor.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_actor_best.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_critic_best.pth"))

        if solved:
            tqdm.write(f'[Solved! Episode {i_episode}] steps={steps} return={episode_return:.1f}')

        if i_episode % 10 == 0:
            recent_ret = np.mean(return_list[-10:])
            recent_steps = np.mean(step_list[-10:])
            recent_solved = np.mean(solved_list[-10:])
            pbar.set_postfix({
                'return': f'{recent_ret:.1f}',
                'steps': f'{recent_steps:.1f}',
                'solved%': f'{recent_solved:.0%}',
            })
            tqdm.write(
                f'[Episode {i_episode}] '
                f'return: {recent_ret:.1f}, '
                f'steps: {recent_steps:.1f}, '
                f'solved%: {recent_solved:.0%}'
            )

        if save_interval > 0 and i_episode % save_interval == 0:
            torch.save(agent.actor.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_actor_latest.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_critic_latest.pth"))
            plot_results(return_list, step_list, solved_list, results_dir,
                         show_plot=False, filename="ppo_deepcubea_results_latest.png")

        if i_episode % eval_interval == 0:
            eval_env = make_huarongdao_env()
            avg_ret, win_rate, avg_steps = evaluate(
                eval_env, agent, model, num_eval_episodes, max_steps, beta, gamma, j_max)
            eval_env.close()
            tqdm.write(
                f'[Eval  episode {i_episode}] '
                f'return: {avg_ret:.1f}, '
                f'win%: {win_rate:.0%}, '
                f'avg_steps(solved): {avg_steps:.1f}'
            )

    torch.save(agent.actor.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_actor_final.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(results_dir, "models", "ppo_deepcubea_critic_final.pth"))

    return return_list, step_list, solved_list, j_values_all


# ==================== Plotting ====================

def plot_results(return_list, step_list, solved_list, results_dir="results",
                  show_plot=True, filename="ppo_deepcubea_results.png"):
    os.makedirs(os.path.join(results_dir, "imgs"), exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Return
    ax = axes[0]
    mv_ret = moving_average(return_list, 9)
    ax.plot(return_list, alpha=0.4, color='steelblue', label='episode return')
    ax.plot(mv_ret, color='steelblue', label='moving avg (9)')
    ax.set_ylabel('Return')
    ax.set_title('PPO + DeepCubeA J(s) Shaping')
    ax.legend()

    # Steps
    ax = axes[1]
    mv_steps = moving_average(step_list, 9)
    ax.plot(step_list, alpha=0.4, color='darkorange', label='episode steps')
    ax.plot(mv_steps, color='darkorange', label='moving avg (9)')
    ax.set_ylabel('Steps')
    ax.legend()

    # Win rate (window)
    ax = axes[2]
    win_rate_window = 100
    win_rates = [np.mean(solved_list[max(0, i - win_rate_window + 1):i + 1])
                 for i in range(len(solved_list))]
    ax.plot(win_rates, color='seagreen', label=f'win rate (window={win_rate_window})')
    ax.set_ylabel('Win Rate')
    ax.set_xlabel('Episode')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "imgs", filename), dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ==================== Main ====================

def main_smoke():
    """冒烟测试：5 episode，验证模型加载、J(s) 分布、r_total 计算。"""
    device = torch.device("cuda" if PPO_DEEPCUBEA_USE_GPU else "cpu")
    print(f"[Smoke] device={device}")

    env = make_huarongdao_env()
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("[Smoke] Loading DeepCubeA model...")
    model = load_official_model(PPO_DEEPCUBEA_MODEL_PATH, use_gpu=PPO_DEEPCUBEA_USE_GPU)
    print("[Smoke] Model loaded.")

    agent = PPO(
        state_dim, PPO_DEEPCUBEA_HIDDEN_DIM, action_dim,
        PPO_DEEPCUBEA_ACTOR_LR, PPO_DEEPCUBEA_CRITIC_LR,
        PPO_DEEPCUBEA_LMBDA, PPO_DEEPCUBEA_EPOCHS, PPO_DEEPCUBEA_EPS,
        PPO_DEEPCUBEA_GAMMA, PPO_DEEPCUBEA_ENTROPY_COEF, device,
        PPO_DEEPCUBEA_RESNET_BLOCKS,
    )

    gamma = PPO_DEEPCUBEA_GAMMA
    beta = PPO_DEEPCUBEA_BETA
    j_max = PPO_DEEPCUBEA_J_MAX

    j_all = []
    r_total_samples = []

    print(f"\n[Smoke] Running {PPO_DEEPCUBEA_NUM_EPISODES} episodes with MAX_STEPS={PPO_DEEPCUBEA_MAX_STEPS}")
    print(f"[Smoke] gamma={gamma}, beta={beta}, j_max={j_max}\n")

    for ep in range(1, min(5, PPO_DEEPCUBEA_NUM_EPISODES) + 1):
        transition_dict, ep_return, solved, steps = run_episode(
            env, agent, model, PPO_DEEPCUBEA_MAX_STEPS, beta, gamma, j_max)

        j_vals = [jn * j_max for jn in transition_dict['j_norms']]
        j_all.extend(j_vals)
        r_total_samples.extend(transition_dict['rewards'])

        print(f"[Smoke Ep{ep}] return={ep_return:.1f}  steps={steps}  "
              f"solved={solved}  J min/mean/max={np.min(j_vals):.1f}/{np.mean(j_vals):.1f}/{np.max(j_vals):.1f}  "
              f"r_total min/mean/max={np.min(transition_dict['rewards']):.1f}/{np.mean(transition_dict['rewards']):.1f}/{np.max(transition_dict['rewards']):.1f}")

    print(f"\n[Smoke Summary]")
    print(f"  J(s)   overall: min={np.min(j_all):.1f}  mean={np.mean(j_all):.1f}  max={np.max(j_all):.1f}")
    print(f"  r_total overall: min={np.min(r_total_samples):.1f}  mean={np.mean(r_total_samples):.1f}  max={np.max(r_total_samples):.1f}")
    print(f"  j_max={j_max} (adjust if J(s) max significantly different)")

    env.close()
    print("[Smoke] Done. No crash = pass.")


def main():
    device = torch.device("cuda" if PPO_DEEPCUBEA_USE_GPU else "cpu")
    print(f"Device: {device}")

    env = make_huarongdao_env()
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Loading DeepCubeA model...")
    model = load_official_model(PPO_DEEPCUBEA_MODEL_PATH, use_gpu=PPO_DEEPCUBEA_USE_GPU)
    print("Model loaded.")

    agent = PPO(
        state_dim, PPO_DEEPCUBEA_HIDDEN_DIM, action_dim,
        PPO_DEEPCUBEA_ACTOR_LR, PPO_DEEPCUBEA_CRITIC_LR,
        PPO_DEEPCUBEA_LMBDA, PPO_DEEPCUBEA_EPOCHS, PPO_DEEPCUBEA_EPS,
        PPO_DEEPCUBEA_GAMMA, PPO_DEEPCUBEA_ENTROPY_COEF, device,
        PPO_DEEPCUBEA_RESNET_BLOCKS,
    )

    results_dir = f"results/train_ppo_deepcubea_{datetime.now():%Y%m%d_%H%M%S}"

    return_list, step_list, solved_list, j_values_all = train_on_policy_agent(
        env, agent, model,
        PPO_DEEPCUBEA_NUM_EPISODES, PPO_DEEPCUBEA_MAX_STEPS,
        PPO_DEEPCUBEA_BETA, PPO_DEEPCUBEA_GAMMA, PPO_DEEPCUBEA_J_MAX,
        PPO_DEEPCUBEA_EVAL_INTERVAL, PPO_DEEPCUBEA_NUM_EVAL_EPISODES,
        PPO_DEEPCUBEA_SAVE_INTERVAL, results_dir,
    )
    env.close()

    if j_values_all:
        j_arr = np.array(j_values_all)
        print(f"\nJ(s) distribution over training: "
              f"min={j_arr.min():.1f} mean={j_arr.mean():.1f} max={j_arr.max():.1f}")

    plot_results(return_list, step_list, solved_list, results_dir, show_plot=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main_smoke()
    else:
        main()
