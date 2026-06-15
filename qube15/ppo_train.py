"""QUBE-15 PPO 训练集成：Hamiltonian 密集奖励 + 四阶段 curriculum learning。

使用 ResBlock+LN 网络（对齐 ppo_deepcubea_agent.py），通过外部
HamiltonianRewardWrapper 提供密集奖励，不修改现有代码。
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym

from config import (
    QUBE15_B, QUBE15_J, QUBE15_LAMBDA, QUBE15_ALPHA,
    QUBE15_EPSILON, QUBE15_SLIDING_WINDOW, QUBE15_PHASE_SWITCH_K,
    QUBE15_SHUFFLE_STEPS_P1, QUBE15_SHUFFLE_STEPS_P2,
    QUBE15_SHUFFLE_STEPS_P3, QUBE15_SHUFFLE_STEPS_P4,
    QUBE15_MAX_STEPS_P1, QUBE15_MAX_STEPS_P2,
    QUBE15_MAX_STEPS_P3, QUBE15_MAX_STEPS_P4,
    QUBE15_HIDDEN_DIM, QUBE15_ACTOR_LR, QUBE15_CRITIC_LR,
    QUBE15_GAMMA, QUBE15_LMBDA, QUBE15_EPOCHS, QUBE15_EPS,
    QUBE15_NUM_EPISODES, QUBE15_EVAL_INTERVAL, QUBE15_NUM_EVAL_EPISODES,
    QUBE15_ENTROPY_COEF, QUBE15_RESNET_BLOCKS, QUBE15_USE_GPU,
    QUBE15_SAVE_INTERVAL,
)
from env_digital_huarongdao import DigitalHuarongdaoEnv
from qube15.hamiltonian import compute_reward_from_obs


# ==================== Network Modules ====================

class ResBlock(nn.Module):
    """FC → LN → ReLU → FC → LN → +residual → ReLU。"""
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
    """Actor：输入 obs grid，输出 action logits。"""
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
    """Critic：输入 obs grid，输出 scalar 状态价值。"""
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
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, num_resnet_blocks).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, num_resnet_blocks).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state):
        state_t = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        logits = self.actor(state_t)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = (td_target - self.critic(states)).detach().cpu().numpy()
        dones_np = dones.cpu().numpy()
        advantage = compute_gae(self.gamma, self.lmbda, td_delta.flatten(), dones_np.flatten())
        advantage = advantage.to(self.device).view(-1, 1)
        adv_std = advantage.std()
        if adv_std > 1e-8:
            advantage = (advantage - advantage.mean()) / adv_std

        old_log_probs = F.log_softmax(self.actor(states), dim=1).gather(1, actions).detach()

        for _ in range(self.epochs):
            logits = self.actor(states)
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
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# ==================== Hamiltonian Reward Wrapper ====================

class HamiltonianRewardWrapper(gym.Wrapper):
    """外部 wrapper：将 env 的稀疏奖励替换为 Hamiltonian 密集奖励。

    不修改 DigitalHuarongdaoEnv 或任何现有代码。
    通过 unwrapped 属性代理修改 shuffle_steps / max_steps。
    """

    _SHUFFLE_STEPS = {
        1: QUBE15_SHUFFLE_STEPS_P1, 2: QUBE15_SHUFFLE_STEPS_P2,
        3: QUBE15_SHUFFLE_STEPS_P3, 4: QUBE15_SHUFFLE_STEPS_P4,
    }
    _MAX_STEPS = {
        1: QUBE15_MAX_STEPS_P1, 2: QUBE15_MAX_STEPS_P2,
        3: QUBE15_MAX_STEPS_P3, 4: QUBE15_MAX_STEPS_P4,
    }

    def __init__(self, env, B=QUBE15_B, J=QUBE15_J, lmbda=QUBE15_LAMBDA,
                 alpha=QUBE15_ALPHA, phase=1):
        super().__init__(env)
        self.B = B
        self.J = J
        self.lmbda = lmbda
        self.alpha = alpha
        self._phase = phase
        self._last_metric = None
        self._last_k_values = None

    @property
    def phase(self):
        return self._phase

    def set_phase(self, phase):
        self._phase = phase
        self._apply_phase_config()

    def _apply_phase_config(self):
        env = self.unwrapped
        env.shuffle_steps = self._SHUFFLE_STEPS[self._phase]
        env.max_steps = self._MAX_STEPS[self._phase]

    def reset(self, **kwargs):
        self._apply_phase_config()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        r, k_values, metric = compute_reward_from_obs(
            obs, self._phase, self.B, self.J, self.lmbda, self.alpha)
        r = max(r, -10.0)  # 防止极端状态梯度爆炸
        self._last_metric = metric
        self._last_k_values = k_values
        info['hamiltonian_reward'] = r
        info['phase_metric'] = metric
        return obs, r, terminated, truncated, info


# ==================== Training ====================

def run_episode(env, agent):
    """执行单个 episode rollout，返回 transition_dict 和统计。"""
    transition_dict = {
        'states': [], 'actions': [], 'next_states': [],
        'rewards': [], 'dones': [],
    }
    obs, _info = env.reset()
    episode_return = 0.0
    step_count = 0
    terminated = False
    max_steps = env.unwrapped.max_steps

    while step_count < max_steps:
        action = agent.take_action(obs)
        next_obs, reward, _terminated, truncated, info = env.step(action)
        done = _terminated or truncated
        step_count += 1

        transition_dict['states'].append(obs)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(reward)
        transition_dict['next_states'].append(next_obs)
        transition_dict['dones'].append(done)

        episode_return += reward
        obs = next_obs

        if done:
            terminated = _terminated
            break

    metric = info.get('phase_metric', None)
    return transition_dict, episode_return, terminated, step_count, metric


def evaluate(env, agent, num_eval_episodes, phase):
    """多 episode 评估，返回 (avg_return, win_rate, avg_steps_solved, avg_metric)。"""
    env.set_phase(phase)
    total_return = 0.0
    solved_count = 0
    solved_steps = []
    metrics = []

    for _ in range(num_eval_episodes):
        _transition_dict, ep_return, solved, steps, metric = run_episode(env, agent)
        total_return += ep_return
        if solved:
            solved_count += 1
            solved_steps.append(steps)
        if metric is not None:
            metrics.append(metric)

    avg_return = total_return / num_eval_episodes
    win_rate = solved_count / num_eval_episodes
    avg_steps = float(np.mean(solved_steps)) if solved_steps else float('nan')
    avg_metric = float(np.mean(metrics)) if metrics else float('nan')
    return avg_return, win_rate, avg_steps, avg_metric


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes,
                          eval_interval, num_eval_episodes,
                          save_interval, results_dir="results"):
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)

    return_list = []
    step_list = []
    solved_list = []
    metric_list = []
    phase_boundaries = []  # (episode, phase) transitions
    best_original_return = -float('inf')

    current_phase = 1
    max_phase = 4
    env.set_phase(current_phase)
    epsilon = QUBE15_EPSILON
    window = QUBE15_SLIDING_WINDOW
    k_required = QUBE15_PHASE_SWITCH_K
    cooldown = window // 2

    # Phase 切换状态（随 phase 切换重置）
    phase_metrics = []        # 当前 phase 的 metric 历史，切换时清空
    consecutive_passes = 0    # 连续检查通过次数
    episodes_in_phase = 0     # 当前 phase 已运行的 episode 数

    pbar = tqdm(range(1, num_episodes + 1), desc=f'Training QUBE-15 [Phase {current_phase}]')

    for i_episode in pbar:
        transition_dict, episode_return, solved, steps, metric = run_episode(env, agent)
        return_list.append(episode_return)
        step_list.append(steps)
        solved_list.append(solved)
        if metric is not None:
            metric_list.append(metric)
            phase_metrics.append(metric)
        episodes_in_phase += 1

        agent.update(transition_dict)

        if episode_return > best_original_return:
            best_original_return = episode_return
            torch.save(agent.actor.state_dict(),
                       os.path.join(results_dir, "models", "qube15_actor_best.pth"))
            torch.save(agent.critic.state_dict(),
                       os.path.join(results_dir, "models", "qube15_critic_best.pth"))

        if solved:
            tqdm.write(f'[Solved! Episode {i_episode}] phase={current_phase} steps={steps}')

        # Phase 切换检查（冷却期 + 连续 K 次通过）
        if current_phase < max_phase and episodes_in_phase > cooldown:
            if len(phase_metrics) >= window:
                avg = np.mean(phase_metrics[-window:])
                if avg < epsilon:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0

                if consecutive_passes >= k_required:
                    current_phase += 1
                    env.set_phase(current_phase)
                    phase_boundaries.append((i_episode, current_phase))
                    tqdm.write(f'\n>>> Phase {current_phase - 1} -> Phase {current_phase} '
                               f'at episode {i_episode} '
                               f'(metric avg={avg:.3f} < {epsilon}, '
                               f'K={consecutive_passes}/{k_required})\n')
                    pbar.set_description(f'Training QUBE-15 [Phase {current_phase}]')
                    phase_metrics = []
                    consecutive_passes = 0
                    episodes_in_phase = 0

        if i_episode % 10 == 0:
            recent_ret = np.mean(return_list[-10:])
            recent_steps = np.mean(step_list[-10:])
            recent_solved = np.mean(solved_list[-10:])
            phase_str = f'P{current_phase}'
            pbar.set_postfix({
                'phase': phase_str,
                'return': f'{recent_ret:.1f}',
                'steps': f'{recent_steps:.1f}',
                'solved%': f'{recent_solved:.0%}',
            })

        if save_interval > 0 and i_episode % save_interval == 0:
            torch.save(agent.actor.state_dict(),
                       os.path.join(results_dir, "models", "qube15_actor_latest.pth"))
            torch.save(agent.critic.state_dict(),
                       os.path.join(results_dir, "models", "qube15_critic_latest.pth"))
            plot_results(return_list, step_list, solved_list, metric_list,
                         phase_boundaries, results_dir, current_phase,
                         show_plot=False, filename="qube15_results_latest.png")

        if i_episode % eval_interval == 0:
            _avg_ret, win_rate, _avg_steps, avg_metric = evaluate(
                env, agent, num_eval_episodes, current_phase)
            tqdm.write(
                f'[Eval  episode {i_episode}] '
                f'phase={current_phase} '
                f'win%: {win_rate:.0%} '
                f'metric: {avg_metric:.3f}'
            )

    # 最终保存
    torch.save(agent.actor.state_dict(),
               os.path.join(results_dir, "models", "qube15_actor_final.pth"))
    torch.save(agent.critic.state_dict(),
               os.path.join(results_dir, "models", "qube15_critic_final.pth"))

    return return_list, step_list, solved_list, metric_list, phase_boundaries


# ==================== Plotting ====================

def plot_results(return_list, step_list, solved_list, metric_list,
                 phase_boundaries, results_dir="results", current_phase=1,
                 show_plot=True, filename="qube15_results.png"):
    os.makedirs(os.path.join(results_dir, "imgs"), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Return
    ax = axes[0]
    mv_ret = moving_average(return_list, 9)
    ax.plot(return_list, alpha=0.4, color='steelblue', label='episode return')
    ax.plot(mv_ret, color='steelblue', label='moving avg (9)')
    for ep, ph in phase_boundaries:
        ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
        ax.text(ep, ax.get_ylim()[1] * 0.9, f'P{ph}', color='red', fontsize=8)
    ax.set_ylabel('Return')
    ax.set_title(f'QUBE-15 PPO Training (current phase: {current_phase})')
    ax.legend()

    # Steps
    ax = axes[1]
    mv_steps = moving_average(step_list, 9)
    ax.plot(step_list, alpha=0.4, color='darkorange', label='episode steps')
    ax.plot(mv_steps, color='darkorange', label='moving avg (9)')
    for ep, ph in phase_boundaries:
        ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Steps')
    ax.legend()

    # Win rate
    ax = axes[2]
    win_rate_window = 100
    win_rates = [np.mean(solved_list[max(0, i - win_rate_window + 1):i + 1])
                 for i in range(len(solved_list))]
    ax.plot(win_rates, color='seagreen', label=f'win rate (window={win_rate_window})')
    for ep, ph in phase_boundaries:
        ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Win Rate')
    ax.legend()

    # Phase metric ⟨Ĥ_k⟩/|S_k|
    ax = axes[3]
    if metric_list:
        mv_metric = moving_average(metric_list, 9)
        ax.plot(metric_list, alpha=0.4, color='purple', label='⟨Ĥ⟩/|S| per episode')
        ax.plot(mv_metric, color='purple', label='moving avg (9)')
        ax.axhline(y=QUBE15_EPSILON, color='red', linestyle=':', label=f'ε={QUBE15_EPSILON}')
        for ep, ph in phase_boundaries:
            ax.axvline(x=ep, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('⟨Ĥ⟩/|S|')
    ax.set_xlabel('Episode')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "imgs", filename), dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ==================== Main ====================

def main():
    device = torch.device("cuda" if QUBE15_USE_GPU else "cpu")
    print(f"Device: {device}")

    base_env = DigitalHuarongdaoEnv()
    env = HamiltonianRewardWrapper(base_env)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(
        state_dim, QUBE15_HIDDEN_DIM, action_dim,
        QUBE15_ACTOR_LR, QUBE15_CRITIC_LR,
        QUBE15_LMBDA, QUBE15_EPOCHS, QUBE15_EPS,
        QUBE15_GAMMA, QUBE15_ENTROPY_COEF, device,
        QUBE15_RESNET_BLOCKS,
    )

    results_dir = f"results/train_qube15_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Results dir: {results_dir}")
    print(f"Config: B={QUBE15_B}, J={QUBE15_J}, lambda={QUBE15_LAMBDA}, alpha={QUBE15_ALPHA}")
    print(f"Phase switch: epsilon={QUBE15_EPSILON}, window={QUBE15_SLIDING_WINDOW}")
    print(f"Shuffle steps: P1={QUBE15_SHUFFLE_STEPS_P1} P2={QUBE15_SHUFFLE_STEPS_P2} "
          f"P3={QUBE15_SHUFFLE_STEPS_P3} P4={QUBE15_SHUFFLE_STEPS_P4}")
    print(f"Max steps: P1={QUBE15_MAX_STEPS_P1} P2={QUBE15_MAX_STEPS_P2} "
          f"P3={QUBE15_MAX_STEPS_P3} P4={QUBE15_MAX_STEPS_P4}")

    return_list, step_list, solved_list, metric_list, phase_boundaries = \
        train_on_policy_agent(
            env, agent, QUBE15_NUM_EPISODES,
            QUBE15_EVAL_INTERVAL, QUBE15_NUM_EVAL_EPISODES,
            QUBE15_SAVE_INTERVAL, results_dir,
        )
    env.close()

    print(f"\nPhase transitions: {phase_boundaries}")
    if metric_list:
        print(f"Final metric (H/|S|) moving avg: {np.mean(metric_list[-100:]):.3f}")

    plot_results(return_list, step_list, solved_list, metric_list,
                 phase_boundaries, results_dir, show_plot=True)


if __name__ == "__main__":
    main()
