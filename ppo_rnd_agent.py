import os
from datetime import datetime
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    PPO_HIDDEN_DIM, PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_GAMMA,
    PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_EVAL_INTERVAL,
    PPO_ENTROPY_COEF, PPO_USE_GPU,
    RND_HIDDEN_DIM, RND_OUTPUT_DIM, RND_LR, RND_BETA, RND_BETA_END, RND_BETA_DECAY, RND_EPOCHS,
    RND_NUM_EPISODES, RND_BUFFER_SIZE,
)
from env import make_env


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RNDTargetNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RNDPredictorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RNDModule:
    def __init__(self, state_dim, hidden_dim, output_dim, lr, device):
        self.target = RNDTargetNet(state_dim, hidden_dim, output_dim).to(device)
        self.predictor = RNDPredictorNet(state_dim, hidden_dim, output_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.device = device
        self.running_std = 1.0
        self.update_count = 0

    def get_intrinsic_reward(self, states):
        states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        with torch.no_grad():
            target_out = self.target(states_tensor)
            pred_out = self.predictor(states_tensor)
            mse = torch.mean((pred_out - target_out) ** 2, dim=1)
        return mse.cpu().numpy()

    def normalize(self, r_int):
        return r_int / (self.running_std + 1e-8)

    def update(self, states, epochs):
        states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        with torch.no_grad():
            target_out = self.target(states_tensor)
        for _ in range(epochs):
            pred_out = self.predictor(states_tensor)
            loss = F.mse_loss(pred_out, target_out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            pred_out = self.predictor(states_tensor)
            mse = torch.mean((pred_out - target_out) ** 2, dim=1)
            batch_std = mse.std().item()
        self.update_count += 1
        alpha = 1.0 / (self.update_count + 1)
        self.running_std = (1 - alpha) * self.running_std + alpha * batch_std


def compute_gae(gamma, lmbda, td_delta, dones):
    advantages = []
    advantage = 0.0
    for delta, done in zip(reversed(td_delta), reversed(dones)):
        advantage = delta + gamma * lmbda * (1 - done) * advantage
        advantages.append(advantage)
    advantages.reverse()
    return torch.tensor(advantages, dtype=torch.float)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, entropy_coef, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.entropy_coef = entropy_coef
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = (td_target - self.critic(states)).detach().cpu().numpy()
        dones_np = dones.cpu().numpy()
        advantage = compute_gae(self.gamma, self.lmbda, td_delta.flatten(), dones_np.flatten())
        advantage = advantage.to(self.device).view(-1, 1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            probs = self.actor(states)
            action_dist = torch.distributions.Categorical(probs)
            log_probs = torch.log(probs.gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            entropy = action_dist.entropy().mean()
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.entropy_coef * entropy
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def evaluate(env, agent):
    obs, info = env.reset()
    episode_return = 0
    done = False
    while not done:
        action = agent.take_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward
    return episode_return


def train_rnd_ppo(env, ppo, rnd, num_episodes, results_dir):
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    state_buffer = deque(maxlen=RND_BUFFER_SIZE)
    total_return_list = []
    original_return_list = []
    max_pos_list = []
    best_original_return = -float('inf')

    current_beta = RND_BETA
    pbar = tqdm(range(1, num_episodes + 1), desc='Training RND+PPO')
    for i_episode in pbar:
        episode_original_return = 0
        episode_total_return = 0
        max_position = 0
        transition_dict = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': [],
        }
        obs, info = env.reset()
        done = False
        while not done:
            action = ppo.take_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            max_position = max(max_position, obs[0])
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['dones'].append(done)
            obs = next_obs
            episode_original_return += reward

        # RND intrinsic rewards
        episode_states = transition_dict['states']
        r_int = rnd.get_intrinsic_reward(episode_states)
        r_int_norm = rnd.normalize(r_int)

        # Combine rewards: r_total = r_ext + β * r_int_norm
        r_ext = np.full(len(episode_states), -1.0)  # MountainCar: -1 per step
        r_total = r_ext + current_beta * r_int_norm

        episode_total_return = r_total.sum()

        # Replace rewards in transition dict with combined rewards for GAE
        transition_dict['rewards'] = r_total.tolist()

        transition_dict['states'] = episode_states
        transition_dict['next_states'] = transition_dict['next_states']
        transition_dict['actions'] = transition_dict['actions']
        transition_dict['dones'] = transition_dict['dones']

        ppo.update(transition_dict)

        # Add episode states to buffer, train RND
        state_buffer.append(episode_states)
        all_buffer_states = np.concatenate(list(state_buffer))
        rnd.update(all_buffer_states, RND_EPOCHS)

        total_return_list.append(episode_total_return)
        original_return_list.append(episode_original_return)
        max_pos_list.append(max_position)

        if episode_original_return > best_original_return:
            best_original_return = episode_original_return
            torch.save(ppo.actor.state_dict(), os.path.join(results_dir, "models", "rnd_ppo_actor_best.pth"))
            torch.save(ppo.critic.state_dict(), os.path.join(results_dir, "models", "rnd_ppo_critic_best.pth"))

        if max_position >= 0.5:
            tqdm.write(f'[Cleared! Episode {i_episode}] max_position={max_position:.3f}')
            torch.save(ppo.actor.state_dict(), os.path.join(results_dir, "models", "rnd_ppo_actor_cleared.pth"))
            torch.save(ppo.critic.state_dict(), os.path.join(results_dir, "models", "rnd_ppo_critic_cleared.pth"))

        if i_episode % 10 == 0:
            recent_total = np.mean(total_return_list[-10:])
            recent_original = np.mean(original_return_list[-10:])
            recent_max_pos = np.mean(max_pos_list[-10:])
            pbar.set_postfix({
                'tot': f'{recent_total:.1f}',
                'orig': f'{recent_original:.1f}',
                'pos': f'{recent_max_pos:.3f}',
                'β': f'{current_beta:.2f}',
            })
            tqdm.write(
                f'[Episode {i_episode}] '
                f'total_return: {recent_total:.1f}, '
                f'original_return: {recent_original:.1f}, '
                f'max_pos: {recent_max_pos:.3f}, '
                f'beta: {current_beta:.2f}'
            )

        if i_episode % PPO_EVAL_INTERVAL == 0:
            eval_env = make_env(render_mode="human")
            eval_return = evaluate(eval_env, ppo)
            eval_env.close()
            tqdm.write(
                f'[Eval  episode {i_episode}] return: {eval_return:.1f}'
            )

        current_beta = max(RND_BETA_END, current_beta - RND_BETA_DECAY)

    return total_return_list, original_return_list


def plot_return(total_return_list, original_return_list, results_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    mv = moving_average(original_return_list, 9)
    ax1.plot(original_return_list, alpha=0.4, color='steelblue', label='episode return')
    ax1.plot(mv, color='steelblue', label='moving avg (9)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (original)')
    ax1.set_title('RND+PPO on MountainCar-v0 (Original Reward)')
    ax1.legend()

    mv2 = moving_average(total_return_list, 9)
    ax2.plot(total_return_list, alpha=0.4, color='darkorange', label='total episode return')
    ax2.plot(mv2, color='darkorange', label='moving avg (9)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Return (total = ext + beta*r_int)')
    ax2.set_title('RND+PPO on MountainCar-v0 (Total Reward)')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.join(results_dir, "imgs"), exist_ok=True)
    fig.savefig(os.path.join(results_dir, "imgs", "rnd_ppo_training_results.png"), dpi=150)
    plt.show()


def main():
    device = torch.device("cuda" if PPO_USE_GPU else "cpu")

    env = make_env()
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, PPO_HIDDEN_DIM, action_dim, PPO_ACTOR_LR, PPO_CRITIC_LR,
              PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_GAMMA, PPO_ENTROPY_COEF, device)

    rnd = RNDModule(state_dim, RND_HIDDEN_DIM, RND_OUTPUT_DIM, RND_LR, device)

    results_dir = f"results/rnd_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"[RND+PPO] Results will be saved to: {results_dir}/")

    total_return_list, original_return_list = train_rnd_ppo(
        env, ppo, rnd, RND_NUM_EPISODES, results_dir,
    )
    env.close()

    plot_return(total_return_list, original_return_list, results_dir)


if __name__ == "__main__":
    main()
