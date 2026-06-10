import os
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    PPO_HIDDEN_DIM, PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_GAMMA,
    PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_NUM_EPISODES, PPO_EVAL_INTERVAL,
    PPO_ENTROPY_COEF, PPO_USE_GPU,
)
from env_digital_huarongdao import make_huarongdao_env


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
        logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=logits)
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


def train_on_policy_agent(env, agent, num_episodes, results_dir="results"):
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    return_list = []
    placed_list = []
    step_list = []
    best_original_return = -float('inf')
    pbar = tqdm(range(1, num_episodes + 1), desc='Training PPO')
    for i_episode in pbar:
        episode_return = 0
        max_placed = 0
        transition_dict = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': [],
        }
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.take_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            max_placed = max(max_placed, info.get('placed_count', 0))
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            obs = next_obs
            episode_return += reward
        return_list.append(episode_return)
        placed_list.append(max_placed)
        step_list.append(info.get('step_count', 0))
        agent.update(transition_dict)

        if episode_return > best_original_return:
            best_original_return = episode_return
            torch.save(agent.actor.state_dict(), os.path.join(results_dir, "models", "ppo_actor_best.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(results_dir, "models", "ppo_critic_best.pth"))

        if terminated:
            tqdm.write(f'[Solved! Episode {i_episode}] placed={max_placed} steps={info.get("step_count", 0)}')
            torch.save(agent.actor.state_dict(), os.path.join(results_dir, "models", "ppo_actor_cleared.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(results_dir, "models", "ppo_critic_cleared.pth"))

        if i_episode % 10 == 0:
            recent_avg = np.mean(return_list[-10:])
            recent_placed = np.mean(placed_list[-10:])
            recent_steps = np.mean(step_list[-10:])
            pbar.set_postfix({
                'return': f'{recent_avg:.1f}',
                'placed': f'{recent_placed:.1f}',
                'steps': f'{recent_steps:.1f}',
            })
            tqdm.write(
                f'[Episode {i_episode}] '
                f'return: {recent_avg:.1f}, '
                f'placed: {recent_placed:.1f}, '
                f'steps: {recent_steps:.1f}'
            )

        if i_episode % PPO_EVAL_INTERVAL == 0:
            eval_env = make_huarongdao_env(render_mode="human")
            eval_return = evaluate(eval_env, agent)
            eval_env.close()
            tqdm.write(
                f'[Eval  episode {i_episode}] return: {eval_return:.1f}'
            )

    return return_list


def plot_return(return_list, results_dir="results"):
    fig, ax = plt.subplots(figsize=(10, 5))

    mv = moving_average(return_list, 9)
    ax.plot(return_list, alpha=0.4, color='steelblue', label='episode return')
    ax.plot(mv, color='steelblue', label='moving avg (9)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('PPO on DigitalHuarongdaoEnv')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.join(results_dir, "imgs"), exist_ok=True)
    fig.savefig(os.path.join(results_dir, "imgs", "ppo_training_results.png"), dpi=150)
    plt.show()


def main():
    device = torch.device("cuda" if PPO_USE_GPU else "cpu")

    env = make_huarongdao_env()
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, PPO_HIDDEN_DIM, action_dim, PPO_ACTOR_LR, PPO_CRITIC_LR,
                PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_GAMMA, PPO_ENTROPY_COEF, device)

    results_dir = f"results/train_{datetime.now():%Y%m%d_%H%M%S}"

    return_list = train_on_policy_agent(
        env, agent, PPO_NUM_EPISODES, results_dir,
    )
    env.close()

    plot_return(return_list, results_dir)


if __name__ == "__main__":
    main()
