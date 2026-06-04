import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    HIDDEN_DIM, ACTOR_LR, CRITIC_LR, GAMMA,
    NUM_EPISODES, EVAL_INTERVAL,
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


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
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


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    pbar = tqdm(range(1, num_episodes + 1), desc='Training')
    for i_episode in pbar:
        episode_return = 0
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
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            obs = next_obs
            episode_return += reward
        return_list.append(episode_return)
        agent.update(transition_dict)

        if i_episode % 10 == 0:
            recent_avg = np.mean(return_list[-10:])
            pbar.set_postfix({
                'return': f'{recent_avg:.1f}',
            })

        if i_episode % EVAL_INTERVAL == 0:
            eval_env = make_env(render_mode="human")
            eval_return = evaluate(eval_env, agent)
            eval_env.close()
            tqdm.write(
                f'[Eval  episode {i_episode}] return: {eval_return:.1f}'
            )

    return return_list


def plot_return(return_list):
    mv = moving_average(return_list, 9)
    plt.plot(return_list, alpha=0.4, color='steelblue', label='episode return')
    plt.plot(mv, color='steelblue', label='moving avg (9)')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Actor-Critic on MountainCar-v0')
    plt.legend()
    plt.show()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = make_env()
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, HIDDEN_DIM, action_dim, ACTOR_LR, CRITIC_LR,
                        GAMMA, device)

    return_list = train_on_policy_agent(env, agent, NUM_EPISODES)
    env.close()

    plot_return(return_list)


if __name__ == "__main__":
    main()
