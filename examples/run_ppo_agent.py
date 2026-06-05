import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from config import (
    PPO_HIDDEN_DIM, PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_GAMMA,
    PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_ENTROPY_COEF, PPO_INFERENCE_EPISODES,
)
from ppo_agent import PPO


def main():
    os.makedirs("results/videos", exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, "results/videos", episode_trigger=lambda x: True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, PPO_HIDDEN_DIM, action_dim, PPO_ACTOR_LR, PPO_CRITIC_LR,
                PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_GAMMA, PPO_ENTROPY_COEF, device)

    agent.actor.load_state_dict(torch.load("results/models/ppo_actor_best.pth", map_location=device))
    agent.critic.load_state_dict(torch.load("results/models/ppo_critic_best.pth", map_location=device))

    for episode in range(PPO_INFERENCE_EPISODES):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        done = False
        while not done:
            action = agent.take_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
        print(f'Episode {episode + 1}: return={episode_return}, length={episode_length}')

    env.close()


if __name__ == "__main__":
    main()
