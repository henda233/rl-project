import os
from datetime import datetime

import torch
from gymnasium.wrappers import RecordVideo

from config import (
    PPO_HIDDEN_DIM, PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_GAMMA,
    PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_ENTROPY_COEF,
    PPO_INFERENCE_EPISODES, PPO_USE_GPU,
    RND_INFERENCE_ACTOR_PATH, RND_INFERENCE_CRITIC_PATH,
)
from ppo_rnd_agent import PPO
from env import make_env


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"results/videos/{timestamp}"
    os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if PPO_USE_GPU else "cpu")

    env = make_env(render_mode="rgb_array")
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, PPO_HIDDEN_DIM, action_dim, PPO_ACTOR_LR, PPO_CRITIC_LR,
                PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_GAMMA, PPO_ENTROPY_COEF, device)

    agent.actor.load_state_dict(torch.load(RND_INFERENCE_ACTOR_PATH, map_location=device))
    agent.critic.load_state_dict(torch.load(RND_INFERENCE_CRITIC_PATH, map_location=device))

    for episode in range(PPO_INFERENCE_EPISODES):
        obs, info = env.reset()
        episode_return = 0
        episode_length = 0
        max_position = 0
        done = False
        while not done:
            action = agent.take_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
            max_position = max(max_position, obs[0])
        cleared = "Cleared!" if max_position >= 0.5 else ""
        print(f'Episode {episode + 1}: return={episode_return}, length={episode_length}, max_pos={max_position:.3f} {cleared}')

    env.close()


if __name__ == "__main__":
    main()
