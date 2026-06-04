import gymnasium as gym
from config import ENV_NAME, RENDER_MODE


def make_env(render_mode=None):
    mode = render_mode if render_mode is not None else RENDER_MODE
    return gym.make(ENV_NAME, render_mode=mode)


if __name__ == "__main__":
    env = make_env()

    for episode in range(5):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if done:
                print(
                    f"episode {episode + 1}: "
                    f"steps={steps}, "
                    f"total_reward={total_reward}, "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )
                break

    env.close()
    print("self-test passed: 5 episodes completed")
