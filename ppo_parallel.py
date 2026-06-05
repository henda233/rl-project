import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

from config import (
    PPO_HIDDEN_DIM, PPO_ACTOR_LR, PPO_CRITIC_LR, PPO_GAMMA,
    PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_NUM_EPISODES,
    PPO_ENTROPY_COEF, PPO_USE_GPU, PPO_NUM_ENVS, POTENTIAL_K_START, POTENTIAL_K_INTERVAL,
)
from env import make_env
from ppo_agent import PolicyNet, ValueNet, PPO, moving_average


def train_single_agent(k_value, seed, result_queue):
    device = torch.device("cuda" if PPO_USE_GPU else "cpu")
    torch.manual_seed(seed)

    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, PPO_HIDDEN_DIM, action_dim, PPO_ACTOR_LR, PPO_CRITIC_LR,
                PPO_LMBDA, PPO_EPOCHS, PPO_EPS, PPO_GAMMA, PPO_ENTROPY_COEF, device)

    return_list = []
    shaped_return_list = []
    best_shaped_return = -float('inf')
    best_state = None
    best_episode = 0

    for i_episode in range(1, PPO_NUM_EPISODES + 1):
        episode_return = 0
        shaped_episode_return = 0
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
            shaped_reward = reward + k_value * (PPO_GAMMA * next_obs[0] - obs[0])
            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(shaped_reward)
            transition_dict['dones'].append(done)
            obs = next_obs
            episode_return += reward
            shaped_episode_return += shaped_reward

        return_list.append(episode_return)
        shaped_return_list.append(shaped_episode_return)
        agent.update(transition_dict)

        if shaped_episode_return > best_shaped_return:
            best_shaped_return = shaped_episode_return
            best_episode = i_episode
            best_state = {
                'actor': {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()},
                'critic': {k: v.cpu().clone() for k, v in agent.critic.state_dict().items()},
            }

        if i_episode % 10 == 0:
            recent_avg = np.mean(return_list[-10:])
            shaped_avg = np.mean(shaped_return_list[-10:])
            print(
                f'[k={k_value:.1f}][Ep {i_episode:4d}/{PPO_NUM_EPISODES}] '
                f'return(original): {recent_avg:7.1f}  return(shaped): {shaped_avg:7.1f}'
            )

    env.close()

    os.makedirs('results/models', exist_ok=True)
    model_path = f'results/models/ppo_k{k_value:.1f}.pth'
    torch.save(best_state, model_path)

    result_queue.put({
        'k': k_value,
        'seed': seed,
        'return_list': return_list,
        'shaped_return_list': shaped_return_list,
        'best_shaped_return': best_shaped_return,
        'best_episode': best_episode,
    })


def plot_comparison(all_results):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_results)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for r, c in zip(all_results, colors):
        k = r['k']
        mv_orig = moving_average(r['return_list'], 9)
        mv_shaped = moving_average(r['shaped_return_list'], 9)

        ax1.plot(r['return_list'], alpha=0.15, color=c, linewidth=0.8)
        ax1.plot(mv_orig, color=c, linewidth=1.5, label=f'k={k:.1f}')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Return (original)')
        ax1.set_title('PPO Parallel Training — Original Reward Comparison')
        ax1.legend(fontsize=8)

        ax2.plot(r['shaped_return_list'], alpha=0.15, color=c, linewidth=0.8)
        ax2.plot(mv_shaped, color=c, linewidth=1.5, label=f'k={k:.1f}')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return (shaped)')
        ax2.set_title('PPO Parallel Training — Shaped Reward Comparison')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    save_path = 'results/comparison.png'
    fig.savefig(save_path, dpi=150)
    print(f'Comparison plot saved to: {save_path}')
    plt.close(fig)


def main():
    k_values = [POTENTIAL_K_START + i * POTENTIAL_K_INTERVAL for i in range(PPO_NUM_ENVS)]

    print(f'PPO Parallel Training: {PPO_NUM_ENVS} environments, {PPO_NUM_EPISODES} episodes each')
    print(f'k values: {[f"{k:.1f}" for k in k_values]}')
    print()

    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []

    for i, k in enumerate(k_values):
        p = mp.Process(target=train_single_agent, args=(k, i, result_queue))
        p.start()
        processes.append(p)

    all_results = [result_queue.get() for _ in range(len(processes))]

    for p in processes:
        p.join()

    all_results.sort(key=lambda x: x['k'])

    plot_comparison(all_results)

    best = max(all_results, key=lambda x: x['best_shaped_return'])
    print(f'\n===== Best k = {best["k"]:.1f} =====')
    print(f'  Best shaped return: {best["best_shaped_return"]:.1f} (episode {best["best_episode"]})')
    print(f'  Model saved: results/models/ppo_k{best["k"]:.1f}.pth')

    print(f'\n{"k":>6s}  {"best_shaped":>12s}  {"best_ep":>8s}  {"final_orig_avg":>15s}')
    print('-' * 50)
    for r in all_results:
        final_orig_avg = np.mean(r['return_list'][-50:])
        print(f'{r["k"]:6.1f}  {r["best_shaped_return"]:12.1f}  {r["best_episode"]:8d}  {final_orig_avg:15.1f}')


if __name__ == '__main__':
    main()
