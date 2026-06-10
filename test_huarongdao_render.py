"""数字华容道渲染测试：随机步可视化。"""
from env_digital_huarongdao import make_huarongdao_env
import numpy as np

np.random.seed(42)

env = make_huarongdao_env(render_mode='human')
obs, info = env.reset()
print('=== 初始状态 (reset 后) ===')
print(f'观测向量: {obs}')
print()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    dirs = ['上', '下', '左', '右']
    print(f'--- Step {step+1}: 动作={dirs[action]}, reward={reward:.0f} ---')
    if terminated:
        print('>>> 已通关!')
        break
    if truncated:
        print('>>> 已达最大步数!')
        break

env.close()
print()
print('=== 最终观测 ===')
print(obs)
