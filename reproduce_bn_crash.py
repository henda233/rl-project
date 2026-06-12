"""复现 BN batch_size=1 崩溃。

预期结果：
  ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])
"""
import torch
from deepcubea_network import DeepCubeANetwork, encode

# 1. 构造网络，设为训练模式
net = DeepCubeANetwork()
net.train()

# 2. 构造单样本输入（模拟末尾 batch 仅 1 样本）
goal = torch.arange(16, dtype=torch.int32)  # (16,) 模拟一个 state
x = encode(goal.numpy())
x = x.unsqueeze(0)  # (1, 256) — batch_size=1

print(f"input shape: {x.shape}")
print("forward...")

try:
    pred = net(x)
    print(f"pred: {pred}")
except ValueError as e:
    print(f"\n>>> 复现成功: {e}")
