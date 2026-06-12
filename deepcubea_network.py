import numpy as np
import torch
import torch.nn as nn
from config import HUARONGDAO_N, DEEPCUBEA_HIDDEN_DIM

N = HUARONGDAO_N
N2 = N * N           # 16 positions
INPUT_DIM = N2 * N2  # 256-dim one-hot

_DR = [-1, 1, 0, 0]
_DC = [0, 0, -1, 1]


def get_children(grid, blank_idx):
    """返回所有合法子节点，一次调用消除 4 次 np.where。

    Args:
        grid: (N2,) numpy int32 array，当前状态（只读，不会被修改）
        blank_idx: 空格在 grid 中的索引

    Returns:
        [(child_grid, action, new_blank_idx), ...] 仅合法子节点
    """
    r, c = blank_idx // N, blank_idx % N
    children = []
    for action, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < N and 0 <= nc < N:
            new_idx = nr * N + nc
            child = grid.copy()
            child[blank_idx] = grid[new_idx]
            child[new_idx] = 0
            children.append((child, action, new_idx))
    return children


def encode(grid_flat):
    """Convert flat grid (N2,) int → one-hot (INPUT_DIM,) tensor.

    Position i (row-major), value v → bit i*N2 + v = 1.
    """
    one_hot = np.zeros(INPUT_DIM, dtype=np.float32)
    for i, v in enumerate(grid_flat):
        one_hot[i * N2 + int(v)] = 1.0
    return torch.from_numpy(one_hot)


def encode_batch(grids_flat):
    """Vectorized batch encode: (B, N2) numpy → (B, INPUT_DIM) tensor."""
    batch_size = grids_flat.shape[0]
    one_hot = np.zeros((batch_size, INPUT_DIM), dtype=np.float32)
    positions = np.arange(N2)
    indices = positions * N2 + grids_flat
    one_hot[np.arange(batch_size)[:, None], indices] = 1.0
    return torch.from_numpy(one_hot)


def decode(one_hot):
    """Convert one-hot (INPUT_DIM,) tensor/numpy → flat grid (N2,) int array."""
    if isinstance(one_hot, torch.Tensor):
        one_hot = one_hot.detach().cpu().numpy()
    grid_flat = np.zeros(N2, dtype=np.int32)
    for i in range(N2):
        segment = one_hot[i * N2 : (i + 1) * N2]
        grid_flat[i] = np.argmax(segment)
    return grid_flat


def transition(grid_flat, action):
    """Pure-function state transition. Returns new grid_flat or None if illegal."""
    idx = int(np.where(grid_flat == 0)[0][0])
    r, c = idx // N, idx % N

    dr, dc = _DR[action], _DC[action]
    nr, nc = r + dr, c + dc

    if 0 <= nr < N and 0 <= nc < N:
        new_grid = grid_flat.copy()
        new_idx = nr * N + nc
        new_grid[idx] = new_grid[new_idx]
        new_grid[new_idx] = 0
        return new_grid
    return None


class ResidualBlock(nn.Module):
    """FC → ReLU → FC → +input → ReLU."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + residual)


class DeepCubeANetwork(nn.Module):
    """DeepCubeA heuristic network J(s).

    Architecture: 256 → FC+ReLU → FC+ReLU → 4×ResBlock → Linear → scalar J.
    """

    def __init__(self, hidden_dim=None, num_res_blocks=4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = DEEPCUBEA_HIDDEN_DIM
        self.fc1 = nn.Linear(INPUT_DIM, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: (B, INPUT_DIM) one-hot tensor → (B,) scalar J values."""
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.relu(self.bn2(self.fc2(out)))
        for res_block in self.res_blocks:
            out = res_block(out)
        return self.output(out).squeeze(-1)

    def predict_j(self, grid_flat):
        """Single grid → scalar J (no_grad)."""
        self.eval()
        device = next(self.parameters()).device
        one_hot = encode(grid_flat).to(device)
        with torch.inference_mode():
            return self.forward(one_hot.unsqueeze(0)).item()

    def predict_j_batch(self, grids_flat):
        """Batch grids (B, N2) numpy → (B,) tensor J (no_grad)."""
        self.eval()
        device = next(self.parameters()).device
        one_hot = encode_batch(grids_flat).to(device)
        with torch.inference_mode():
            return self.forward(one_hot)
