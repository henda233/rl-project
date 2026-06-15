"""QUBE-15 量子态模块：15-puzzle 瓦片的 Hilbert 空间嵌入。

纯函数式，仅 numpy 依赖。将 15-puzzle 的经典 grid 状态映射为量子位置量子数
(k_x, k_y)，为 Hamiltonian 期望值计算提供输入。
"""

import numpy as np

# 目标位置 (0-indexed): tile_id -> (row_goal, col_goal)
_TARGET = {tid: ((tid - 1) // 4, (tid - 1) % 4) for tid in range(1, 16)}

# 四阶段瓦片分组
PHASE_TILES = {
    1: [1, 2, 3, 4],
    2: [5, 6, 7, 8],
    3: [9, 10, 11, 12],
    4: [13, 14, 15],
}


def compute_k_values(grid_flat):
    """从 flattened observation 计算所有 15 个瓦片的位置量子数。

    Args:
        grid_flat: (16,) int/float array，env observation，0=空格。

    Returns:
        dict: tile_id (1..15) -> (k_x, k_y)
            k_x = col - col_goal
            k_y = row - row_goal
    """
    grid_2d = grid_flat.reshape(4, 4).astype(int)
    k_values = {}
    for r in range(4):
        for c in range(4):
            tid = grid_2d[r, c]
            if tid == 0:
                continue
            r_goal, c_goal = _TARGET[tid]
            k_values[tid] = (c - c_goal, r - r_goal)
    return k_values


def is_solved(k_values, phase=None):
    """检查是否已解（全部或指定阶段的瓦片位于目标位置）。

    Args:
        k_values: compute_k_values 的输出。
        phase: None=全解，1-4=仅检查该阶段瓦片。

    Returns:
        bool
    """
    tiles = range(1, 16) if phase is None else PHASE_TILES[phase]
    return all(k_values[tid] == (0, 0) for tid in tiles)


def get_tile_target(tile_id):
    """返回 tile 的目标位置 (row, col)，0-indexed。"""
    return _TARGET[tile_id]
