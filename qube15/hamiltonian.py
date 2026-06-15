"""QUBE-15 Hamiltonian 模块：4 个 Ising Hamiltonian + 期望值 + 阶段奖励。

纯函数式，仅 numpy 依赖。基于 QUBE 论文的量子力学框架，为 15-puzzle
提供密集奖励信号。

Hamiltonian 统一形式（Phase k）:
    Ĥ_k = B Σ_{i∈S_k} (K_x² + K_y²) + J Σ_{i<j∈S_k} (K_ix²K_jx² + K_iy²K_jy²)

软约束有效 Hamiltonian:
    Ĥ_k^eff = Ĥ_k + λ Σ_{j<k} Ĥ_j

奖励:
    r_k = -α · ⟨Ĥ_k^eff⟩ / |S_k|
"""

import numpy as np
from .quantum_state import compute_k_values, PHASE_TILES


def hamiltonian_expectation(k_values, phase, B=1.0, J=0.1):
    """计算单个阶段的 Hamiltonian 期望值 ⟨Ĥ_k⟩。

    ⟨Ĥ_k⟩ = B Σ(k_x² + k_y²) + J Σ_{i<j} (k_ix²k_jx² + k_iy²k_jy²)

    Args:
        k_values: compute_k_values 的输出，tile_id -> (k_x, k_y)。
        phase: 阶段 1-4。
        B: 局域场强度，默认 1.0。
        J: Ising 耦合常数，默认 0.1。

    Returns:
        float: 期望值 ⟨Ĥ_k⟩ ≥ 0。⟨Ĥ_k⟩ = 0 当且仅当该阶段全部瓦片归位。
    """
    tiles = PHASE_TILES[phase]
    n = len(tiles)

    k_sq = {}  # tile_id -> (k_x², k_y²)
    local_sum = 0.0
    for tid in tiles:
        kx, ky = k_values[tid]
        kx2, ky2 = kx * kx, ky * ky
        k_sq[tid] = (kx2, ky2)
        local_sum += kx2 + ky2

    coupling_sum = 0.0
    for i in range(n):
        tid_i = tiles[i]
        kix2, kiy2 = k_sq[tid_i]
        for j in range(i + 1, n):
            tid_j = tiles[j]
            kjx2, kjy2 = k_sq[tid_j]
            coupling_sum += kix2 * kjx2 + kiy2 * kjy2

    return B * local_sum + J * coupling_sum


def effective_hamiltonian(k_values, phase, B=1.0, J=0.1, lmbda=5.0):
    """计算有效 Hamiltonian 期望值 ⟨Ĥ_k^eff⟩ = ⟨Ĥ_k⟩ + λ Σ_{j<k} ⟨Ĥ_j⟩。

    Args:
        k_values: compute_k_values 的输出。
        phase: 当前阶段 1-4。
        B, J: Hamiltonian 参数。
        lmbda: 软约束权重，默认 5.0。

    Returns:
        float: 有效期望值 ≥ 0。
    """
    total = hamiltonian_expectation(k_values, phase, B, J)
    for prev_phase in range(1, phase):
        total += lmbda * hamiltonian_expectation(k_values, prev_phase, B, J)
    return total


def phase_reward(k_values, phase, B=1.0, J=0.1, lmbda=5.0, alpha=0.1):
    """计算阶段奖励 r_k = -α · ⟨Ĥ_k^eff⟩ / |S_k|。

    Args:
        k_values: compute_k_values 的输出。
        phase: 当前阶段 1-4。
        B, J, lmbda: Hamiltonian 参数。
        alpha: 奖励缩放系数，默认 0.1。

    Returns:
        float: 奖励值，范围约 [-4, 0]。
    """
    h_eff = effective_hamiltonian(k_values, phase, B, J, lmbda)
    n_tiles = len(PHASE_TILES[phase])
    return float(-alpha * h_eff / n_tiles)


def phase_metric(k_values, phase, B=1.0, J=0.1):
    """计算阶段切换指标 ⟨Ĥ_k⟩ / |S_k|（per-tile 平均 Hamiltonian）。

    用于判断当前阶段是否收敛：⟨Ĥ_k⟩ / |S_k| < ε 时切换。

    Args:
        k_values: compute_k_values 的输出。
        phase: 阶段 1-4。
        B, J: Hamiltonian 参数。

    Returns:
        float: per-tile 平均 ⟨Ĥ_k⟩。
    """
    return hamiltonian_expectation(k_values, phase, B, J) / len(PHASE_TILES[phase])


def compute_reward_from_obs(obs, phase, B=1.0, J=0.1, lmbda=5.0, alpha=0.1):
    """从环境 observation 直接计算奖励（一步便捷函数）。

    Args:
        obs: (16,) env observation。
        phase: 当前阶段 1-4。
        B, J, lmbda, alpha: Hamiltonian 参数。

    Returns:
        tuple: (reward, k_values, metric)
            reward: float 奖励值
            k_values: dict 位置量子数（可用于后续计算避免重复）
            metric: float per-tile 指标（用于 phase 切换判断）
    """
    k_values = compute_k_values(obs)
    reward = phase_reward(k_values, phase, B, J, lmbda, alpha)
    metric = phase_metric(k_values, phase, B, J)
    return reward, k_values, metric
