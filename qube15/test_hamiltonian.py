"""QUBE-15 Hamiltonian 单元测试。

验证：
1. 已解状态 ⟨Ĥ_k⟩ = 0
2. 单瓦片位移
3. 两瓦片耦合项
4. 软约束奖励数值
5. α 缩放后值域
6. 连续状态过渡
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qube15.quantum_state import compute_k_values, PHASE_TILES, is_solved, get_tile_target
from qube15.hamiltonian import (
    hamiltonian_expectation,
    effective_hamiltonian,
    phase_reward,
    phase_metric,
    compute_reward_from_obs,
)


def _make_solved_grid():
    """返回已解状态的 (4,4) grid。"""
    grid = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0],
    ], dtype=int)
    return grid


def _grid_to_obs(grid):
    """将 (4,4) grid 转为 flatten observation。"""
    return grid.flatten().astype(np.float32)


class TestQuantumState(unittest.TestCase):
    """量子态坐标映射测试。"""

    def test_solved_state_all_zeros(self):
        """已解状态：所有 (k_x, k_y) = (0, 0)。"""
        obs = _grid_to_obs(_make_solved_grid())
        k = compute_k_values(obs)
        for tid in range(1, 16):
            self.assertEqual(k[tid], (0, 0), f"Tile {tid} should be at target")

    def test_single_tile_displaced_right(self):
        """瓦片 1 右移 1 步 → k_x=1, k_y=0。"""
        grid = _make_solved_grid()
        grid[0, 0], grid[0, 1] = grid[0, 1], grid[0, 0]  # swap tile1 and tile2
        obs = _grid_to_obs(grid)
        k = compute_k_values(obs)
        self.assertEqual(k[1], (1, 0))   # tile1 moved right: col=1, target col=0
        self.assertEqual(k[2], (-1, 0))  # tile2 moved left

    def test_single_tile_displaced_down(self):
        """瓦片 1 下移 1 步 → k_x=0, k_y=1。"""
        grid = _make_solved_grid()
        grid[0, 0], grid[1, 0] = grid[1, 0], grid[0, 0]  # swap tile1 and tile5
        obs = _grid_to_obs(grid)
        k = compute_k_values(obs)
        self.assertEqual(k[1], (0, 1))   # tile1 moved down
        self.assertEqual(k[5], (0, -1))  # tile5 moved up

    def test_is_solved_phase(self):
        """is_solved phase 参数。"""
        grid = _make_solved_grid()
        obs = _grid_to_obs(grid)
        k = compute_k_values(obs)
        self.assertTrue(is_solved(k, phase=1))
        self.assertTrue(is_solved(k, phase=None))

        # 扰动 Phase 1 瓦片
        grid[0, 0], grid[0, 1] = grid[0, 1], grid[0, 0]
        obs = _grid_to_obs(grid)
        k = compute_k_values(obs)
        self.assertFalse(is_solved(k, phase=1))
        self.assertTrue(is_solved(k, phase=2))   # Phase 2 未受影响

    def test_target_positions(self):
        """目标位置正确性。"""
        self.assertEqual(get_tile_target(1), (0, 0))
        self.assertEqual(get_tile_target(4), (0, 3))
        self.assertEqual(get_tile_target(5), (1, 0))
        self.assertEqual(get_tile_target(15), (3, 2))


class TestHamiltonian(unittest.TestCase):
    """Hamiltonian 期望值测试。"""

    def test_solved_hamiltonian_zero(self):
        """已解状态：所有 ⟨Ĥ_k⟩ = 0。"""
        obs = _grid_to_obs(_make_solved_grid())
        k = compute_k_values(obs)
        for phase in range(1, 5):
            self.assertAlmostEqual(hamiltonian_expectation(k, phase), 0.0, places=10,
                                   msg=f"Phase {phase} should be 0 for solved state")

    def test_single_tile_local_term(self):
        """单瓦片位移 1 步：⟨Ĥ⟩ = B × 1（仅局域项，耦合项为 0）。"""
        grid = _make_solved_grid()
        grid[0, 0], grid[0, 1] = grid[0, 1], grid[0, 0]  # tile1 right 1 step
        obs = _grid_to_obs(grid)
        k = compute_k_values(obs)

        # Phase 1: tile1 displaced, tiles 2-4 at target
        # tile1: k_x=1, k_y=0 → k_x²+k_y²=1
        # tile2: k_x=-1, k_y=0 → k_x²+k_y²=1
        # tiles 3,4: 0
        # local = B*(1+1+0+0) = 2B = 2.0
        # coupling: only tile1-tile2 have non-zero:
        #   k1x²·k2x² + k1y²·k2y² = 1·1 + 0·0 = 1
        # coupling_sum = J*1 = 0.1
        expected = 2.0 + 0.1  # 2.1
        self.assertAlmostEqual(hamiltonian_expectation(k, 1), expected, places=10)

    def test_single_tile_only(self):
        """仅一个瓦片偏离（其他 Phase 1 瓦片归位）→ 无耦合项。"""
        grid = _make_solved_grid()
        # tile1 at (0,1), empty at (0,0)
        grid[0, 0] = 0
        grid[0, 1] = 1
        grid[0, 3] = 4
        # This is not a valid puzzle state easily... let me do a simpler test.

        # Actually let's just verify: if only tile1 is off by 1 step, and tiles 2-4 are at target:
        # That can't happen in a real puzzle (swaps are mutual), but for unit test we
        # just check the formula computes correctly with arbitrary k_values.
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)  # only tile1 displaced
        # local = B*(1+0) = 1.0
        # coupling: 0 (only one tile non-zero)
        self.assertAlmostEqual(hamiltonian_expectation(k, 1), 1.0, places=10)

    def test_two_tiles_coupling(self):
        """两瓦片各位移 1→ 局域 B×2 + 耦合 J×1。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)  # k_x²=1
        k[2] = (0, 1)  # k_y²=1
        # local = B*(1+0 + 0+1) = 2B = 2.0
        # coupling: k1x²*k2x² + k1y²*k2y² = 1*0 + 0*1 = 0
        # total = 2.0
        self.assertAlmostEqual(hamiltonian_expectation(k, 1), 2.0, places=10)

        # Both displaced in same axis
        k[1] = (1, 0)
        k[2] = (2, 0)
        # local = B*(1 + 4) = 5B = 5.0
        # coupling: k1x²*k2x² + k1y²*k2y² = 1*4 + 0*0 = 4
        # total = 5.0 + 0.1*4 = 5.4
        self.assertAlmostEqual(hamiltonian_expectation(k, 1), 5.4, places=10)

    def test_coupling_zero_when_one_at_target(self):
        """一个瓦片在目标位置时，耦合项为 0。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (2, 0)  # only tile1 displaced
        k[2] = (0, 0)  # at target
        # local = B*(4+0) = 4.0
        # coupling: k1x²*k2x² = 4*0 = 0
        self.assertAlmostEqual(hamiltonian_expectation(k, 1), 4.0, places=10)


class TestEffectiveHamiltonian(unittest.TestCase):
    """有效 Hamiltonian 与软约束测试。"""

    def test_phase1_equals_base(self):
        """Phase 1 无前驱阶段：⟨Ĥ_1^eff⟩ = ⟨Ĥ_1⟩。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)
        base = hamiltonian_expectation(k, 1)
        eff = effective_hamiltonian(k, 1)
        self.assertAlmostEqual(base, eff, places=10)

    def test_phase2_includes_soft_constraint(self):
        """Phase 2：⟨Ĥ_2^eff⟩ = ⟨Ĥ_2⟩ + λ⟨Ĥ_1⟩。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)  # Phase 1 tile displaced
        k[5] = (1, 0)  # Phase 2 tile displaced
        h1 = hamiltonian_expectation(k, 1)  # should be 1.0
        h2 = hamiltonian_expectation(k, 2)  # should be 1.0
        expected = h2 + 5.0 * h1  # λ=5
        self.assertAlmostEqual(effective_hamiltonian(k, 2), expected, places=10)

    def test_phase4_includes_all_prior(self):
        """Phase 4：包含 Phase 1-3 的软约束。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)
        k[5] = (1, 0)
        k[9] = (1, 0)
        h1 = hamiltonian_expectation(k, 1)
        h2 = hamiltonian_expectation(k, 2)
        h3 = hamiltonian_expectation(k, 3)
        h4 = hamiltonian_expectation(k, 4)
        expected = h4 + 5.0 * (h1 + h2 + h3)
        self.assertAlmostEqual(effective_hamiltonian(k, 4), expected, places=10)


class TestPhaseReward(unittest.TestCase):
    """阶段奖励函数测试。"""

    def test_solved_reward_zero(self):
        """已解状态：r = 0。"""
        obs = _grid_to_obs(_make_solved_grid())
        k = compute_k_values(obs)
        for phase in range(1, 5):
            self.assertAlmostEqual(phase_reward(k, phase), 0.0, places=10)

    def test_reward_negative_for_unsolved(self):
        """未解状态：奖励为负。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)
        r = phase_reward(k, 1)
        self.assertLess(r, 0.0)

    def test_phase1_reward_numerical(self):
        """Phase 1 奖励数值验证：
        tile1 displaced (1,0), others at target.
        ⟨Ĥ₁⟩ = 1.0
        r = -α · ⟨Ĥ₁⟩ / 4 = -0.1 × 1.0 / 4 = -0.025
        """
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)
        r = phase_reward(k, 1, alpha=0.1)
        expected = -0.1 * 1.0 / 4
        self.assertAlmostEqual(r, expected, places=10)

    def test_phase2_reward_with_soft_constraint(self):
        """Phase 2 奖励包含软约束：
        tile1 displaced (1,0), tile5 displaced (1,0).
        ⟨Ĥ₁⟩ = 1.0, ⟨Ĥ₂⟩ = 1.0
        ⟨Ĥ₂^eff⟩ = 1.0 + 5.0×1.0 = 6.0
        r = -0.1 × 6.0 / 4 = -0.15
        """
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (1, 0)
        k[5] = (1, 0)
        r = phase_reward(k, 2, alpha=0.1)
        expected = -0.1 * (1.0 + 5.0 * 1.0) / 4
        self.assertAlmostEqual(r, expected, places=10)

    def test_alpha_scaling(self):
        """α 缩放：r ∝ α。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[1] = (2, 0)
        r_default = phase_reward(k, 1, alpha=0.1)
        r_double = phase_reward(k, 1, alpha=0.2)
        self.assertAlmostEqual(r_double, 2 * r_default, places=10)

    def test_phase3_three_tiles(self):
        """Phase 3：|S₃|=4 瓦片。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[9] = (1, 0)  # Phase 3 tile displaced
        r = phase_reward(k, 3)
        expected = -0.1 * 1.0 / 4
        self.assertAlmostEqual(r, expected, places=10)

    def test_phase4_three_tiles_normalization(self):
        """Phase 4：|S₄|=3 瓦片。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        k[13] = (1, 0)  # Phase 4 tile displaced
        r = phase_reward(k, 4)
        expected = -0.1 * 1.0 / 3
        self.assertAlmostEqual(r, expected, places=10)


class TestRewardValueRange(unittest.TestCase):
    """奖励值域验证。"""

    def test_reward_within_range(self):
        """随机扰动状态下，奖励有限且非正（不爆炸）。"""
        np.random.seed(42)
        for _ in range(100):
            tiles = list(range(1, 16)) + [0]
            np.random.shuffle(tiles)
            grid = np.array(tiles).reshape(4, 4)
            obs = _grid_to_obs(grid)
            k = compute_k_values(obs)

            for phase in range(1, 5):
                r = phase_reward(k, phase)
                self.assertTrue(np.isfinite(r),
                    f"Reward should be finite, got {r} for phase {phase}")
                self.assertLessEqual(r, 0.0,
                    f"Reward {r:.4f} above 0 for phase {phase}")

    def test_worst_case_reward(self):
        """最坏情况（所有瓦片在最远位置）奖励有限不爆炸。"""
        k = {}
        for tid in range(1, 16):
            r_goal, c_goal = get_tile_target(tid)
            k[tid] = (3 - c_goal, 3 - r_goal)

        for phase in range(1, 5):
            r = phase_reward(k, phase)
            self.assertTrue(np.isfinite(r),
                f"Worst-case reward should be finite, got {r} for phase {phase}")
            self.assertLessEqual(r, 0.0)


class TestPhaseMetric(unittest.TestCase):
    """阶段切换指标测试。"""

    def test_solved_metric_zero(self):
        """已解状态：metric = 0。"""
        obs = _grid_to_obs(_make_solved_grid())
        k = compute_k_values(obs)
        for phase in range(1, 5):
            self.assertAlmostEqual(phase_metric(k, phase), 0.0, places=10)

    def test_metric_scales_with_displacement(self):
        """metric 随位移单调递增。"""
        k = {tid: (0, 0) for tid in range(1, 16)}
        m1 = phase_metric(k, 1)
        k[1] = (1, 0)
        m2 = phase_metric(k, 1)
        k[1] = (2, 0)
        m3 = phase_metric(k, 1)
        self.assertLess(m1, m2)
        self.assertLess(m2, m3)


class TestConvenienceFunction(unittest.TestCase):
    """便捷函数测试。"""

    def test_compute_reward_from_obs(self):
        """一步计算：从 obs 直接得 reward/k_values/metric。"""
        obs = _grid_to_obs(_make_solved_grid())
        r, k, m = compute_reward_from_obs(obs, phase=1)
        self.assertAlmostEqual(r, 0.0, places=10)
        self.assertAlmostEqual(m, 0.0, places=10)
        self.assertEqual(len(k), 15)

    def test_continuous_transition(self):
        """连续状态过渡：奖励变化连续（非离散跳变）。"""
        # 从已解状态逐步移动 tile1
        r_prev = None
        for dist in range(4):
            k = {tid: (0, 0) for tid in range(1, 16)}
            k[1] = (dist, 0)
            r = phase_reward(k, 1)
            if r_prev is not None:
                # 奖励应单调递减（距离越大越负）
                self.assertLess(r, r_prev,
                    f"Reward should decrease with distance: {r} >= {r_prev}")
            r_prev = r


if __name__ == "__main__":
    unittest.main(verbosity=2)
