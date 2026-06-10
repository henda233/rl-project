import gymnasium as gym
from gymnasium import spaces
import numpy as np
from config import (HUARONGDAO_N, HUARONGDAO_MAX_STEPS,
                     HUARONGDAO_SHUFFLE_STEPS, HUARONGDAO_FIRST_PLACEMENT_REWARD)


class DigitalHuarongdaoEnv(gym.Env):
    """n×n 数字华容道 Gymnasium 环境。

    空格相邻方块滑入空格，目标：数字 1..n²-1 按序排列，空格在末尾。
    """
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    _DR = [-1, 1, 0, 0]
    _DC = [0, 0, -1, 1]

    def __init__(self, render_mode=None):
        super().__init__()
        self.n = HUARONGDAO_N
        self.max_steps = HUARONGDAO_MAX_STEPS
        self.shuffle_steps = HUARONGDAO_SHUFFLE_STEPS
        self.first_placement_reward = HUARONGDAO_FIRST_PLACEMENT_REWARD

        n2 = self.n ** 2
        self.observation_space = spaces.Box(
            low=0, high=n2 - 1, shape=(n2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._grid = None
        self._goal_grid = None
        self._rewarded_mask = None
        self._step_count = 0
        self._empty_pos = None

    def _get_obs(self):
        return self._grid.flatten().astype(np.float32)

    def _get_info(self):
        placed = int(np.sum((self._grid == self._goal_grid) & (self._grid != 0)))
        return {"step_count": self._step_count, "placed_count": placed}

    def _valid_actions(self):
        er, ec = self._empty_pos
        n = self.n
        valid = []
        for a in range(4):
            tr = er + self._DR[a]
            tc = ec + self._DC[a]
            if 0 <= tr < n and 0 <= tc < n:
                valid.append(a)
        return valid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = self.n

        self._goal_grid = np.arange(1, n**2 + 1, dtype=np.int32).reshape(n, n)
        self._goal_grid[n - 1, n - 1] = 0

        # 正向打乱：目标状态执行 N 次随机合法移动
        while True:
            self._grid = self._goal_grid.copy()
            self._empty_pos = (n - 1, n - 1)

            for _ in range(self.shuffle_steps):
                valid = self._valid_actions()
                a = self.np_random.choice(valid)
                dr, dc = self._DR[a], self._DC[a]
                er, ec = self._empty_pos
                tr, tc = er + dr, ec + dc
                self._grid[er, ec] = self._grid[tr, tc]
                self._grid[tr, tc] = 0
                self._empty_pos = (tr, tc)

            if not np.array_equal(self._grid, self._goal_grid):
                break

        self._rewarded_mask = np.zeros((n, n), dtype=bool)
        self._step_count = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        er, ec = self._empty_pos
        dr, dc = self._DR[action], self._DC[action]
        tr, tc = er + dr, ec + dc
        n = self.n

        in_bounds = (0 <= tr < n) and (0 <= tc < n)

        if in_bounds:
            self._grid[er, ec] = self._grid[tr, tc]
            self._grid[tr, tc] = 0
            self._empty_pos = (tr, tc)

        self._step_count += 1

        # 奖励: -1/步 + 首次归位奖励
        reward = -1.0
        new_placements = 0
        for r in range(n):
            for c in range(n):
                if (self._grid[r, c] == self._goal_grid[r, c]
                        and not self._rewarded_mask[r, c]
                        and self._grid[r, c] != 0):
                    self._rewarded_mask[r, c] = True
                    new_placements += 1
        reward += self.first_placement_reward * new_placements

        terminated = np.array_equal(self._grid, self._goal_grid)
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            print(self._render_ansi())

    def _render_ansi(self):
        n = self.n
        lines = []
        sep = "+" + "---+" * n
        lines.append(sep)
        for r in range(n):
            row = "|"
            for c in range(n):
                val = self._grid[r, c]
                if val == 0:
                    row += "   |"
                else:
                    row += f"{val:^3d}|"
            lines.append(row)
            lines.append(sep)
        return "\n".join(lines)

    # ---- 5×3 点阵字体 ----
    _FONT = {
        0: np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        1: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
        2: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
        3: np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
        4: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
        5: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
        6: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        7: np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
        8: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        9: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    }

    def _render_number(self, num, scale=18):
        if num == 0:
            return np.kron(self._FONT[0], np.ones((scale, scale), dtype=bool))

        digits = []
        n = num
        while n > 0:
            digits.append(self._FONT[n % 10])
            n //= 10
        digits.reverse()

        out = np.kron(digits[0], np.ones((scale, scale), dtype=bool))
        gap = np.zeros((5 * scale, scale // 2), dtype=bool)
        for d in digits[1:]:
            out = np.hstack([out, gap, np.kron(d, np.ones((scale, scale), dtype=bool))])
        return out

    def _render_rgb_array(self):
        n = self.n
        cell = 80
        size = n * cell
        img = np.full((size, size, 3), (40, 40, 40), dtype=np.uint8)

        TILE = np.array([70, 130, 180], dtype=np.uint8)
        BORDER = np.array([160, 160, 160], dtype=np.uint8)
        TEXT = np.array([255, 255, 255], dtype=np.uint8)

        for r in range(n):
            for c in range(n):
                val = self._grid[r, c]
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell

                if val != 0:
                    img[y0:y1, x0:x1] = TILE
                    img[y0, x0:x1] = BORDER
                    img[y1 - 1, x0:x1] = BORDER
                    img[y0:y1, x0] = BORDER
                    img[y0:y1, x1 - 1] = BORDER

                    bitmap = self._render_number(val)
                    bh, bw = bitmap.shape
                    dy = (cell - bh) // 2
                    dx = (cell - bw) // 2
                    region = img[y0 + dy:y0 + dy + bh, x0 + dx:x0 + dx + bw]
                    region[bitmap] = TEXT

        return img


def make_huarongdao_env(render_mode=None):
    return DigitalHuarongdaoEnv(render_mode=render_mode)


if __name__ == "__main__":
    print(f"=== DigitalHuarongdaoEnv {HUARONGDAO_N}x{HUARONGDAO_N} self-test ===")

    # 随机动作测试（asi、rgb_array）
    for mode in [None, "ansi", "rgb_array"]:
        label = mode if mode else "no_render"
        env = make_huarongdao_env(render_mode=mode)
        for ep in range(3):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    print(f"  [{label}] ep={ep+1}: steps={steps}, reward={total_reward:.0f}, "
                          f"terminated={terminated}, truncated={truncated}")
                    break
        env.close()

    # Gymnasium 合规性检查
    print("\n=== gymnasium env_checker ===")
    env = make_huarongdao_env()
    from gymnasium.utils.env_checker import check_env
    check_env(env)
    env.close()
    print("env_checker passed")
    print("=== self-test passed ===")
