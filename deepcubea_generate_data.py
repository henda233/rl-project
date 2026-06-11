"""DeepCubeA training data generation.

Generates training states via reverse random walks from the goal state,
deduplicates, and saves as .npy for the training script.
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import (
    DEEPCUBEA_T_MIN,
    DEEPCUBEA_T_MAX,
    DEEPCUBEA_TRAIN_SET_SIZE,
    DEEPCUBEA_TRAIN_DATA_PATH,
    HUARONGDAO_N,
)
from deepcubea_network import N, N2, transition

GOAL_STATE = np.arange(N2, dtype=np.int32)

_DR = np.array([-1, 1, 0, 0])
_DC = np.array([0, 0, -1, 1])


def _legal_actions(grid_flat):
    idx = int(np.where(grid_flat == 0)[0][0])
    r, c = idx // N, idx % N
    actions = []
    for a in range(4):
        nr, nc = r + _DR[a], c + _DC[a]
        if 0 <= nr < N and 0 <= nc < N:
            actions.append(a)
    return actions


def main():
    output_path = Path(DEEPCUBEA_TRAIN_DATA_PATH)

    if output_path.exists():
        print(f"Data file already exists: {output_path}")
        print("Delete it manually to regenerate.")
        return

    print(f"Generating {DEEPCUBEA_TRAIN_SET_SIZE} raw states "
          f"(T ~ U({DEEPCUBEA_T_MIN}, {DEEPCUBEA_T_MAX}))...")

    rng = np.random.default_rng(42)
    raw_states = [GOAL_STATE]

    for _ in tqdm(range(DEEPCUBEA_TRAIN_SET_SIZE - 1), desc="Generating"):
        t = rng.integers(DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX + 1)
        current = GOAL_STATE.copy()
        for _step in range(t):
            actions = _legal_actions(current)
            a = actions[rng.integers(len(actions))]
            current = transition(current, a)
        raw_states.append(current.copy())

    raw = np.array(raw_states, dtype=np.int32)
    unique = np.unique(raw, axis=0)

    print(f"Unique states: {len(unique)} (from {len(raw)} raw)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, unique)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
