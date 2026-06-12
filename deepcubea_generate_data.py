"""DeepCubeA training data generation.

Generates training states via random walks from the goal state,
deduplicates, and saves as .npy for the training script.
"""

import numpy as np
from pathlib import Path
from config import (
    DEEPCUBEA_T_MIN,
    DEEPCUBEA_T_MAX,
    DEEPCUBEA_TRAIN_SET_SIZE,
    DEEPCUBEA_TRAIN_DATA_PATH,
)
from deepcubea_network import N2
from deepcubea_utils import generate_scrambled_states

GOAL_STATE = np.arange(N2, dtype=np.int32)


def main():
    output_path = Path(DEEPCUBEA_TRAIN_DATA_PATH)

    if output_path.exists():
        print(f"Data file already exists: {output_path}")
        print("Delete it manually to regenerate.")
        return

    print(f"Generating {DEEPCUBEA_TRAIN_SET_SIZE} raw states "
          f"(T ~ U({DEEPCUBEA_T_MIN}, {DEEPCUBEA_T_MAX}))...")

    states = generate_scrambled_states(
        DEEPCUBEA_TRAIN_SET_SIZE - 1, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX, seed=42,
    )
    raw = np.concatenate([[GOAL_STATE], states])
    unique = np.unique(raw, axis=0)

    print(f"Unique states: {len(unique)} (from {len(raw)} raw)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, unique)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
