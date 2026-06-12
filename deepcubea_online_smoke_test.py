"""Smoke test for online sampling + validation metrics.

Small-scale end-to-end test:
  1. Generate a tiny base dataset
  2. Run online mixed sampling training
  3. Run stratified Bellman MSE (方案 B, always)
  4. Run greedy expansion (方案 C)
  5. Run full A* three-tier (方案 D)

Usage:
  uv run python deepcubea_online_smoke_test.py
"""

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch

import config
from deepcubea_network import N2, DeepCubeANetwork
from deepcubea_utils import generate_scrambled_states


# ═══════════════════════════════════════════
# Smoke-scale config overrides
# ═══════════════════════════════════════════

config.DEEPCUBEA_T_MIN = 10
config.DEEPCUBEA_T_MAX = 30

# Training (tiny)
config.DEEPCUBEA_TRAIN_SET_SIZE = 500
config.DEEPCUBEA_LR = 1e-3
config.DEEPCUBEA_BATCH_SIZE = 64
config.DEEPCUBEA_OUTER_ITER = 3
config.DEEPCUBEA_INNER_EPOCHS = 10
config.DEEPCUBEA_INNER_PATIENCE = 3
config.DEEPCUBEA_HIDDEN_DIM = 128
config.DEEPCUBEA_USE_GPU = False
config.DEEPCUBEA_ONLINE_BATCH = 100
config.DEEPCUBEA_BASE_BATCH = 50
config.DEEPCUBEA_OUTER_SEED = 42
config.DEEPCUBEA_SEED_OVERLAP = 0.3

# Validation
config.DEEPCUBEA_VAL_SIZE = 30
config.DEEPCUBEA_VAL_SEED = 999
config.DEEPCUBEA_VAL_NUM_STRATA = 3
config.DEEPCUBEA_VAL_GREEDY_EXPAND = 100
config.DEEPCUBEA_LAMBDA = 0.8

# Full A*
config.DEEPCUBEA_NUM_TEST_STATES = 30
config.DEEPCUBEA_MAX_EXPAND_NODES = 500

# Inference
config.DEEPCUBEA_INFERENCE_USE_GPU = False


def main():
    print("=" * 60)
    print("DeepCubeA Online Sampling + Validation Smoke Test")
    print("=" * 60)
    print(f"Train: {config.DEEPCUBEA_OUTER_ITER} outer × "
          f"max {config.DEEPCUBEA_INNER_EPOCHS} inner, "
          f"B={config.DEEPCUBEA_ONLINE_BATCH}, B'={config.DEEPCUBEA_BASE_BATCH}")
    print(f"Val:   {config.DEEPCUBEA_VAL_SIZE} states, "
          f"{config.DEEPCUBEA_VAL_NUM_STRATA} strata")
    print()

    # ── Step 1: Generate small base dataset ──
    print("[1/4] Generating base dataset...")
    GOAL_STATE = np.arange(N2, dtype=np.int32)
    base_states = generate_scrambled_states(
        config.DEEPCUBEA_TRAIN_SET_SIZE - 1,
        config.DEEPCUBEA_T_MIN, config.DEEPCUBEA_T_MAX, seed=42,
    )
    base_states = np.concatenate([[GOAL_STATE], base_states])
    base_states = np.unique(base_states, axis=0)
    print(f"  Base dataset: {len(base_states)} unique states")

    # Write to temp path and point config at it
    tmp_dir = tempfile.mkdtemp(prefix="deepcubea_smoke_")
    tmp_data = Path(tmp_dir) / "train_states.npy"
    np.save(tmp_data, base_states)
    original_path = config.DEEPCUBEA_TRAIN_DATA_PATH
    config.DEEPCUBEA_TRAIN_DATA_PATH = str(tmp_data)

    try:
        # ── Step 2: Run training ──
        print("\n[2/4] Training with online mixed sampling...")
        t0 = time.perf_counter()
        from deepcubea_train import train
        network, losses = train()
        train_time = time.perf_counter() - t0
        print(f"  Training finished in {train_time:.1f}s, "
              f"{len(losses)} epochs, final loss={losses[-1]:.6f}")

        # Save model to temp path
        model_path = Path(tmp_dir) / "smoke_model.pt"
        torch.save(network.state_dict(), model_path)
        config.DEEPCUBEA_MODEL_PATH = str(model_path)

        # ── Step 3: Load model for validation ──
        print("\n[3/4] Loading model for validation...")
        from deepcubea_search import (
            load_model, stratified_bellman_mse,
            greedy_expansion_eval, evaluate,
        )
        model = load_model(str(model_path))

        # ── Step 4a: 方案 B — Stratified Bellman MSE ──
        print("\n[4a/4] 方案 B: Stratified Bellman MSE")
        stratified_bellman_mse(model)

        # ── Step 4b: 方案 C — Greedy expansion ──
        print("\n[4b/4] 方案 C: Greedy Expansion")
        greedy_expansion_eval(
            model,
            max_expand=config.DEEPCUBEA_VAL_GREEDY_EXPAND,
        )

        # ── Step 4c: 方案 D — Full A* three-tier ──
        print("\n[4c/4] 方案 D: Full A* Three-tier Evaluation")
        evaluate(
            model, model_path=str(model_path),
            max_expand=config.DEEPCUBEA_MAX_EXPAND_NODES,
        )

    finally:
        # Restore original config path
        config.DEEPCUBEA_TRAIN_DATA_PATH = original_path

        # Cleanup temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Smoke test completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
