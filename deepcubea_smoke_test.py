"""Smoke test for batch-update + fixed-target AVI training.

Validates:
  1. Early stopping triggers correctly (patience exhausted).
  2. Outer checkpoints saved.
  3. Final model saved.
  4. Saw-tooth loss curve generated.

Uses small config: 100 states, 3 outer × 10 inner, patience=3.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# ── Override config for smoke test ──
import config

config.DEEPCUBEA_T_MIN = 5
config.DEEPCUBEA_T_MAX = 15
config.DEEPCUBEA_TRAIN_SET_SIZE = 100
config.DEEPCUBEA_OUTER_ITER = 3
config.DEEPCUBEA_INNER_EPOCHS = 10
config.DEEPCUBEA_INNER_PATIENCE = 3
config.DEEPCUBEA_LR = 1e-3
config.DEEPCUBEA_BATCH_SIZE = 32
config.DEEPCUBEA_HIDDEN_DIM = 64
config.DEEPCUBEA_USE_GPU = False

# ── Generate tiny dataset ──
from deepcubea_network import N, N2, transition

GOAL_STATE = np.arange(N2, dtype=np.int32)

def _legal_actions(grid_flat):
    idx = int(np.where(grid_flat == 0)[0][0])
    r, c = idx // N, idx % N
    _DR = np.array([-1, 1, 0, 0])
    _DC = np.array([0, 0, -1, 1])
    actions = []
    for a in range(4):
        nr, nc = r + _DR[a], c + _DC[a]
        if 0 <= nr < N and 0 <= nc < N:
            actions.append(a)
    return actions

print("Generating 100 states...")
states_set = set()
states_set.add(tuple(GOAL_STATE))
pbar = None
try:
    from tqdm import tqdm
    pbar = tqdm(total=config.DEEPCUBEA_TRAIN_SET_SIZE, desc="Generating")
except ImportError:
    pass

rng = np.random.default_rng(42)
while len(states_set) < config.DEEPCUBEA_TRAIN_SET_SIZE:
    t = rng.integers(config.DEEPCUBEA_T_MIN, config.DEEPCUBEA_T_MAX + 1)
    current = GOAL_STATE.copy()
    for _step in range(t):
        actions = _legal_actions(current)
        a = actions[rng.integers(len(actions))]
        current = transition(current, a)
    if not np.array_equal(current, GOAL_STATE):
        states_set.add(tuple(current))
    if pbar:
        pbar.set_postfix(states=len(states_set))

if pbar:
    pbar.close()

states = np.array(list(states_set), dtype=np.int32)
data_dir = Path("results/train_data")
data_dir.mkdir(parents=True, exist_ok=True)
np.save(data_dir / "train_states.npy", states)
print(f"Saved {len(states)} states to results/train_data/train_states.npy")

# ── Run training ──
from deepcubea_train import train

print("\n=== Smoke test training ===")
network, losses = train()

# ── Verify results ──
print("\n=== Verification ===")

total_epochs = len(losses)
print(f"Total epochs: {total_epochs}")
max_epochs = config.DEEPCUBEA_OUTER_ITER * config.DEEPCUBEA_INNER_EPOCHS
print(f"Max possible: {max_epochs}")

# Check early stopping reduced epoch count
if total_epochs < max_epochs:
    print(f"[PASS] Early stopping triggered (saved {max_epochs - total_epochs} epochs)")
else:
    print(f"[WARN] Early stopping never triggered")

# Check checkpoints
import glob
checkpoints = sorted(glob.glob("results/train_*/models/deepcubea_heuristic_epoch_*.pt"))
print(f"Checkpoints: {len(checkpoints)}")
for cp in checkpoints:
    print(f"  {cp}")
if len(checkpoints) == config.DEEPCUBEA_OUTER_ITER:
    print(f"[PASS] All {config.DEEPCUBEA_OUTER_ITER} outer checkpoints saved")
else:
    print(f"[FAIL] Expected {config.DEEPCUBEA_OUTER_ITER} checkpoints, got {len(checkpoints)}")

# Check final model
final_models = sorted(glob.glob("results/train_*/models/deepcubea_heuristic_final.pt"))
if len(final_models) >= 1:
    print(f"[PASS] Final model saved: {final_models[-1]}")
else:
    print(f"[FAIL] No final model found")

# Check loss curve
loss_imgs = sorted(glob.glob("results/train_*/imgs/loss_curve.png"))
if len(loss_imgs) >= 1:
    print(f"[PASS] Loss curve saved: {loss_imgs[-1]}")
else:
    print(f"[FAIL] No loss curve found")

# Quick model sanity check
goal_j = network.predict_j(GOAL_STATE)
print(f"J(goal) = {goal_j:.6f} (should be near 0)")

print("\n=== Smoke test done ===")
