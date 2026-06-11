"""Quick end-to-end test for deepcubea_train.py pipeline.

Run: uv run python test_deepcubea_training.py
"""

import numpy as np
import torch
from config import (
    DEEPCUBEA_T_MIN,
    DEEPCUBEA_T_MAX,
    DEEPCUBEA_TRAIN_SET_SIZE,
    DEEPCUBEA_LR,
    DEEPCUBEA_BATCH_SIZE,
    HUARONGDAO_N,
)
from deepcubea_network import DeepCubeANetwork, N2, encode_batch, transition
from deepcubea_train import generate_training_data, compute_targets, GOAL_STATE


def test_data_generation():
    print("=" * 50)
    print("TEST 1: generate_training_data")
    print("=" * 50)

    states = generate_training_data(20, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX, seed=0)
    assert states.shape == (20, N2), f"Shape mismatch: {states.shape}"
    assert states.dtype == np.int32, f"Dtype mismatch: {states.dtype}"
    assert (states >= 0).all() and (states < N2).all(), "Values out of range"

    # Goal state should be first
    assert np.array_equal(states[0], GOAL_STATE), "First state not goal state"
    print(f"  Shape: {states.shape}, dtype: {states.dtype}")
    print(f"  Goal state at index 0: OK")

    # Count unique states
    unique = np.unique(states, axis=0)
    print(f"  Unique states: {len(unique)} / {len(states)}")
    print("  PASS\n")
    return True


def test_compute_targets():
    print("=" * 50)
    print("TEST 2: compute_targets")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    states = generate_training_data(50, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX, seed=1)
    network = DeepCubeANetwork().to(device)

    targets = compute_targets(states, network, device)
    assert targets.shape == (50,), f"Targets shape: {targets.shape}"

    # Goal state target should be 0
    assert targets[0].item() == 0.0, f"Goal target != 0: {targets[0].item()}"

    # All targets should be >= 0 and finite
    assert (targets >= 0).all(), "Negative targets found"
    assert torch.isfinite(targets).all(), "Non-finite targets"

    print(f"  Targets shape: {targets.shape}")
    print(f"  Goal state target: {targets[0].item():.1f} (expected 0)")
    print(f"  Target range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
    print("  PASS\n")
    return True


def test_transition_consistency():
    print("=" * 50)
    print("TEST 3: transition function consistency")
    print("=" * 50)

    # All states from data generation should be reachable
    states = generate_training_data(100, 1, 30, seed=2)
    for s in states[:10]:  # Check first 10
        blank_pos = np.where(s == 0)[0][0]
        r, c = blank_pos // HUARONGDAO_N, blank_pos % HUARONGDAO_N
        legal_count = 0
        for a in range(4):
            result = transition(s, a)
            if result is not None:
                legal_count += 1
                # Verify transition is valid: single swap
                diff = (s != result).sum()
                assert diff == 2, f"Transition changed {diff} tiles, expected 2"
        assert 2 <= legal_count <= 4, f"Unexpected legal count: {legal_count}"

    print(f"  Checked 10 states, transitions OK")
    print("  PASS\n")
    return True


def test_mini_training():
    print("=" * 50)
    print("TEST 4: mini training (5 epochs)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    states = generate_training_data(200, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX, seed=3)
    states = np.unique(states, axis=0)
    print(f"  Training states: {len(states)}")

    states_onehot = encode_batch(states).to(device)
    network = DeepCubeANetwork().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=DEEPCUBEA_LR)

    param_count = sum(p.numel() for p in network.parameters())
    print(f"  Parameters: {param_count:,}")

    initial_j_goal = network.predict_j(states[0])
    print(f"  J(goal) before training: {initial_j_goal:.4f}")

    losses = []
    for epoch in range(1, 6):
        targets = compute_targets(states, network, device).to(device)
        network.train()
        perm = torch.randperm(len(states), device=device)

        total_loss = 0.0
        for i in range(0, len(states), DEEPCUBEA_BATCH_SIZE):
            idx = perm[i : i + DEEPCUBEA_BATCH_SIZE]
            batch_x = states_onehot[idx]
            batch_y = targets[idx]

            pred = network(batch_x)
            loss = torch.nn.functional.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(states) // DEEPCUBEA_BATCH_SIZE)
        losses.append(avg_loss)

        j_goal = network.predict_j(states[0])
        print(
            f"  Epoch {epoch}: loss={avg_loss:.6f}, J(goal)={j_goal:.4f} "
            f"(target=0)"
        )

    # Loss should decrease
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )
    # J(goal) should move toward 0
    final_j_goal = network.predict_j(states[0])
    assert abs(final_j_goal) < abs(initial_j_goal) or abs(final_j_goal) < 10, (
        f"J(goal) not converging to 0: {initial_j_goal:.4f} -> {final_j_goal:.4f}"
    )

    print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f} (decreasing: OK)")
    print("  PASS\n")
    return True


if __name__ == "__main__":
    all_passed = True
    for test in [
        test_data_generation,
        test_compute_targets,
        test_transition_consistency,
        test_mini_training,
    ]:
        try:
            test()
        except Exception as e:
            print(f"  FAIL: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("=" * 50)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 50)
