"""Smoke test: verify weighted A* finds correct solutions on near-goal states.

Usage:
    uv run python deepcubea_search_smoke_test.py <model_path>
"""
import sys
import numpy as np
from deepcubea_search import load_model, weighted_astar, _GOAL_GRID, GOAL_STATE
from deepcubea_network import transition


def _random_walk_from_goal(steps, rng):
    """Generate a solvable start state by walking `steps` actions from goal."""
    grid = _GOAL_GRID.copy()
    for _ in range(steps):
        action = rng.integers(0, 4)
        child = transition(grid, action)
        if child is not None:
            grid = child
    return grid


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_path:
        print("Usage: uv run python deepcubea_search_smoke_test.py <model_path>")
        sys.exit(1)

    model = load_model(model_path)
    rng = np.random.default_rng(123)

    test_cases = [
        (3, "3-step walk"),
        (5, "5-step walk"),
        (7, "7-step walk"),
        (10, "10-step walk"),
    ]

    all_ok = True
    for steps, desc in test_cases:
        start_grid = _random_walk_from_goal(steps, rng)
        # Use λ=1.0 (standard A*) for correctness testing
        path, expanded = weighted_astar(start_grid, model, lambda_weight=1.0)

        if path is None:
            print(f"FAIL [{desc}]: search returned None (expanded={expanded})")
            all_ok = False
            continue

        # Verify: applying path from start reaches goal
        grid = start_grid.copy()
        for a in path:
            grid = transition(grid, a)
            if grid is None:
                print(f"FAIL [{desc}]: illegal action {a} in solution path")
                all_ok = False
                break

        if grid is not None:
            at_goal = tuple(int(x) for x in grid) == GOAL_STATE
            status = "OK" if at_goal else "FAIL (path does not reach goal)"
            print(f"{status} [{desc}]: length={len(path)}, expanded={expanded}")
            if not at_goal:
                all_ok = False

    if all_ok:
        print("\nAll smoke tests passed.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)
