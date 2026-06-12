import heapq
import numpy as np
import torch
from config import (
    HUARONGDAO_N, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX,
    DEEPCUBEA_LAMBDA, DEEPCUBEA_MODEL_PATH, DEEPCUBEA_MAX_EXPAND_NODES,
    DEEPCUBEA_NUM_TEST_STATES, DEEPCUBEA_INFERENCE_USE_GPU,
    DEEPCUBEA_VAL_SIZE, DEEPCUBEA_VAL_SEED, DEEPCUBEA_VAL_NUM_STRATA,
    DEEPCUBEA_VAL_GREEDY_EXPAND, DEEPCUBEA_VAL_ASTAR_FLAG, DEEPCUBEA_VAL_GREEDY_FLAG,
)
from deepcubea_network import DeepCubeANetwork, transition, get_children, encode_batch
from deepcubea_utils import generate_stratified_states

N2 = HUARONGDAO_N * HUARONGDAO_N
_GOAL_GRID = np.array(list(range(1, N2)) + [0], dtype=np.int32)
_GOAL_BYTES = _GOAL_GRID.tobytes()

_DR = np.array([-1, 1, 0, 0])
_DC = np.array([0, 0, -1, 1])


def load_model(model_path=None, use_gpu=None):
    path = model_path or DEEPCUBEA_MODEL_PATH
    if not path:
        raise ValueError("DEEPCUBEA_MODEL_PATH is empty, must specify model path")
    if use_gpu is None:
        use_gpu = DEEPCUBEA_INFERENCE_USE_GPU
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = DeepCubeANetwork().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def weighted_astar(start_grid_flat, model, lambda_weight=None, max_expand=None):
    """Weighted A* search: f(s) = λ·g(s) + h(s).

    Returns (solution_path, expanded_nodes) or (None, expanded_nodes) if failed.
    solution_path is list of actions (0=up,1=down,2=left,3=right) from start to goal.
    """
    if lambda_weight is None:
        lambda_weight = DEEPCUBEA_LAMBDA
    if max_expand is None:
        max_expand = DEEPCUBEA_MAX_EXPAND_NODES

    start_bytes = start_grid_flat.tobytes()

    if start_bytes == _GOAL_BYTES:
        return [], 0

    start_blank_idx = int(np.where(start_grid_flat == 0)[0][0])

    h_start = model.predict_j(start_grid_flat)
    open_set = [(lambda_weight * 0 + h_start, 0, start_bytes)]
    tiebreaker = 1

    # g_score: state_bytes → (g, parent_bytes, action, blank_idx)
    g_score = {start_bytes: (0, None, None, start_blank_idx)}
    closed = set()
    expanded = 0

    while open_set and expanded < max_expand:
        f_val, _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        g_current, _, _, blank_idx = g_score[current]
        current_grid = np.frombuffer(current, dtype=np.int32).copy()

        raw_children = get_children(current_grid, blank_idx)

        to_predict = []
        for child_grid, action, new_blank_idx in raw_children:
            child_bytes = child_grid.tobytes()

            if child_bytes in closed:
                continue

            g_child = g_current + 1

            if child_bytes in g_score and g_child >= g_score[child_bytes][0]:
                continue

            if child_bytes == _GOAL_BYTES:
                path = [action]
                state = current
                while g_score[state][1] is not None:
                    _, parent, a, _ = g_score[state]
                    path.append(a)
                    state = parent
                path.reverse()
                return path, expanded

            g_score[child_bytes] = (g_child, current, action, new_blank_idx)
            to_predict.append((child_bytes, child_grid, action))

        if not to_predict:
            continue

        child_grids = np.stack([c[1] for c in to_predict])
        h_values = model.predict_j_batch(child_grids).cpu().numpy()

        for (child_bytes, _, _), h_child in zip(to_predict, h_values):
            g_child = g_score[child_bytes][0]
            f_child = lambda_weight * g_child + float(h_child)
            heapq.heappush(open_set, (f_child, tiebreaker, child_bytes))
            tiebreaker += 1

    return None, expanded


def _generate_test_states(num_states, min_steps, max_steps, rng):
    """Generate solvable test states by random walk from goal."""
    states = []
    for _ in range(num_states):
        t = rng.integers(min_steps, max_steps + 1)
        grid = _GOAL_GRID.copy()
        for _ in range(t):
            action = rng.integers(0, 4)
            child = transition(grid, action)
            if child is not None:
                grid = child
        states.append(grid)
    return states


def _compute_bellman_errors(states, model):
    """Compute per-state squared Bellman error: (J(s) - min_a(1+J(A(s,a))))^2."""
    B = len(states)
    N = HUARONGDAO_N
    blank_pos = np.argmin(states, axis=1)
    blank_r = blank_pos // N
    blank_c = blank_pos % N

    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        x = encode_batch(states).to(device)
        j_s = model(x).cpu().numpy()

        best_j = np.full(B, np.inf, dtype=np.float32)

        for a in range(4):
            nr = blank_r + _DR[a]
            nc = blank_c + _DC[a]
            legal = (0 <= nr) & (nr < N) & (0 <= nc) & (nc < N)
            if not legal.any():
                continue

            new_idx = nr * N + nc
            legal_idx = np.where(legal)[0]

            next_states = states.copy()
            next_states[legal_idx, blank_pos[legal_idx]] = states[legal_idx, new_idx[legal_idx]]
            next_states[legal_idx, new_idx[legal_idx]] = 0

            x_next = encode_batch(next_states).to(device)
            j_vals = model(x_next).cpu().numpy()
            best_j[legal] = np.minimum(best_j[legal], j_vals[legal] + 1.0)

    goal_mask = np.all(states == _GOAL_GRID, axis=1)
    best_j[goal_mask] = 0.0
    j_s[goal_mask] = 0.0
    best_j[np.isinf(best_j)] = 0.0

    return (j_s - best_j) ** 2


def stratified_bellman_mse(model, val_size=None, t_min=None, t_max=None,
                           num_strata=None, val_seed=None):
    """方案 B: Stratified Bellman MSE — always runs.

    Generates validation states stratified by scramble distance K,
    computes per-state Bellman MSE, and reports per-stratum statistics.
    K is a proxy for true optimal distance (not exact).
    """
    if val_size is None:
        val_size = DEEPCUBEA_VAL_SIZE
    if t_min is None:
        t_min = DEEPCUBEA_T_MIN
    if t_max is None:
        t_max = DEEPCUBEA_T_MAX
    if num_strata is None:
        num_strata = DEEPCUBEA_VAL_NUM_STRATA
    if val_seed is None:
        val_seed = DEEPCUBEA_VAL_SEED

    states, k_values, strata_bounds = generate_stratified_states(
        val_size, t_min, t_max, num_strata, seed=val_seed,
        start_state=_GOAL_GRID,
    )

    errors = _compute_bellman_errors(states, model)

    print("=" * 72)
    print("方案 B — Stratified Bellman MSE (K proxy)")
    print("=" * 72)
    print(f"States: {len(states)}  Strata: {num_strata}  K range: [{t_min}, {t_max}]")
    print(f"Note: K = scramble steps, NOT true optimal distance")
    print()
    header = f"{'Stratum':<10} {'K Range':<16} {'Count':<8} {'Mean MSE':<12} {'Median MSE':<12} {'Std MSE':<12}"
    print(header)
    print("-" * len(header))

    for i, (lo, hi) in enumerate(strata_bounds):
        mask = (k_values >= lo) & (k_values <= hi)
        stratum_errors = errors[mask]
        print(f"{i+1:<10} [{lo}, {hi}]{'':<8} {len(stratum_errors):<8} "
              f"{np.mean(stratum_errors):<12.6f} {np.median(stratum_errors):<12.6f} "
              f"{np.std(stratum_errors):<12.6f}")

    print("-" * len(header))
    print(f"{'Overall':<10} {'':<16} {len(errors):<8} "
          f"{np.mean(errors):<12.6f} {np.median(errors):<12.6f} "
          f"{np.std(errors):<12.6f}")
    print("=" * 72)


def greedy_expansion_eval(model, val_size=None, t_min=None, t_max=None,
                          num_strata=None, val_seed=None, max_expand=None,
                          lambda_weight=None):
    """方案 C: Greedy expansion evaluation (flag-controlled).

    Runs truncated weighted A* (max_expand limit) on stratified
    validation states, reporting per-stratum solve rate and
    average expanded nodes.
    """
    if val_size is None:
        val_size = DEEPCUBEA_VAL_SIZE
    if t_min is None:
        t_min = DEEPCUBEA_T_MIN
    if t_max is None:
        t_max = DEEPCUBEA_T_MAX
    if num_strata is None:
        num_strata = DEEPCUBEA_VAL_NUM_STRATA
    if val_seed is None:
        val_seed = DEEPCUBEA_VAL_SEED
    if max_expand is None:
        max_expand = DEEPCUBEA_VAL_GREEDY_EXPAND
    if lambda_weight is None:
        lambda_weight = DEEPCUBEA_LAMBDA

    states, k_values, strata_bounds = generate_stratified_states(
        val_size, t_min, t_max, num_strata, seed=val_seed,
        start_state=_GOAL_GRID,
    )

    print("=" * 78)
    print("方案 C — Greedy Expansion Evaluation (truncated A*)")
    print("=" * 78)
    print(f"States: {len(states)}  Max Expand: {max_expand}  Lambda: {lambda_weight}")
    print()
    header = f"{'Stratum':<10} {'K Range':<16} {'Count':<8} {'Solved':<10} {'Rate %':<10} {'Avg Expand':<12} {'Avg Len':<10}"
    print(header)
    print("-" * len(header))

    total_solved = 0
    total_expanded = 0
    total_length = 0

    for i, (lo, hi) in enumerate(strata_bounds):
        mask = (k_values >= lo) & (k_values <= hi)
        stratum_states = states[mask]
        stratum_ks = k_values[mask]

        solved = 0
        expanded = 0
        length = 0

        for grid in stratum_states:
            path, exp = weighted_astar(grid, model, lambda_weight, max_expand)
            expanded += exp
            if path is not None:
                solved += 1
                length += len(path)

        total_solved += solved
        total_expanded += expanded
        total_length += length

        n = len(stratum_states)
        rate = solved / n * 100 if n > 0 else 0.0
        avg_exp = expanded / n if n > 0 else 0.0
        avg_len = length / solved if solved > 0 else float("nan")

        print(f"{i+1:<10} [{lo}, {hi}]{'':<8} {n:<8} {solved:<10} {rate:<10.1f} "
              f"{avg_exp:<12.1f} {avg_len:<10.1f}")

    print("-" * len(header))
    n_total = len(states)
    overall_rate = total_solved / n_total * 100 if n_total > 0 else 0.0
    overall_exp = total_expanded / n_total if n_total > 0 else 0.0
    print(f"{'Overall':<10} {'':<16} {n_total:<8} {total_solved:<10} {overall_rate:<10.1f} "
          f"{overall_exp:<12.1f} {'':<10}")
    print("=" * 78)


def evaluate(model, model_path=None, num_states=None, lambda_weight=None, max_expand=None):
    """Batch evaluation across short/medium/long difficulty tiers.

    Tiers are computed by splitting [T_MIN, T_MAX] into 3 equal ranges.
    """
    if num_states is None:
        num_states = DEEPCUBEA_NUM_TEST_STATES
    if lambda_weight is None:
        lambda_weight = DEEPCUBEA_LAMBDA
    if max_expand is None:
        max_expand = DEEPCUBEA_MAX_EXPAND_NODES

    tier_size = num_states // 3
    remainder = num_states % 3
    sizes = [tier_size] * 3
    for i in range(remainder):
        sizes[i] += 1

    step_range = DEEPCUBEA_T_MAX - DEEPCUBEA_T_MIN
    step_delta = step_range / 3.0

    tiers = [
        ("Short",  DEEPCUBEA_T_MIN,                int(DEEPCUBEA_T_MIN + step_delta)),
        ("Medium", int(DEEPCUBEA_T_MIN + step_delta) + 1, int(DEEPCUBEA_T_MIN + 2 * step_delta)),
        ("Long",   int(DEEPCUBEA_T_MIN + 2 * step_delta) + 1, DEEPCUBEA_T_MAX),
    ]

    rng = np.random.default_rng(42)

    model_path_str = model_path or DEEPCUBEA_MODEL_PATH

    print("=" * 60)
    print("DeepCubeA Weighted A* Evaluation")
    print("=" * 60)
    print(f"Model:  {model_path_str}")
    print(f"Lambda: {lambda_weight}")
    print(f"Max Expand: {max_expand:,}")
    print()

    total_solved = 0
    total_tested = 0
    total_expanded = 0

    for idx, (tier_name, t_min, t_max) in enumerate(tiers):
        test_states = _generate_test_states(sizes[idx], t_min, t_max, rng)
        print(f"[{tier_name}] {len(test_states)} states, step range [{t_min}, {t_max}]", flush=True)

        tier_solved = 0
        tier_expanded = 0
        tier_length = 0

        for i, grid in enumerate(test_states):
            path, expanded = weighted_astar(grid, model, lambda_weight, max_expand)
            if path is not None:
                tier_solved += 1
                tier_length += len(path)
            tier_expanded += expanded
            if (i + 1) % 10 == 0 or i == len(test_states) - 1:
                print(f"  {i + 1}/{len(test_states)} done", flush=True)

        total_solved += tier_solved
        total_tested += len(test_states)
        total_expanded += tier_expanded

        rate = tier_solved / len(test_states) * 100
        avg_len = tier_length / tier_solved if tier_solved > 0 else float("nan")
        avg_expand = tier_expanded / len(test_states)

        print(f"--- {tier_name} ({t_min}-{t_max} steps, {len(test_states)} states) ---")
        print(f"Solved:               {tier_solved}/{len(test_states)} ({rate:.2f}%)")
        print(f"Avg Solution Length:  {avg_len:.1f}")
        print(f"Avg Expanded Nodes:   {avg_expand:,.1f}")
        print()

    overall_rate = total_solved / total_tested * 100
    overall_expand = total_expanded / total_tested
    print(f"--- Overall ---")
    print(f"Solved:             {total_solved}/{total_tested} ({overall_rate:.2f}%)")
    print(f"Avg Expanded Nodes: {overall_expand:,.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import time

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEEPCUBEA_MODEL_PATH
    if not model_path:
        print("ERROR: DEEPCUBEA_MODEL_PATH is empty.")
        print("Set it in config.py or pass as argument:")
        print("  uv run python deepcubea_search.py <model_path>")
        sys.exit(1)

    t0 = time.perf_counter()
    model = load_model(model_path)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s\n")

    # 方案 B — Always: Stratified Bellman MSE
    stratified_bellman_mse(model)

    # 方案 C — Optional: Greedy expansion
    if DEEPCUBEA_VAL_GREEDY_FLAG:
        print()
        greedy_expansion_eval(model)

    # 方案 D — Optional: Full A* three-tier evaluation
    if DEEPCUBEA_VAL_ASTAR_FLAG:
        print()
        evaluate(model, model_path=model_path)
