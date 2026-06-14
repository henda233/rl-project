import heapq
import numpy as np
import torch
from config import (
    HUARONGDAO_N,
    DEEPCUBEA_OFFICIAL_MODEL_PATH, DEEPCUBEA_OFFICIAL_DATA_DIR,
    DEEPCUBEA_OFFICIAL_LAMBDA, DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES,
    DEEPCUBEA_OFFICIAL_NUM_STRATA, DEEPCUBEA_OFFICIAL_GREEDY_MAX_STEPS,
    DEEPCUBEA_OFFICIAL_GREEDY_FLAG, DEEPCUBEA_OFFICIAL_ASTAR_FLAG,
    DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA,
    DEEPCUBEA_OFFICIAL_T_MIN, DEEPCUBEA_OFFICIAL_T_MAX,
    DEEPCUBEA_OFFICIAL_NUM_TEST_STATES,
    DEEPCUBEA_INFERENCE_USE_GPU, DEEPCUBEA_VAL_SEED,
)
from deepcubea_network import transition, get_children
from deepcubea_official_network import load_official_model
from deepcubea_official_data import load_official_test_data

N2 = HUARONGDAO_N * HUARONGDAO_N
N = HUARONGDAO_N
_GOAL_GRID = np.array(list(range(1, N2)) + [0], dtype=np.int32)
_GOAL_BYTES = _GOAL_GRID.tobytes()


def _generate_test_states(num_states, min_steps, max_steps, rng):
    """Generate solvable test states by random walk from goal."""
    states = []
    k_values = np.empty(num_states, dtype=np.int32)
    for i in range(num_states):
        t = int(rng.integers(min_steps, max_steps + 1))
        k_values[i] = t
        grid = _GOAL_GRID.copy()
        for _ in range(t):
            action = int(rng.integers(0, 4))
            child = transition(grid, action)
            if child is not None:
                grid = child
        states.append(grid)
    return np.array(states, dtype=np.int64), k_values


def _compute_bellman_errors_official(states, model):
    """Compute per-state squared Bellman error: (J(s) - min_a(1+J(s')))^2.

    Reference: deepcubea_search.py:_compute_bellman_errors (line 129-168).
    Key difference: no encode_batch — raw tiles are passed directly to model.
    """
    B = len(states)
    blank_pos = np.argmin(states, axis=1)
    blank_r = blank_pos // N
    blank_c = blank_pos % N

    # J(s) for all states in one batch
    j_s = model.predict_j_batch(states).cpu()

    # Collect all legal next states in one array
    all_next_states = []
    offsets = [0]
    for i in range(B):
        for action in range(4):
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nr, nc = blank_r[i] + dr, blank_c[i] + dc
            if 0 <= nr < N and 0 <= nc < N:
                child = states[i].copy()
                new_idx = nr * N + nc
                old_idx = blank_pos[i]
                child[old_idx] = child[new_idx]
                child[new_idx] = 0
                all_next_states.append(child)
        offsets.append(len(all_next_states))

    all_next = np.array(all_next_states, dtype=np.int64) if all_next_states else np.zeros((0, N2), dtype=np.int64)
    if len(all_next) > 0:
        j_next = model.predict_j_batch(all_next).cpu()
    else:
        j_next = torch.tensor([])

    # Compute per-state squared Bellman error
    errors = np.empty(B, dtype=np.float64)
    for i in range(B):
        start, end = offsets[i], offsets[i + 1]
        if start == end:
            errors[i] = (j_s[i].item() - 0) ** 2
        else:
            best_j = (1.0 + j_next[start:end]).min().item()
            errors[i] = (j_s[i].item() - best_j) ** 2
    return errors


def stratified_bellman_mse_official(tiles, solution_lengths, model, num_strata,
                                    stratify_label="Len"):
    """Stratified Bellman MSE by solution length — always executed.

    Sorts by solution_lengths, splits into num_strata equal groups,
    computes per-state Bellman MSE for each stratum.
    """
    n = len(tiles)
    sorted_idx = np.argsort(solution_lengths)
    strata_size = n // num_strata

    print("=" * 72)
    print("方案 A — Stratified Bellman MSE (Official Model)")
    print("=" * 72)
    print(f"States: {n}  Strata: {num_strata}  Stratify by: {stratify_label}")
    print()

    overall_errors = []
    for s in range(num_strata):
        start = s * strata_size
        if s == num_strata - 1:
            end = n
        else:
            end = start + strata_size
        idx = sorted_idx[start:end]
        stratum_tiles = tiles[idx]
        stratum_lens = solution_lengths[idx]

        errors = _compute_bellman_errors_official(stratum_tiles, model)
        overall_errors.append(errors)

        count = len(stratum_tiles)
        lo, hi = stratum_lens.min(), stratum_lens.max()
        print(f"Stratum {s+1} [{lo}, {hi}] (n={count}): "
              f"mean={np.mean(errors):.6f}  median={np.median(errors):.6f}  std={np.std(errors):.6f}")

    all_errors = np.concatenate(overall_errors)
    print("-" * 72)
    print(f"Overall (n={n}): mean={np.mean(all_errors):.6f}  "
          f"median={np.median(all_errors):.6f}  std={np.std(all_errors):.6f}")
    print("=" * 72)


def _greedy_expand_official(start_grid_flat, model, max_steps):
    """Pure greedy: at each step, pick child with argmin J(s').

    Returns (solution_path, num_steps) or (None, num_steps) if failed.
    """
    current = start_grid_flat.copy()
    blank_idx = int(np.argmin(current))
    path = []
    for step in range(max_steps):
        if current.tobytes() == _GOAL_BYTES:
            return path, step
        children_info = get_children(current, blank_idx)
        if not children_info:
            return None, step
        grids = np.array([c[0] for c in children_info], dtype=np.int64)
        actions = [c[1] for c in children_info]
        new_blank_idxs = [c[2] for c in children_info]
        j_vals = model.predict_j_batch(grids).cpu().numpy()
        best_idx = int(np.argmin(j_vals))
        current = grids[best_idx]
        path.append(actions[best_idx])
        blank_idx = new_blank_idxs[best_idx]
    return None, max_steps


def greedy_expansion_eval_official(tiles, solution_lengths, model, max_steps,
                                   stratify_label="Len"):
    """Greedy expansion stratified by solution length — flag-controlled."""
    n = len(tiles)
    num_strata = DEEPCUBEA_OFFICIAL_NUM_STRATA
    sorted_idx = np.argsort(solution_lengths)
    strata_size = n // num_strata

    print()
    print("=" * 72)
    print("方案 B — Greedy Expansion Evaluation (Official Model, argmin J(s'))")
    print("=" * 72)
    print(f"States: {n}  Max Steps: {max_steps}")
    print()

    range_header = f"{stratify_label} Range"
    header = f"{'Stratum':<10} {range_header:<16} {'Count':<8} {'Solved':<10} {'Rate %':<10} {'Avg Steps':<12} {'Avg Len':<10}"
    print(header)
    print("-" * len(header))

    total_solved = 0
    total_steps = 0
    total_length = 0

    for s in range(num_strata):
        start = s * strata_size
        end = start + strata_size if s < num_strata - 1 else n
        idx = sorted_idx[start:end]
        stratum_tiles = tiles[idx]
        stratum_lens = solution_lengths[idx]

        solved = 0
        steps_sum = 0
        length_sum = 0

        for grid in stratum_tiles:
            path, num_steps = _greedy_expand_official(grid, model, max_steps)
            steps_sum += num_steps
            if path is not None:
                solved += 1
                length_sum += len(path)

        total_solved += solved
        total_steps += steps_sum
        total_length += length_sum

        count = len(stratum_tiles)
        rate = solved / count * 100 if count > 0 else 0.0
        avg_steps = steps_sum / count if count > 0 else 0.0
        avg_len = length_sum / solved if solved > 0 else float("nan")

        print(f"{s+1:<10} [{stratum_lens.min()}, {stratum_lens.max()}]{'':<8} "
              f"{count:<8} {solved:<10} {rate:<10.1f} {avg_steps:<12.1f} {avg_len:<10.1f}")

    print("-" * len(header))
    overall_rate = total_solved / n * 100
    overall_steps = total_steps / n
    print(f"{'Overall':<10} {'':<16} {n:<8} {total_solved:<10} {overall_rate:<10.1f} "
          f"{overall_steps:<12.1f} {'':<10}")
    print("=" * 72)


def weighted_astar_official(start_grid_flat, model, lambda_weight=None, max_expand=None):
    """Weighted A* search: f(s) = λ·g(s) + h(s).

    Reference: deepcubea_search.py:weighted_astar (line 34-111).
    Key difference: passes raw tiles to model.predict_j / predict_j_batch
    (ResnetModel does one_hot internally).
    """
    if lambda_weight is None:
        lambda_weight = DEEPCUBEA_OFFICIAL_LAMBDA
    if max_expand is None:
        max_expand = DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES

    h_start = model.predict_j(start_grid_flat)
    start_bytes = start_grid_flat.tobytes()
    open_set = [(lambda_weight * 0 + h_start, 0, start_bytes, start_grid_flat, None, None, 0)]
    heapq.heapify(open_set)
    g_score = {start_bytes: (0, None, None, 0)}

    expanded = 0

    while open_set:
        f_val, g_val, current_bytes, current_grid, _, _, _ = heapq.heappop(open_set)

        if current_bytes == _GOAL_BYTES:
            path = []
            cur_bytes = current_bytes
            while cur_bytes != start_bytes:
                _, parent_bytes, action, _ = g_score[cur_bytes]
                path.append(action)
                cur_bytes = parent_bytes
            path.reverse()
            return path, expanded + 1

        expanded += 1
        if expanded >= max_expand:
            return None, expanded

        entry = g_score.get(current_bytes)
        if entry is None or entry[0] < g_val:
            continue

        _, _, _, blank_idx = entry
        children_info = get_children(current_grid, blank_idx)

        if children_info:
            child_grids = np.array([c[0] for c in children_info], dtype=np.int64)
            child_h = model.predict_j_batch(child_grids).cpu().numpy()
            for idx, (child, action, new_blank_idx) in enumerate(children_info):
                new_g = g_val + 1
                child_bytes = child.tobytes()
                existing = g_score.get(child_bytes)
                if existing is None or new_g < existing[0]:
                    g_score[child_bytes] = (new_g, current_bytes, action, new_blank_idx)
                    f = lambda_weight * new_g + child_h[idx]
                    heapq.heappush(open_set, (f, new_g, child_bytes, child, current_bytes, action, new_blank_idx))

    return None, expanded


def evaluate_official(tiles, solution_lengths, model, max_expand=None, lambda_weight=None,
                      stratify_label="Len"):
    """Full weighted A* stratified by solution length — flag-controlled."""
    if lambda_weight is None:
        lambda_weight = DEEPCUBEA_OFFICIAL_LAMBDA
    if max_expand is None:
        max_expand = DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES

    n = len(tiles)
    num_strata = DEEPCUBEA_OFFICIAL_NUM_STRATA
    sorted_idx = np.argsort(solution_lengths)
    strata_size = n // num_strata

    print()
    print("=" * 72)
    print("方案 C — Full Weighted A* Evaluation (Official Model)")
    print("=" * 72)
    print(f"Lambda: {lambda_weight}  Max Expand: {max_expand:,}  States: {n}")
    print()

    range_header = f"{stratify_label} Range"
    header = f"{'Stratum':<10} {range_header:<16} {'Count':<8} {'Solved':<10} {'Rate %':<10} {'Avg Expand':<14} {'Avg Len':<10}"
    print(header)
    print("-" * len(header))

    total_solved = 0
    total_expanded = 0
    total_length = 0

    for s in range(num_strata):
        start = s * strata_size
        end = start + strata_size if s < num_strata - 1 else n
        idx = sorted_idx[start:end]
        stratum_tiles = tiles[idx]
        stratum_lens = solution_lengths[idx]

        solved = 0
        expanded_sum = 0
        length_sum = 0

        count = len(stratum_tiles)
        for i, grid in enumerate(stratum_tiles):
            path, expanded = weighted_astar_official(grid, model, lambda_weight, max_expand)
            expanded_sum += expanded
            if path is not None:
                solved += 1
                length_sum += len(path)
            if (i + 1) % 50 == 0 or i == count - 1:
                print(f"  [{s+1}] {i+1}/{count} done", flush=True)

        total_solved += solved
        total_expanded += expanded_sum
        total_length += length_sum

        rate = solved / count * 100 if count > 0 else 0.0
        avg_exp = expanded_sum / count if count > 0 else 0.0
        avg_len = length_sum / solved if solved > 0 else float("nan")

        print(f"{s+1:<10} [{stratum_lens.min()}, {stratum_lens.max()}]{'':<8} "
              f"{count:<8} {solved:<10} {rate:<10.1f} {avg_exp:<14.1f} {avg_len:<10.1f}")
        print()

    print("-" * len(header))
    overall_rate = total_solved / n * 100
    overall_exp = total_expanded / n
    print(f"{'Overall':<10} {'':<16} {n:<8} {total_solved:<10} {overall_rate:<10.1f} "
          f"{overall_exp:<14.1f} {'':<10}")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    import time

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEEPCUBEA_OFFICIAL_MODEL_PATH
    data_dir = sys.argv[2] if len(sys.argv) > 2 else DEEPCUBEA_OFFICIAL_DATA_DIR
    max_expand = int(sys.argv[3]) if len(sys.argv) > 3 else DEEPCUBEA_OFFICIAL_MAX_EXPAND_NODES
    lambda_weight = float(sys.argv[4]) if len(sys.argv) > 4 else DEEPCUBEA_OFFICIAL_LAMBDA

    # Load model
    t0 = time.perf_counter()
    model = load_official_model(model_path, use_gpu=DEEPCUBEA_INFERENCE_USE_GPU)
    load_time = time.perf_counter() - t0
    print(f"Official model loaded in {load_time:.2f}s (device={model._device})\n")

    # Load or generate test data
    if DEEPCUBEA_OFFICIAL_USE_OFFICIAL_DATA:
        tiles, solution_lengths = load_official_test_data(data_dir)
        stratify_label = "Len"
        print(f"Test data: {tiles.shape}, solution lengths [{solution_lengths.min()}, {solution_lengths.max()}] "
              f"mean={solution_lengths.mean():.1f}\n")
    else:
        rng = np.random.default_rng(DEEPCUBEA_VAL_SEED)
        num_states = DEEPCUBEA_OFFICIAL_NUM_TEST_STATES
        tiles, k_values = _generate_test_states(num_states,
                                                 DEEPCUBEA_OFFICIAL_T_MIN,
                                                 DEEPCUBEA_OFFICIAL_T_MAX, rng)
        solution_lengths = k_values
        stratify_label = "K"
        print(f"Generated test data: {tiles.shape}, K scramble steps [{k_values.min()}, {k_values.max()}] "
              f"mean={k_values.mean():.1f}\n")

    # 方案 A — Bellman MSE (always)
    stratified_bellman_mse_official(tiles, solution_lengths, model,
                                    num_strata=DEEPCUBEA_OFFICIAL_NUM_STRATA,
                                    stratify_label=stratify_label)

    # 方案 B — Greedy Expansion (flag-controlled)
    if DEEPCUBEA_OFFICIAL_GREEDY_FLAG:
        greedy_expansion_eval_official(tiles, solution_lengths, model,
                                       max_steps=DEEPCUBEA_OFFICIAL_GREEDY_MAX_STEPS,
                                       stratify_label=stratify_label)

    # 方案 C — Full A* (flag-controlled)
    if DEEPCUBEA_OFFICIAL_ASTAR_FLAG:
        evaluate_official(tiles, solution_lengths, model,
                          max_expand=max_expand, lambda_weight=lambda_weight,
                          stratify_label=stratify_label)
