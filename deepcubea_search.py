import heapq
import numpy as np
import torch
from config import (
    HUARONGDAO_N, DEEPCUBEA_T_MIN, DEEPCUBEA_T_MAX,
    DEEPCUBEA_LAMBDA, DEEPCUBEA_MODEL_PATH, DEEPCUBEA_MAX_EXPAND_NODES,
    DEEPCUBEA_NUM_TEST_STATES, DEEPCUBEA_INFERENCE_USE_GPU,
)
from deepcubea_network import DeepCubeANetwork, transition, get_children

N2 = HUARONGDAO_N * HUARONGDAO_N
_GOAL_GRID = np.array(list(range(1, N2)) + [0], dtype=np.int32)
_GOAL_BYTES = _GOAL_GRID.tobytes()


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
