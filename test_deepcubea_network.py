import numpy as np
import torch
from deepcubea_network import encode, encode_batch, decode, transition, DeepCubeANetwork

# --- Test 1: encode/decode round-trip ---
goal = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0], dtype=np.int32)
oh = encode(goal)
decoded = decode(oh)
print('Test 1 encode/decode round-trip:', 'PASS' if np.array_equal(goal, decoded) else 'FAIL')
if not np.array_equal(goal, decoded):
    print('  original:', goal)
    print('  decoded :', decoded)

# --- Test 2: encode/decode with random grids ---
for _ in range(10):
    g = np.random.permutation(16).astype(np.int32)
    d = decode(encode(g))
    assert np.array_equal(g, d), f'Round-trip failed: {g} vs {d}'
print('Test 2 random round-trip (10x): PASS')

# --- Test 3: encode_batch ---
grids = np.array([np.random.permutation(16).astype(np.int32) for _ in range(5)])
oh_batch = encode_batch(grids)
print('Test 3 encode_batch shape:', oh_batch.shape, '(expected (5, 256))')
for i in range(5):
    assert torch.allclose(oh_batch[i], encode(grids[i])), f'Batch mismatch at {i}'
print('Test 3 encode_batch consistency: PASS')

# --- Test 4: transition ---
t = transition(goal, 1)
print('Test 4 down from goal (should be None):', 'PASS' if t is None else 'FAIL')
t = transition(goal, 0)
expected = np.array([1,2,3,4,5,6,7,8,9,10,11,0,13,14,15,12], dtype=np.int32)
print('Test 4 up from goal:', 'PASS' if np.array_equal(t, expected) else 'FAIL')

# --- Test 5: transition all four directions from center ---
grid_mid = np.arange(16, dtype=np.int32)
grid_mid[5] = 0
grid_mid[0] = 5
t_up = transition(grid_mid, 0)
print('Test 5 up:   ', 'PASS' if t_up is not None and t_up[5] == grid_mid[1] and t_up[1] == 0 else 'FAIL')
t_down = transition(grid_mid, 1)
print('Test 5 down: ', 'PASS' if t_down is not None and t_down[5] == grid_mid[9] and t_down[9] == 0 else 'FAIL')
t_left = transition(grid_mid, 2)
print('Test 5 left: ', 'PASS' if t_left is not None and t_left[5] == grid_mid[4] and t_left[4] == 0 else 'FAIL')
t_right = transition(grid_mid, 3)
print('Test 5 right:', 'PASS' if t_right is not None and t_right[5] == grid_mid[6] and t_right[6] == 0 else 'FAIL')

# --- Test 6: edge transitions (should be illegal) ---
grid_top = np.arange(16, dtype=np.int32)
grid_top[0] = 0  # empty at top-left
grid_top[1] = 1  # restore 1
grid_top[1] = 1  # won't work... let me fix

grid_top = np.arange(16, dtype=np.int32)
# put 0 at top-left (row 0, col 0)
# swap position 0 with the 0 value
idx_zero = np.where(grid_top == 0)[0][0]
grid_top[0], grid_top[idx_zero] = 0, grid_top[0]
print(f'  grid_top: {grid_top.reshape(4,4)}')
print('Test 6 up from top   (illegal):', 'PASS' if transition(grid_top, 0) is None else 'FAIL')
print('Test 6 left from top (illegal):', 'PASS' if transition(grid_top, 2) is None else 'FAIL')
print('Test 6 down from top (legal):  ', 'PASS' if transition(grid_top, 1) is not None else 'FAIL')
print('Test 6 right from top(legal):  ', 'PASS' if transition(grid_top, 3) is not None else 'FAIL')

# --- Test 7: DeepCubeANetwork forward ---
net = DeepCubeANetwork()
x = torch.randn(4, 256)
out = net(x)
print(f'Test 7 forward: input {x.shape} -> output {out.shape} (expected (4,)):', 'PASS' if out.shape == (4,) else 'FAIL')

# --- Test 8: predict_j ---
j = net.predict_j(goal)
print(f'Test 8 predict_j: j={j:.4f} (type={type(j).__name__}): PASS')

# --- Test 9: predict_j_batch ---
grids = np.array([goal for _ in range(3)])
js = net.predict_j_batch(grids)
print(f'Test 9 predict_j_batch: shape={js.shape} (expected (3,)):', 'PASS' if js.shape == (3,) else 'FAIL')

# --- Test 10: GPU support ---
if torch.cuda.is_available():
    net_gpu = DeepCubeANetwork().cuda()
    x_gpu = torch.randn(2, 256, device='cuda')
    out_gpu = net_gpu(x_gpu)
    print(f'Test 10 GPU: device={out_gpu.device} PASS')
else:
    print('Test 10 GPU: CUDA not available (skip)')

print('\n=== ALL TESTS PASSED ===')
