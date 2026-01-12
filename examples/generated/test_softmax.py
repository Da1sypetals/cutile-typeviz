"""
Test Softmax Implementation
Computes softmax along the last dimension: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions - use tile-aligned size
BATCH = 32
SEQ_LEN = 64
DIM = 256
TILE_DIM = 64
# Note: cutile doesn't support float('-inf') as a constant in kernel code,
# so we use a very negative number as a workaround
NEG_INF_APPROX = -1e38


@ct.kernel
def softmax_kernel(X, Y, TILE_DIM: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, SEQ_LEN, DIM), dtype="float32")
    MockTensor((BATCH, SEQ_LEN, DIM), dtype="float32")
    64
    </typecheck>
    Softmax kernel: computes softmax along the last dimension.
    """
    bid_batch = ct.bid(0)
    bid_seq = ct.bid(1)

    DIM_SIZE = X.shape[2]
    num_tiles = ct.num_tiles(X, axis=2, shape=(1, 1, TILE_DIM))

    # Step 1: Find max value for numerical stability
    # Using NEG_INF_APPROX (-1e38) as cutile doesn't support float('-inf')
    max_val = ct.full((1, 1, TILE_DIM), -1e38, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_batch, bid_seq, j), shape=(1, 1, TILE_DIM), padding_mode=PAD_ZERO)
        mask = j * TILE_DIM + ct.arange(TILE_DIM, dtype=ct.int32) < DIM_SIZE
        tx = ct.where(mask, tx, -1e38)
        max_val = ct.maximum(max_val, tx)
    max_val = ct.max(max_val, axis=2, keepdims=True)  # Scalar max

    # Step 2: Compute sum of exp(x - max)
    sum_exp = ct.full((1, 1, TILE_DIM), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_batch, bid_seq, j), shape=(1, 1, TILE_DIM), padding_mode=PAD_ZERO)
        mask = j * TILE_DIM + ct.arange(TILE_DIM, dtype=ct.int32) < DIM_SIZE
        exp_val = ct.exp(tx - max_val)
        exp_val = ct.where(mask, exp_val, 0.0)
        sum_exp += exp_val
    sum_exp = ct.sum(sum_exp, axis=2, keepdims=True)  # Scalar sum

    # Step 3: Compute softmax output
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_batch, bid_seq, j), shape=(1, 1, TILE_DIM), padding_mode=PAD_ZERO)
        exp_val = ct.exp(tx - max_val)
        softmax_val = exp_val / sum_exp
        ct.store(Y, index=(bid_batch, bid_seq, j), tile=softmax_val)


# cutile-typeviz: end


def run_test():
    """Run softmax test and compare with numpy/torch."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing Softmax")
    print(f"  Shape: ({BATCH}, {SEQ_LEN}, {DIM}), Tile: {TILE_DIM}")
    print("=" * 60)

    # Create random input
    np.random.seed(42)
    x = np.random.randn(BATCH, SEQ_LEN, DIM).astype(np.float32)
    y = np.zeros_like(x)

    tmp_dir = Path("ir_artifacts") / "generated" / "softmax"

    # Launch kernel
    launch_numpy(
        softmax_kernel,
        [x, y, TILE_DIM],
        grid=(BATCH, SEQ_LEN, 1),
        tmp_dir=tmp_dir,
    )

    # Compute expected with numpy
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Compare results
    mae = np.abs(y - expected).mean()
    max_diff = np.abs(y - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")
    print(f"  Output sum (should be ~1): {y[0, 0].sum():.6f}")
    print(f"  Expected sum: {expected[0, 0].sum():.6f}")

    # Check if test passes
    passed = mae < 1e-5 and max_diff < 1e-4
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
