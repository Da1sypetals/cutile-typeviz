"""
Test Reduction Operations (Sum, Mean, Max, Min)
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions - use tile-aligned size
BATCH = 64
DIM = 4096
TILE_SIZE = 1024
# Note: cutile doesn't support float('-inf') as a constant in kernel code,
# so we use a very negative number as a workaround
NEG_INF_APPROX = -1e38


@ct.kernel
def reduce_sum_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, DIM), dtype="float32")
    MockTensor((BATCH,), dtype="float32")
    1024
    </typecheck>
    Sum reduction kernel along the last dimension.
    """
    bid = ct.bid(0)

    DIM_SIZE = X.shape[1]
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_SIZE))

    acc = ct.full((1, TILE_SIZE), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid, j), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
        mask = j * TILE_SIZE + ct.arange(TILE_SIZE, dtype=ct.int32) < DIM_SIZE
        tx = ct.where(mask, tx, 0.0)
        acc += tx

    result = ct.sum(acc, axis=1)
    ct.store(Y, index=(bid,), tile=result)


@ct.kernel
def reduce_max_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, DIM), dtype="float32")
    MockTensor((BATCH,), dtype="float32")
    1024
    </typecheck>
    Max reduction kernel along the last dimension.
    """
    bid = ct.bid(0)

    DIM_SIZE = X.shape[1]
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_SIZE))

    # Using -1e38 as cutile doesn't support float('-inf')
    acc = ct.full((1, TILE_SIZE), -1e38, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid, j), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
        mask = j * TILE_SIZE + ct.arange(TILE_SIZE, dtype=ct.int32) < DIM_SIZE
        tx = ct.where(mask, tx, -1e38)
        acc = ct.maximum(acc, tx)

    result = ct.max(acc, axis=1)
    ct.store(Y, index=(bid,), tile=result)


# cutile-typeviz: end


def run_test():
    """Run reduction tests."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing Reduction Operations")
    print(f"  Shape: ({BATCH}, {DIM}), Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    x = np.random.randn(BATCH, DIM).astype(np.float32)

    all_passed = True

    # Test sum reduction
    print("\n  --- Sum Reduction ---")
    y_sum = np.zeros(BATCH, dtype=np.float32)
    tmp_dir = Path("ir_artifacts") / "generated" / "reduce_sum"

    launch_numpy(
        reduce_sum_kernel,
        [x, y_sum, TILE_SIZE],
        grid=(BATCH, 1, 1),
        tmp_dir=tmp_dir,
    )

    expected_sum = np.sum(x, axis=-1)
    mae_sum = np.abs(y_sum - expected_sum).mean()
    max_diff_sum = np.abs(y_sum - expected_sum).max()

    print(f"    MAE: {mae_sum:.6e}")
    print(f"    Max Abs Diff: {max_diff_sum:.6e}")
    passed_sum = mae_sum < 1e-4 and max_diff_sum < 1e-3
    print(f"    Status: {'PASSED' if passed_sum else 'FAILED'}")
    all_passed &= passed_sum

    # Test max reduction
    print("\n  --- Max Reduction ---")
    y_max = np.zeros(BATCH, dtype=np.float32)
    tmp_dir = Path("ir_artifacts") / "generated" / "reduce_max"

    launch_numpy(
        reduce_max_kernel,
        [x, y_max, TILE_SIZE],
        grid=(BATCH, 1, 1),
        tmp_dir=tmp_dir,
    )

    expected_max = np.max(x, axis=-1)
    mae_max = np.abs(y_max - expected_max).mean()
    max_diff_max = np.abs(y_max - expected_max).max()

    print(f"    MAE: {mae_max:.6e}")
    print(f"    Max Abs Diff: {max_diff_max:.6e}")
    passed_max = mae_max < 1e-6 and max_diff_max < 1e-5
    print(f"    Status: {'PASSED' if passed_max else 'FAILED'}")
    all_passed &= passed_max

    print(f"\n  Overall Status: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
