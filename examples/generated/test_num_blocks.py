"""
Test num_blocks operation
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/cutile_typeviz/cutile_utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import cuda.tile as ct
from cutile_typeviz.transpiler.transpile import launch_numpy

ConstInt = int

# Test dimensions
BATCH = 32
DIM = 64

tmp_dir = os.path.join(os.path.dirname(__file__), "../../ir_artifacts/num_blocks_test")


@ct.kernel
def num_blocks_kernel(X, Y):
    """
    <typecheck>
    MockTensor((BATCH, DIM), dtype="float32")
    MockTensor((BATCH,), dtype="float32")
    </typecheck>
    Test kernel that uses num_blocks to verify grid dimensions.
    Stores num_blocks_x * 10 + num_blocks_y as the result.
    """
    bid_x = ct.bid(0)

    # Get number of blocks in each dimension
    num_blocks_x = ct.num_blocks(0)
    num_blocks_y = ct.num_blocks(1)
    num_blocks_z = ct.num_blocks(2)

    # Store the computed value based on num_blocks
    # Result: num_blocks_x * 100 + num_blocks_y * 10 + num_blocks_z
    result_val = num_blocks_x * 100 + num_blocks_y * 10 + num_blocks_z

    # Create a tile with the result value
    result_tile = ct.full((1,), result_val, dtype=ct.float32)

    out_idx = (bid_x,)
    ct.store(Y, out_idx, result_tile)


def test_num_blocks():
    print("=" * 50)
    print("Testing num_blocks")
    print("=" * 50)

    np.random.seed(42)
    x = np.random.randn(BATCH, DIM).astype(np.float32)
    y = np.zeros(BATCH, dtype=np.float32)

    # Grid: (32, 4, 2) -> num_blocks = (32, 4, 2)
    # Expected result: 32 * 100 + 4 * 10 + 2 = 3242
    grid_x, grid_y, grid_z = 32, 4, 2
    expected_val = grid_x * 100 + grid_y * 10 + grid_z
    ref = np.full(BATCH, expected_val, dtype=np.float32)

    launch_numpy(
        num_blocks_kernel,
        [x, y],
        grid=(grid_x, grid_y, grid_z),
        tmp_dir=tmp_dir,
    )

    # Compute metrics
    mae = np.mean(np.abs(y - ref))
    max_diff = np.max(np.abs(y - ref))

    print(f"Expected value: {expected_val}")
    print(f"Actual values: {y[:5]}...")
    print(f"MAE: {mae:.6e}")
    print(f"Max Abs Diff: {max_diff:.6e}")

    # Check correctness
    passed = max_diff < 1e-4
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    success = test_num_blocks()
    sys.exit(0 if success else 1)


def run_test():
    return test_num_blocks()
