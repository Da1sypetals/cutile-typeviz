"""
Test Tile Item Implementation
Extract scalar value from a tile using tile.item()
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
SIZE = 1024
TILE_SIZE = 32


@ct.kernel
def tile_item_kernel(A, B, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((SIZE,), dtype="float32")
    MockTensor((SIZE,), dtype="float32")
    32
    </typecheck>
    Tile item kernel: Extract first element from each tile and broadcast it
    """
    bid = ct.bid(0)

    # Load a tile
    a = ct.load(A, index=(bid,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)

    # Extract the first element as a scalar using item()
    # First reshape to get a single element tile, then use item()
    # first_elem_tile = ct.extract(a, index=(0,), shape=(1,))  # Get first element as a tile
    first_elem_tile = ct.extract(a, index=(0,), shape=(1,))  # Get first element as a tile
    scalar_val = first_elem_tile.item()  # Convert to scalar

    # Broadcast the scalar back to a tile
    result = ct.full((TILE_SIZE,), scalar_val, dtype=ct.float32)

    ct.store(B, index=(bid,), tile=result)


# cutile-typeviz: end


def run_test():
    """Run tile item test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing Tile Item")
    print(f"  Size: {SIZE}, Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    a = np.random.randn(SIZE).astype(np.float32)
    b = np.zeros(SIZE, dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "generated" / "tile_item"

    num_blocks = SIZE // TILE_SIZE

    launch_numpy(
        tile_item_kernel,
        [a, b, TILE_SIZE],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    # Expected: each tile should be filled with the first element of that tile
    expected = np.zeros(SIZE, dtype=np.float32)
    for i in range(num_blocks):
        start_idx = i * TILE_SIZE
        end_idx = start_idx + TILE_SIZE
        first_val = a[start_idx]
        expected[start_idx:end_idx] = first_val

    mae = np.abs(b - expected).mean()
    max_diff = np.abs(b - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-6
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
