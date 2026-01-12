"""Test num_tiles operation"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
BATCH = 4
DIM = 256
TILE_SIZE = 64


@ct.kernel
def num_tiles_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((4, 256), dtype="float32")
    MockTensor((4, 256), dtype="float32")
    64
    </typecheck>
    Test num_tiles operation
    """
    bid = ct.bid(0)

    # Use num_tiles to get the number of tiles along axis 1
    n_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_SIZE))

    # Simple loop using num_tiles
    for j in range(n_tiles):
        tx = ct.load(X, index=(bid, j), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
        ct.store(Y, (bid, j), tx)


# cutile-typeviz: end


def run_test():
    """Run num_tiles test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing num_tiles")
    print(f"  Batch: {BATCH}, Dim: {DIM}, Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(BATCH, DIM).astype(np.float32)
    Y = np.zeros_like(X)

    tmp_dir = Path("ir_artifacts") / "generated" / "num_tiles"

    launch_numpy(
        num_tiles_kernel,
        [X, Y, TILE_SIZE],
        grid=(BATCH, 1, 1),
        tmp_dir=tmp_dir,
    )

    mae = np.abs(Y - X).mean()
    max_diff = np.abs(Y - X).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-6
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
