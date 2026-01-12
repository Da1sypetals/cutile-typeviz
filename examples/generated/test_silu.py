"""
Test SiLU (Swish) Activation Implementation
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
BATCH = 64
DIM = 8192
TILE_SIZE = 2048


@ct.kernel
def silu_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, DIM), dtype="float32")
    MockTensor((BATCH, DIM), dtype="float32")
    2048
    </typecheck>
    SiLU activation kernel: Y = X * sigmoid(X)
    """
    bid_batch = ct.bid(0)
    bid_tile = ct.bid(1)

    x = ct.load(X, index=(bid_batch, bid_tile), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)

    # SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x))
    y = x * sigmoid_x

    ct.store(Y, index=(bid_batch, bid_tile), tile=y)


# cutile-typeviz: end


def run_test():
    """Run SiLU test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing SiLU (Swish) Activation")
    print(f"  Shape: ({BATCH}, {DIM}), Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    x = np.random.randn(BATCH, DIM).astype(np.float32)
    y = np.zeros_like(x)

    tmp_dir = Path("ir_artifacts") / "generated" / "silu"

    num_tiles = DIM // TILE_SIZE

    launch_numpy(
        silu_kernel,
        [x, y, TILE_SIZE],
        grid=(BATCH, num_tiles, 1),
        tmp_dir=tmp_dir,
    )

    # Compute expected SiLU
    sigmoid_x = 1.0 / (1.0 + np.exp(-x))
    expected = x * sigmoid_x

    mae = np.abs(y - expected).mean()
    max_diff = np.abs(y - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-6 and max_diff < 1e-5
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
