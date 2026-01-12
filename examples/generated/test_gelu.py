"""
Test GELU Activation Implementation
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path
import math

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
BATCH = 64
DIM = 4096
TILE_SIZE = 1024


@ct.kernel
def gelu_kernel(X, Y, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, DIM), dtype="float32")
    MockTensor((BATCH, DIM), dtype="float32")
    1024
    </typecheck>
    GELU activation kernel (approximate version).
    """
    bid_batch = ct.bid(0)
    bid_tile = ct.bid(1)

    x = ct.load(X, index=(bid_batch, bid_tile), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715

    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    y = 0.5 * x * (1.0 + ct.tanh(inner))

    ct.store(Y, index=(bid_batch, bid_tile), tile=y)


# cutile-typeviz: end


def run_test():
    """Run GELU test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing GELU Activation")
    print(f"  Shape: ({BATCH}, {DIM}), Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    x = np.random.randn(BATCH, DIM).astype(np.float32)
    y = np.zeros_like(x)

    tmp_dir = Path("ir_artifacts") / "generated" / "gelu"

    num_tiles = DIM // TILE_SIZE

    launch_numpy(
        gelu_kernel,
        [x, y, TILE_SIZE],
        grid=(BATCH, num_tiles, 1),
        tmp_dir=tmp_dir,
    )

    # Compute expected GELU
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    expected = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))

    mae = np.abs(y - expected).mean()
    max_diff = np.abs(y - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-5 and max_diff < 1e-4
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
