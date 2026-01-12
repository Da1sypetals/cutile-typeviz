"""
Test Vector Add Implementation
Simple element-wise addition: C = A + B
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
SIZE = 65536
TILE_SIZE = 1024


@ct.kernel
def vector_add_kernel(A, B, C, TILE_SIZE: ConstInt):
    """
    <typecheck>
    MockTensor((SIZE,), dtype="float32")
    MockTensor((SIZE,), dtype="float32")
    MockTensor((SIZE,), dtype="float32")
    1024
    </typecheck>
    Vector add kernel: C = A + B
    """
    bid = ct.bid(0)

    a = ct.load(A, index=(bid,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)
    b = ct.load(B, index=(bid,), shape=(TILE_SIZE,), padding_mode=PAD_ZERO)

    c = a + b

    ct.store(C, index=(bid,), tile=c)


# cutile-typeviz: end


def run_test():
    """Run vector add test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing Vector Add")
    print(f"  Size: {SIZE}, Tile: {TILE_SIZE}")
    print("=" * 60)

    np.random.seed(42)
    a = np.random.randn(SIZE).astype(np.float32)
    b = np.random.randn(SIZE).astype(np.float32)
    c = np.zeros(SIZE, dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "generated" / "vector_add"

    num_blocks = SIZE // TILE_SIZE

    launch_numpy(
        vector_add_kernel,
        [a, b, c, TILE_SIZE],
        grid=(num_blocks, 1, 1),
        tmp_dir=tmp_dir,
    )

    expected = a + b

    mae = np.abs(c - expected).mean()
    max_diff = np.abs(c - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-6
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
