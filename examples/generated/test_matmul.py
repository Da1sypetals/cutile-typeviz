"""
Test Matrix Multiplication (Simple 2D)
C = A @ B
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions
M = 256
K = 128
N = 512
TM = 32
TN = 64
TK = 32


@ct.kernel
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    <typecheck>
    MockTensor((M, K), dtype="float32")
    MockTensor((K, N), dtype="float32")
    MockTensor((M, N), dtype="float32")
    32
    64
    32
    </typecheck>
    Simple 2D matrix multiplication kernel.
    """
    pidx = ct.bid(0)  # M dimension
    pidy = ct.bid(1)  # N dimension

    num_k_tiles = ct.cdiv(A.shape[1], tk)

    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)

    for k in range(num_k_tiles):
        a = ct.load(A, index=(pidx, k), shape=(tm, tk), padding_mode=PAD_ZERO)
        b = ct.load(B, index=(k, pidy), shape=(tk, tn), padding_mode=PAD_ZERO)
        acc = ct.mma(a, b, acc=acc)

    ct.store(C, index=(pidx, pidy), tile=acc)


# cutile-typeviz: end


def run_test():
    """Run matmul test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing Matrix Multiplication")
    print(f"  Shape: ({M}, {K}) @ ({K}, {N}) -> ({M}, {N})")
    print(f"  Tiles: TM={TM}, TN={TN}, TK={TK}")
    print("=" * 60)

    np.random.seed(42)
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    tmp_dir = Path("ir_artifacts") / "generated" / "matmul_2d"

    launch_numpy(
        matmul_kernel,
        [a, b, c, TM, TN, TK],
        grid=(M // TM, N // TN, 1),
        tmp_dir=tmp_dir,
    )

    expected = np.matmul(a, b)

    mae = np.abs(c - expected).mean()
    max_diff = np.abs(c - expected).max()

    print(f"  MAE: {mae:.6e}")
    print(f"  Max Abs Diff: {max_diff:.6e}")

    passed = mae < 1e-4 and max_diff < 1e-3
    print(f"  Status: {'PASSED' if passed else 'FAILED'}")

    return passed


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
