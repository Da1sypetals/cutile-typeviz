"""
Test RMSNorm Implementation (used in LLaMA, etc.)
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
"""

import cuda.tile as ct
import numpy as np
from pathlib import Path

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Test dimensions - use tile-aligned size
BATCH = 32
SEQ_LEN = 64
DIM = 512
TILE_DIM = 128
EPS = 1e-6


@ct.kernel
def rmsnorm_kernel(X, W, Y, eps, TILE_DIM: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH, SEQ_LEN, DIM), dtype="float32")
    MockTensor((DIM,), dtype="float32")
    MockTensor((BATCH, SEQ_LEN, DIM), dtype="float32")
    1e-6
    128
    </typecheck>
    RMSNorm kernel: Y = X * rsqrt(mean(X^2) + eps) * W
    """
    bid_batch = ct.bid(0)
    bid_seq = ct.bid(1)

    DIM_SIZE = X.shape[2]
    num_tiles = ct.num_tiles(X, axis=2, shape=(1, 1, TILE_DIM))

    # Step 1: Compute mean of squares
    sum_sq = ct.full((1, 1, TILE_DIM), 0.0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_batch, bid_seq, j), shape=(1, 1, TILE_DIM), padding_mode=PAD_ZERO)
        mask = j * TILE_DIM + ct.arange(TILE_DIM, dtype=ct.int32) < DIM_SIZE
        sq = tx * tx
        sq = ct.where(mask, sq, 0.0)
        sum_sq += sq

    mean_sq = ct.sum(sum_sq, axis=2, keepdims=True) / DIM_SIZE
    rrms = ct.rsqrt(mean_sq + eps)  # Reciprocal RMS

    # Step 2: Normalize and scale
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_batch, bid_seq, j), shape=(1, 1, TILE_DIM), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_DIM,), padding_mode=PAD_ZERO)

        ty = tx * rrms * tw
        ct.store(Y, index=(bid_batch, bid_seq, j), tile=ty)


# cutile-typeviz: end


def run_test():
    """Run RMSNorm test."""
    from cutile_typeviz.transpiler import launch_numpy

    print("=" * 60)
    print("Testing RMSNorm")
    print(f"  Shape: ({BATCH}, {SEQ_LEN}, {DIM}), Tile: {TILE_DIM}")
    print("=" * 60)

    np.random.seed(42)
    x = np.random.randn(BATCH, SEQ_LEN, DIM).astype(np.float32)
    w = np.random.randn(DIM).astype(np.float32) * 0.1 + 1.0  # Around 1.0
    y = np.zeros_like(x)

    tmp_dir = Path("ir_artifacts") / "generated" / "rmsnorm"

    launch_numpy(
        rmsnorm_kernel,
        [x, w, y, EPS, TILE_DIM],
        grid=(BATCH, SEQ_LEN, 1),
        tmp_dir=tmp_dir,
    )

    # Compute expected RMSNorm
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)
    rrms = 1.0 / np.sqrt(mean_sq + EPS)
    expected = x * rrms * w

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
