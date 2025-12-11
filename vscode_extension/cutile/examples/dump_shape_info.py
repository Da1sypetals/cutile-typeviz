import json
import cuda.tile as ct
from ir_dump.mock_tensor import MockTensor
from ir_dump.shape_check import get_kernel_shapes_info

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


M = 1024  # Batch size (flattened)
N = 2048  # Feature dimension
TILE_N = 1024  # Tile size along N dimension
GROUP_SIZE_M = 64  # Group size for backward pass
TILE_M = 32  # Tile size for final reduction in backward pass

# Forward pass tensors
X = MockTensor((M, N), dtype="float16")
W = MockTensor((N,), dtype="float16")
B = MockTensor((N,), dtype="float16")
Y = MockTensor((M, N), dtype="float16")
Mean = MockTensor((M,), dtype="float32")
Rstd = MockTensor((M,), dtype="float32")

# Backward pass tensors
DX = MockTensor((M, N), dtype="float16")
DY = MockTensor((M, N), dtype="float16")
DW = MockTensor((GROUP_SIZE_M, N), dtype="float32")
DB = MockTensor((GROUP_SIZE_M, N), dtype="float32")
Locks = MockTensor((GROUP_SIZE_M,), dtype="int32")
FINAL_DW = MockTensor((N,), dtype="float16")
FINAL_DB = MockTensor((N,), dtype="float16")


@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    """
    Forward pass: computes mean/var, normalizes input, and applies affine transform.

    Args:
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        B: Bias tensor (N,).
        Y: Output tensor (M, N).
        Mean: Output mean tensor (M,).
        Rstd: Output reciprocal standard deviation tensor (M,).
        eps: Epsilon for numerical stability.
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute mean
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute variance
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx**2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    for j in range(num_tiles):
        # Normalize and apply affine transformation
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


def bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N):
    """Helper to load data and compute common backward terms."""
    tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
    tdy = ct.load(DY, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    xhat = (tx - mean) * rstd
    wdy = tw * tdy
    mask = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32) < N
    xhat = ct.where(mask, xhat, 0)
    wdy = ct.where(mask, wdy, 0)
    return tdy, xhat, wdy


@ct.kernel
def layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, TILE_N: ConstInt):
    """
    Backward pass part 1: computes dX and partial dW/dB.
    Accumulates partial gradients using atomic locks.

    Args:
        DX: Output gradient with respect to X (M, N).
        DY: Input gradient with respect to Y (M, N).
        DW: Partial gradient with respect to W (GROUP_SIZE_M, N).
        DB: Partial gradient with respect to B (GROUP_SIZE_M, N).
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        Mean: Mean tensor (M,).
        Rstd: Reciprocal standard deviation tensor (M,).
        Locks: Lock tensor for atomic operations (GROUP_SIZE_M,).
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    GROUP_SIZE_M = DW.shape[0]
    group_bid_m = bid_m % GROUP_SIZE_M

    mean = ct.load(Mean, index=(bid_m,), shape=(1,))
    rstd = ct.load(Rstd, index=(bid_m,), shape=(1,))

    c1 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute reduction terms for dX
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 += xhat * wdy
        c2 += wdy
    c1 = ct.sum(c1, axis=1) / N
    c2 = ct.sum(c2, axis=1) / N

    for j in range(num_tiles):
        # Compute dX and partial dW, dB
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy - (xhat * c1 + c2)) * rstd
        ct.store(DX, index=(bid_m, j), tile=tdx.astype(DX.dtype))

        partial_dw = (tdy * xhat).astype(DW.dtype)
        partial_db = tdy.astype(DB.dtype)

        while ct.atomic_cas(Locks, group_bid_m, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1:
            pass

        # Accumulate partial weight/bias gradients
        partial_dw += ct.load(DW, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        partial_db += ct.load(DB, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        ct.store(DW, index=(group_bid_m, j), tile=partial_dw)
        ct.store(DB, index=(group_bid_m, j), tile=partial_db)

        ct.atomic_xchg(Locks, group_bid_m, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ConstInt, TILE_N: ConstInt):
    """
    Backward pass part 2: Final reduction for dW and dB.

    Args:
        DW: Partial gradient with respect to W (TILE_M, N).
        DB: Partial gradient with respect to B (TILE_M, N).
        FINAL_DW: Final gradient with respect to W (N,).
        FINAL_DB: Final gradient with respect to B (N,).
        TILE_M: Number of partial gradients to reduce.
        TILE_N: Tile size along N dimension.
    """
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))

    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        # Sum partial gradients
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)

    ct.store(FINAL_DW, index=(bid_n,), tile=sum_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(bid_n,), tile=sum_db.astype(FINAL_DB.dtype))


if __name__ == "__main__":
    import os

    # 直接调用 get_kernel_shapes_info 获取形状信息，等同于原来的 typecheck 装饰器功能
    ops = [
        *get_kernel_shapes_info(layer_norm_fwd, [X, W, B, Y, Mean, Rstd, 1e-5, TILE_N]),
        *get_kernel_shapes_info(
            layer_norm_bwd_dx_partial_dwdb, [DX, DY, DW, DB, X, W, Mean, Rstd, Locks, TILE_N]
        ),
        *get_kernel_shapes_info(layer_norm_bwd_dwdb, [DW, DB, FINAL_DW, FINAL_DB, TILE_M, TILE_N]),
    ]

    dump_path = os.path.join("ir_artifacts", "shape_info.json")

    with open(dump_path, "w") as f:
        json.dump(ops, f, indent=4)

    print(f"Dumped shape info to {dump_path}")
