import cuda.tile as ct

ConstInt = ct.Constant[int]


BATCH_DIM = 4
M_DIM = 512
K_DIM = 256
N_DIM = 1024


@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    <typecheck>
    MockTensor((BATCH_DIM, M_DIM, K_DIM), dtype="bfloat16")
    MockTensor((BATCH_DIM, K_DIM, N_DIM), dtype="bfloat16")
    MockTensor((BATCH_DIM, M_DIM, N_DIM), dtype="bfloat16")
    32
    64
    128
    </typecheck>
    CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Each thread block computes one (tm x tn) tile for a specific batch item.
    The grid is 3D: (Batch_idx, M_tile_idx, N_tile_idx).
    """
    pid_batch = ct.bid(0)  # Batch dimension
    pidx = ct.bid(1)  # M dimension
    pidy = ct.bid(2)  # N dimension

    # Calculate number of K tiles
    # A is (Batch, M, K), so K is axis 2
    # Use A.shape[2] for the total K dimension and ct.cdiv for ceiling division
    num_k_tiles = ct.cdiv(A.shape[2], tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        # A is (Batch, M, K), load (1, tm, tk) tile
        a = ct.load(
            A,
            index=(pid_batch, pidx, k),
            shape=(1, tm, tk),
            padding_mode=zero_pad,
        )
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D for ct.mma

        # B is (Batch, K, N), load (1, tk, tn) tile
        b = ct.load(
            B,
            index=(pid_batch, k, pidy),
            shape=(1, tk, tn),
            padding_mode=zero_pad,
        )
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D for ct.mma

        accumulator = ct.mma(a, b, acc=accumulator)

    # Convert to output dtype and store
    result = ct.astype(accumulator, C.dtype)
    # Store with 3D index and 3D shape, C is (Batch, M, N)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, pidx, pidy), tile=result_3d)


# cutile-typeviz: end

import numpy as np
from cutile_typeviz.transpiler import launch_numpy
from pathlib import Path
from icecream import ic

a = np.random.randn(BATCH_DIM, M_DIM, K_DIM).astype(np.float32)
b = np.random.randn(BATCH_DIM, K_DIM, N_DIM).astype(np.float32)
c = np.zeros((BATCH_DIM, M_DIM, N_DIM), dtype=np.float32)

tm = 32
tn = 64
tk = 128

tmp_dir = Path("ir_artifacts") / "matmul"

launch_numpy(
    batch_matmul_kernel,
    [a, b, c, tm, tn, tk],
    grid=(BATCH_DIM, M_DIM // tm, N_DIM // tn),
    tmp_dir=tmp_dir,
)

expected = np.matmul(a, b)

mae = np.abs(expected - c).mean()
print(f"MAE: {mae}")
ic(expected[0, :3, :3])
ic(c[0, :3, :3])
