# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# code taken from https://github.com/raphaelamorim/cutilesample/blob/main/matmul.py

import cuda.tile as ct
from math import ceil  # Required for host-side grid calculation


ConstInt = ct.Constant[int]


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given CUDA block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(
    A,
    B,
    C,
    tm: ConstInt,  # Tile size along M dimension (rows of C)
    tn: ConstInt,  # Tile size along N dimension (columns of C)
    tk: ConstInt,
):  # Tile size along K dimension (inner product dimension)
    """
    <typecheck>
    MockTensor((1024, 768), dtype="float16")
    MockTensor((768, 512), dtype="float16")
    MockTensor((1024, 512), dtype="float16")
    128
    256
    64
    </typecheck>
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each CUDA thread block (CTA)
    computes a `tm` x `tn` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `tk`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `tk` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


# cutile-typeviz: end


@ct.kernel
def persistent_matmul_kernel(
    A,
    B,
    C,
    tm: ConstInt,  # Tile size along M dimension (rows of C)
    tn: ConstInt,  # Tile size along N dimension (columns of C)
    tk: ConstInt,
):  # Tile size along K dimension
    """
    <typecheck>
    MockTensor((1024, 768), dtype="float16")
    MockTensor((768, 512), dtype="float16")
    MockTensor((1024, 512), dtype="float16")
    128
    256
    64
    </typecheck>
    cuTile persistent kernel for performing matrix multiplication C = A @ B.

    This kernel uses a persistent approach, where NUM_SMS tile blocks are launched
    and each tile block processes multiple output tiles.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        tm (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        tn (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        tk (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8

    bid = ct.bid(0)
    M = A.shape[0]
    N = B.shape[1]

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `tk` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # Number of tiles along M and N
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    upper_bound = num_bid_m * num_bid_n

    # Static persistent loop: each program processes multiple tiles.
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        # Initialize an accumulator for the current output tile (tm x tn).
        # It's common practice to use `float32` for accumulation even with `float16` inputs
        # to maintain higher precision during the sum-reduction of the matrix multiplication.
        accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
        bidx, bidy = swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, current_bid)

        # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
        # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
        # are loaded, multiplied, and accumulated.
        for k in range(num_tiles_k):
            # Load tile from matrix A.
            # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
            # from global memory A. `shape=(tm, tk)` defines the size of this tile.
            a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)

            # Load tile from matrix B.
            # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
            # from global memory B. `shape=(tk, tn)` defines the size of this tile.
            b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)

            # Perform Matrix Multiplication for the current tiles.
            # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
            accumulator = ct.mma(a, b, accumulator)

        # Cast result back to C.dtype and store
        accumulator = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bidx, bidy), tile=accumulator)
