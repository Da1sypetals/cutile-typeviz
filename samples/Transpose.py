# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
from math import ceil  # Required for host-side grid calculation


ConstInt = ct.Constant[int]


@ct.kernel
def transpose_kernel(x, y,
                     tm: ConstInt,  # Tile size along M dimension (rows of original x)
                     tn: ConstInt):  # Tile size along N dimension (columns of original x)
    """
    cuTile kernel to transpose a 2D matrix by processing data in tiles.

    Each CUDA thread block (CTA) is responsible for computing a `tn` x `tm` tile
    of the output (transposed) matrix `y`. This involves loading a `tm` x `tn`
    tile from the input matrix `x`, transposing it locally, and then storing
    the `tn` x `tm` result to the correct location in `y`.

    Args:
        x: Input matrix (M x N).
        y: Output matrix (N x M), which will be the transpose of x.
        tm (ConstInt): The height of the input tile (number of rows from x)
                       processed by this block.
        tn (ConstInt): The width of the input tile (number of columns from x)
                       processed by this block.
    """
    # Get the global IDs of the current CUDA block (CTA) in a 2D grid.
    # `ct.bid(0)` gives the block ID along the X-axis of the grid, which corresponds
    # to the M-tile index (rows) of the original input matrix `x`.
    # `ct.bid(1)` gives the block ID along the Y-axis of the grid, which corresponds
    # to the N-tile index (columns) of the original input matrix `x`.
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    # Load a tile from the input matrix 'x'.
    # `ct.load` reads a `tm` x `tn` chunk of data from global memory `x`
    # at the specified `index=(bidx, bidy)`. This data is brought into
    # the thread block's local scope (e.g., shared memory or registers).
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))

    # Transpose the loaded tile.
    # `ct.transpose` without explicit axes defaults to swapping the last two dimensions.
    # For a 2D tile of shape (tm, tn), this operation transforms it into a (tn, tm) tile.
    transposed_tile = ct.transpose(input_tile)

    # Store the transposed tile to the output matrix 'y'.
    # Crucially, the store index for `y` must be swapped (`bidy`, `bidx`)
    # because `y` is the transpose of `x`. The `tile` argument provides
    # the `tn` x `tm` data to be written to global memory.
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)


def cutile_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix transposition C = X.T using a cuTile kernel.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `transpose_kernel`.

    Args:
        x (torch.Tensor): The input matrix (M x N).
                          This tensor *must* be 2D and on a CUDA device.

    Returns:
        torch.Tensor: The transposed matrix (N x M) on the same CUDA device.

    Raises:
        ValueError: If the input tensor is not on CUDA or is not 2D.
    """
    # --- Input Validation ---
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if x.ndim != 2:
        raise ValueError("Transpose kernel currently supports only 2D tensors.")

    # --- Get Matrix Dimensions ---
    m, n = x.shape  # Original dimensions of the input matrix: M rows, N columns

    # --- Determine Tile Shapes for Optimization ---
    # These are the tile sizes (`tm`, `tn`) for the *input* matrix `x`.
    # cuTile allows for specifying tile sizes as compile-time constants.
    # For half-precision data types (like `torch.float16` or `torch.bfloat16`,
    # where `itemsize` is 2 bytes), larger tile sizes (e.g., 128x128) might be
    # chosen to leverage Tensor Cores for improved performance.
    # For full-precision (e.g., `torch.float32`, `itemsize` is 4 bytes),
    # smaller, more general tile sizes (e.g., 32x32) are often used.
    if x.dtype.itemsize == 2:  # Likely torch.float16 or torch.bfloat16
        tm, tn = 128, 128  # Example optimal tile sizes for 16-bit data
    else:  # Likely torch.float32 or other
        tm, tn = 32, 32  # Example optimal tile sizes for 32-bit data

    # --- Calculate Grid Dimensions for Kernel Launch (2D Grid) ---
    # The grid defines how many CUDA blocks (CTAs) will be launched.
    # Each block computes one `(tn x tm)` output tile, which corresponds to
    # processing one `(tm x tn)` input tile.
    # `grid_x`: Number of blocks needed along the M dimension (rows of original x).
    # `grid_y`: Number of blocks needed along the N dimension (columns of original x).
    # `ceil(dimension / tile_size)` ensures that enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ceil(m / tm)
    grid_y = ceil(n / tn)
    grid = (grid_x, grid_y, 1)  # cuTile expects a 3-tuple for grid dimensions (x, y, z)

    # --- Create Output Tensor y ---
    # The output tensor `y` will have transposed dimensions (N x M)
    # and the same device and data type as the input `x`.
    y = torch.empty((n, m), device=x.device, dtype=x.dtype)

    # --- Launch the cuTile Kernel ---
    # The `transpose_kernel` is launched on the GPU with the calculated grid dimensions.
    # `tm` and `tn` are passed as Constant integer parameters to the kernel.
    ct.launch(torch.cuda.current_stream(), grid, transpose_kernel, (x, y, tm, tn))

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    print("--- Running cuTile Matrix Transposition Examples ---")

    # Define common matrix dimensions for the examples
    M_dim = 1024
    N_dim = 512

    # --- Test Case 1: float16 (Half-Precision) ---
    print("\n--- Test Case 1: Matrix Transposition with float16 (Half-Precision) ---")
    # Create a random input matrix with float16 data type on the CUDA device.
    x_fp16 = torch.randn(M_dim, N_dim, dtype=torch.float16, device='cuda')
    print(f"Input x shape: {x_fp16.shape}, dtype: {x_fp16.dtype}")

    # Perform transposition using the cuTile wrapper function.
    y_fp16_cutile = cutile_transpose(x_fp16)
    print(f"cuTile Output y shape: {y_fp16_cutile.shape}, dtype: {y_fp16_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_fp16_cutile, x_fp16.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: float32 (Single-Precision) ---
    print("\n--- Test Case 2: Matrix Transposition with float32 (Single-Precision) ---")
    # Create a random input matrix with float32 data type on the CUDA device.
    x_fp32 = torch.randn(M_dim, N_dim, dtype=torch.float32, device='cuda')
    print(f"Input x shape: {x_fp32.shape}, dtype: {x_fp32.dtype}")

    # Perform transposition using the cuTile wrapper function.
    y_fp32_cutile = cutile_transpose(x_fp32)
    print(f"cuTile Output y shape: {y_fp32_cutile.shape}, dtype: {y_fp32_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_fp32_cutile, x_fp32.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 3: Non-square matrix with non-multiple dimensions ---
    print("\n--- Test Case 3: Matrix Transposition with Non-Square, Non-Multiple Dimensions ---")
    # Define matrix dimensions that are not exact multiples of the default tile sizes (32x32).
    # Demonstration that the `ceil` function in grid calculation correctly handles partial tiles.
    M_dim_non_mult = 1000
    N_dim_non_mult = 500
    x_non_mult = torch.randn(M_dim_non_mult, N_dim_non_mult, dtype=torch.float32, device='cuda')
    print(f"Input x shape: {x_non_mult.shape}, dtype: {x_non_mult.dtype}")

    y_non_mult_cutile = cutile_transpose(x_non_mult)
    print(f"cuTile Output y shape: {y_non_mult_cutile.shape}, dtype: {y_non_mult_cutile.dtype}")
    if args.correctness_check:
        torch.testing.assert_close(y_non_mult_cutile, x_non_mult.T)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- All cuTile matrix transposition examples completed. ---")
