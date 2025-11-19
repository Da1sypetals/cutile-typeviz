# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import math

from test.kernels.fft import fft_kernel


# --- Helper function for generating DFT matrices (W-matrices) ---
def twiddles(rows: int, cols: int, factor: int, device: torch.device, precision: torch.dtype):
    """
    Generates a matrix of complex exponentials ($$W_{\text{factor}}^{i dot j}$$),
    which are the core components of Discrete Fourier Transform (DFT) matrices.
    Returns it as an interleaved real/imaginary tensor.

    Args:
        rows (int): Number of rows for the generated matrix.
        cols (int): Number of columns for the generated matrix.
        factor (int): The factor used in the complex exponential denominator.
        device (torch.device): The PyTorch device to create the tensor on (e.g., 'cuda').
        precision (torch.dtype): The desired floating-point precision (e.g., torch.float32).

    Returns:
        torch.Tensor: A tensor of shape (rows, cols, 2) with interleaved real and
                      imaginary parts of the complex exponential matrix.
    """
    # Create 2D grids for indices I and J using torch.meshgrid.
    I, J = torch.meshgrid(torch.arange(rows, device=device),
                          torch.arange(cols, device=device),
                          indexing='ij')
    # Compute the complex exponential based on the FFT formula.
    W_complex = torch.exp(-2 * math.pi * 1j * (I * J) / factor)
    # Convert the complex tensor to a real tensor where the last dimension
    # stores the real and imaginary parts (e.g., [real_part, imag_part]).
    return torch.view_as_real(W_complex).to(precision).contiguous()


def make_twiddles(decomp: tuple, precision: torch.dtype, device: torch.device):
    """
    Generates mathematically correct W (rotation) and T (twiddle) matrices
    for a multi-dimensional FFT decomposition, with interleaved real/imaginary parts.

    These matrices are pre-computed on the host (CPU) and then transferred to the GPU
    for use by the cuTile kernel, avoiding re-computation on the device for each batch.

    Args:
        decomp (tuple): A tuple (F0, F1, F2) representing the factors of N,
                        where N is the total FFT size.
        precision (torch.dtype): The desired floating-point precision for the matrices.
        device (torch.device): The PyTorch device to create the tensors on.

    Returns:
        tuple: A tuple containing (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri),
               where each element is a tensor with interleaved real/imaginary parts.
    """
    F0, F1, F2 = decomp
    N = F0 * F1 * F2  # Total FFT size, product of factors
    F1F2 = F1 * F2   # Product of F1 and F2, used for T0 matrix dimensions

    # Generate W matrices (rotation matrices for each dimension's DFT).
    W0_ri = twiddles(F0, F0, F0, device, precision)
    W1_ri = twiddles(F1, F1, F1, device, precision)
    W2_ri = twiddles(F2, F2, F2, device, precision)

    # Generate T matrices (twiddle factors for dimension transitions/permutations).
    # T0 applies across F0 and F1F2, with N as the overall factor.
    T0_ri = twiddles(F0, F1F2, N, device, precision)
    # T1 applies across F1 and F2, with F1F2 as the factor.
    T1_ri = twiddles(F1, F2, F1F2, device, precision)

    return (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri)


# --- Wrapper function to launch the fft_kernel ---
def cutile_fft(
    x: torch.Tensor,
    factors: tuple,  # (F0, F1, F2) - factors of N
    atom_packing_dim: int = 64  # The 'D' parameter for data packing/unpacking
) -> torch.Tensor:
    """
    Performs a Batched 1D Fast Fourier Transform (FFT) using a cuTile kernel
    based on multi-dimensional factorization (similar to a Cooley-Tukey algorithm).

    This function prepares the input data, generates necessary pre-computed
    matrices, launches the cuTile kernel, and unpacks the results.

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, N) containing complex64 numbers.
                          This tensor *must* be on a CUDA device.
                          N (the FFT size) must be factorable into factors[0]*factors[1]*factors[2].
        factors (tuple): A tuple (F0, F1, F2) representing the factors of N.
                         The product F0 * F1 * F2 must equal N. These factors define
                         the logical 3D shape for the FFT decomposition within the kernel.
        atom_packing_dim (int): The dimension 'D' used for data packing/unpacking
                                in the kernel. This value affects memory access patterns.
                                The total number of real/imaginary elements (N*2) must be
                                divisible by this dimension. Default is 64.

    Returns:
        torch.Tensor: Output tensor of shape (Batch, N) containing the FFT results.
                      The output data type will be torch.complex64.

    Raises:
        ValueError: If input tensor dimensions, device, or data type are incorrect,
                    if the provided factors do not multiply to N, or if N*2 is not
                    divisible by atom_packing_dim.
    """
    # --- Input Validation ---
    if x.ndim != 2:
        raise ValueError("Input tensor must be 2D (Batch, N).")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    if x.dtype != torch.complex64:
        raise ValueError("Input tensor dtype must be torch.complex64.")

    BS = x.shape[0]  # Extract Batch Size from the input tensor's shape.
    N = x.shape[1]   # Extract Total FFT size from the input tensor's shape.

    F0, F1, F2 = factors
    # Validate that the provided factors correctly decompose the total FFT size N.
    if F0 * F1 * F2 != N:
        raise ValueError(f"Factors ({F0}*{F1}*{F2}={F0*F1*F2}) do not multiply to N={N}. "
                         f"Please provide factors that correctly decompose N.")

    # Determine the underlying floating-point precision (e.g., float32) for
    # the real and imaginary parts of the complex numbers.
    PRECISION_DTYPE = x.real.dtype

    # --- Prepare Input Data for Kernel (Split real/imag, pack) ---
    # Convert the complex input tensor (BS, N) to a real tensor (BS, N, 2)
    # where the last dimension explicitly separates real and imaginary parts.
    x_ri = torch.view_as_real(x)

    # Reshape the real/imaginary tensor to the packed format (BS, N*2 // D, D)
    # that the kernel expects for efficient memory access.
    # This step assumes that the total number of real/imaginary elements (N*2)
    # is perfectly divisible by the `atom_packing_dim` (D).
    if (N * 2) % atom_packing_dim != 0:
        raise ValueError(f"Total real/imag elements (N*2 = {N*2}) must be divisible by "
                         f"atom_packing_dim ({atom_packing_dim}) for kernel packing.")
    x_packed_in = x_ri.reshape(BS, N * 2 // atom_packing_dim, atom_packing_dim).contiguous()

    # --- Generate W (Rotation) and T (Twiddle) Matrices ---
    # These matrices are pre-computed mathematically based on the FFT decomposition.
    # They are generated on the same device as the input tensor (CUDA) to avoid
    # costly host-to-device transfers during kernel execution.
    W0_gmem, W1_gmem, W2_gmem, T0_gmem, T1_gmem = make_twiddles(factors, PRECISION_DTYPE, x.device)

    # --- Create Output Tensor ---
    # Initialize an empty tensor with the same shape and properties as the packed input.
    # This tensor will store the results computed by the kernel.
    y_packed_out = torch.empty_like(x_packed_in)

    # --- Calculate Grid Dimensions ---
    # For this FFT kernel, one thread block is launched for each item in the batch.
    # The grid is a 3-tuple (grid_x, grid_y, grid_z).
    grid = (BS, 1, 1)

    # --- Launch the cuTile Kernel ---
    # The `fft_kernel` is launched on the GPU with the calculated grid dimensions.
    # All necessary input tensors (packed data, W and T matrices) and constant parameters
    # (N, F0, F1, F2, BS, D) are passed to the kernel.
    ct.launch(torch.cuda.current_stream(), grid, fft_kernel,
              (x_packed_in, y_packed_out,
               W0_gmem, W1_gmem, W2_gmem,
               T0_gmem, T1_gmem,
               N, F0, F1, F2, BS, atom_packing_dim))

    # --- Unpack Output from Kernel (Reshape, combine real/imag) ---
    # Reshape the packed output tensor back to (BS, N, 2) to separate real/imaginary parts.
    y_ri = y_packed_out.reshape(BS, N, 2)
    # Convert the real/imaginary pair tensor back to a complex tensor (torch.complex64).
    y_complex = torch.view_as_complex(y_ri)

    return y_complex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()
    print("--- Running cuTile FFT Example ---")

    # --- User Configuration ---
    BATCH_SIZE = 2
    # Total FFT size (N) must be factorable into factors[0] * factors[1] * factors[2].
    # For example, N = 1024 can be factored as (8, 16, 8).
    FFT_SIZE = 8  # A smaller FFT size for demonstration purposes.
    FFT_FACTORS = (2, 2, 2)  # Factors for N=8 (2 * 2 * 2 = 8).
    # The 'D' parameter for data packing/unpacking in the kernel.
    # For N=8, N*2=16. ATOM_PACKING_DIM=2 is a valid divisor (16 % 2 == 0).
    ATOM_PACKING_DIM = 2

    # Data type for the real/imaginary components (e.g., float32 for complex64).
    PRECISION_DTYPE = torch.float32

    # --- Create Sample Input Data ---
    # Generate a random input tensor of complex64 numbers, placed on the CUDA device.
    # `torch.manual_seed(0)` ensures reproducibility of the random numbers.
    torch.manual_seed(0)
    input_data_complex = torch.randn(BATCH_SIZE, FFT_SIZE, dtype=torch.complex64, device='cuda')

    print("  Configuration:")
    print(f"  FFT Size (N): {FFT_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  FFT Factors (F0,F1,F2): {FFT_FACTORS}")
    print(f"  Atom Packing Dimension (D): {ATOM_PACKING_DIM}")
    print(f"Input data shape: {input_data_complex.shape}, dtype: {input_data_complex.dtype}")

    # Perform FFT using the custom cuTile kernel.
    output_fft_cutile = cutile_fft(
        x=input_data_complex,
        factors=FFT_FACTORS,
        atom_packing_dim=ATOM_PACKING_DIM
    )
    print(
        f"""\ncuTile FFT Output shape: {output_fft_cutile.shape},
        dtype: {output_fft_cutile.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(output_fft_cutile, torch.fft.fft(input_data_complex, axis=-1))
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile FFT example execution complete ---")
