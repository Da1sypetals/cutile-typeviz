# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from conftest import dtype_id, shape_id

import pytest
import torch
import cuda.tile as ct
from math import ceil
from util import estimate_bench_iter
from kernels.rms_norm import (
    rms_norm_kernel, rms_norm_kernel_gather, rms_norm_kernel_static_persistent
)
from autotuner.autotuner import Autotuner, Config, SearchSpace, autotune


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


@pytest.fixture(params=[
    (262144, 1024),
    (262144, 2048),
    (262144, 4096),
    (65536, 8192),
    (65536, 16384),
], ids=shape_id)
def shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.float32, torch.bfloat16
], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='rms_norm')
@pytest.mark.parametrize('static_persistent', [True, False])
@pytest.mark.parametrize('gather', [True, False])
def bench_rms_norm(shape, dtype, static_persistent, gather, backend, benchmark):
    if backend is cutile_autotune_rms_norm and static_persistent:
        pytest.xfail("cutile autotune backend on static persistent mode hangs on arm64")
    if shape[1] == 16384:
        pytest.xfail("It uses too much memory and hangs. This previously created a PTXAS Error")
    if backend is torch_rms_norm and (static_persistent or gather):
        pytest.skip("torch backend does not distinguish between standard and static persistent")
    if static_persistent and gather:
        pytest.skip("static persistent does not support gather mode")
    x_shape = shape
    w_shape = (shape[1], )
    x = torch.rand(x_shape, dtype=dtype, device="cuda")
    weight = torch.randn(w_shape, dtype=dtype, device="cuda")

    eps = 1e-5

    o = backend(x, weight, eps, static_persistent, gather)
    ref = ref_rms_norm(x, weight, eps)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(
        backend, (x, weight, eps, static_persistent, gather),
    )

    benchmark.pedantic(
        backend, (x, weight, eps, static_persistent, gather),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
    )

    M, N = x.shape
    flop_count = M * (4 * N + 2)
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (x, weight, o)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def cutile_rms_norm(x, weight, eps, static_persistent, gather):
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)
    M, N = x.shape

    if static_persistent:
        NUM_SMS = torch.cuda.get_device_properties(
            "cuda"
        ).multi_processor_count
        TILE_SIZE_M = 4  # Default value, could be made configurable
        TILE_SIZE_N = next_power_of_2(N)

        # Other tile sizes are more optimal when other dimension is too large/too small
        if TILE_SIZE_N <= 1024:
            TILE_SIZE_M = 16
        elif TILE_SIZE_N >= 16384:
            TILE_SIZE_M = 2

        grid_size = min(
            NUM_SMS,
            ceil(M / TILE_SIZE_M) * ceil(N / TILE_SIZE_N),
        )
        grid = (grid_size,)
        ct.launch(torch.cuda.current_stream(), grid, rms_norm_kernel_static_persistent, (
            x,
            y,
            weight,
            TILE_SIZE_M,
            TILE_SIZE_N,
            eps,
        ))
    else:
        # Standard RMSNorm kernel
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        MAX_FUSED_SIZE = 2048 // x.element_size()
        TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
        grid = (M,)
        kernel = rms_norm_kernel_gather if gather else rms_norm_kernel
        ct.launch(torch.cuda.current_stream(), grid, kernel, (
            x,
            weight,
            y,
            rstd,
            N,
            eps,
            TILE_SIZE,
        ))
    return y.view(*x.shape)


def _static_persistent_autotune_grid(X, TILE_SIZE_M, TILE_SIZE_N):
    """Grid function for static persistent RMS Norm autotuning"""
    NUM_SMS = torch.cuda.get_device_properties(
        "cuda"
    ).multi_processor_count
    M, N = X.shape[0], X.shape[1]
    grid_size = min(
        NUM_SMS,
        ceil(M / TILE_SIZE_M) * ceil(N / TILE_SIZE_N),
    )
    return (grid_size,)


def _static_persistent_autotune_configs():
    """Get autotune configurations for RMS Norm kernel"""
    configs = [
        Config(
            TILE_SIZE_M=ts_m,
            TILE_SIZE_N=ts_n,
            num_ctas=s,
            occupancy=w,
        )
        for ts_m in [2, 4, 8, 16]
        for ts_n in [2**9, 2**10, 2**11, 2**12, 2**13, 2**14]
        for s in [1, 2]
        for w in [1, 2, 4, 8, 16, 32]
    ]
    return configs


def _static_persistent_autotune_predicate(X, TILE_SIZE_N):
    """Predicate function for static persistent RMS Norm autotuning"""
    return X.shape[1] * 2 > TILE_SIZE_N >= X.shape[1]


def _standard_autotune_grid(x):
    """Grid function for standard RMS Norm autotuning"""
    return (x.shape[0],)


def _standard_autotune_configs():
    """Get autotune configurations for RMS Norm kernel"""
    configs = [
        Config(
            TILE_SIZE=ts,
            num_ctas=s,
            occupancy=w,
        )
        for ts in [2**7, 2**8, 2**9, 2**10, 2**11, 2**12]
        for s in [1, 2]
        for w in [1, 2, 4, 8, 16, 32]
    ]
    return configs


@autotune(
    search_space=SearchSpace(
        _static_persistent_autotune_configs(),
        predicate_fn=_static_persistent_autotune_predicate,
    ),
)
def _rms_norm_static_persistent_base(stream, x, y, weight, eps, autotuner: Autotuner):
    autotuner(
        stream,
        grid_fn=_static_persistent_autotune_grid,
        kernel=rms_norm_kernel_static_persistent,
        args_fn=lambda TILE_SIZE_M, TILE_SIZE_N: (x, y, weight, TILE_SIZE_M, TILE_SIZE_N, eps),
        max_iter=20,  # reduce the number of configurations to test for faster benchmarking.
    )
    return y


@autotune(
    search_space=_standard_autotune_configs(),
)
def _rms_norm_standard_gather_base(stream, x, weight, y, rstd, N, eps, autotuner: Autotuner):
    autotuner(
        stream,
        grid_fn=_standard_autotune_grid,
        kernel=rms_norm_kernel_gather,
        args_fn=lambda TILE_SIZE: (x, weight, y, rstd, N, eps, TILE_SIZE),
        max_iter=20,  # reduce the number of configurations to test for faster benchmarking.
    )
    return y


@autotune(
    search_space=_standard_autotune_configs(),
)
def _rms_norm_standard_tiled_base(stream, x, weight, y, rstd, N, eps, autotuner: Autotuner):
    autotuner(
        stream,
        grid_fn=_standard_autotune_grid,
        kernel=rms_norm_kernel,
        args_fn=lambda TILE_SIZE: (x, weight, y, rstd, N, eps, TILE_SIZE),
        max_iter=20,  # reduce the number of configurations to test for faster benchmarking.
    )
    return y


def cutile_autotune_rms_norm(x, weight, eps, static_persistent, gather):
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)
    M, N = x.shape

    if static_persistent:
        _rms_norm_static_persistent_base(torch.cuda.current_stream(), x, y, weight, eps)
    else:
        # Standard RMSNorm kernel
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        if gather:
            _rms_norm_standard_gather_base(torch.cuda.current_stream(), x, weight, y, rstd, N, eps)
        else:
            _rms_norm_standard_tiled_base(
                torch.cuda.current_stream(), x, weight, y, rstd, N, eps
            )
    return y.view(*x.shape)


def torch_rms_norm(input, weight, eps, static_persistent=False, gather=False):
    # layer norm should always be calculated in float32
    normalized_shape = weight.shape
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


def ref_rms_norm(input, weight, eps):
    return torch_rms_norm(input, weight, eps)
