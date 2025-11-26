# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import cuda.tile as ct
import math
import re
from typing import Any
from util import assert_equal

import autotuner.autotuner as autotuner_mod
from autotuner.autotuner import Autotuner, Config, SearchSpace, autotune
from cuda.tile._cext import default_tile_context


@ct.kernel
def inplace_kernel(
    x,
    TILE_SIZE: ct.Constant[int]
):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    tx_updated = tx + 1
    ct.store(x, index=(bid,), tile=tx_updated)


configs = [Config(TILE_SIZE=64), Config(TILE_SIZE=128)]


@ct.kernel
def dummy_kernel(x, TILE_SIZE: ct.Constant[int]):
    pass


def grid_fn(x, TILE_SIZE):
    return (math.ceil(x.shape[0] / TILE_SIZE), 1, 1)


# ========== Test Predicate Filters ==========#
def test_predicate_filters_all():
    tuner = Autotuner(SearchSpace(configs, predicate_fn=lambda x, TILE_SIZE: False))

    x = torch.empty((256,), device='cuda')
    with pytest.raises(ValueError, match=r"No valid config"):
        tuner(
            stream=torch.cuda.current_stream(),
            grid_fn=grid_fn,
            kernel=inplace_kernel,
            args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
        )


# ========== Test Autotune Raises on Invalid Grid Function / Args Function Parameters ==========#
def test_autotune_raises_on_invalid_grid_function_parameters():
    tuner = Autotuner(configs)
    x = torch.empty((1024,), device="cuda")
    match = re.escape("Function parameter TILE_SIZE0 in grid_fn is not in kernel parameters, "
                      "available parameters are ['x', 'TILE_SIZE']")
    with pytest.raises(TypeError, match=match):
        tuner(
            stream=torch.cuda.current_stream(),
            grid_fn=lambda x, TILE_SIZE0: (math.ceil(x.shape[0] / TILE_SIZE0), 1, 1),
            kernel=dummy_kernel,
            args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
        )


def test_autotune_raises_on_invalid_args_function_parameters():
    tuner = Autotuner(configs)
    x = torch.empty((1024,), device="cuda")
    match = re.escape("Invalid parameters for args_fn, "
                      "should be the same as the search space config argument keys: ['TILE_SIZE']")
    with pytest.raises(TypeError, match=match):
        tuner(
            stream=torch.cuda.current_stream(),
            grid_fn=lambda x, TILE_SIZE: (math.ceil(x.shape[0] / TILE_SIZE), 1, 1),
            kernel=dummy_kernel,
            args_fn=lambda TILE_SIZE0: (x, TILE_SIZE0),
        )


# ========== Test Autotune Allows Keyword Only Parameters ==========#
@ct.kernel
def k_kwonly(x, y, *, TILE_SIZE: ct.Constant[int]):
    bid = ct.bid(0)
    t = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    ct.store(y, index=(bid,), tile=t + 1)


def test_autotune_allows_keyword_only_param_and_runs():
    tuner = Autotuner(configs)

    x = torch.empty((1024,), device="cuda")
    y = torch.zeros_like(x)
    tuner(
        stream=torch.cuda.current_stream(),
        grid_fn=grid_fn,
        kernel=k_kwonly,
        args_fn=lambda TILE_SIZE: (x, y, TILE_SIZE),
    )
    assert_equal(y, x + 1)


@pytest.fixture
def _patch_timer_and_launch(monkeypatch):
    calls = {"count": 0}

    def fake_time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
        calls["count"] += 1
        return 1

    monkeypatch.setattr(autotuner_mod, "_time_ms", fake_time_ms, raising=True)
    monkeypatch.setattr(ct, "launch", lambda *a, **k: None, raising=True)
    return calls


# ========== Test Clear Cache ==========#
def test_clear_cache(_patch_timer_and_launch):
    tuner = autotuner_mod.Autotuner(configs)
    x = torch.empty((256,), device="cuda")

    def args_fn(TILE_SIZE):
        return (x, TILE_SIZE)

    # 1) First tune
    tuner(torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn)
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0, "Expected timing to run on first tune (cache miss)"

    # 2) Second tune with same args → cache hit (no new timings)
    tuner(torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn)
    second_count = _patch_timer_and_launch["count"]
    assert second_count == first_count, "Expected cache hit: no additional timing calls"

    # 3) Clear entire cache → next tune should re-benchmark
    tuner.clear_cache()
    tuner(torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn)
    third_count = _patch_timer_and_launch["count"]
    assert third_count > second_count, "Expected timing to run after clear_cache()"

    # 4) Clear by key only
    key = autotuner_mod._default_key(dummy_kernel, args_fn(**configs[0].kwargs))
    tuner(torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn)
    before_key_clear = _patch_timer_and_launch["count"]
    tuner.clear_cache(key)
    tuner(torch.cuda.current_stream(), grid_fn, dummy_kernel, args_fn)
    after_key_clear = _patch_timer_and_launch["count"]
    assert after_key_clear > before_key_clear, "Expected re-tune after clear_cache(key)"


# ========== Test custom key function ==========#
@ct.kernel
def kernel_with_scalar_value(x, scalar_value, TILE_SIZE: ct.Constant[int]):
    pass


def test_default_key_includes_scalar_value(_patch_timer_and_launch):
    tuner = autotuner_mod.Autotuner(configs)
    x = torch.empty((256,), device="cuda")

    def custom_key_with_scalar_value(_, args: tuple[Any, ...]):
        scalar_value = args[1]
        return (scalar_value,)

    # 1) First tune
    tuner(
        torch.cuda.current_stream(),
        grid_fn,
        kernel_with_scalar_value,
        args_fn=lambda TILE_SIZE: (x, 0.0, TILE_SIZE),
        key_fn=custom_key_with_scalar_value,
    )
    first_count = _patch_timer_and_launch["count"]
    assert first_count > 0, "Expected timing to run on first tune (cache miss)"

    # 2) Second tune with same args → cache hit (no new timings)
    tuner(
        torch.cuda.current_stream(),
        grid_fn,
        kernel_with_scalar_value,
        args_fn=lambda TILE_SIZE: (x, 0.0, TILE_SIZE),
        key_fn=custom_key_with_scalar_value,
    )
    second_count = _patch_timer_and_launch["count"]
    assert second_count == first_count, "Expected cache hit: no additional timing calls"

    # 3) Different scalar value -> cache miss (re-tune)
    tuner(
        torch.cuda.current_stream(),
        grid_fn,
        kernel_with_scalar_value,
        args_fn=lambda TILE_SIZE: (x, 1.0, TILE_SIZE),
        key_fn=custom_key_with_scalar_value,
    )
    third_count = _patch_timer_and_launch["count"]
    assert third_count > second_count, "Expected timing to run after scalar value change"


# ========== Test Arg Policy: custom transforms ==========#
def test_custom_transforms(monkeypatch):
    # Record the packed args passed to ct.launch
    launches = []
    monkeypatch.setattr(ct, "launch", lambda *a: launches.append(a), raising=True)

    x = torch.empty((256,), device="cuda")
    # Custom value: a recognizable tensor
    custom_x = torch.full_like(x, 7)

    tuner = autotuner_mod.Autotuner(configs)
    tuned_result = tuner(
        stream=torch.cuda.current_stream(),
        grid_fn=grid_fn,
        kernel=dummy_kernel,
        args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
        transforms={"x": lambda x: custom_x},
    )

    # At least two launches should have occurred
    assert len(launches) >= 2, "ct.launch was not called during tuning"

    # Check that the y argument passed to launch is our scratch (not the real y)
    # packed order for dummy_kernel is (x, TILE_SIZE)
    # Notice the last launch is the one with the best config so it does not run our custom transform
    _, _, _, packed_tune = launches[-2]  # (stream, grid, kernel, packed_args)
    assert packed_tune[0] is custom_x
    assert packed_tune[0] is not x

    _, _, _, packed_best = launches[-1]  # (stream, grid, kernel, packed_args)
    assert packed_best[0] is x

    # Then test the tuned result - we can still use reguar ct.launch with the tuned result
    num_launches = len(launches)
    ct.launch(
        torch.cuda.current_stream(),
        tuned_result.grid,
        tuned_result.kernel,
        (x, tuned_result.TILE_SIZE)
    )
    assert len(launches) == num_launches + 1


# ========== Test timeout / failed configs handling ==========#
def test_autotune_handles_timeout_and_raises_when_all_configs_fail(monkeypatch, caplog):
    old_timeout = default_tile_context.config.compiler_timeout_sec

    tuner = autotuner_mod.Autotuner(configs)
    x = torch.empty((256,), device="cuda")

    def fake_time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
        if default_tile_context.config.compiler_timeout_sec <= 1:
            raise RuntimeError("simulated compiler timeout")
        return 1

    monkeypatch.setattr(autotuner_mod, "_time_ms", fake_time_ms, raising=True)

    # No timeout
    with caplog.at_level("DEBUG"):
        tuner(
            stream=torch.cuda.current_stream(),
            grid_fn=grid_fn,
            kernel=dummy_kernel,
            args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
        )
    assert "failed to run" not in caplog.text

    # Timeout
    caplog.clear()
    with caplog.at_level("DEBUG", logger=autotuner_mod.logger.name):
        with pytest.raises(ValueError, match=r"No valid config"):
            tuner(
                stream=torch.cuda.current_stream(),
                grid_fn=grid_fn,
                kernel=dummy_kernel,
                args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
                compiler_time_limit_sec=1,
                force_retune=True,
            )
    assert "failed to run" in caplog.text
    # Make sure the timeout is restored
    assert default_tile_context.config.compiler_timeout_sec == old_timeout


# ========== Real use case with decorator: test Inplace Plus One with clone policy ==========#
@autotune(
    search_space=SearchSpace(configs),
)
def inplace_plus_one_base(stream, x, autotuner: Autotuner):
    autotuner(
        stream,
        grid_fn=grid_fn,
        kernel=inplace_kernel,
        args_fn=lambda TILE_SIZE: (x, TILE_SIZE),
        transforms={"x": lambda x: x.clone()}
    )
    return x


def test_inplace_plus_one():
    x = torch.empty((1024,), device="cuda")
    original_x = x.clone()
    inplace_plus_one_base(torch.cuda.current_stream(), x)
    assert_equal(x, original_x + 1)
