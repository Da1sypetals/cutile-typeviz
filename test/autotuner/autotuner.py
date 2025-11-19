# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import functools
from typing import Callable, Sequence
import cuda.tile as ct
from cuda.tile._execution import TileDispatcher
import random
import torch
import inspect
import logging


logger = logging.getLogger(__name__)


class Config:
    """One kernel variant: meta-params in kwargs (e.g., TILE)."""
    def __init__(self, *, num_ctas=None, occupancy=None, opt_level=3, **kwargs):
        self.kwargs = dict(kwargs)
        self.num_ctas = num_ctas
        self.occupancy = occupancy
        self.opt_level = opt_level

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}={v}")
        res.append(f"num_ctas={self.num_ctas}")
        res.append(f"occupancy={self.occupancy}")
        res.append(f"opt_level={self.opt_level}")
        return f"Config({', '.join(res)})"


class SearchSpace:
    def __init__(self, configs: list[Config], predicate_fn: Callable | None = None):
        if len(configs) <= 1:
            raise ValueError(
                "At least two configurations in the search space are required for autotuning"
            )
        self.kwargs_keys = set(configs[0].kwargs.keys())
        for config in configs[1:]:
            if set(config.kwargs.keys()) != self.kwargs_keys:
                raise ValueError(
                    "All configurations must have the same set of keyword arguments"
                )
        self.configs = configs
        self.predicate_fn = predicate_fn

    def __iter__(self):
        return iter(self.configs)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, index):
        return self.configs[index]

    def filter(self, named_args: dict[str, Any]) -> bool:
        if self.predicate_fn is None:
            return True
        predicate_sig = inspect.signature(self.predicate_fn)
        predicate_keys = set(predicate_sig.parameters.keys())
        kwargs = {k: named_args[k] for k in predicate_keys}
        return self.predicate_fn(**kwargs)


def _shape_dtype_stride(arg: Any) -> tuple[tuple[int, ...], str, tuple[int, ...] | None]:
    shape = tuple(arg.shape)
    dtype = arg.dtype
    stride = None
    if hasattr(arg, "stride"):                     # PyTorch, etc.
        s = arg.stride() if callable(arg.stride) else arg.stride
        stride = tuple(int(x) for x in s)
    elif hasattr(arg, "strides"):                  # NumPy, etc. (bytes)
        itemsize = getattr(arg, "itemsize", 1)
        stride = tuple(int(b // itemsize) for b in arg.strides)

    return shape, dtype, stride


def _default_key(kernel: TileDispatcher, args: tuple[Any, ...]):
    """Default cache key for autotune.
    The key(for now) is a tuple of:
    - kernel function name
    - tuple of (shape, dtype, stride) for each argument in the runtime argument (tensor),
    - or its type name for each argument in the runtime argument (other types).
    """
    tinfo = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            shape, dtype, stride = _shape_dtype_stride(arg)
            tinfo.append((shape, dtype, stride))
        else:
            tinfo.append(type(arg).__name__)
    return (kernel._pyfunc.__name__, tuple(tinfo))


def _time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
    stream.synchronize()
    for _ in range(warmup):
        run_once(get_args())

    args_per_run = [get_args() for _ in range(rep)]
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for i in range(rep):
        run_once(args_per_run[i])
    end.record(stream)
    end.synchronize()

    ms = start.elapsed_time(end)
    return ms / max(1, rep)


def _get_grid(grid_fn, named_args: dict[str, Any]) -> tuple[int, ...]:
    grid_sig = inspect.signature(grid_fn)
    grid_keys = set(grid_sig.parameters.keys())
    kwargs = {}
    for k in grid_keys:
        if k not in named_args:
            raise TypeError(
                f"Function parameter {k} in grid_fn is not in kernel parameters, "
                f"available parameters are {list(named_args.keys())}"
            )
        kwargs[k] = named_args[k]
    return grid_fn(**kwargs)


@dataclass
class TunedResult:
    # The tuned parameters
    tuned_params: dict[str, Any]
    # The grid to be used for launching the kernel
    grid: tuple[int, ...]
    # The updated tile dispatcher to be used for launching the kernel
    kernel: TileDispatcher
    num_ctas: int
    occupancy: int
    opt_level: int

    def __getattr__(self, name):
        if name in self.tuned_params:
            return self.tuned_params[name]
        raise AttributeError(f"Attribute {name} not found in {self.tuned_params}")


def _make_trial_args(
    args_fn: Callable, kwargs: dict[str, Any], kernel, transforms: dict[str, Callable[[Any], Any]]
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Make trial runtime arguments applying the transforms."""
    args = args_fn(**kwargs)

    trial_named_args = {}
    trial_args = []
    kernel_sig = inspect.signature(kernel._pyfunc)
    for kernel_key, arg in zip(kernel_sig.parameters.keys(), args, strict=True):
        if kernel_key in transforms:
            trial_named_args[kernel_key] = transforms[kernel_key](arg)
        else:
            trial_named_args[kernel_key] = arg
        trial_args.append(trial_named_args[kernel_key])
    return trial_named_args, tuple(trial_args)


def _normalize_search_space(space: SearchSpace | Sequence[Config]) -> SearchSpace:
    if isinstance(space, SearchSpace):
        return space

    # Allow sequence of Configs
    if isinstance(space, Sequence) and all(isinstance(c, Config) for c in space):
        return SearchSpace(list(space))

    raise TypeError(
        "search_space must be a SearchSpace, or a sequence of Configs"
    )


def _safe_args_fn(args_fn: Callable, kwargs: dict[str, Any]) -> tuple[Any, ...]:
    try:
        return args_fn(**kwargs)
    except TypeError:
        raise TypeError(
            f"Invalid parameters for args_fn, "
            f"should be the same as the search space config argument keys: {list(kwargs.keys())}"
        )


class Autotuner:
    def __init__(self, search_space: SearchSpace | Sequence[Config]):
        self._search_space = _normalize_search_space(search_space)
        self._cache = {}

    def clear_cache(self, key=None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __call__(self,
                 stream, grid_fn, kernel,
                 args_fn: Callable,
                 transforms: dict[str, Callable] = {},
                 *,
                 key_fn=_default_key,
                 max_iter: int = 60,
                 time_limit_ms: int = 30_000,
                 seed: int | None = None,
                 force_retune: bool = False) -> TunedResult:
        """
        It performs the following steps:
        1) picks or reuses the cached config and kernel,
        2) runs the kernel with the best config,
        3) returns the tuned result.

        Args:
            stream: The stream.
            grid_fn: The grid function.
            kernel: The kernel.
            args_fn: The function from the search space parameters to the runtime arguments.
            transforms: The transforms functions for runtime arguments if needed.
            key_fn: The key function.
            max_iter: The maximum number of valid condigurations to sample from the search space.
            time_limit_ms: The time limit for each kernel run (including compilation)
                           in milliseconds. Default is 30000ms (30s).
            seed: The seed for the random number generator. Default is None.
            force_retune: Force retuning even if the config is found in the cache. Default is False.
        """
        key = key_fn(kernel, _safe_args_fn(args_fn, self._search_space.configs[0].kwargs))
        if not force_retune and key in self._cache:
            best_idx, best_grid, best_kernel = self._cache[key]
            logger.debug(f"Using cached config for key {key}: {self._search_space[best_idx]}")
        else:
            rng = random.Random(seed)
            indices = rng.sample(range(len(self._search_space)), len(self._search_space))
            best_time_ms, best_idx, best_kernel = float("inf"), None, None
            successes = 0
            for cfg_idx in indices:
                if successes >= max_iter:
                    break
                cfg = self._search_space[cfg_idx]
                trial_named_args, trial_args = _make_trial_args(
                    args_fn, cfg.kwargs, kernel, transforms
                )
                if not self._search_space.filter(trial_named_args):
                    logger.debug(f"Config {cfg} filtered out by predicate function")
                    continue

                grid = _get_grid(grid_fn, trial_named_args)
                updated_kernel = ct.kernel(
                    kernel._pyfunc,
                    num_ctas=cfg.num_ctas,
                    occupancy=cfg.occupancy,
                    opt_level=cfg.opt_level
                )

                def run_once(args):
                    ct.launch(stream, grid, updated_kernel, args)

                try:
                    time_ms = _time_ms(
                        run_once,
                        get_args=lambda: _make_trial_args(args_fn, cfg.kwargs, kernel, transforms)[1], # noqa
                        stream=stream,
                    )
                except Exception as e:
                    logger.debug(f"{cfg} failed to run: {e}")
                    continue

                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_idx, best_grid, best_kernel = cfg_idx, grid, updated_kernel
                    logger.debug(
                        f"Iteration {successes} updated best config to {cfg}: {best_time_ms} ms"
                    )
                successes += 1

            # Save the best config and kernel.
            if best_idx is None:
                raise ValueError("No valid config found")
            self._cache[key] = (best_idx, best_grid, best_kernel)

        best_cfg = self._search_space[best_idx]

        # Use the original runtime arguments to run the kernel with the best config
        best_packed_args = args_fn(**best_cfg.kwargs)
        ct.launch(stream, best_grid, best_kernel, best_packed_args)

        # Return the tuned result
        return TunedResult(best_cfg.kwargs, best_grid, best_kernel,
                           num_ctas=best_cfg.num_ctas,
                           occupancy=best_cfg.occupancy,
                           opt_level=best_cfg.opt_level)


def autotune(search_space):

    def decorator(func):
        tuner = Autotuner(search_space)   # single, device-agnostic instance

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inject the tuner into the function arguments
            kwargs.setdefault("autotuner", tuner)
            return func(*args, **kwargs)

        return wrapper

    return decorator
