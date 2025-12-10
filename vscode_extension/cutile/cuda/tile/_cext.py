# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Mock _cext module for compilation without CUDA driver.
This module provides stub implementations that allow IR generation
without actually calling CUDA C libraries.
"""

from cuda.tile._context import TileContextConfig


def launch(stream, grid, kernel, kernel_args, /):
    """Mock launch function - does not actually launch kernel"""
    raise RuntimeError(
        "Cannot launch kernel: CUDA driver not available. This mock module only supports IR generation."
    )


class TileDispatcher:
    """Mock TileDispatcher class"""

    def __new__(cls, *args, **kwargs):
        # 正确处理 __new__ 方法
        instance = object.__new__(cls)
        return instance

    def __init__(self, arg_constant_flags, compile_func):
        self._arg_constant_flags = arg_constant_flags
        self._compile_func = compile_func


class TileContext:
    """Mock TileContext class"""

    def __init__(self, config=None):
        if config is None:
            from cuda.tile._context import init_context_config_from_env

            config = init_context_config_from_env()
        self._config = config

    @property
    def config(self):
        return self._config


# Create default tile context
default_tile_context = TileContext()


def get_compute_capability():
    """Mock get_compute_capability - returns a default value"""
    # Return Hopper (9.0) as default
    # Users can modify this if needed
    return (9, 0)


class ArraySpecialization:
    """Mock ArraySpecialization class for array type analysis"""

    def __init__(self, base_ptr, dtype_bitwidth, shape, strides):
        self.base_ptr = base_ptr
        self.dtype_bitwidth = dtype_bitwidth
        self.shape = shape
        self.strides = strides

        # 简单的启发式规则：假设所有 stride 都是静态的
        self.stride_is_static = tuple(True for _ in strides)

        # 假设元素不重叠
        self.elements_disjoint = True

        # 对齐信息：假设基本对齐
        self.base_ptr_div_by = dtype_bitwidth // 8  # 按字节对齐
        self.stride_div_by = tuple(1 for _ in strides)
        self.shape_div_by = tuple(1 for _ in shape)
