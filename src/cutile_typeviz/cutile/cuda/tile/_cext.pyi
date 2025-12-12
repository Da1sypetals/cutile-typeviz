# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence
from cuda.tile._context import TileContextConfig


def launch(stream,
           grid: tuple[int] | tuple[int, int] | tuple[int, int, int],
           kernel,
           kernel_args: tuple[Any, ...],
           /):
    ...


class TileDispatcher:
    def __init__(self, arg_constant_flags: Sequence[bool], compile_func):
        ...


class TileContext:
    def __init__(self, config: TileContextConfig):
        ...

    @property
    def config(self) -> TileContextConfig:
        ...


default_tile_context: TileContext
