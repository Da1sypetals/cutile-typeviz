# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import functools
from types import FunctionType

from cuda.tile._by_target import ByTarget
from cuda.tile._cext import TileDispatcher


__all__ = ("function", "kernel")


###############################################################################
# Decorators


def function(func=None, /, *, host=False, tile=True):
    """*Tile functions* are functions that are usable in |tile code|.

    This decorator indicates what |execution spaces| a function can be called from.
    With no arguments, it denotes a tile-only function.

    When an unannotated function is called by a |tile function|, tile shall be added to the
    unannotated function's execution space.
    This process is recursive.
    No explicit annotation is required.

    The types usable as parameters to a |tile function| are described in the |data model|.

    Args:
        host (bool, optional): Whether the function can be called from |host code|.
            Default is False.
        tile (bool, optional): Whether the function can be called from |tile code|.
            Default is True.
    """
    def decorator(func):
        if host:
            return func
        else:
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError('Tile functions can only be called from tile code.')
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


class kernel(TileDispatcher):
    """A *tile kernel* is a function executed by each |block| in a |grid|.

    Functions with this decorator are |kernels|.

    |Kernels| are the entry points of |tile code|.
    Their |execution space| shall be only |tile code|; they cannot be called from |host code|.

    Kernels cannot be called directly. Instead, use :py:func:`launch` to
    queue a kernel for execution over a grid.

    The types usable as parameters to a |kernel| are described in the |data model|.

    Args:
        num_ctas: Number of CTAs in a CGA. Must be a power of 2 between 1 and 16, inclusive.
            Default: None (auto).
        occupancy: Expected number of active CTAs per SM, [1, 32]. Default: None (auto).
        opt_level: Optimization level [0, 3], default 3.

    Target-specific values for the compiler options above can be provided
    using a :py:class:`ByTarget` object.

    Examples::

        @ct.kernel
        def f(a, b, c):
            pass

        grid = (8, 8)
        ct.launch(stream, grid, f, (A, B, C))
    """
    def __new__(cls, function=None, /, **kwargs):
        if function is None:
            def decorate(func):
                return kernel(func, **kwargs)
            return decorate

        return super().__new__(cls, function, **kwargs)

    def __init__(self,
                 function=None,
                 /, *,
                 num_ctas: None | int | ByTarget[int] = None,
                 occupancy: None | int | ByTarget[int] = None,
                 opt_level: None | int | ByTarget[int] = 3):
        if not isinstance(function, FunctionType):
            raise TypeError("`kernel` decorator must be applied to a Python function")

        from cuda.tile._compiler_options import CompilerOptions
        from cuda.tile._const_utils import get_constant_arg_flags
        from cuda.tile import _compile

        constant_flags = get_constant_arg_flags(function)
        compiler_options = CompilerOptions(
            num_ctas=num_ctas,
            occupancy=occupancy,
            opt_level=opt_level
        )
        compile = _compile.CompileCallback(function, compiler_options)
        super().__init__(constant_flags, compile)
        self._pyfunc = function

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")
