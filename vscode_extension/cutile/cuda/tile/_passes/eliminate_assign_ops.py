# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .._ir import ir
from .._ir.ops import Assign


def eliminate_assign_ops(func: ir.Function):
    def walk(block):
        new_ops = []
        for op in block:
            if isinstance(op, Assign):
                var = orig_var.get(op.value.name, op.value)
                orig_var[op.result_var.name] = var
                mapper.set_var(op.result_var, var)
            else:
                for nested_block in op.nested_blocks:
                    walk(nested_block)
                new_ops.append(op)
        block[:] = new_ops

    mapper = ir.Mapper(func.root_block.ctx, preserve_vars=True)
    orig_var = dict()
    walk(func.root_block)
    func.root_block[:] = [op.clone(mapper) for op in func.root_block]
