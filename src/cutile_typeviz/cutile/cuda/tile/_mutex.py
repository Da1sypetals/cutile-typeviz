# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import cuda.tile._bytecode as bc
from cuda.tile._ir2bytecode import BytecodeContext


_MUTEX_UNLOCKED = 1
_MUTEX_LOCKED = 0


@contextlib.contextmanager
def tile_mutex(mutex_name: str, ctx: BytecodeContext):
    """
    Helper context manager that tries to acquire the mutex at the specified global
    memory location and then executes the critical section. The memory
    location is a !cuda_tile.tensor<cuda_tile.ptr<i32>>. A value of "1"
    indicates that the mutex is available. A value of "0" indicates that
    another block tile currently holds the mutex.
    """

    tt = ctx.type_table
    if not ctx.global_section.is_defined(mutex_name):
        ctx.global_section.define_global(mutex_name, tt.tile(tt.I32, (1,)),
                                         _MUTEX_UNLOCKED.to_bytes(4, "little"))

    ptr_tile_ty = tt.tile(tt.pointer(tt.I32), ())
    ptr = bc.encode_GetGlobalOp(ctx.builder, ptr_tile_ty, mutex_name)
    tile_ty = tt.tile(tt.I32, ())

    locked, unlocked = (bc.encode_ConstantOp(ctx.builder, tile_ty, x.to_bytes(4, "little"))
                        for x in (_MUTEX_LOCKED, _MUTEX_UNLOCKED))

    # Busy loop to acquire the mutex
    loop_builder = bc.encode_LoopOp(ctx.builder, (), ())
    with loop_builder.new_block(()):
        prev, _ = bc.encode_AtomicCASTkoOp(
            ctx.builder,
            result_type=tile_ty,
            result_token_type=tt.Token,
            pointers=ptr,
            cmp=unlocked,
            val=locked,
            mask=None,
            token=None,
            memory_ordering_semantics=bc.MemoryOrderingSemantics.ACQ_REL,
            memory_scope=bc.MemoryScope.DEVICE,
        )

        # Exit busy loop if the mutex was acquired.
        was_unlocked = bc.encode_CmpIOp(ctx.builder,
                                        result_type=tt.tile(tt.I1, ()),
                                        lhs=prev,
                                        rhs=unlocked,
                                        comparison_predicate=bc.ComparisonPredicate.EQUAL,
                                        signedness=bc.Signedness.Unsigned)
        branch_builder = bc.encode_IfOp(ctx.builder, (), was_unlocked)
        with branch_builder.new_block(()):
            bc.encode_BreakOp(ctx.builder, ())
        with branch_builder.new_block(()):
            bc.encode_YieldOp(ctx.builder, ())
        branch_builder.done()

        # Continue busy loop if the mutex was not acquired.
        bc.encode_ContinueOp(ctx.builder, ())
    loop_builder.done()

    yield

    # Release the mutex
    # TODO: use correct scope and order
    bc.encode_AtomicRMWTkoOp(
        ctx.builder,
        result_type=tile_ty,
        result_token_type=tt.Token,
        pointers=ptr,
        arg=unlocked,
        mask=None,
        token=None,
        memory_ordering_semantics=bc.MemoryOrderingSemantics.ACQ_REL,
        memory_scope=bc.MemoryScope.DEVICE,
        mode=bc.AtomicRMWMode.XCHG
    )
