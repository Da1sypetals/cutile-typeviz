# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union, TYPE_CHECKING, Tuple, Optional

from cuda.tile._ir.type import TileTy, TupleTy
from cuda.tile import PaddingMode
from cuda.tile._bytecode.attribute import make_load_store_hints
from cuda.tile._ir import ir
from cuda.tile._ir2bytecode import BytecodeContext, typeid
from cuda.tile._exception import TileValueError, TileTypeError
import cuda.tile._bytecode as bc

if TYPE_CHECKING:
    from cuda.tile._ir.ops import (
        TileLoad, TileLoadTokenOrdered, TileStore, TileStoreTokenOrdered,
        LoadPointer, LoadPointerTokenOrdered, StorePointer, StorePointerTokenOrdered,
    )


def check_load_store_hints(latency_value: int | None, allow_tma_value: bool | None = None) -> None:
    if latency_value is not None:
        if not (1 <= latency_value <= 10):
            raise TileValueError(f"Latency must be between 1 and 10, got {latency_value}")
    if allow_tma_value is not None:
        if not isinstance(allow_tma_value, bool):
            raise TileTypeError(f"Allow TMA must be a boolean, got {allow_tma_value}")


def _create_optimization_hints(op, sm_arch: str) -> Optional[bc.OptimizationHints]:
    latency = getattr(op, "latency", None)
    allow_tma = getattr(op, "allow_tma", None)
    if all(x is None for x in (latency, allow_tma)):
        return None
    if allow_tma is None:
        allow_tma = True
    load_store_hints = bc.LoadStoreHints(latency=latency, allow_tma=allow_tma)
    return make_load_store_hints({sm_arch: load_store_hints})


def _create_common_kwargs(op, ctx: BytecodeContext):
    token = getattr(op, "token", None)
    token = None if token is None else ctx.get_optional_value(token)
    return dict(optimization_hints=_create_optimization_hints(op, ctx.sm_arch),
                token=token,
                memory_scope=None)


def _get_index_tuple(index: ir.Var, ctx: BytecodeContext) -> Tuple[bc.Value, ...]:
    i32_tile_ty = ctx.type_table.tile(ctx.type_table.I32, ())
    index_ty = ctx.typeof(index)
    item_types = index_ty.value_types if isinstance(index_ty, TupleTy) else (index_ty,)
    index_values = ctx.get_value_tuple(index)
    return tuple(bc.encode_TruncIOp(ctx.builder, i32_tile_ty, v, bc.IntegerOverflow.NONE)
                 if (t.dtype if isinstance(t, TileTy) else t).bitwidth > 32 else v
                 for v, t in zip(index_values, item_types, strict=True))


def tile_load_generate_bytecode(op: Union["TileLoad", "TileLoadTokenOrdered"],
                                ctx: BytecodeContext) -> tuple[tuple[bc.Value], tuple[bc.Value]]:
    tile_type: TileTy = op.result_vars[0].get_type()
    shape = tile_type.shape_value
    padding_mode = op.padding_mode
    partition = ctx.make_partition_view(op.array, op.order, shape, padding_mode=padding_mode)
    res, res_token = bc.encode_LoadViewTkoOp(
        ctx.builder,
        tile_type=typeid(ctx.type_table, tile_type),
        result_token_type=ctx.type_table.Token,
        memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
        view=partition,
        index=_get_index_tuple(op.index, ctx),
        **_create_common_kwargs(op, ctx)
    )
    return (res,), (res_token,)


def tile_store_generate_bytecode(op: Union["TileStore", "TileStoreTokenOrdered"],
                                 ctx: BytecodeContext) -> bc.Value:
    tile_ty = op.tile.get_type()
    tile_shape = tuple(x.value for x in tile_ty.shape)

    partition = ctx.make_partition_view(op.array, op.order, tile_shape,
                                        padding_mode=PaddingMode.UNDETERMINED)

    return bc.encode_StoreViewTkoOp(
        ctx.builder,
        result_token_type=ctx.type_table.Token,
        tile=ctx.get_value(op.tile),
        view=partition,
        index=_get_index_tuple(op.index, ctx),
        memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
        **_create_common_kwargs(op, ctx)
    )


def load_pointer_lowering(
    op: Union["LoadPointer", "LoadPointerTokenOrdered"],
    ctx: BytecodeContext
) -> tuple[bc.Value, bc.Value]:
    result_type_id = ctx.typeid_of(op.result_vars[0])
    return bc.encode_LoadPtrTkoOp(
        ctx.builder,
        result_type=result_type_id,
        result_token_type=ctx.type_table.Token,
        source=ctx.get_value(op.pointer),
        mask=None if op.mask is None else ctx.get_value(op.mask),
        paddingValue=ctx.get_value(op.padding_value),
        memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
        **_create_common_kwargs(op, ctx)
    )


def store_pointer_lowering(
    op: Union["StorePointer", "StorePointerTokenOrdered"],
    ctx: BytecodeContext
) -> bc.Value:
    mask_value = None if op.mask is None else ctx.get_value(op.mask)
    return bc.encode_StorePtrTkoOp(
        ctx.builder,
        result_token_type=ctx.type_table.Token,
        destination=ctx.get_value(op.pointer),
        value=ctx.get_value(op.value),
        mask=None if op.mask is None else mask_value,
        memory_ordering_semantics=bc.MemoryOrderingSemantics.WEAK,
        **_create_common_kwargs(op, ctx)
    )
