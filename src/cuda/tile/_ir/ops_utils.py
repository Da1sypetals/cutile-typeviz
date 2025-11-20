# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from cuda.tile import _datatype as datatype
from cuda.tile._numeric_semantics import RoundingMode, PaddingMode
from cuda.tile._exception import Loc, TileTypeError
from cuda.tile._memory_model import MemoryOrder, MemoryScope
import cuda.tile._bytecode as bc

from .ir import Operation
from .type import TileTy, PointerTy


class ComparisonPredicates(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


@dataclass
class MathOpDef:
    impl: callable    # Python scalar fallback
    supported_rounding_modes: Tuple[RoundingMode, ...] = ()
    support_flush_to_zero: bool = False


BINOP_REGISTRY = {
    "add": MathOpDef(lambda x, y: x + y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "sub": MathOpDef(lambda x, y: x - y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "mul": MathOpDef(lambda x, y: x * y,
                     (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP),
                     support_flush_to_zero=True),
    "floordiv": MathOpDef(lambda x, y: x // y),
    "cdiv": MathOpDef(lambda x, y: (x + y - 1) // y),
    "truediv": MathOpDef(lambda x, y: x / y,
                         (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP,
                          RoundingMode.FULL, RoundingMode.APPROX),
                         support_flush_to_zero=True),
    "mod": MathOpDef(lambda x, y: x % y),
    "pow": MathOpDef(lambda x, y: x ** y),
    "max": MathOpDef(max, (), support_flush_to_zero=True),
    "min": MathOpDef(min, (), support_flush_to_zero=True),
    "and_": MathOpDef(lambda x, y: x & y),
    "or_": MathOpDef(lambda x, y: x | y),
    "xor": MathOpDef(lambda x, y: x ^ y),
    "eq": MathOpDef(lambda x, y: x == y),
    "ne": MathOpDef(lambda x, y: x != y),
    "ge": MathOpDef(lambda x, y: x >= y),
    "gt": MathOpDef(lambda x, y: x > y),
    "le": MathOpDef(lambda x, y: x <= y),
    "lt": MathOpDef(lambda x, y: x < y),
    "is": MathOpDef(lambda x, y: x is y),
    "lshift": MathOpDef(lambda x, y: x << y),
    "rshift": MathOpDef(lambda x, y: x >> y),
}

for name in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow',
             'and_', 'or_', 'xor']:
    BINOP_REGISTRY["i" + name] = BINOP_REGISTRY[name]


UNARYOP_REGISTRY = {
    "abs": MathOpDef(abs),
    "neg": MathOpDef(lambda x: -x),
    "exp": MathOpDef(math.exp),
    "exp2": MathOpDef(lambda x: 2 ** x, (), support_flush_to_zero=True),
    "sin": MathOpDef(math.sin),
    "sinh": MathOpDef(math.sinh),
    "cos": MathOpDef(math.cos),
    "cosh": MathOpDef(math.cosh),
    "tan": MathOpDef(math.tan),
    "tanh": MathOpDef(math.tanh),
    "log": MathOpDef(math.log),
    "log2": MathOpDef(math.log2),
    "sqrt": MathOpDef(math.sqrt,
                      (RoundingMode.RN, RoundingMode.RZ, RoundingMode.RM, RoundingMode.RP,
                       RoundingMode.APPROX),
                      support_flush_to_zero=True),
    "rsqrt": MathOpDef(lambda x: x ** -0.5, (), support_flush_to_zero=True),
    "invert": MathOpDef(lambda x: ~x),
    "not_": MathOpDef(lambda x: not x),
    "floor": MathOpDef(math.floor),
    "ceil": MathOpDef(math.ceil),
}


def get_default_rounding_mode():
    return RoundingMode.RN


rounding_mode_to_bytecode = {
    RoundingMode.RN: bc.RoundingMode.NEAREST_EVEN,
    RoundingMode.RZ: bc.RoundingMode.ZERO,
    RoundingMode.RM: bc.RoundingMode.NEGATIVE_INF,
    RoundingMode.RP: bc.RoundingMode.POSITIVE_INF,
    RoundingMode.FULL: bc.RoundingMode.FULL,
    RoundingMode.APPROX: bc.RoundingMode.APPROX,
    RoundingMode.RZI: bc.RoundingMode.NEAREST_INT_TO_ZERO
}

rounding_mode_to_bytecode[None] = rounding_mode_to_bytecode[get_default_rounding_mode()]


def get_rounding_mode(op: Operation, constants: Dict[str, Any]) -> Optional[RoundingMode]:
    return (
        constants[op.rounding_mode.name]
        if "rounding_mode" in op.operands
        else None
    )


def get_flush_to_zero(op: Operation, constants: Dict[str, Any]) -> bool:
    return (
        constants[op.flush_to_zero.name]
        if "flush_to_zero" in op.operands
        else False
    )


def check_rd_and_ftz(fn: str, rounding_mode: Optional[RoundingMode], flush_to_zero: bool,
                     dtype: datatype.DType):
    if rounding_mode is None and flush_to_zero is False:
        return

    math_op_def = BINOP_REGISTRY[fn] if fn in BINOP_REGISTRY else UNARYOP_REGISTRY[fn]
    if rounding_mode is not None:
        if rounding_mode not in math_op_def.supported_rounding_modes:
            raise TileTypeError(
                f'Rounding mode {rounding_mode.value} is not supported for {fn}')
        if not datatype.is_float(dtype):
            raise TileTypeError(
                f'Rounding mode can only be used for float types, '
                f'but got {dtype}')
        if rounding_mode in [RoundingMode.APPROX, RoundingMode.FULL]:
            if dtype != datatype.float32:
                raise TileTypeError(
                    f'Rounding mode {rounding_mode.value} can only be used for float32 type, '
                    f'but got {dtype}')
    if flush_to_zero:
        if flush_to_zero and not math_op_def.support_flush_to_zero:
            raise TileTypeError(f'Flush to zero is not supported for {fn}')
        if dtype != datatype.float32:
            raise TileTypeError(
                f'Flush to zero can only be used for float32 type, '
                f'but got {dtype}')


memory_scope_to_bytecode = {
    MemoryScope.TL_BLK: bc.MemoryScope.TL_BLK,
    MemoryScope.DEVICE: bc.MemoryScope.DEVICE,
    MemoryScope.SYS: bc.MemoryScope.SYS
}


memory_order_to_bytecode = {
    MemoryOrder.RELAXED: bc.MemoryOrderingSemantics.RELAXED,
    MemoryOrder.ACQUIRE: bc.MemoryOrderingSemantics.ACQUIRE,
    MemoryOrder.RELEASE: bc.MemoryOrderingSemantics.RELEASE,
    MemoryOrder.ACQ_REL: bc.MemoryOrderingSemantics.ACQ_REL,
}


def memory_order_has_acquire(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.ACQUIRE, MemoryOrder.ACQ_REL)


def memory_order_has_release(memory_order: MemoryOrder):
    return memory_order in (MemoryOrder.RELEASE, MemoryOrder.ACQ_REL)


def get_dtype(ty: TileTy | datatype.DType) -> datatype.DType | PointerTy:
    if isinstance(ty, TileTy):
        return ty.dtype
    elif isinstance(ty, datatype.DType):
        return ty
    elif isinstance(ty, PointerTy):
        return ty
    else:
        raise TypeError(f"Cannot get dtype from {ty}")


def change_dtype(ty: TileTy | datatype.DType | PointerTy,
                 new_dtype: datatype.DType | PointerTy) \
        -> TileTy | datatype.DType | PointerTy:
    if isinstance(ty, TileTy):
        return TileTy(new_dtype, ty.shape)
    else:
        assert isinstance(ty, datatype.DType | PointerTy)
        return new_dtype


def check_dtype_autocast(
    from_d: datatype.DType,
    to_d: datatype.DType,
) -> None:
    if not datatype.can_autocast_dtypes(from_d, to_d):
        raise TileTypeError(f"Autocast from value of type {from_d} to "
                            f"{to_d} is not allowed. "
                            f"Please perform explicit cast using `astype`.")


def check_shapes_eq(a: TileTy, b: TileTy,
                    a_name: str, b_name: str, loc: Loc) -> None:
    if a.shape != b.shape:
        raise TileTypeError(f"{a_name} and {b_name} shapes must match, "
                            f"got {a.shape} and {b.shape}", loc)


class CompareOrdering(Enum):
    ORDERED = "ordered"
    UNORDERED = "unordered"


padding_mode_to_bytecode = {
    PaddingMode.UNDETERMINED: bc.PaddingValue.Missing,
    PaddingMode.ZERO: bc.PaddingValue.Zero,
    PaddingMode.NEG_ZERO: bc.PaddingValue.NegZero,
    PaddingMode.NAN: bc.PaddingValue.Nan,
    PaddingMode.POS_INF: bc.PaddingValue.PosInf,
    PaddingMode.NEG_INF: bc.PaddingValue.NegInf,
}
