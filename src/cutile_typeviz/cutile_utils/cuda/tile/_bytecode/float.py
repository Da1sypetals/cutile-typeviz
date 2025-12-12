# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import struct
from typing import NamedTuple

from .type import SimpleType


class _FloatSpec(NamedTuple):
    bitwidth: int
    emin: int
    emax: int
    exp_bits: int
    precision: int
    finite_only: bool = False


_specs = {
    SimpleType.F16: _FloatSpec(16, -14, 15, 5, 10),
    SimpleType.BF16: _FloatSpec(16, -126, 127, 8, 7),
    SimpleType.F32: _FloatSpec(32, -126, 127, 8, 23),
    SimpleType.TF32: _FloatSpec(19, -126, 127, 8, 10),
    SimpleType.F8E4M3FN: _FloatSpec(8, -6, 8, 4, 3, finite_only=True),
    SimpleType.F8E5M2: _FloatSpec(8, -14, 15, 5, 2),
}


def float_max_value(ty: SimpleType) -> float:
    return _specs[ty].max_value()


def float_bit_size(ty: SimpleType) -> int:
    return 64 if ty == SimpleType.F64 else _specs[ty].bitwidth


def float_to_bits(val: float, ty: SimpleType) -> int:
    if ty == SimpleType.F64:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    else:
        spec = _specs[ty]
        return _convert_float(val, *spec)


def _convert_float(val: float,
                   bitwidth: int,
                   emin: int,
                   emax: int,
                   exp_bits: int,
                   precision: int,
                   finite_only: bool) -> int:
    if val == 0.0:
        sign = math.copysign(1.0, val) < 0.0
        return sign << (bitwidth - 1)
    elif not math.isfinite(val):
        return _convert_nonfinite(val, bitwidth, exp_bits, precision, finite_only)

    sign, val = (1, -val) if (val < 0) else (0, val)
    m, e = math.frexp(val)
    m *= 2   # [1, 2)
    e -= 1

    if e > emax:
        return _convert_nonfinite(-math.inf if sign else math.inf,
                                  bitwidth, exp_bits, precision, finite_only)

    if e < emin:
        m = math.ldexp(m, e - emin)
        e = 0
    else:
        m -= 1.0
        e += -emin + 1

    # Round to nearest, ties to even
    m = round(m * (1 << precision))
    if m == (1 << precision):
        m = 0
        e += 1
        if e > emax - emin + 1:
            return _convert_nonfinite(-math.inf if sign else math.inf,
                                      bitwidth, exp_bits, precision, finite_only)
    bits = (sign << (bitwidth - 1)) | (e << precision) | m
    return bits


def _convert_nonfinite(val, bitwidth, exp_bits, precision, finite_only) -> int:
    if finite_only:
        # NaN is encoded as all ones
        sign = math.copysign(1.0, val) < 0.0
        return (sign << (bitwidth - 1)) | ((1 << (bitwidth - 1)) - 1)
    else:
        # Exponent is all ones. Truncate the low bits, preserve the rest of the payload
        float64_bits, = struct.unpack("<Q", struct.pack("<d", val))
        payload = (float64_bits >> (52 - precision)) & ((1 << precision) - 1)
        hi_bits = (float64_bits >> (63 - exp_bits)) << precision
        return hi_bits | payload
