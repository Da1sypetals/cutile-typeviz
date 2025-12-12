# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .attribute import (  # noqa: F401
    Bool, DivBy, Bounded, EntryHints, Float, Integer,
    LoadStoreHints, OptimizationHints, TaggedAttribute
)
from .writer import BytecodeWriter, FunctionBuilder, GlobalSection, write_bytecode  # noqa: F401
from .type import SimpleType, TypeTable, TypeId, PaddingValue  # noqa: F401
from .debug_info import DebugAttrId, DebugAttrTable, MISSING_DEBUG_ATTR_ID  # noqa: F401
from .code_builder import CodeBuilder, Value  # noqa: F401
from .float import float_to_bits, float_bit_size  # noqa: F401
from .encodings import *  # noqa: F401 F403

DYNAMIC_SHAPE = -1 << 63   # INT64_MIN
