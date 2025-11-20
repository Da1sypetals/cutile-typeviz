# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RoundingMode(Enum):
    # Rounds the nearest (ties to even)
    RN = "nearest_even"
    # Round towards zero (truncate)
    RZ = "zero"
    # Round towards negative infinity
    RM = "negative_inf"
    # Round towards positive infinity
    RP = "positive_inf"
    # Full precision rounding mode
    FULL = "full"
    # Approximate rounding mode
    APPROX = "approx"
    # Round towards zero to the nearest integer
    RZI = "nearest_int_to_zero"


class PaddingMode(Enum):
    UNDETERMINED = "undetermined"
    ZERO = "zero"
    NEG_ZERO = "neg_zero"
    NAN = "nan"
    POS_INF = "pos_inf"
    NEG_INF = "neg_inf"
