# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from cuda.tile._by_target import ByTarget, UNSPECIFIED


@dataclass(frozen=True)
class CompilerOptions:
    num_ctas: None | int | ByTarget[int] = None
    occupancy: None | int | ByTarget[int] = None
    opt_level: int | ByTarget[int] = 3

    def __post_init__(self):
        for field in dataclasses.fields(self):
            validator = globals()[f"_validate_{field.name}"]
            value = getattr(self, field.name)
            if isinstance(value, ByTarget):
                for target_val in value._by_target.values():
                    validator(target_val)
                if value._default is not UNSPECIFIED:
                    validator(value._default)
            else:
                validator(value)

    def specialize_for_target(self, target_name: str) -> "CompilerOptions":
        specialized = []
        for field in dataclasses.fields(CompilerOptions):
            value = getattr(self, field.name)
            if isinstance(value, ByTarget):
                value = value._by_target.get(target_name, value._default)
                if value is UNSPECIFIED:
                    value = field.default
            specialized.append(value)
        return CompilerOptions(*specialized)


def _validate_num_ctas(num_ctas: None | int):
    if num_ctas is not None:
        if num_ctas > 16 or num_ctas < 1:
            raise ValueError(f'num_ctas should be [1, 16], got {num_ctas}')
        if (num_ctas & (num_ctas - 1)) != 0:
            raise ValueError(f'num_ctas should be power of 2, got {num_ctas}')


def _validate_occupancy(occupancy: None | int):
    if occupancy is not None:
        if occupancy < 1 or occupancy > 32:
            raise ValueError(f'occupancy should be [1, 32], got {occupancy}')


def _validate_opt_level(opt_level: None | int):
    if opt_level is not None:
        if opt_level < 0 or opt_level > 3:
            raise ValueError(f'opt_level should be [0, 3], got {opt_level}')
