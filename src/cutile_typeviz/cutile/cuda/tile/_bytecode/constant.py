# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.tile._bytecode.basic import encode_varint, Table


@dataclass
class ConstantId:
    constant_id: int


class ConstantTable(Table[bytes, ConstantId]):
    _wrapper_type = ConstantId

    def dense_constant(self, data: bytes) -> ConstantId:
        buf = bytearray()
        encode_varint(len(data), buf)
        buf.extend(data)
        return self[bytes(buf)]

    def _unwrap_id(self, id: ConstantId) -> int:
        return id.constant_id
