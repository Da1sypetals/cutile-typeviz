# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Sequence, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class Table(dict[K, V]):
    _starting_id = 0

    def __missing__(self, key) -> V:
        ret = self._wrapper_type(len(self) + self._starting_id)
        self[key] = ret
        return ret

    def _unwrap_id(self, id: V) -> int:
        raise NotImplementedError()


@dataclass
class StringId:
    string_id: int


class StringTable(Table[bytes, StringId]):
    _wrapper_type = StringId

    def _unwrap_id(self, id: StringId) -> int:
        return id.string_id


def encode_varint(x: int, buf: bytearray):
    assert x >= 0
    for i in range((x.bit_length() - 1) // 7):
        buf.append((x & 0x7f) | 0x80)
        x >>= 7
    buf.append(x)


def encode_signed_varint(x: int, buf: bytearray):
    x <<= 1
    if x < 0:
        x = ~x
    encode_varint(x, buf)


def encode_int_list(lst: Sequence[int], byte_width: int, buf: bytearray):
    encode_varint(len(lst), buf)
    for x in lst:
        buf.extend(x.to_bytes(byte_width, "little", signed=True))


def encode_varint_list(lst: Sequence[int], buf: bytearray):
    encode_varint(len(lst), buf)
    for x in lst:
        encode_varint(x, buf)
