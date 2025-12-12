# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import enum
from dataclasses import dataclass

from .basic import Table, StringTable, encode_varint


@dataclass(frozen=True)
class DebugAttrId:
    debug_attr_id: int


MISSING_DEBUG_ATTR_ID = DebugAttrId(0)


class DebugTag(enum.Enum):
    DICompileUnit = b"\x01"
    DIFile = b"\x02"
    DILexicalBlock = b"\x03"
    DILoc = b"\x04"
    DISubprogram = b"\x05"
    CallSite = b"\x06"


class DebugAttrTable(Table[bytes, DebugAttrId]):
    _wrapper_type = DebugAttrId
    _starting_id = 1

    def __init__(self, string_table: StringTable):
        super().__init__()
        self._string_table = string_table

    def _unwrap_id(self, id: DebugAttrId) -> int:
        return id.debug_attr_id

    def file(self, name: str, directory: str) -> DebugAttrId:
        buf = bytearray(DebugTag.DIFile._value_)
        encode_varint(self._string_table[name.encode()].string_id, buf)
        encode_varint(self._string_table[directory.encode()].string_id, buf)
        return self[bytes(buf)]

    def compile_unit(self, file: DebugAttrId) -> DebugAttrId:
        buf = bytearray(DebugTag.DICompileUnit._value_)
        encode_varint(file.debug_attr_id, buf)
        return self[bytes(buf)]

    def lexical_block(self,
                      parent_scope: DebugAttrId,
                      file: DebugAttrId,
                      line: int,
                      column: int) -> DebugAttrId:
        buf = bytearray(DebugTag.DILexicalBlock._value_)
        encode_varint(parent_scope.debug_attr_id, buf)
        encode_varint(file.debug_attr_id, buf)
        encode_varint(line, buf)
        encode_varint(column, buf)
        return self[bytes(buf)]

    def subprogram(self,
                   file: DebugAttrId,
                   line: int,
                   name: str,
                   linkage_name: str,
                   compile_unit: DebugAttrId,
                   scope_line: int) -> DebugAttrId:
        buf = bytearray(DebugTag.DISubprogram._value_)
        encode_varint(file.debug_attr_id, buf)
        encode_varint(line, buf)
        encode_varint(self._string_table[name.encode()].string_id, buf)
        encode_varint(self._string_table[linkage_name.encode()].string_id, buf)
        encode_varint(compile_unit.debug_attr_id, buf)
        encode_varint(scope_line, buf)
        return self[bytes(buf)]

    def call_site(self, callee: DebugAttrId, caller: DebugAttrId) -> DebugAttrId:
        buf = bytearray(DebugTag.CallSite._value_)
        encode_varint(callee.debug_attr_id, buf)
        encode_varint(caller.debug_attr_id, buf)
        return self[bytes(buf)]

    def loc(self, scope: DebugAttrId, filename: str, line: int, column: int) -> DebugAttrId:
        buf = bytearray(DebugTag.DILoc._value_)
        encode_varint(scope.debug_attr_id, buf)
        encode_varint(self._string_table[filename.encode()].string_id, buf)
        encode_varint(line, buf)
        encode_varint(column, buf)
        return self[bytes(buf)]
