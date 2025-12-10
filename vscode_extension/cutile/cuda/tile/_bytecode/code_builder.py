# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Sequence, Tuple

from .attribute import TaggedAttribute, OptimizationHints, encode_dictionary_untagged
from .basic import encode_varint, encode_int_list, StringTable
from .constant import ConstantTable
from .debug_info import DebugAttrId
from .type import TypeId, encode_typeid


@dataclass
class Value:
    value_id: int


class NestedBlockBuilder:
    def __init__(self,
                 code_builder: "CodeBuilder",
                 num_results: int,
                 num_blocks: int):
        self._code_builder = code_builder
        self._num_results = num_results
        self._num_blocks = num_blocks

    @contextmanager
    def new_block(self, arg_type_ids: Sequence[TypeId]):
        assert self._num_blocks > 0
        self._code_builder.buf.append(1)  # number of blocks in region (always 1)
        encode_varint(len(arg_type_ids), self._code_builder.buf)
        for t in arg_type_ids:
            encode_typeid(t, self._code_builder.buf)
        orig_buf = self._code_builder.buf
        orig_next_value_id = self._code_builder.next_value_id
        orig_num_ops = self._code_builder.num_ops
        self._code_builder.num_ops = 0
        self._code_builder.buf = bytearray()
        try:
            yield self._code_builder._make_value_tuple(len(arg_type_ids))
            encode_varint(self._code_builder.num_ops, orig_buf)
            orig_buf.extend(self._code_builder.buf)
            self._num_blocks -= 1
        finally:
            self._code_builder.next_value_id = orig_next_value_id
            self._code_builder.num_ops = orig_num_ops
            self._code_builder.buf = orig_buf

    def done(self) -> Tuple[Value, ...]:
        assert self._num_blocks == 0
        return self._code_builder._make_value_tuple(self._num_results)


@dataclass
class CodeBuilder:
    buf: bytearray
    string_table: StringTable
    constant_table: ConstantTable
    debug_attr_per_op: List[DebugAttrId]
    next_value_id: int = 0
    cur_debug_attr: DebugAttrId = DebugAttrId(0)
    num_ops: int = 0

    def new_op(self, num_results: Optional[int] = None) -> Value | Tuple[Value, ...] | None:
        self.debug_attr_per_op.append(self.cur_debug_attr)
        self.num_ops += 1
        if num_results is None:
            ret = Value(self.next_value_id)
            self.next_value_id += 1
            return ret
        elif num_results == 0:
            return None
        else:
            return self._make_value_tuple(num_results)

    def new_op_with_nested_blocks(self,
                                  num_results: int,
                                  num_blocks: int) -> NestedBlockBuilder:
        self.debug_attr_per_op.append(self.cur_debug_attr)
        self.num_ops += 1
        encode_varint(num_blocks, self.buf)
        return NestedBlockBuilder(self, num_results=num_results, num_blocks=num_blocks)

    def _make_value_tuple(self, length: int) -> Tuple[Value, ...]:
        end = self.next_value_id + length
        ret = tuple(Value(i) for i in range(self.next_value_id, end))
        self.next_value_id = end
        return ret

    @contextmanager
    def debug_attr(self, debug_attr_id: DebugAttrId):
        old = self.cur_debug_attr
        self.cur_debug_attr = debug_attr_id
        try:
            yield
        finally:
            self.cur_debug_attr = old

    def encode_opattr_bool(self, val: bool):
        self.buf.append(bool(val))

    def encode_opattr_int(self, val: int):
        encode_varint(val, self.buf)

    def encode_opattr_enum(self, expected_type, val):
        assert isinstance(val, expected_type)
        self.buf.extend(val._value_)

    def encode_opattr_str(self, val: str):
        encode_varint(self.string_table[val.encode()].string_id, self.buf)

    def encode_opattr_typeid(self, type_id: TypeId):
        encode_typeid(type_id, self.buf)

    def encode_opattr_array(self, arr: Sequence[TaggedAttribute]):
        encode_varint(len(arr), self.buf)
        for x in arr:
            self.encode_opattr_tagged(TaggedAttribute, x)

    def encode_opattr_tagged(self, attr_class, val):
        assert isinstance(val, attr_class)
        val.encode_tagged(self.string_table, self.buf)

    def encode_opattr_dense_int_or_fp_elements(self, val: bytes):
        cid = self.constant_table.dense_constant(val)
        encode_varint(cid.constant_id, self.buf)

    def encode_opattr_dense_int32_array(self, val: Sequence[int]):
        encode_int_list(val, 4, self.buf)

    def encode_opattr_optimization_hints(self, val: OptimizationHints):
        encode_dictionary_untagged(val.hints_by_arch, self.string_table, self.buf)


def encode_optional_operand(val: Optional[Value], buf: bytearray):
    if val is not None:
        encode_varint(val.value_id, buf)


def encode_unsized_variadic_operands(vals: Sequence[Value], buf: bytearray):
    for v in vals:
        encode_varint(v.value_id, buf)


def encode_sized_variadic_operands(vals: Sequence[Value], buf: bytearray):
    encode_varint(len(vals), buf)
    encode_unsized_variadic_operands(vals, buf)


def encode_operand(val: Value, buf: bytearray):
    encode_varint(val.value_id, buf)
