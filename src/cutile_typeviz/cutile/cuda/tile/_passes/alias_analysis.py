# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import FrozenSet, Optional, Dict

from cuda.tile._ir.ir import Var, Function, Block
from cuda.tile._ir.ops import Assign, CarriedVariables, ListItemOperation, IfElseResults, \
    Loop, IfElse, Continue, Break, EndBranch, PointerOffset, GetArrayBasePtr, ScalarToTile, \
    TileBroadcast, TileReshape


class AliasUniverseClass:
    # Union with other set always gives the universe
    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    # Intersection with other set always gives the other set
    def __and__(self, other):
        return other

    def __rand__(self, other):
        return other

    def __bool__(self):
        return True

    def __repr__(self):
        return "UNIVERSE"


ALIAS_UNIVERSE = AliasUniverseClass()

AliasSet = FrozenSet[str] | AliasUniverseClass


@dataclass
class AliasResult:
    aliases: Dict[str, AliasSet]

    def __getitem__(self, var_name: str) -> AliasSet:
        return self.aliases.get(var_name, ALIAS_UNIVERSE)


def alias_analysis_pass(function: Function) -> AliasResult:
    alias_tracker = _AliasTracker()
    for p in function.parameters:
        alias_tracker[p.name] = frozenset([p.name])

    _analyze_aliases_in_block(function.root_block, alias_tracker, False)

    while alias_tracker.dirty:
        alias_tracker.dirty = False
        _analyze_aliases_in_block(function.root_block, alias_tracker, False)

    return AliasResult(alias_tracker.finalize())


class _AliasTracker:
    def __init__(self):
        self.dirty = False
        self._aliases: Dict[str, AliasSet] = dict()

    def __getitem__(self, var_name: str) -> AliasSet:
        return self._aliases[var_name]

    def __setitem__(self, var_name: str, alias_set: AliasSet):
        if var_name not in self._aliases or self._aliases[var_name] != alias_set:
            self.dirty = True
        self._aliases[var_name] = alias_set

    def get(self, var_name: str, default: AliasSet) -> AliasSet:
        return self._aliases.get(var_name, default)

    def finalize(self):
        return self._aliases


def _propagate(alias_tracker: _AliasTracker,
               src: Var,
               dst: Var):
    if src.is_undefined():
        alias_tracker[src.name] = frozenset()

    src_aliases = alias_tracker[src.name]
    dst_aliases = alias_tracker.get(dst.name, frozenset())
    alias_tracker[dst.name] = dst_aliases | src_aliases


def _analyze_aliases_in_block(block: Block,
                              alias_tracker: _AliasTracker,
                              for_loop: bool):
    for op in block.operations:
        if isinstance(op, Assign):
            _propagate(alias_tracker, op.value, op.result_var)
        elif isinstance(op, ListItemOperation):
            _propagate(alias_tracker, op.x, op.result_var)
            # TODO: more granular array list get item alias analysis
        elif isinstance(op, PointerOffset):
            _propagate(alias_tracker, op.pointer, op.result_var)
        elif isinstance(op, GetArrayBasePtr):
            _propagate(alias_tracker, op.array, op.result_var)
        elif isinstance(op, ScalarToTile | TileBroadcast | TileReshape):
            # Needed for tiles of pointers produced by gather/scatter
            _propagate(alias_tracker, op.x, op.result_var)
        elif isinstance(op, Loop):
            if op.for_loop is not None:
                alias_tracker[op.for_loop.induction_var.name] = ALIAS_UNIVERSE

            for init, body, result in _get_carried_var_triplets(op.carried_vars):
                # Loop initial values flow into body values.
                _propagate(alias_tracker, init, body)

                # `For` loop initial values can flow into result values if
                # loop runs for 0 iteration.
                if op.for_loop is not None:
                    _propagate(alias_tracker, init, result)

            _analyze_aliases_in_block(op.body, alias_tracker,
                                      op.for_loop is not None)

        elif isinstance(op, Continue):
            for next, (_, body, result) in zip(
                op.next_vars, _get_carried_var_triplets(op.loop_vars), strict=True
            ):
                # Loop next values can flow into body values
                _propagate(alias_tracker, next, body)

                # `For` loop next values can flow into result values when
                # the iterator is exhausted.
                if for_loop:
                    _propagate(alias_tracker, next, result)

        elif isinstance(op, Break):
            for output, (_, _, result) in zip(
                op.output_vars, _get_carried_var_triplets(op.loop_vars), strict=True
            ):
                _propagate(alias_tracker, output, result)

        elif isinstance(op, IfElse):
            _analyze_aliases_in_block(op.then_block, alias_tracker, for_loop)

            _analyze_aliases_in_block(op.else_block, alias_tracker, for_loop)

        elif isinstance(op, EndBranch):
            for output, result in zip(
                op.outputs, _get_ifelse_result_vars(op.ifelse_results), strict=True
            ):
                _propagate(alias_tracker, output, result)

        else:
            assert len(op.nested_blocks) == 0
            for v in op.result_vars:
                alias_tracker[v.name] = ALIAS_UNIVERSE


def _get_carried_var_triplets(carried_vars: Optional[CarriedVariables]):
    return carried_vars.zipped_triplets() if carried_vars else ()


def _get_ifelse_result_vars(ifelse_results: Optional[IfElseResults]):
    return ifelse_results.vars if ifelse_results else ()
