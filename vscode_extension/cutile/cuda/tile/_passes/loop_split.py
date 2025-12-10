# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional, Set, Dict, DefaultDict, Mapping, NamedTuple, Sequence

from cuda.tile._ir.ir import Block, Var, Mapper, IRContext
from cuda.tile._ir.ops import (Loop, IfElse, RawBinaryArithmeticOperation, RawComparisonOperation,
                               Assign, UnpackRange, Range, EndBranch, Continue, ForLoopInfo,
                               CarriedVariables, TypedConst)


class _Condition(NamedTuple):
    cmp: str
    rhs: Var


_FLIP = {"ge": "le", "gt": "lt", "le": "ge", "lt": "gt"}


def _find_splittable_loops(block: Block,
                           def_depth: Dict[str, int],
                           depth: int,
                           for_loop: Optional[Loop],
                           induction_var: Optional[str],
                           comparisons: Dict[str, _Condition],
                           equiv_map: Dict[str, Var],
                           result: DefaultDict[Loop, Dict[IfElse, _Condition]]):
    for op in block:
        if isinstance(op, RawComparisonOperation):
            if op.fn in ("ge", "gt", "le", "lt"):
                lhs = equiv_map.get(op.lhs.name, op.lhs)
                rhs = equiv_map.get(op.rhs.name, op.rhs)
                if lhs.name == induction_var and def_depth[rhs.name] < depth:
                    comparisons[op.result_var.name] = _Condition(op.fn, rhs)
                elif rhs.name == induction_var and def_depth[lhs.name] < depth:
                    comparisons[op.result_var.name] = _Condition(_FLIP[op.fn], lhs)
        elif isinstance(op, IfElse):
            cond = equiv_map.get(op.cond.name, op.cond)
            if cond.name in comparisons:
                assert for_loop is not None
                result[for_loop][op] = comparisons[cond.name]
            _find_splittable_loops(op.then_block, def_depth, depth + 1, None, None, dict(),
                                   equiv_map, result)
            _find_splittable_loops(op.else_block, def_depth, depth + 1, None, None, dict(),
                                   equiv_map, result)
        elif isinstance(op, Assign):
            equiv_map[op.result_var.name] = equiv_map.get(op.value.name, op.value)
        elif isinstance(op, Loop):
            good_loop = (op.for_loop is not None
                         and op.for_loop.iterable.has_range_info()
                         and op.for_loop.iterable.get_range_info().known_step == 1)
            _find_splittable_loops(
                op.body,
                def_depth,
                depth + 1,
                op if good_loop else None,
                op.for_loop.induction_var.name if good_loop else None,
                dict(),
                equiv_map,
                result
            )

        for v in op.result_vars:
            def_depth[v.name] = depth


_NEED_TO_ADJUST_RANGE = {"ge": False, "gt": True, "le": True, "lt": False}
_BRANCH_TO_KEEP = {"ge": ("else_block", "then_block"),
                   "gt": ("else_block", "then_block"),
                   "le": ("then_block", "else_block"),
                   "lt": ("then_block", "else_block")}


def _apply_splits(block: Block,
                  loops_to_split: Mapping[Loop, _Condition],
                  if_ops_to_flatten: Set[IfElse]):
    new_block = Block(block.ctx)
    for op in block:
        for nested in op.nested_blocks:
            _apply_splits(nested, loops_to_split, if_ops_to_flatten)

        if isinstance(op, Loop) and op in loops_to_split:
            _split_loop(op, loops_to_split[op], if_ops_to_flatten, new_block)
        else:
            new_block.append(op)

    block[:] = new_block.detach_all()


# This is horrible
def _split_loop(loop: Loop, cond: _Condition, if_ops_to_flatten: Set[IfElse], new_block: Block):
    typemap = new_block.ctx.typemap

    range_ty = typemap[loop.for_loop.iterable.name]
    range_dtype = range_ty.dtype
    split_value = cond.rhs
    loc = loop.loc
    if _NEED_TO_ADJUST_RANGE[cond.cmp]:
        one_var = new_block.make_temp_var(loc)
        new_block.append(TypedConst(1, one_var, loc))
        typemap[one_var.name] = range_dtype
        plus_one_var = new_block.make_temp_var(loc)
        new_block.append(RawBinaryArithmeticOperation("add", split_value, one_var, None, False,
                                                      plus_one_var, loc))
        typemap[plus_one_var.name] = range_dtype
        split_value = plus_one_var

    orig_start, orig_stop, orig_step = new_block.make_temp_vars(loc, 3)
    new_block.append(UnpackRange(loop.for_loop.iterable,
                                 (orig_start, orig_stop, orig_step), loc))

    first_loop_stop = new_block.make_temp_var(loc)
    new_block.append(RawBinaryArithmeticOperation("min", orig_stop, split_value, None, False,
                                                  first_loop_stop, loc))

    second_loop_start = new_block.make_temp_var(loc)
    new_block.append(RawBinaryArithmeticOperation("max", orig_start, split_value, None, False,
                                                  second_loop_start, loc))

    for var in orig_start, orig_stop, orig_step, first_loop_stop, second_loop_start:
        typemap[var.name] = range_dtype

    first_range, second_range = new_block.make_temp_vars(loc, 2)
    new_block.append(Range(orig_start, first_loop_stop, orig_step, first_range, loc))
    new_block.append(Range(second_loop_start, orig_stop, orig_step, second_range, loc))
    for var in first_range, second_range:
        typemap[var.name] = range_ty

    first_branch, second_branch = _BRANCH_TO_KEEP[cond.cmp]

    intermediate_vars = tuple(new_block.ctx.make_var_like(v) for v in loop.result_vars)
    for old_var, new_var in zip(loop.result_vars, intermediate_vars, strict=True):
        typemap[new_var.name] = typemap[old_var.name]

    new_block.append(_clone_loop(loop, first_range, loop.carried_vars.initial, intermediate_vars,
                                 if_ops_to_flatten, first_branch, new_block.ctx))

    second_loop = _clone_loop(loop, second_range, intermediate_vars, loop.result_vars,
                              if_ops_to_flatten, second_branch, new_block.ctx)
    new_block.append(second_loop)


def _clone_loop(loop: Loop, new_range: Var, initial_vars: Sequence[Var], result_vars: Sequence[Var],
                if_ops_to_flatten: Set[IfElse], branch_to_keep: str, ctx: IRContext) -> Loop:
    mapper = Mapper(ctx)
    new_body_vars = mapper.clone_vars(loop.carried_vars.body)
    new_induction_var = mapper.clone_var(loop.for_loop.induction_var)
    new_for_loop = ForLoopInfo(new_induction_var, new_range)
    new_carried_vars = CarriedVariables(names=loop.carried_vars.names,
                                        initial=initial_vars,
                                        body=new_body_vars,
                                        results=result_vars)
    mapper.set_object(loop.carried_vars, new_carried_vars)

    new_body = Block(ctx)
    for body_op in loop.body:
        if isinstance(body_op, IfElse) and body_op in if_ops_to_flatten:
            early_continue = False
            branch = getattr(body_op, branch_to_keep)
            for branch_op in branch:
                if isinstance(branch_op, EndBranch):
                    for old_res, branch_res in zip(body_op.result_vars, branch_op.outputs,
                                                   strict=True):
                        new_res = mapper.get_var(branch_res)
                        mapper.set_var(old_res, new_res)
                    break

                new_body.append(branch_op.clone(mapper))
                if isinstance(branch_op, Continue):
                    early_continue = True
                    break
            if early_continue:
                break
        else:
            new_body.append(body_op.clone(mapper))

    return Loop(new_body, loop.loc, new_for_loop, new_carried_vars)


def split_loops(block: Block):
    splittable_loops = defaultdict(dict)
    _find_splittable_loops(block, dict(), 0, None, None, dict(), dict(), splittable_loops)

    loops_to_split = dict()
    if_ops_to_flatten = set()
    for loop, if_ops in splittable_loops.items():
        # For now, only split if there is exactly one splittable `if`
        if len(if_ops) != 1:
            continue
        if_op, condition = next(iter(if_ops.items()))
        loops_to_split[loop] = condition
        if_ops_to_flatten.add(if_op)

    if len(loops_to_split) > 0:
        _apply_splits(block, loops_to_split, if_ops_to_flatten)
