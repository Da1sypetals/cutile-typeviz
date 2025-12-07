import ast
import inspect
import textwrap
from typing import Callable
import cuda.tile as ct
from cuda.tile._ir.ir import Function, Operation
from cuda.tile._ir.ops import IfElse, Loop
from cuda.tile._ir.ops import Assign
from cuda.tile._ast2ir import get_function_ir
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._passes.typeinfer import infer_types_pass
from cuda.tile._cext import default_tile_context
from cuda.tile._execution import kernel as cutile_kernel
from cuda.tile._ir import ir
from enum import StrEnum
from functools import wraps


def typecheck(*kernel_args, dump_json: bool = True):
    def decorator(kernel):
        @wraps(kernel)
        def wrapper(*args, **kwargs):
            ops = get_kernel_shapes_info(
                kernel_func=kernel,
                args=[*kernel_args],
            )
            if dump_json:
                import json

                return json.dumps(ops)
            else:
                return ops

        return wrapper

    return decorator


def get_kernel_shapes_info(kernel_func: cutile_kernel, args: list) -> list[Operation]:
    useful_ops = _get_ops_for_shapes_info(kernel_func, args)
    assignment_ops_list = [
        {
            "line": op.loc.line,
            "col_start": op.loc.col,
            "col_end": op.loc.end_col,
            "ty": str(op.result_var.get_type()),
        }
        for op in useful_ops
    ]
    return assignment_ops_list


class ControlFlowToken(StrEnum):
    If = "if"
    Elif = "elif"
    Else = "else"
    While = "while"
    For = "for"


def _get_ops_for_shapes_info(kernel_func: cutile_kernel, args: list) -> list[Operation]:
    func_ir = _get_kernel_shapecheck_ir(kernel_func, args)
    flattened_ops = _flatten_operations(func_ir.root_block._operations)
    assignment_ops = []

    control_flow_lines_mapping = _get_control_flow_lines_mapping(kernel_func._pyfunc)
    control_flow_lines = set(control_flow_lines_mapping.keys())
    # print(control_flow_lines)

    for op in flattened_ops:
        if isinstance(op, Assign):
            if not str(op.result_var).startswith("$") and op.loc.line not in control_flow_lines:
                # print(f"{op.loc}--{op.loc.end_col} | {op}")
                assignment_ops.append(op)

    return assignment_ops


def _get_kernel_shapecheck_ir(kernel_func: cutile_kernel, args: list):
    # 获取原始 Python 函数
    pyfunc = kernel_func._pyfunc

    # 创建 IR 上下文并获取函数 IR
    ir_ctx = ir.IRContext()
    func_ir = get_function_ir(pyfunc, ir_ctx, call_site=None)

    # 绑定参数
    ir_args = func_ir.bind_arguments(args, get_constant_annotations(pyfunc))

    # 执行类型推断 pass
    func_ir = infer_types_pass(func_ir, ir_args, pyfunc, default_tile_context)

    return func_ir


def _flatten_operations(operations: list[Operation]) -> list[Operation]:
    flattened = []
    for i in range(len(operations)):
        op = operations[i]
        if isinstance(op, Loop):
            body = _flatten_operations(op.body._operations)
            flattened.extend(body)

        elif isinstance(op, IfElse):
            then_block = _flatten_operations(op.then_block._operations)
            else_block = _flatten_operations(op.else_block._operations)
            flattened.extend(then_block)
            flattened.extend(else_block)

        else:
            flattened.append(op)
    return flattened


def _get_control_flow_lines_mapping(func: Callable) -> dict[int, ControlFlowToken]:
    """
    Find line numbers where control flow statements appear using AST.

    Args:
        func: A Python function object

    Returns:
        Dictionary with statement types as keys and lists of absolute line numbers
        in the source file as values

    Raises:
        TypeError: If func is not a function
        OSError: If source code cannot be retrieved
    """
    if not callable(func):
        raise TypeError(f"Expected a function, got {type(func)}")

    try:
        source_code, starting_line = inspect.getsourcelines(func)
    except OSError as e:
        raise OSError(f"Cannot retrieve source code for {func.__name__}: {e}")

    # Join source lines and dedent
    source_code = "".join(source_code)
    source_code = textwrap.dedent(source_code)

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        raise SyntaxError(f"Invalid Python code for {func.__name__}")

    results = {
        ControlFlowToken.If: [],
        ControlFlowToken.Elif: [],
        ControlFlowToken.Else: [],
        ControlFlowToken.While: [],
        ControlFlowToken.For: [],
    }

    class ControlFlowVisitor(ast.NodeVisitor):
        def visit_If(self, node):
            # Add starting_line - 1 to get absolute line number
            results[ControlFlowToken.If].append(node.lineno + starting_line - 1)

            # Handle elif chains
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # This is an elif
                    results[ControlFlowToken.Elif].append(node.orelse[0].lineno + starting_line - 1)
                elif node.orelse:
                    # This is an else (could be else-if or just else)
                    # Check if it's a bare else or contains statements
                    first_stmt = node.orelse[0]
                    if not isinstance(first_stmt, ast.If):
                        # It's a genuine else block
                        results[ControlFlowToken.Else].append(first_stmt.lineno - 1 + starting_line - 1)

            self.generic_visit(node)

        def visit_While(self, node):
            results[ControlFlowToken.While].append(node.lineno + starting_line - 1)
            self.generic_visit(node)

        def visit_For(self, node):
            results[ControlFlowToken.For].append(node.lineno + starting_line - 1)
            self.generic_visit(node)

    visitor = ControlFlowVisitor()
    visitor.visit(tree)

    revmap = dict()

    for token, lines in results.items():
        for line in lines:
            revmap[line] = token

    return revmap
