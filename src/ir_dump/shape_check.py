import cuda.tile as ct
from cuda.tile._ir.ir import Function, Operation
from cuda.tile._ir.ops import IfElse, Loop
from cuda.tile._ir.ops import Assign
from cuda.tile._ast2ir import get_function_ir
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._passes.typeinfer import infer_types_pass
from cuda.tile._cext import default_tile_context
from cuda.tile._ir import ir


def get_kernel_shapes_info(kernel_func, args: list) -> list[Operation]:
    func_ir = _get_kernel_shapecheck_ir(kernel_func, args)
    flattened_ops = _list_all_operations(func_ir)
    assignment_ops = []

    for op in flattened_ops:
        if isinstance(op, Assign):
            if not str(op.result_var).startswith("$"):
                # print(f"{op.loc}--{op.loc.end_col} | {op}")
                op_dict = {
                    "line": op.loc.line,
                    "col_start": op.loc.col,
                    "col_end": op.loc.end_col,
                    "ty": str(op.result_var.get_type()),
                }
                assignment_ops.append(op_dict)

    return assignment_ops


def _get_kernel_shapecheck_ir(kernel_func, args: list):
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


def _list_all_operations(func: Function) -> list[Operation]:
    operations = func.root_block._operations
    return _flatten_operations(operations)


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


def _filter_out_functions_with_multiple_calls(operation: list[Operation]) -> list[Operation]:
    """
    functions here means device functions, not kernel
    """
    