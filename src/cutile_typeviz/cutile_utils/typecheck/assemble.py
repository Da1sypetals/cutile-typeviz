# PYTHONPATH=/root/dev/cutile-python/src python assemble_source_file.py
from typing import Callable
import importlib.util
from pathlib import Path
from dataclasses import dataclass
import textwrap
import inspect
import hashlib
import ast

CACHE_DIR_NAME = ".cutile_typeviz"
TYPECHECK_INFO_FILENAME = "typecheck.json"

head = """
import json
import traceback
from ir_dump.mock_tensor import MockTensor
from typecheck.shape_check import get_kernel_shapes_info
from pathlib import Path
from cuda.tile._exception import TileError, Loc

hints = []
diagnostics = []
"""


entrance = """
if __name__ == "__main__":
    main()
"""

TYPECHECK_START = "<typecheck>"
TYPECHECK_END = "</typecheck>"


class TypecheckSyntaxError(Exception):
    """当typecheck块中的某一行语法错误时抛出"""

    def __init__(self, line: int, col: int, message: str):
        super().__init__()
        self.line = line
        self.col = col
        self.message = message


class TypecheckParamCountError(Exception):
    """当typecheck参数个数与函数定义的参数个数不匹配时抛出"""

    def __init__(self, line: int, col: int, message: str):
        super().__init__()
        self.line = line
        self.col = col
        self.message = message


def space(n: int):
    return " " * n


def parse_typecheck_params(docstring: str | None, source_lines: list[str] = None, func_start_line: int = 0):
    """
    Process a string by:
    1. Splitting by '<typecheck>' and taking the last part
    2. Splitting by '</typecheck>' and taking the first part
    3. Trimming whitespace from the resulting string
    4. Splitting into lines, trimming whitespace from each line, and returning as a list
    5. Validate each line's Python syntax

    Args:
        docstring: The docstring to process
        source_lines: Source code lines of the function (for calculating line numbers)
        func_start_line: The starting line number of the function in the source file

    Returns:
        list: List of whitespace-trimmed lines from the extracted content

    Raises:
        ValueError: If input args is not annotated
        TypecheckSyntaxError: If a line in the typecheck block has invalid Python syntax
    """
    if docstring is None or TYPECHECK_START not in docstring or TYPECHECK_END not in docstring:
        raise ValueError("Input args is not annotated. Typecheck will not be performed.")
    # Split by <typecheck> and get the last part
    after_typecheck = docstring.split(TYPECHECK_START, 1)[-1]

    # Split by </typecheck> and get the first part
    before_close = after_typecheck.rsplit(TYPECHECK_END, 1)[0]

    # Trim whitespace from the extracted content
    trimmed_content = before_close.strip()

    # Handle empty content
    if not trimmed_content:
        return []

    # Split into lines, trim each line, and filter out empty lines if desired
    # (keeping empty lines that were between non-empty content)
    lines = [line.strip() for line in trimmed_content.splitlines()]

    # 计算typecheck块在源文件中的起始行号
    typecheck_start_line_in_file = 0
    if source_lines is not None:
        # 在源代码中找到 <typecheck> 的位置
        for i, source_line in enumerate(source_lines):
            if TYPECHECK_START in source_line:
                typecheck_start_line_in_file = func_start_line + i
                break

    # 验证每一行的Python语法
    for idx, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue  # 跳过空行

        try:
            # 尝试以eval模式解析（表达式）
            ast.parse(line, mode="eval")
        except SyntaxError as e:
            # 计算实际行号：typecheck起始行 + 当前行偏移 + 1（因为<typecheck>占一行）
            actual_line = typecheck_start_line_in_file + idx + 1
            col = e.offset - 1 if e.offset else 0
            raise TypecheckSyntaxError(
                line=actual_line, col=col, message=f"Invalid Python syntax in typecheck parameter: {e.msg}"
            )

    # Filter out empty lines
    lines = [line for line in lines if len(line) > 0]

    return lines


@dataclass
class Kernel:
    name: str
    args_str: list[str]


def var_def_code(kernel: Kernel):
    identifiers = [f"tmp_{kernel.name}_param_{param_index}" for param_index in range(len(kernel.args_str))]
    var_def_lines = [f"{ident} = {param}" for ident, param in zip(identifiers, kernel.args_str)]

    code = "\n".join(var_def_lines)
    return code


def launch_code(kernel: Kernel):
    args = ", ".join(kernel.args_str)
    ops_name = f"ops_{kernel.name}"
    code = f"""
try:
    {ops_name} = get_kernel_shapes_info(
        kernel_func={kernel.name},
        args=[{args}],
    )
    hints.extend({ops_name})
except TileError as e:
    # 如果遇到TileError，添加诊断信息
    
    # 序列化Loc信息，只包含必要的字段
    loc_info = {{
        "message": e.message,
        "line": e.loc.line,
        "col": e.loc.col
    }}
    
    # 添加可选的字段（如果存在）
    if e.loc.last_line is not None:
        loc_info["last_line"] = e.loc.last_line
    if e.loc.end_col is not None:
        loc_info["end_col"] = e.loc.end_col
    if e.loc.filename is not None:
        loc_info["filename"] = e.loc.filename
    
    diagnostics.append(loc_info)
"""
    return code


def generate_typecheck_code(file_path, uri, module_name="custom_module"):
    # 1. Define the path to the file
    file_path = Path(file_path)

    # 2. 基于 URI 计算子目录名
    uri_hash = hashlib.sha256(uri.encode()).hexdigest()[:16]
    cache_dir = Path.home() / CACHE_DIR_NAME / uri_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 3. 设置子目录中的文件路径
    output_file_path = cache_dir / "main.py"
    typecheck_info_path = cache_dir / TYPECHECK_INFO_FILENAME

    # 4. Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    code_parts = []
    code_parts.append(head)

    # 在main函数开始时定义result变量
    code_parts.append("result = {'hints': [], 'diagnostics': []}")

    for name in module.__dir__():
        item = getattr(module, name)

        if "<class 'cuda.tile._execution.kernel'>" in str(type(item)):
            func_name = name
            func = item

            # print(f"Examining {func_name}")
            pyfunc: Callable = func._pyfunc

            # 获取函数的起始行号和源码
            source_lines, start_line = inspect.getsourcelines(pyfunc)

            docs = pyfunc.__doc__

            try:
                args_str = parse_typecheck_params(docs, source_lines, start_line)

                # 检查参数个数是否与函数定义的参数个数匹配
                sig = inspect.signature(pyfunc)
                func_param_count = len(sig.parameters)
                typecheck_param_count = len(args_str)

                if func_param_count != typecheck_param_count:
                    raise TypecheckParamCountError(
                        line=start_line,
                        col=0,
                        message=f"Parameter count mismatch in kernel '{func_name}': function has {func_param_count} parameters, but <typecheck> has {typecheck_param_count} parameters.",
                    )

                kernel = Kernel(name=func_name, args_str=args_str)
                code_parts.append(launch_code(kernel))
            except TypecheckParamCountError as e:
                # 参数个数不匹配时，记录诊断信息，但继续处理其他kernel
                param_count_error_diagnostic = f"""
# typecheck 参数个数不匹配 for kernel {func_name}
diagnostics.append({{
    "message": "{e.message}",
    "line": {e.line},
    "col": {e.col},
    "filename": "{file_path.name}"
}})
"""
                code_parts.append(param_count_error_diagnostic)
            except TypecheckSyntaxError as e:
                # 语法错误时，记录诊断信息，但继续处理其他kernel
                syntax_error_diagnostic = f"""
# typecheck 语法错误 for kernel {func_name}
diagnostics.append({{
    "message": "{e.message}",
    "line": {e.line},
    "col": {e.col},
    "filename": "{file_path.name}"
}})
"""
                code_parts.append(syntax_error_diagnostic)
                # 不再 break，继续处理后续 kernel
            except ValueError as e:
                # 如果参数未正确注释，添加诊断信息
                diagnostic_code = f"""
# 为未正确注释的 kernel {func_name} 添加诊断信息
diagnostics.append({{
    "message": "Kernel '{func_name}' is not properly annotated with typecheck parameters. Please add <typecheck>...</typecheck> tags in the docstring.",
    "line": {start_line},
    "col": 0,
    "filename": "{file_path.name}"
}})
"""
                code_parts.append(diagnostic_code)

    tail = f"""
# 序列化结果，只包含必要的字段
result["hints"] = hints
result["diagnostics"] = diagnostics
result_str = json.dumps(result)
typecheck_info_path = Path("{str(typecheck_info_path)}").resolve()
typecheck_info_path.write_text(result_str)
"""
    code_parts.append(tail)

    main_code = "\n\n".join(code_parts)

    # 直接读取文件内容（comment annotation 已经在语言服务器入口处理过了）
    file_content = file_path.read_text()

    code = file_content
    code += "\n"
    code += "\ndef main():\n"
    code += textwrap.indent(main_code, space(4))
    code += entrance

    return code, str(output_file_path), str(typecheck_info_path)


if __name__ == "__main__":
    # code = generate_typecheck_code("/root/dev/cutile-python/src/new_attn.cutile.py")
    # code = generate_typecheck_code("/root/dev/cutile-python/src/simple_attn.cutile.py")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--uri", "-u", type=str, required=True)
    args = parser.parse_args()

    code, output_file_path, typecheck_info_path = generate_typecheck_code(args.file, args.uri)

    # 将代码写入到子目录中的文件
    Path(output_file_path).write_text(code)

    print(f"Code saved to {output_file_path}")
    print(f"Typecheck info will be saved to {typecheck_info_path}")
