# PYTHONPATH=/root/dev/cutile-python/src python assemble_source_file.py
from typing import Callable
import importlib.util
from pathlib import Path
from dataclasses import dataclass
import textwrap

CACHE_DIR_NAME = ".cutile-typeviz"
TYPECHECK_INFO_FILENAME = "typecheck.json"
TYPECHECK_INFO_PATH = Path.home() / CACHE_DIR_NAME / TYPECHECK_INFO_FILENAME

head = """
import json
import traceback
from ir_dump.mock_tensor import MockTensor
from typecheck.shape_check import get_kernel_shapes_info
from pathlib import Path
from cuda.tile._exception import TileError, Loc

ops = []
"""

tail = f"""
# 序列化结果，只包含必要的字段
result = {{"success": True, "content": ops}}
result_str = json.dumps(result)
typecheck_info_path = Path("{TYPECHECK_INFO_PATH}").resolve()
typecheck_info_path.write_text(result_str)
"""

entrance = """
if __name__ == "__main__":
    main()
"""

TYPECHECK_START = "<typecheck>"
TYPECHECK_END = "</typecheck>"


def space(n: int):
    return " " * n


def parse_typecheck_params(docstring: str):
    """
    Process a string by:
    1. Splitting by '<typecheck>' and taking the last part
    2. Splitting by '</typecheck>' and taking the first part
    3. Trimming whitespace from the resulting string
    4. Splitting into lines, trimming whitespace from each line, and returning as a list

    Args:
        s (str): Input string to process

    Returns:
        list: List of whitespace-trimmed lines from the extracted content
    """
    if TYPECHECK_START not in docstring or TYPECHECK_END not in docstring:
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
    ops.extend({ops_name})
except TileError as e:
    # 如果遇到TileError，设置失败标志并序列化错误信息
    result["success"] = False
    
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
    
    result["content"] = loc_info
    content_json = json.dumps(result)
    typecheck_info_path = Path("{TYPECHECK_INFO_PATH}").resolve()
    typecheck_info_path.write_text(content_json)
    return
"""
    return code


def generate_typecheck_code(file_path, module_name="custom_module"):
    # 1. Define the path to the file
    file_path = Path(file_path)

    # 2. Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    code_parts = []
    code_parts.append(head)

    # 在main函数开始时定义result变量
    code_parts.append("result = {'success': True, 'content': []}")

    for name in module.__dir__():
        item = getattr(module, name)

        if "<class 'cuda.tile._execution.kernel'>" in str(type(item)):
            func_name = name
            func = item

            # print(f"Examining {func_name}")
            pyfunc: Callable = func._pyfunc

            # 4. Get the source code of the function

            # source_code = inspect.getsource(pyfunc)
            # print(f"line of source code is {len(source_code.splitlines())}")

            # source_lines, start_line = inspect.getsourcelines(pyfunc)
            # print("Starting line number:", start_line)

            docs = pyfunc.__doc__

            args_str = parse_typecheck_params(docs)

            kernel = Kernel(name=func_name, args_str=args_str)

            # print(var_def_code(kernel))

            # print(launch_code(kernel))

            # code_parts.append(var_def_code(kernel))

            code_parts.append(launch_code(kernel))

    code_parts.append(tail)

    main_code = "\n\n".join(code_parts)

    file_content = file_path.read_text()

    code = file_content
    code += "\n"
    code += "\ndef main():\n"
    code += textwrap.indent(main_code, space(4))
    code += entrance

    return code


if __name__ == "__main__":
    # code = generate_typecheck_code("/root/dev/cutile-python/src/new_attn.cutile.py")
    # code = generate_typecheck_code("/root/dev/cutile-python/src/simple_attn.cutile.py")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()

    code = generate_typecheck_code(args.file)

    cache_dir = Path.home() / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_file = cache_dir / "main.py"
    output_file.write_text(code)

    print(f"Code saved to {output_file}")
