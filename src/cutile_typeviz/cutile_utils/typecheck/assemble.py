# PYTHONPATH=/root/dev/cutile-python/src python assemble_source_file.py
from typing import Callable
import importlib.util
from pathlib import Path
from dataclasses import dataclass
import textwrap
import inspect

CACHE_DIR_NAME = ".cutile_typeviz"
TYPECHECK_INFO_FILENAME = "typecheck.json"
TYPECHECK_INFO_PATH = Path.home() / CACHE_DIR_NAME / TYPECHECK_INFO_FILENAME

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

tail = f"""
# 序列化结果，只包含必要的字段
result["hints"] = hints
result["diagnostics"] = diagnostics
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


def parse_typecheck_params(docstring: str | None):
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


@dataclass
class CommentAnnotation:
    content: str
    # line, col
    start: tuple[int, int]
    # line, col
    end: tuple[int, int]


def apply_comment_annotations(source: str) -> str:
    import tokenize
    import io

    comment_annotations: list[CommentAnnotation] = []

    # 将源代码转换为字节流供tokenize使用
    source_bytes = source.encode("utf-8")
    source_io = io.BytesIO(source_bytes)

    # with open("/Users/daisy/develop/cutile-typeviz/a.txt", "w") as f:
    #     f.write("")

    try:
        # 使用tokenize生成tokens
        tokens = tokenize.tokenize(source_io.readline)

        for token in tokens:
            # 检查是否是注释token
            if token.type == tokenize.COMMENT:
                comment_content = token.string.strip()
                comment_content = comment_content[1:].strip()  # keep only the content after #

                # with open("/Users/daisy/develop/cutile-typeviz/a.txt", "a") as f:
                #     start_line, start_col = token.start
                #     end_line, end_col = token.end
                #     f.write(f"{start_line}:{start_col} - {end_line}:{end_col} | ")
                #     f.write(comment_content)
                #     f.write("\n")

                # 检查是否以"cutile-typeviz:"开头，允许任意空格
                if comment_content.startswith("cutile-typeviz"):
                    annotation_str = comment_content[14:].strip()
                    if annotation_str.startswith(":"):
                        annotation_str = annotation_str[1:].strip()

                        # 计算注释的起始和结束位置
                        start_line, start_col = token.start
                        end_line, end_col = token.end

                        # 创建CommentAnnotation对象
                        annotation = CommentAnnotation(
                            content=annotation_str,
                            start=(start_line, start_col),
                            end=(end_line, end_col),
                        )
                        comment_annotations.append(annotation)

    except tokenize.TokenError:
        # 如果tokenize失败，返回空列表
        pass

    # Apply cutile-typeviz: end
    end_annotation: CommentAnnotation | None = None
    for annotation in comment_annotations:
        if annotation.content == "end":
            if end_annotation is None:
                # Apply the FIRST end annotation
                end_annotation = annotation

    processed_source = source
    if end_annotation is not None:
        end_line = end_annotation.start[0] - 1
        processed_source = "\n".join(source.splitlines()[:end_line])

    return processed_source


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
    code_parts.append("result = {'hints': [], 'diagnostics': []}")

    for name in module.__dir__():
        item = getattr(module, name)

        if "<class 'cuda.tile._execution.kernel'>" in str(type(item)):
            func_name = name
            func = item

            # print(f"Examining {func_name}")
            pyfunc: Callable = func._pyfunc

            # 获取函数的起始行号
            source_lines, start_line = inspect.getsourcelines(pyfunc)

            docs = pyfunc.__doc__

            try:
                args_str = parse_typecheck_params(docs)
                kernel = Kernel(name=func_name, args_str=args_str)
                code_parts.append(launch_code(kernel))
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

    code_parts.append(tail)

    main_code = "\n\n".join(code_parts)

    file_content = file_path.read_text()

    source = apply_comment_annotations(file_content)

    code = source
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
