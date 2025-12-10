# PYTHONPATH=/root/dev/cutile-python/src python import_example.py
from typing import Callable
import importlib.util
import inspect
from pathlib import Path
from icecream import ic


def process_typecheck_string(s):
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
    # Split by <typecheck> and get the last part
    after_typecheck = s.split("<typecheck>")[-1]

    # Split by </typecheck> and get the first part
    before_close = after_typecheck.split("</typecheck>")[0]

    # Trim whitespace from the extracted content
    trimmed_content = before_close.strip()

    # Handle empty content
    if not trimmed_content:
        return []

    # Split into lines, trim each line, and filter out empty lines if desired
    # (keeping empty lines that were between non-empty content)
    lines = [line.strip() for line in trimmed_content.splitlines()]

    return lines


# 1. Define the path to the file
file_path = Path("/root/dev/cutile-python/src/new_fmha.py")
func_name = "fmha_kernel"  # renamed to avoid overwriting later

# 2. Load the module dynamically
spec = importlib.util.spec_from_file_location("new_fmha", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 3. Get the function from the module
func = getattr(module, func_name)
print(str(type(func)))

pyfunc: Callable = func._pyfunc

# 4. Get the source code of the function
source_code = inspect.getsource(pyfunc)
print(f"line of source code is {len(source_code.splitlines())}")

source_lines, start_line = inspect.getsourcelines(pyfunc)
print("Starting line number:", start_line)

docs = pyfunc.__doc__

doc_lines = process_typecheck_string(docs)

ic(doc_lines)
