"""
Where we parse the typecheck parameter annotations
"""

import ast
from typing import Callable


TYPECHECK_START = "<typecheck>"
TYPECHECK_END = "</typecheck>"

VALID_DTYPE = {
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "tfloat32",
    "float8_e4m3fn",
    "float8_e5m2",
}


class TypecheckParseError(Exception):
    def __init__(self) -> None:
        super().__init__()


class TypecheckParamNotFound(Exception):
    def __init__(self) -> None:
        super().__init__()


class TypecheckInvalidArg(Exception):
    def __init__(self, line: int, arg: str) -> None:
        super().__init__()
        self.line = line
        self.arg = arg

    def __str__(self) -> str:
        return f"Line {self.line} does not contain a valid argument. Only constant numbers, a tuple of numbers or tensor is allowed."


def space(n: int):
    return " " * n


def parse_typecheck_params(docstring: Callable):
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
        raise TypecheckParamNotFound()
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
    lines = [line.strip() for line in trimmed_content.splitlines()]
    lines = [line for line in lines if len(line) > 0]

    return lines
