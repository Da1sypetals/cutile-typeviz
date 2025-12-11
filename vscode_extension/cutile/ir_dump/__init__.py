"""
CuTile IR Dump Tool Library

Standalone functions for compiling cuda.tile kernels and dumping various IR formats.
"""

from .dumper import (
    dump_cutileir,
    dump_bytecode,
    dump_typechecked_ir,
    dump_mlir,
    get_function_repr,
    create_mock_tensor,
    save_ir_to_file,
)
from .mock_tensor import MockTensor

__all__ = [
    "dump_cutileir",
    "dump_bytecode",
    "dump_typechecked_ir",
    "dump_mlir",
    "get_function_repr",
    "create_mock_tensor",
    "save_ir_to_file",
    "MockTensor",
]
__version__ = "0.1.0"
