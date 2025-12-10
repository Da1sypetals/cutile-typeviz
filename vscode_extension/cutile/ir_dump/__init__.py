"""
CuTile IR Dump 工具库

用于编译 cuda.tile kernel 并导出各种 IR 格式
"""

from .dumper import CutileIrDump
from .mock_tensor import MockTensor

__all__ = ["CutileIrDump", "MockTensor"]
__version__ = "0.1.0"
