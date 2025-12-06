"""
CutileIrDump 类

用于编译 cuda.tile kernel 并导出各种 IR 格式
"""

import os
import sys
import functools
from typing import List, Dict, Any, Optional, Tuple

import cuda.tile as ct
from cuda.tile._compile import _get_final_ir
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._cext import default_tile_context, get_compute_capability
from cuda.tile._ir import ir
import cuda.tile._bytecode as bc
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel

from .mock_tensor import MockTensor


class CutileIrDump:
    """
    CuTile IR Dump 工具类
    用于编译 cuda.tile kernel 并导出各种 IR 格式
    """

    def __init__(
        self,
        output_dir: str = "./ir_artifacts",
        dump_cutileir: bool = True,
        dump_bytecode: bool = True,
        dump_mlir: bool = False,
        compute_capability: Optional[Tuple[int, int]] = None,
    ):
        """
        初始化 IR Dump 工具

        Args:
            output_dir: IR 文件输出目录
            dump_cutileir: 是否导出 CuTile IR (.cutileir)
            dump_bytecode: 是否导出 Bytecode (.cutile)
            dump_mlir: 是否导出 MLIR (.cuda_tile.mlir)
            compute_capability: GPU 计算能力，如 (8, 9)，None 则自动检测
        """
        self.output_dir = output_dir
        self.dump_cutileir = dump_cutileir
        self.dump_bytecode = dump_bytecode
        self.dump_mlir = dump_mlir
        self.compute_capability = compute_capability
        self.compiler_options = CompilerOptions(num_ctas=None, occupancy=None, opt_level=3)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def compile_kernel(
        self,
        kernel_func,
        args: List[Any],
        kernel_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        编译 kernel 并导出 IR

        Args:
            kernel_func: 使用 @ct.kernel 装饰的 kernel 函数
            args: kernel 参数列表（可以是 MockTensor 或真实 tensor）
            constants: 常量参数字典，如 {"hidden_size": 64, "br": 32}
            grid: grid 维度，如 (32, 1, 1)
            block: block 维度，如 (128, 1, 1)
            kernel_name: 导出的文件名前缀，默认使用 kernel 函数名

        Returns:
            生成的文件路径字典，如：
            {
                "cutileir": "/path/to/kernel.cutileir",
                "bytecode": "/path/to/kernel.cutile",
                "mlir": "/path/to/kernel.cuda_tile.mlir"
            }
        """
        # 获取 kernel 名称
        if kernel_name is None:
            kernel_name = kernel_func._pyfunc.__name__

        # 获取原始 Python 函数
        pyfunc = kernel_func._pyfunc

        # 获取 sm_arch
        if self.compute_capability is not None:
            major, minor = self.compute_capability
        else:
            major, minor = get_compute_capability()
        sm_arch = f"sm_{major}{minor}"

        # 获取 IR
        func_ir = _get_final_ir(pyfunc, args, default_tile_context)

        # 生成 IR 字符串
        ir_string = func_ir.to_string(include_loc=False)

        # 生成 bytecode
        bytecode_generator = functools.partial(
            generate_bytecode_for_kernel, func_ir, self.compiler_options, sm_arch
        )

        bytecode_buf = bytearray()
        with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
            bytecode_generator(writer, anonymize_debug_attr=False)

        # 保存文件
        result = {}

        if self.dump_cutileir:
            ir_path = os.path.join(self.output_dir, f"{kernel_name}.cutileir")
            with open(ir_path, "w") as f:
                f.write(ir_string)
            result["cutileir"] = ir_path

        if self.dump_bytecode:
            bytecode_path = os.path.join(self.output_dir, f"{kernel_name}.cutile")
            with open(bytecode_path, "wb") as f:
                f.write(bytecode_buf)
            result["bytecode"] = bytecode_path

        if self.dump_mlir:
            try:
                from cuda.tile_internal._internal_cext import bytecode_to_mlir_text

                mlir_text = bytecode_to_mlir_text(bytecode_buf)
                mlir_path = os.path.join(self.output_dir, f"{kernel_name}.cuda_tile.mlir")
                with open(mlir_path, "w") as f:
                    f.write(mlir_text)
                result["mlir"] = mlir_path
            except ImportError:
                pass

        return result

    def create_mock_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: str = "float32",
    ):
        """
        创建用于类型推断的 Mock Tensor

        Args:
            shape: tensor 形状
            dtype: 数据类型，如 "float32", "float16", "int32"

        Returns:
            MockTensor 对象
        """
        return MockTensor(shape, dtype)

    def dump_ir_to_string(
        self,
        kernel_func,
        args: List[Any],
        constants: Optional[Dict[str, Any]] = None,
        ir_type: str = "cutileir",
    ) -> str:
        """
        编译 kernel 并返回 IR 字符串（不写入文件）

        Args:
            kernel_func: kernel 函数
            args: 参数列表
            constants: 常量参数
            ir_type: IR 类型，可选 "cutileir", "bytecode", "mlir"

        Returns:
            IR 的字符串表示
        """
        # 获取原始 Python 函数
        pyfunc = kernel_func._pyfunc

        # 获取 sm_arch
        if self.compute_capability is not None:
            major, minor = self.compute_capability
        else:
            major, minor = get_compute_capability()
        sm_arch = f"sm_{major}{minor}"

        # 获取 IR
        func_ir = _get_final_ir(pyfunc, args, default_tile_context)

        if ir_type == "cutileir":
            return func_ir.to_string(include_loc=False)
        elif ir_type == "bytecode":
            bytecode_generator = functools.partial(
                generate_bytecode_for_kernel, func_ir, self.compiler_options, sm_arch
            )
            bytecode_buf = bytearray()
            with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
                bytecode_generator(writer, anonymize_debug_attr=False)
            return bytecode_buf.hex()
        elif ir_type == "mlir":
            try:
                from cuda.tile_internal._internal_cext import bytecode_to_mlir_text

                bytecode_generator = functools.partial(
                    generate_bytecode_for_kernel, func_ir, self.compiler_options, sm_arch
                )
                bytecode_buf = bytearray()
                with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
                    bytecode_generator(writer, anonymize_debug_attr=False)
                return bytecode_to_mlir_text(bytecode_buf)
            except ImportError:
                raise ImportError("MLIR conversion requires cuda.tile_internal")
        else:
            raise ValueError(f"Unknown ir_type: {ir_type}")

    @staticmethod
    def from_env() -> "CutileIrDump":
        """
        从环境变量创建 CutileIrDump 实例

        读取以下环境变量：
        - CUDA_TILE_DUMP_TILEIR: CuTile IR 输出目录
        - CUDA_TILE_DUMP_BYTECODE: Bytecode 输出目录
        - CUDA_TILE_DUMP_MLIR: MLIR 输出目录

        Returns:
            配置好的 CutileIrDump 实例
        """
        output_dir = os.environ.get("CUDA_TILE_DUMP_TILEIR", "./ir_dump")
        dump_cutileir = "CUDA_TILE_DUMP_TILEIR" in os.environ
        dump_bytecode = "CUDA_TILE_DUMP_BYTECODE" in os.environ
        dump_mlir = "CUDA_TILE_DUMP_MLIR" in os.environ

        return CutileIrDump(
            output_dir=output_dir,
            dump_cutileir=dump_cutileir,
            dump_bytecode=dump_bytecode,
            dump_mlir=dump_mlir,
        )

    def set_compiler_options(self, **options) -> "CutileIrDump":
        """
        设置编译器选项（链式调用）

        Args:
            **options: 编译器选项，传递给 CompilerOptions

        Returns:
            self，支持链式调用
        """
        self.compiler_options = CompilerOptions(**options)
        return self

    def get_function_repr(
        self,
        kernel_func,
        args: List[Any],
    ) -> str:
        from cuda.tile._ast2ir import get_function_ir
        from cuda.tile._const_utils import get_constant_annotations
        from cuda.tile._passes.typeinfer import infer_types_pass

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

    def dump_typechecked_ir(
        self,
        kernel_func,
        args: List[Any],
        constants: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None,
    ) -> str:
        """
        编译 kernel 并输出 infer_types_pass 后的 IR（类型检查后的 IR）

        Args:
            kernel_func: 使用 @ct.kernel 装饰的 kernel 函数
            args: kernel 参数列表（可以是 MockTensor 或真实 tensor）
            constants: 常量参数字典，如 {"hidden_size": 64, "br": 32}
            output_file: 可选的输出文件路径，如果提供则写入文件

        Returns:
            类型检查后的 IR 字符串
        """
        func_ir = self.get_function_repr(kernel_func, args)

        # 生成 IR 字符串
        ir_string = func_ir.to_string(include_loc=False)

        # 如果提供了输出文件路径，则写入文件
        if output_file is not None:
            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, "w") as f:
                f.write(ir_string)

        return ir_string
