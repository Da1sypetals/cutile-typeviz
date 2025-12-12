import functools

from cuda.tile._compile import _get_final_ir
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._cext import default_tile_context, get_compute_capability
from cuda.tile._ir import ir
import cuda.tile._bytecode as bc
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel


def _get_sm_arch(compute_capability: tuple[int, int] | None = None) -> str:
    """
    Get the SM architecture string.

    Args:
        compute_capability: GPU compute capability, e.g., (8, 9). None for auto-detection.

    Returns:
        SM architecture string, e.g., "sm_89"
    """
    if compute_capability is not None:
        major, minor = compute_capability
    else:
        major, minor = get_compute_capability()
    return f"sm_{major}{minor}"


def get_function_repr(
    kernel_func,
    args: list,
):
    """
    Get the FunctionIR object after type inference pass.

    Args:
        kernel_func: Kernel function decorated with @ct.kernel
        args: List of kernel arguments (can be MockTensor or real tensors)

    Returns:
        FunctionIR object after type inference
    """
    from cuda.tile._ast2ir import get_function_ir
    from cuda.tile._const_utils import get_constant_annotations
    from cuda.tile._passes.typeinfer import infer_types_pass

    pyfunc = kernel_func._pyfunc

    ir_ctx = ir.IRContext()
    func_ir = get_function_ir(pyfunc, ir_ctx, call_site=None)

    ir_args = func_ir.bind_arguments(args, get_constant_annotations(pyfunc))

    func_ir = infer_types_pass(func_ir, ir_args, pyfunc, default_tile_context)

    return func_ir


def dump_typechecked_ir(
    kernel_func,
    args: list,
    include_loc: bool = False,
) -> str:
    """
    Compile kernel and return IR string after type inference pass (typechecked IR).

    Args:
        kernel_func: Kernel function decorated with @ct.kernel
        args: List of kernel arguments (can be MockTensor or real tensors)
        include_loc: Whether to include source location information in the IR

    Returns:
        IR string after type checking
    """
    func_ir = get_function_repr(kernel_func, args)
    return func_ir.to_string(include_loc=include_loc)


def dump_cutileir(
    kernel_func,
    args: list,
    include_loc: bool = True,
) -> str:
    """
    Compile kernel and return the final CuTile IR string.

    Args:
        kernel_func: Kernel function decorated with @ct.kernel
        args: List of kernel arguments (can be MockTensor or real tensors)
        include_loc: Whether to include source location information in the IR

    Returns:
        CuTile IR string
    """
    pyfunc = kernel_func._pyfunc

    func_ir = _get_final_ir(pyfunc, args, default_tile_context)

    return func_ir.to_string(include_loc=include_loc)


def dump_bytecode(
    kernel_func,
    args: list,
    compute_capability: tuple[int, int] | None = None,
    as_hex: bool = True,
) -> bytes:
    """
    Compile kernel and return the bytecode.

    Args:
        kernel_func: Kernel function decorated with @ct.kernel
        args: List of kernel arguments (can be MockTensor or real tensors)
        compute_capability: GPU compute capability, e.g., (8, 9). None for auto-detection.
        as_hex: If True, return hex string; otherwise return raw bytes as string

    Returns:
        Bytecode as hex string (or raw bytes string if as_hex=False)
    """
    pyfunc = kernel_func._pyfunc

    sm_arch = _get_sm_arch(compute_capability)

    func_ir = _get_final_ir(pyfunc, args, default_tile_context)

    compiler_options = CompilerOptions(num_ctas=None, occupancy=None, opt_level=3)
    bytecode_generator = functools.partial(generate_bytecode_for_kernel, func_ir, compiler_options, sm_arch)

    bytecode_buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=bytecode_buf) as writer:
        bytecode_generator(writer, anonymize_debug_attr=False)

    return bytes(bytecode_buf)
