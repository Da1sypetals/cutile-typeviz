import numpy as np
from cutile_typeviz.cutile_utils.ir_dump.dumper import get_function_repr
from cutile_typeviz.cutile_utils.ir_dump.mock_tensor import MockTensor
from cutile_typeviz.transpiler import simplify_for_numpy, serialize_function
from cutile_typeviz.transpiler.numpy_transpiler import NumpyTranspiler
from cutile_typeviz.transpiler.logging import get_logger
from pathlib import Path
import json
import tempfile
import importlib.util
import sys

logger = get_logger(__name__)


def get_tensor_metadata(args: list[np.ndarray]):
    results = list(args)
    for i, arg in enumerate(results):
        if isinstance(arg, np.ndarray):
            results[i] = MockTensor(arg.shape, arg.dtype.name)

    return results


def launch_numpy(
    kernel,
    args: list[np.ndarray],
    grid: tuple[int, int, int],
    tmp_dir: str | None = None,
):
    """
    Transpile and launch a CuTile kernel on NumPy.

    Args:
        kernel: The CuTile kernel function to transpile
        args: List of numpy arrays as kernel arguments
        grid: Tuple (grid_x, grid_y, grid_z) specifying the grid dimensions
        tmp_dir: Optional temporary directory for transpiled code. If None, creates a temp dir.
    """
    # Create temporary directory if not provided
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="cutile_numpy_")
        logger.info(f"Created temporary directory: {tmp_dir}")

    # Transpile kernel to NumPy code
    transpile(
        kernel,
        args,
        out_dir=tmp_dir,
        save_cutileir=True,
        save_json=True,
        save_kernel=True,
    )

    # Dynamically import the generated module
    numpy_code_path = Path(tmp_dir) / "numpy_code.py"
    module_name = f"cutile_numpy_{kernel._pyfunc.__name__}"

    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, numpy_code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {numpy_code_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the generated function
    func_name = kernel._pyfunc.__name__
    if not hasattr(module, func_name):
        raise AttributeError(f"Module does not contain function '{func_name}'")

    numpy_func = getattr(module, func_name)

    # Call the function with args and grid
    logger.info(f"Launching {func_name} with grid={grid}")
    numpy_func(*args, grid=grid)


def transpile(
    kernel,
    args: list[np.ndarray] | list[MockTensor],
    out_dir: str,
    save_cutileir: bool = True,
    save_json: bool = True,
    save_kernel: bool = True,
):
    # Convert numpy arrays to MockTensor if necessary
    args = get_tensor_metadata(args)

    # Create output directory
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    # Generate function representation
    func_repr = get_function_repr(kernel, args, optimized=True)
    simplify_for_numpy(func_repr)

    if save_cutileir:
        ir_path = out_dir / f"{kernel._pyfunc.__name__}.cutileir"
        with open(ir_path, "w") as f:
            f.write(func_repr.to_string(include_loc=False))
        logger.info(f"cuTileIR saved to {ir_path}")

    # Serialize function to dictionary
    func_dict = serialize_function(func_repr)

    if save_json:
        json_path = out_dir / "ir.json"
        with open(json_path, "w") as f:
            json.dump(func_dict, f, indent=2)
        logger.info(f"Intermediate JSON results saved to {json_path}")

    # Transpile serialized dictionary to NumPy code
    transpiler = NumpyTranspiler(func_dict)
    code = transpiler.transpile()

    if save_kernel:
        numpy_path = out_dir / "numpy_code.py"
        numpy_path.write_text(code)
        logger.info(f"NumPy code saved to {numpy_path}")
