import numpy as np
from cutile_typeviz.cutile_utils.ir_dump.dumper import get_function_repr
from cutile_typeviz.cutile_utils.ir_dump.mock_tensor import MockTensor
from cutile_typeviz.transpiler import simplify_for_numpy, serialize_function
from cutile_typeviz.transpiler.numpy_transpiler import NumpyTranspiler
from pathlib import Path
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def get_tensor_metadata(args: list[np.ndarray]):
    result = []

    for arg in args:
        assert isinstance(arg, np.ndarray), f"args must be a list of numpy arrays, got {type(arg)}"
        result.append(MockTensor(arg.shape, arg.dtype.name))

    return result


def transpile(
    kernel,
    args: list[np.ndarray] | list[MockTensor],
    out_dir: str,
    save_ir: bool = True,
    save_json: bool = True,
    save_kernel: bool = True,
):
    if isinstance(args[0], np.ndarray):
        args = get_tensor_metadata(args)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    func_repr = get_function_repr(kernel, args, optimized=True)
    simplify_for_numpy(func_repr)

    if save_ir:
        ir_path = out_dir / f"{kernel._pyfunc.__name__}.cutileir"
        with open(ir_path, "w") as f:
            f.write(func_repr.to_string(include_loc=False))
        logger.info(f"cuTileIR saved to {ir_path}")

    func_dict = serialize_function(func_repr)

    if save_json:
        json_path = out_dir / "ir.json"
        with open(json_path, "w") as f:
            json.dump(func_dict, f, indent=2)
        logger.info(f"Intermediate JSON results saved to {json_path}")

    transpiler = NumpyTranspiler(func_dict)
    code = transpiler.transpile()

    if save_kernel:
        numpy_path = out_dir / "numpy_code.py"
        numpy_path.write_text(code)
        logger.info(f"NumPy code saved to {numpy_path}")
