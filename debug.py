import os
import subprocess

os.environ["PYTHONPATH"] = "src/cutile_typeviz/cutile_utils/"


subprocess.run(
    ["python", "examples/run_numpy_transpiler.py"],
    check=True,
)
