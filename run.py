import os
import subprocess

os.environ["PYTHONPATH"] = "src/cutile_typeviz/cutile_utils/"


subprocess.run(
    ["python", "examples/sinkhorn.cutile.py"],
    check=True,
)
