# cuTile-typeviz: cuTile Type Hints

## Build

1. Install this repo as a package in your current python environment with `pip install -e .`
2. Build and install the extension

## Quick Build
```sh
bash build_ext.sh
```


## Usage

- Activate the extension
- Choose a python interpreter in `ms-python` extension. You should have this repo installed as a package in your environment
- Change the file extension to `.cutile.py`
- Refer to `vscode_extension/cutile/matmul.cutile.py` to mark the parameters used by the kernel at the beginning of the cutile kernel's docstring with `<typecheck>` tags

## TODO

- Errors other than tile errors should cause the plugin to report error
- Hints and diagnostics for lines exceeding the original file's line count should not be displayed (if the end line exceeds the limit, only display up to the last character of the last line)
