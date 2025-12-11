# cuTile-typeviz: cuTile Type Hints

## Build
```sh
bash build_ext.sh
```


## Usage

- Activate the extension
- Choose a python interpreter in `ms-python` extension. `typing-extensions` should be installed in its environment
- Change the file extension to `.cutile.py`
- Refer to `vscode_extension/cutile/matmul.cutile.py` to mark the parameters used by the kernel at the beginning of the cutile kernel's docstring with `<typecheck>` tags

## TODO

- Errors other than tile errors should cause the plugin to crash
- Hints and diagnostics for lines exceeding the original file's line count should not be displayed (if the end line exceeds the limit, only display up to the last character of the last line)
