# cuTile 类型提示

## 构建
```sh
bash build_ext.sh
```


## 使用

- 激活extension
- 将文件后缀改为`.cutile.py`
- 参照`vscode_extension/cutile/matmul.cutile.py`, 在cutile kernel的docstring开头标识kernel所使用的参数

## TODO

- 除了tile error之外，其他错误应该导致插件崩溃
- 行数超过原文件行数的hint和diagnostic不显示（如果是end line超过了，那就只显示到最后一行的最后一个字符）
