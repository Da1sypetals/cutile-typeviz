# cutile_typeviz

## 运行

1. 修改server.ts的CUTILE_SRC_PATH到你的cuTile目录的src/子目录下,需要包含CutileIrDump扩展
2. npm install
3. npm run compile （在server/和根目录都要运行）
4. 在 VSCode 中按 F5 启动扩展开发宿主 

## 安装

```bash
bash build_ext.sh
```

## 注意

- 只为.cutile.py扩展名的文件启用提示

## 已知问题

- 简单起见当前设计为每行只显示一个inlay hint, 导致问题：
    - 由于kernel内调用自定义函数tile function会完全inline，函数的签名不是固定的，如果不同的输入tensor metadata都可以跑通则都可以编译通过。因此，如果一个函数被不同tensor metadata调用多次，则无法正常显示。
    - 多返回值无法正常显示。
    - 也就是，只有该行代码的**等号左边只有一个identifier，且其有确定的shape**的时候才能正常显示。
