# cuTile-typeviz

## 运行

1. 修改server.ts的CUTILE_SRC_PATH到你的cuTile目录的src/子目录下,需要包含CutileIrDump扩展
2. npm install
3. npm run compile （在server/和根目录都要运行）
4. 在 VSCode 中按 F5 启动扩展开发宿主 

## 已知问题

- 暂时不支持kernel内部调用自定义函数
