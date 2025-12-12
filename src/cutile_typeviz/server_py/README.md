# cuTile typeviz Python LSP Server

这是 cuTile typeviz 的 Python LSP 服务器实现，使用 pygls v2。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行服务器（独立测试）

```bash
python server.py
```

服务器通过 stdin/stdout 与客户端通信。

## 集成到 VS Code 扩展

扩展会自动启动此 Python 服务器。确保：

1. 安装了 Python 3.9+
2. 安装了 pygls 和 lsprotocol 依赖
3. 在 VS Code 中选择了正确的 Python 解释器

## 功能

- **Inlay Hints**: 显示 cuTile 类型信息
- **Diagnostics**: 显示类型检查错误
- **文档同步**: 在保存和修改时自动更新类型信息

## 架构

```
┌─────────────────┐     ┌──────────────────────┐
│  VS Code Client │────│  Python LSP Server   │
│  (extension.ts) │    │  (server.py)         │
└─────────────────┘    └──────────────────────┘
                              │
                              ▼
                       ┌──────────────────────┐
                       │  cuTile typecheck    │
                       │  (assemble.py, etc)  │
                       └──────────────────────┘
```
