# Char Count LSP

一个简单的 LSP Server 示例，在每行行首显示该行中出现最多的字符的个数（通过 Inlay Hint 展示）。

## 功能

- 对于每一行文本，在行首以 Inlay Hint 的形式显示该行中出现次数最多的字符的个数
- 例如，如果一行是 `hello world`，字母 `l` 出现了 3 次（最多），则显示 `[3]`

## 项目结构

```
lsp-example/
├── package.json          # 扩展配置
├── tsconfig.json         # 客户端 TypeScript 配置
├── src/
│   └── extension.ts      # VSCode 扩展客户端
└── server/
    ├── package.json      # 服务器配置
    ├── tsconfig.json     # 服务器 TypeScript 配置
    └── src/
        └── server.ts     # LSP Server 实现
```

## 安装和运行

### 1. 安装依赖

```bash
# 安装客户端依赖
npm install

# 安装服务器依赖
cd server && npm install && cd ..
```

### 2. 编译

```bash
# 编译客户端
npm run compile

# 编译服务器
cd server && npm run compile && cd ..
```

### 3. 在 VSCode 中调试

1. 用 VSCode 打开此项目
2. 按 `F5` 启动调试
3. 在新打开的 Extension Development Host 窗口中打开任意文本文件
4. 查看每行行首的 Inlay Hint

### 4. 打包扩展（可选）

```bash
# 安装 vsce
npm install -g @vscode/vsce

# 打包
vsce package
```

## 原理说明

### LSP (Language Server Protocol)

LSP 是一种协议，用于在编辑器（客户端）和语言服务器之间进行通信。本项目实现了以下功能：

1. **客户端 (extension.ts)**：负责启动 LSP Server，并将其连接到 VSCode

2. **服务器 (server.ts)**：
   - 接收文档内容
   - 计算每行出现最多的字符个数
   - 返回 Inlay Hint 信息

### Inlay Hint

Inlay Hint 是 LSP 3.17 引入的功能，可以在代码中显示额外的提示信息，常用于：
- 显示参数名称
- 显示类型推断结果
- 显示其他辅助信息

本项目使用 Inlay Hint 在每行行首显示字符统计结果。

## 配置

可以在 VSCode 设置中配置：

```json
{
  "charCountLsp.enable": true
}
```
