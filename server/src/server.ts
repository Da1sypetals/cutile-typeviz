import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    InitializeResult,
    TextDocumentSyncKind,
    InlayHint,
    InlayHintParams,
    Position,
    InlayHintKind
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';

// 创建连接
const connection = createConnection(ProposedFeatures.all);

// 创建文档管理器
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

// 初始化处理
connection.onInitialize((params: InitializeParams): InitializeResult => {
    console.log('LSP Server 初始化中...');

    return {
        capabilities: {
            // 文档同步方式：完整同步
            textDocumentSync: TextDocumentSyncKind.Full,
            // 启用 Inlay Hint 功能
            inlayHintProvider: true
        }
    };
});

// 初始化完成
connection.onInitialized(() => {
    console.log('LSP Server 初始化完成');
});

/**
 * 计算字符串中出现最多的字符的个数
 * @param line 输入的字符串行
 * @returns 出现最多的字符的个数
 */
function getMaxCharCount(line: string): number {
    if (line.length === 0) {
        return 0;
    }

    // 使用 Map 统计每个字符出现的次数
    const charCount = new Map<string, number>();

    for (const char of line) {
        // 忽略空白字符（可选，根据需求调整）
        // if (char === ' ' || char === '\t') continue;

        charCount.set(char, (charCount.get(char) || 0) + 1);
    }

    // 找出最大值
    let maxCount = 0;
    for (const count of charCount.values()) {
        if (count > maxCount) {
            maxCount = count;
        }
    }

    return maxCount;
}

// Inlay Hint 请求处理
connection.languages.inlayHint.on((params: InlayHintParams): InlayHint[] => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }

    const hints: InlayHint[] = [];
    const text = document.getText();
    const lines = text.split('\n');

    // 获取请求的范围
    const startLine = params.range.start.line;
    const endLine = params.range.end.line;

    // 遍历每一行
    for (let lineIndex = startLine; lineIndex <= endLine && lineIndex < lines.length; lineIndex++) {
        const line = lines[lineIndex];

        // 跳过空行
        if (line.trim().length === 0) {
            continue;
        }

        // 计算该行出现最多的字符的个数
        const maxCount = getMaxCharCount(line);

        // 创建 Inlay Hint
        const hint: InlayHint = {
            // 位置：行首
            position: Position.create(lineIndex, 0),
            // 显示内容：出现最多的字符的个数
            label: `[${maxCount}] `,
            // 类型：设置为 Type 类型（也可以用 Parameter）
            kind: InlayHintKind.Type,
            // 设置为在左侧显示（paddingRight 在右侧添加间距）
            paddingRight: true
        };

        hints.push(hint);
    }

    return hints;
});

// 监听文档变化，触发 Inlay Hint 刷新
documents.onDidChangeContent(change => {
    // 当文档内容变化时，通知客户端刷新 Inlay Hint
    connection.languages.inlayHint.refresh();
});

// 文档管理器监听连接
documents.listen(connection);

// 启动连接监听
connection.listen();

console.log('Char Count LSP Server 已启动');
