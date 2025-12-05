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
import { execSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

// 创建连接
const connection = createConnection(ProposedFeatures.all);

// 创建文档管理器
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

const CUTILE_SRC_PATH = "/root/dev/cutile-python/src";

/**
 * Python 脚本输出的结果类型
 */
interface Hint {
    line: number;
    col_start: number;
    col_end: number;
    ty: string;
}

/**
 * 调用 Python 脚本计算每行的最大字符出现次数
 * 如果 Python 执行失败，直接抛出错误崩溃
 * 
 * @param text 要分析的文本内容
 * @param scriptPath 要运行的 Python 脚本路径（正在监控的文件）
 * @returns Python 脚本输出的 JSON 结果
 */
function callPythonCutileTypecheck(text: string, scriptPath: string): Array<Hint> {
    try {
        // 使用 execSync 同步调用正在监控的 Python 文件
        // 通过 stdin 传入文本内容
        const result = execSync(`PYTHONPATH=${CUTILE_SRC_PATH} python3 "${scriptPath}"`, {
            input: text,
            encoding: 'utf-8',
            maxBuffer: 10 * 1024 * 1024, // 10MB buffer
            timeout: 30000 // 30 seconds
        });

        // 解析 JSON 输出
        const jsonResult: Array<Hint> = JSON.parse(result);

        return jsonResult;
    } catch (error: any) {
        // Python 执行失败，直接崩溃
        const errorMessage = `Python script execution failed: ${error.message}`;
        console.error(errorMessage);

        // 输出详细错误信息
        if (error.stderr) {
            console.error('Python stderr:', error.stderr.toString());
        }
        if (error.stdout) {
            console.error('Python stdout:', error.stdout.toString());
        }

        // 直接抛出错误，不进行 fallback
        throw new Error(errorMessage);
    }
}

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

// Inlay Hint 请求处理
connection.languages.inlayHint.on((params: InlayHintParams): InlayHint[] => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }

    const text = document.getText();

    // 将 URI 转换为文件路径（这就是正在监控的文件）
    const filePath = fileURLToPath(params.textDocument.uri);

    // 调用正在监控的文件作为 Python 脚本运行
    // 如果失败会直接抛出错误崩溃
    const pythonResult = callPythonCutileTypecheck(text, filePath);


    const hints: InlayHint[] = [];

    // 获取请求的范围
    const startLine = params.range.start.line;
    const endLine = params.range.end.line;

    // 遍历 Python 返回的结果
    for (const item of pythonResult) {
        // 只处理请求范围内的行
        if (item.line < startLine || item.line > endLine) {
            continue;
        }

        // 创建 Inlay Hint，使用 Python 返回的 marker 来区分
        const hint: InlayHint = {
            // 位置：行首
            position: Position.create(item.line - 1, item.col_start),
            // 显示内容：使用 Python 的 marker + 最大字符数
            label: `${item.ty}`,
            // 类型：设置为 Type 类型
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

console.log('cuTile typeviz Server 已启动 (Python backend)');
