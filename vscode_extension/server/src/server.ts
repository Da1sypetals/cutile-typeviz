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
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';

// 创建连接
const connection = createConnection(ProposedFeatures.all);

// 创建文档管理器
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

// 日志工具函数 - 带时间戳
function getTimestamp(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${ms}`;
}

function log(message: string, ...args: any[]): void {
    console.log(`[${getTimestamp()}] ${message}`, ...args);
}

function logError(message: string, ...args: any[]): void {
    console.error(`[${getTimestamp()}] ERROR: ${message}`, ...args);
}

// const CUTILE_SRC_PATH = "/Users/daisy/develop/cutile-python/src";
// const PYTHON_EXECUTABLE = "/Users/daisy/miniconda3/bin/python";

// TODO: use current selected python interpreter
const PYTHON_EXECUTABLE = "/root/miniconda3/bin/python";

const CUTILE_SRC_PATH = path.join(__dirname, '..', '..', 'cutile');
const ASSEMBLE_SCRIPT_PATH = path.join(CUTILE_SRC_PATH, "typecheck", "assemble.py");
const OUTPUT_PATH = path.join(os.homedir(), ".cutile-typeviz", "main.py");
const TYPECHECK_INFO_PATH = path.join(os.homedir(), ".cutile-typeviz", "typecheck.json");

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
 * 执行流程：
 * 1. 先执行 ASSEMBLE_SCRIPT_PATH (typecheck/assemble.py) 处理输入
 * 2. 然后执行 OUTPUT_PATH (~/.cutile-typeviz/main.py) 获取结果
 * 
 * @param text 要分析的文本内容
 * @param scriptPath 要运行的 Python 脚本路径（正在监控的文件）
 * @returns Python 脚本输出的 JSON 结果
 */
function callPythonCutileTypecheck(text: string, scriptPath: string): Array<Hint> {
    try {
        // 第一步：执行 assemble.py 脚本处理输入
        // 通过 stdin 传入文本内容
        const assembleScriptFullPath = ASSEMBLE_SCRIPT_PATH;
        execSync(`PYTHONPATH=${CUTILE_SRC_PATH} ${PYTHON_EXECUTABLE} "${assembleScriptFullPath}" -f "${scriptPath}"`, {
            input: text,
            encoding: 'utf-8',
            maxBuffer: 10 * 1024 * 1024, // 10MB buffer
            timeout: 30000 // 30 seconds
        });

        // 第二步：执行 OUTPUT_PATH 文件获取最终结果
        execSync(`PYTHONPATH=${CUTILE_SRC_PATH} ${PYTHON_EXECUTABLE} "${OUTPUT_PATH}"`, {
            encoding: 'utf-8',
            maxBuffer: 10 * 1024 * 1024, // 10MB buffer
            timeout: 30000 // 30 seconds
        });

        // 第三步：读取 TYPECHECK_INFO_PATH 文件获取结果
        const typecheckResult = fs.readFileSync(TYPECHECK_INFO_PATH, 'utf-8');
        const jsonResult: Array<Hint> = JSON.parse(typecheckResult);

        return jsonResult;
    } catch (error: any) {
        // Python 执行失败，直接崩溃
        const errorMessage = `Python script execution failed: ${error.message}`;
        logError(errorMessage);

        // 输出详细错误信息
        if (error.stderr) {
            logError('Python stderr:', error.stderr.toString());
        }
        if (error.stdout) {
            logError('Python stdout:', error.stdout.toString());
        }

        // 直接抛出错误，不进行 fallback
        throw new Error(errorMessage);
    }
}

// 初始化处理
connection.onInitialize((params: InitializeParams): InitializeResult => {
    log('LSP Server 初始化中...');

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
    log('LSP Server 初始化完成');
});

// Inlay Hint 请求处理
connection.languages.inlayHint.on((params: InlayHintParams): InlayHint[] => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }

    // 将 URI 转换为文件路径（这就是正在监控的文件）
    const filePath = fileURLToPath(params.textDocument.uri);

    // 只为 .cutile.py 扩展名的文件提供 Inlay Hints
    if (!filePath.endsWith('.cutile.py')) {
        return [];
    }

    const text = document.getText();


    // 调用正在监控的文件作为 Python 脚本运行
    // 如果失败会直接抛出错误崩溃
    const pythonResult = callPythonCutileTypecheck(text, filePath);



    // 获取请求的范围
    const startLine = params.range.start.line;
    const endLine = params.range.end.line;

    // 过滤结果：如果一行有2个或更多提示，检查它们的ty和起始列是否相同
    // 如果不同，则不显示该行的任何提示
    const lineGroups = new Map<number, Array<Hint>>();

    // 首先按行分组
    for (const item of pythonResult) {
        if (item.line < startLine || item.line > endLine) {
            continue;
        }

        if (!lineGroups.has(item.line)) {
            lineGroups.set(item.line, []);
        }
        lineGroups.get(item.line)!.push(item);
    }

    const filteredResult: Array<Hint> = [];

    // 检查每行的提示
    for (const [, hints] of lineGroups.entries()) {
        if (hints.length >= 2) {
            // 如果有2个或更多提示，检查它们的ty和起始列是否都相同
            const firstTy = hints[0].ty;
            const firstColStart = hints[0].col_start;
            let allSame = true;

            for (let i = 1; i < hints.length; i++) {
                if (hints[i].ty !== firstTy || hints[i].col_start !== firstColStart) {
                    allSame = false;
                    break;
                }
            }

            // 只有当所有提示的ty和起始列都相同时才保留，只保留一个
            if (allSame) {
                filteredResult.push(hints[0]);
            }
        } else {
            // 只有一个提示，直接保留
            filteredResult.push(...hints);
        }
    }

    const hints: InlayHint[] = [];

    // 遍历过滤后的结果创建 Inlay Hint
    for (const item of filteredResult) {
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

    log(`Provided ${hints.length} inlay hints for file ${document.uri}`)

    return hints;
});

// 监听文档变化，触发 Inlay Hint 刷新
documents.onDidSave(change => {
    // 当文档内容变化时，通知客户端刷新 Inlay Hint
    connection.languages.inlayHint.refresh();
});
documents.onDidChangeContent(change => {
    // 当文档内容变化时，通知客户端刷新 Inlay Hint
    connection.languages.inlayHint.refresh();
});

// 文档管理器监听连接
documents.listen(connection);

// 启动连接监听
connection.listen();

log('cuTile typeviz Server 已启动 (Python backend)');
