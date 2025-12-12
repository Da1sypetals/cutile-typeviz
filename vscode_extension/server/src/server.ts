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
    InlayHintKind,
    Diagnostic,
    DiagnosticSeverity,
    Range
} from 'vscode-languageserver/node';

import { TextDocument } from 'vscode-languageserver-textdocument';
import { execSync, exec } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { fileURLToPath } from 'url';
import { integer } from 'vscode-languageclient';

// 创建连接
const connection = createConnection(ProposedFeatures.all);

// 创建文档管理器
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

// 缓存：存储每个文件的 inlay hints
const hintsCache: Map<string, Array<Hint>> = new Map();

// 缓存：存储每个文件的诊断信息
const diagnosticsCache: Map<string, Diagnostic[]> = new Map();

// 记录正在运行的 Python 任务，避免重复启动
const runningTasks: Map<string, boolean> = new Map();

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

// Python 解释器路径（从客户端获取，如果未设置则不提供 hints）
let pythonExecutable: string | undefined = undefined;

const RECOGNIZED_EXTENSION = ".cutile.py";
const CACHE_DIR_NAME = ".cutile_typeviz";
const CUTILE_SRC_PATH = path.join(__dirname, '..', '..', 'cutile');
const ASSEMBLE_SCRIPT_PATH = path.join(CUTILE_SRC_PATH, "typecheck", "assemble.py");
const OUTPUT_PATH = path.join(os.homedir(), CACHE_DIR_NAME, "main.py");
const TYPECHECK_INFO_PATH = path.join(os.homedir(), CACHE_DIR_NAME, "typecheck.json");

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
 * TileError 错误信息接口
 */
interface TileErrorInfo {
    message: string;
    line: number;
    col: number;
    last_line?: number;
    end_col?: number;
    filename?: string;
}

/**
 * 新的结果格式接口
 */
interface PythonResult {
    success: boolean;
    content: Hint[] | TileErrorInfo;
}

/**
 * Call Python script to perform cuTile type checking (synchronous version)
 * If Python execution fails, throw error
 * 
 * @param text The text content to analyze
 * @param scriptPath The Python script path to run (the file being monitored)
 * @returns JSON result output from Python script, or null if no Python interpreter
 */
function callPythonCutileTypecheckSync(text: string, scriptPath: string): PythonResult | null {
    // 检查是否有 Python 解释器
    if (!pythonExecutable) {
        log('No Python interpreter configured, skipping typecheck');
        return null;
    }

    // 第一步：执行 assemble.py 脚本处理输入
    const assembleScriptFullPath = ASSEMBLE_SCRIPT_PATH;
    execSync(`PYTHONPATH=${CUTILE_SRC_PATH} ${pythonExecutable} "${assembleScriptFullPath}" -f "${scriptPath}"`, {
        input: text,
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        timeout: 30000 // 30 seconds
    });

    // 第二步：执行 OUTPUT_PATH 文件获取最终结果
    execSync(`PYTHONPATH=${CUTILE_SRC_PATH} ${pythonExecutable} "${OUTPUT_PATH}"`, {
        encoding: 'utf-8',
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        timeout: 30000 // 30 seconds
    });

    // 第三步：读取 TYPECHECK_INFO_PATH 文件获取结果
    const typecheckResult = fs.readFileSync(TYPECHECK_INFO_PATH, 'utf-8');
    const jsonResult: PythonResult = JSON.parse(typecheckResult);

    return jsonResult;
}

/**
 * Call Python script to perform cuTile type checking (asynchronous version)
 * Updates cache and triggers refresh when done
 * 
 * @param text The text content to analyze
 * @param scriptPath The Python script path to run (the file being monitored)
 * @param uri The document URI for cache key
 */
function callPythonCutileTypecheckAsync(text: string, scriptPath: string, uri: string): void {
    // 检查是否有 Python 解释器
    if (!pythonExecutable) {
        log('No Python interpreter configured, skipping typecheck');
        // 清空缓存和诊断
        hintsCache.set(uri, []);
        diagnosticsCache.set(uri, []);
        connection.sendDiagnostics({ uri, diagnostics: [] });
        return;
    }

    // 如果该文件已有正在运行的任务，不重复启动
    if (runningTasks.get(uri)) {
        log(`Python task already running for ${uri}, skipping`);
        return;
    }

    runningTasks.set(uri, true);
    const totalStartTime = Date.now();
    log(`Starting async Python task for ${uri}`);

    const assembleScriptFullPath = ASSEMBLE_SCRIPT_PATH;

    // 第一步：异步执行 assemble.py 脚本处理输入
    const assembleStartTime = Date.now();
    const assembleProcess = exec(
        `PYTHONPATH=${CUTILE_SRC_PATH} ${pythonExecutable} "${assembleScriptFullPath}" -f "${scriptPath}"`,
        {
            encoding: 'utf-8',
            maxBuffer: 10 * 1024 * 1024, // 10MB buffer
            timeout: 30000 // 30 seconds
        },
        (error, stdout, stderr) => {
            const assembleEndTime = Date.now();
            const assembleElapsed = assembleEndTime - assembleStartTime;

            if (error) {
                logError(`Assemble script failed (took ${assembleElapsed}ms): ${error.message}`);
                if (stderr) logError('stderr:', stderr);
                runningTasks.set(uri, false);

                // 报告错误给用户
                const errorMessage = stderr ? `Assemble failed: ${stderr.trim()}` : `Assemble failed: ${error.message}`;
                hintsCache.set(uri, []);
                diagnosticsCache.set(uri, [createFileLevelErrorDiagnostic(errorMessage)]);
                connection.sendDiagnostics({ uri, diagnostics: diagnosticsCache.get(uri) || [] });
                return;
            }

            log(`Step 1 - Assemble script completed in ${assembleElapsed}ms`);

            // 第二步：异步执行 OUTPUT_PATH 文件获取最终结果
            const outputStartTime = Date.now();
            exec(
                `PYTHONPATH=${CUTILE_SRC_PATH} ${pythonExecutable} "${OUTPUT_PATH}"`,
                {
                    encoding: 'utf-8',
                    maxBuffer: 10 * 1024 * 1024, // 10MB buffer
                    timeout: 30000 // 30 seconds
                },
                (error2, stdout2, stderr2) => {
                    const outputEndTime = Date.now();
                    const outputElapsed = outputEndTime - outputStartTime;

                    runningTasks.set(uri, false);

                    if (error2) {
                        logError(`Output script failed (took ${outputElapsed}ms): ${error2.message}`);
                        if (stderr2) logError('stderr:', stderr2);

                        // 报告错误给用户
                        const errorMessage = stderr2 ? `Typecheck failed: ${stderr2.trim()}` : `Typecheck failed: ${error2.message}`;
                        hintsCache.set(uri, []);
                        diagnosticsCache.set(uri, [createFileLevelErrorDiagnostic(errorMessage)]);
                        connection.sendDiagnostics({ uri, diagnostics: diagnosticsCache.get(uri) || [] });
                        return;
                    }

                    log(`Step 2 - Output script completed in ${outputElapsed}ms`);

                    try {
                        // 第三步：读取 TYPECHECK_INFO_PATH 文件获取结果
                        const readStartTime = Date.now();
                        const typecheckResult = fs.readFileSync(TYPECHECK_INFO_PATH, 'utf-8');
                        const jsonResult: PythonResult = JSON.parse(typecheckResult);
                        const readEndTime = Date.now();
                        const readElapsed = readEndTime - readStartTime;

                        log(`Step 3 - Read and parse typecheck result completed in ${readElapsed}ms`);

                        // 处理结果
                        if (jsonResult.success && Array.isArray(jsonResult.content)) {
                            // 成功情况：更新inlay hints缓存，清空诊断缓存
                            hintsCache.set(uri, jsonResult.content);
                            diagnosticsCache.set(uri, []);

                            const totalEndTime = Date.now();
                            const totalElapsed = totalEndTime - totalStartTime;
                            log(`Cache updated for ${uri} with ${jsonResult.content.length} hints (total time: ${totalElapsed}ms)`);
                        } else if (!jsonResult.success && typeof jsonResult.content === 'object' && jsonResult.content !== null) {
                            // 失败情况：清空inlay hints缓存，设置诊断信息
                            hintsCache.set(uri, []);

                            const errorInfo = jsonResult.content as TileErrorInfo;
                            // 获取当前文档以创建诊断范围
                            const currentDocument = documents.get(uri);
                            const diagnostics = currentDocument ? createDiagnosticsFromTileError(errorInfo, currentDocument) : [];
                            diagnosticsCache.set(uri, diagnostics);

                            log(`Typecheck failed for ${uri}: ${errorInfo.message}`);
                        } else {
                            // 未知结果格式：清空缓存
                            hintsCache.set(uri, []);
                            diagnosticsCache.set(uri, []);
                            log(`Unknown result format for ${uri}: ${JSON.stringify(jsonResult)}`);
                        }

                        // 触发 inlay hints 刷新和诊断刷新
                        connection.languages.inlayHint.refresh();
                        connection.sendDiagnostics({ uri, diagnostics: diagnosticsCache.get(uri) || [] });
                    } catch (parseError: any) {
                        logError(`Failed to parse typecheck result: ${parseError.message}`);

                        // 报告错误给用户
                        const errorMessage = `Failed to parse typecheck result: ${parseError.message}`;
                        hintsCache.set(uri, []);
                        diagnosticsCache.set(uri, [createFileLevelErrorDiagnostic(errorMessage)]);
                        connection.sendDiagnostics({ uri, diagnostics: diagnosticsCache.get(uri) || [] });
                    }
                }
            );
        }
    );

    // 通过 stdin 传入文本内容
    if (assembleProcess.stdin) {
        assembleProcess.stdin.write(text);
        assembleProcess.stdin.end();
    }
}

/**
 * 从TileError信息创建诊断信息
 */
function createDiagnosticsFromTileError(errorInfo: TileErrorInfo, document: TextDocument): Diagnostic[] {
    const diagnostics: Diagnostic[] = [];

    // 创建主诊断信息
    const diagnostic: Diagnostic = {
        severity: DiagnosticSeverity.Error,
        range: createRangeFromErrorInfo(errorInfo, document),
        message: errorInfo.message,
        source: 'cutile_typeviz'
    };

    diagnostics.push(diagnostic);
    return diagnostics;
}

/**
 * 创建文件级别的错误诊断（当 Python 执行失败时）
 * 错误会显示在文件第一行
 */
function createFileLevelErrorDiagnostic(message: string): Diagnostic {
    return {
        severity: DiagnosticSeverity.Error,
        // 显示在整个文件（第一行到最大行），整数
        range: Range.create(0, 0, 2147483647, 2147483647),
        message: message,
        source: 'cutile_typeviz'
    };
}

/**
 * 从错误信息创建范围
 */
function createRangeFromErrorInfo(errorInfo: TileErrorInfo, document: TextDocument): Range {
    const line = Math.max(0, errorInfo.line - 1); // 转换为0-based索引
    const col = Math.max(0, errorInfo.col);   // col需要的是 1-based索引

    let endLine = line;
    let endCol = col;

    // 如果有结束位置信息，使用它
    if (errorInfo.last_line !== undefined && errorInfo.end_col !== undefined) {
        endLine = Math.max(0, errorInfo.last_line - 1); // 转换为0-based索引
        endCol = Math.max(0, errorInfo.end_col); // col需要的是 1-based索引
    } else {
        // 如果没有结束位置，使用行的末尾
        const lineText = document.getText(Range.create(line, 0, line + 1, 0));
        endCol = lineText.length;
    }

    return Range.create(line, col, endLine, endCol);
}

// 初始化处理
connection.onInitialize((params: InitializeParams): InitializeResult => {
    log('LSP Server initializing...');

    // 从初始化参数中获取 Python 解释器路径
    const initOptions = params.initializationOptions;
    if (initOptions && initOptions.pythonPath) {
        pythonExecutable = initOptions.pythonPath;
        log(`Python interpreter set to: ${pythonExecutable}`);
    } else {
        log('No Python interpreter provided, hints will be disabled');
    }

    return {
        capabilities: {
            // 文档同步方式：完整同步
            textDocumentSync: TextDocumentSyncKind.Full,
            // 启用 Inlay Hint 功能
            inlayHintProvider: true,
            // 启用诊断功能
            diagnosticProvider: {
                documentSelector: [{ scheme: 'file', language: 'python' }],
                interFileDependencies: false,
                workspaceDiagnostics: false
            }
        }
    };
});

// 初始化完成
connection.onInitialized(() => {
    log('LSP Server initialized');
});

// 监听 Python 解释器路径变化的自定义通知
connection.onNotification('cutile/pythonPathChanged', (params: { pythonPath: string | undefined }) => {
    const oldPath = pythonExecutable;
    pythonExecutable = params.pythonPath;
    log(`Python interpreter changed: ${oldPath} -> ${pythonExecutable}`);

    // 如果有新的解释器，触发所有打开文档的刷新
    if (pythonExecutable) {
        documents.all().forEach(doc => {
            const filePath = fileURLToPath(doc.uri);
            if (filePath.endsWith(RECOGNIZED_EXTENSION)) {
                log(`Refreshing hints for ${doc.uri} due to interpreter change`);
                callPythonCutileTypecheckAsync(doc.getText(), filePath, doc.uri);
            }
        });
    } else {
        // 如果解释器被清除，清空所有缓存
        hintsCache.clear();
        diagnosticsCache.clear();
        documents.all().forEach(doc => {
            connection.sendDiagnostics({ uri: doc.uri, diagnostics: [] });
        });
        connection.languages.inlayHint.refresh();
    }
});

// Inlay Hint 请求处理
connection.languages.inlayHint.on((params: InlayHintParams): InlayHint[] => {
    const document = documents.get(params.textDocument.uri);
    if (!document) {
        return [];
    }

    // 将 URI 转换为文件路径（这就是正在监控的文件）
    const filePath = fileURLToPath(params.textDocument.uri);

    // 只为 RECOGNIZED_EXTENSION 扩展名的文件提供 Inlay Hints
    if (!filePath.endsWith(RECOGNIZED_EXTENSION)) {
        return [];
    }

    const uri = params.textDocument.uri;

    // 只从缓存获取结果，不触发 Python 任务（Python 任务只在保存和内容变更时触发）
    const cachedResult = hintsCache.get(uri);

    // 如果缓存中没有结果或者结果不是数组（可能是错误信息），返回空数组
    if (!cachedResult || !Array.isArray(cachedResult)) {
        return [];
    }

    const pythonResult: Hint[] = cachedResult;

    // 获取请求的范围
    const startLine = params.range.start.line;
    const endLine = params.range.end.line;

    // Filter results: if a line has 2 or more hints, check if their ty and starting column are the same
    // If different, don't show any hints for that line
    const lineGroups = new Map<number, Array<Hint>>();

    // First group by line
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

    // Check hints for each line
    for (const [, hints] of lineGroups.entries()) {
        if (hints.length >= 2) {
            // If there are 2 or more hints, check if their ty and starting column are all the same
            const firstTy = hints[0].ty;
            const firstColStart = hints[0].col_start;
            let allSame = true;

            for (let i = 1; i < hints.length; i++) {
                if (hints[i].ty !== firstTy || hints[i].col_start !== firstColStart) {
                    allSame = false;
                    break;
                }
            }

            // Only keep one hint when all hints have the same ty and starting column
            if (allSame) {
                filteredResult.push(hints[0]);
            }
        } else {
            // Only one hint, keep it directly
            filteredResult.push(...hints);
        }
    }

    const hints: InlayHint[] = [];

    // Iterate through filtered results to create Inlay Hints
    for (const item of filteredResult) {
        // Create Inlay Hint with type information from Python
        const hint: InlayHint = {
            // Position: at the specified column
            position: Position.create(item.line - 1, item.col_start),
            // Display content: type information
            label: `${item.ty}`,
            // Kind: set as Type
            kind: InlayHintKind.Type,
            // Display on the left side (paddingRight adds spacing on the right)
            paddingRight: true
        };

        hints.push(hint);
    }

    return hints;
});

/**
 * 处理文档变更事件的通用函数
 * @param document 变更的文档
 * @param eventName 事件名称（用于日志）
 */
function handleDocumentChange(document: TextDocument, eventName: string): void {
    const uri = document.uri;
    const filePath = fileURLToPath(uri);

    // 只为 RECOGNIZED_EXTENSION 扩展名的文件触发刷新
    if (!filePath.endsWith(RECOGNIZED_EXTENSION)) {
        return;
    }

    log(`[Event: ${eventName}] Triggering Python task: ${uri}`);
    const text = document.getText();
    callPythonCutileTypecheckAsync(text, filePath, uri);
}

// 监听文档变化，触发 Inlay Hint 刷新
documents.onDidSave(change => handleDocumentChange(change.document, 'onDidSave'));
documents.onDidChangeContent(change => handleDocumentChange(change.document, 'onDidChangeContent'));

// 监听文档关闭，清理诊断信息
documents.onDidClose(change => {
    const uri = change.document.uri;
    // 清理缓存
    hintsCache.delete(uri);
    diagnosticsCache.delete(uri);
    // 发送空诊断信息以清除显示
    connection.sendDiagnostics({ uri, diagnostics: [] });
});

// 文档管理器监听连接
documents.listen(connection);

// 启动连接监听
connection.listen();

log('cuTile typeviz Server started (Python backend)');
