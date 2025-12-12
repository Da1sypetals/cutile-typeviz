import * as path from 'path';
import { workspace, ExtensionContext, extensions } from 'vscode';

import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

/**
 * 获取 Python 扩展选择的解释器路径
 * @returns Python 解释器路径，如果未选择则返回 undefined
 */
async function getPythonInterpreterPath(): Promise<string | undefined> {
    const pythonExtension = extensions.getExtension('ms-python.python');
    if (!pythonExtension) {
        console.log('Python extension not found');
        return undefined;
    }

    if (!pythonExtension.isActive) {
        await pythonExtension.activate();
    }

    const pythonApi = pythonExtension.exports;

    // 使用 Python 扩展的环境 API 获取当前选择的解释器
    if (pythonApi && pythonApi.environments) {
        const activeEnv = pythonApi.environments.getActiveEnvironmentPath();
        if (activeEnv && activeEnv.path) {
            console.log(`Found Python interpreter: ${activeEnv.path}`);
            return activeEnv.path;
        }
    }

    // 备用方案：从工作区配置中获取
    const pythonConfig = workspace.getConfiguration('python');
    const interpreterPath = pythonConfig.get<string>('defaultInterpreterPath');
    if (interpreterPath && interpreterPath !== 'python') {
        console.log(`Found Python interpreter from config: ${interpreterPath}`);
        return interpreterPath;
    }

    console.log('No Python interpreter selected');
    return undefined;
}

export async function activate(context: ExtensionContext) {
    // 获取 Python 解释器路径
    const pythonPath = await getPythonInterpreterPath();

    // 如果没有 Python 解释器，使用默认的 python 命令
    const pythonExecutable = pythonPath || 'python';

    console.log(`Using Python interpreter: ${pythonExecutable}`);

    // 服务器选项 - 使用 Python 模块方式启动
    const serverOptions: ServerOptions = {
        run: {
            command: pythonExecutable,
            args: ['-m', 'cutile_typeviz'],
            options: {
                env: {
                    ...process.env,
                    // 可以在这里添加额外的环境变量
                }
            }
        },
        debug: {
            command: pythonExecutable,
            args: ['-m', 'cutile_typeviz'],
            options: {
                env: {
                    ...process.env,
                    // 调试时可以启用更详细的日志
                    PYGLS_DEBUG: '1'
                }
            }
        }
    };

    // 客户端选项
    const clientOptions: LanguageClientOptions = {
        // 注册所有文件类型
        documentSelector: [{ scheme: 'file', language: '*' }],
        synchronize: {
            // 监听工作区配置变化
            configurationSection: 'cuTileTypeviz',
            fileEvents: workspace.createFileSystemWatcher('**/*.*')
        },
        // 传递 Python 解释器路径给服务器
        initializationOptions: {
            pythonPath: pythonPath
        }
    };

    // 创建语言客户端
    client = new LanguageClient(
        'cuTileTypeviz',
        'cuTile typeviz',
        serverOptions,
        clientOptions
    );

    // 启动客户端，同时启动服务器
    client.start();

    // 注册 Python 解释器变化监听器
    await registerPythonInterpreterChangeListener(context);

    console.log('cutile_typeviz extension activated');
}

/**
 * 注册 Python 解释器变化监听器
 * 当用户在 VS Code 中切换 Python 解释器时，通知 LSP 服务器更新
 */
async function registerPythonInterpreterChangeListener(context: ExtensionContext): Promise<void> {
    const pythonExtension = extensions.getExtension('ms-python.python');
    if (!pythonExtension) {
        console.log('Python extension not found, cannot listen for interpreter changes');
        return;
    }

    // 确保 Python 扩展已激活
    if (!pythonExtension.isActive) {
        console.log('Waiting for Python extension to activate...');
        await pythonExtension.activate();
    }

    const pythonApi = pythonExtension.exports;
    if (!pythonApi || !pythonApi.environments) {
        console.log('Python extension API not available');
        return;
    }

    // 监听活动环境路径变化
    const onDidChangeActiveEnv = pythonApi.environments.onDidChangeActiveEnvironmentPath;
    if (onDidChangeActiveEnv) {
        const disposable = onDidChangeActiveEnv(async (e: any) => {
            // 获取新的解释器路径
            let newPath: string | undefined;
            if (e && e.path) {
                newPath = e.path;
            } else {
                // 如果事件没有提供路径，重新获取当前解释器
                newPath = await getPythonInterpreterPath();
            }

            console.log(`Python interpreter changed to: ${newPath}`);

            // 发送配置更新通知给服务器
            if (client) {
                client.sendNotification('cutile/pythonPathChanged', { pythonPath: newPath });
            }
        });

        // 注册到扩展的订阅列表，确保扩展停用时自动清理
        context.subscriptions.push(disposable);
        console.log('Registered Python interpreter change listener');
    } else {
        console.log('onDidChangeActiveEnvironmentPath not available');
    }
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
