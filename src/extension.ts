import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';

import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    // LSP Server 模块的路径
    const serverModule = context.asAbsolutePath(
        path.join('server', 'out', 'server.js')
    );

    // 服务器选项
    const serverOptions: ServerOptions = {
        run: {
            module: serverModule,
            transport: TransportKind.ipc
        },
        debug: {
            module: serverModule,
            transport: TransportKind.ipc,
            options: {
                execArgv: ['--nolazy', '--inspect=6009']
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

    console.log('cuTile typeviz 扩展已激活');
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
