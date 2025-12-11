"""
cuTile typeviz LSP Server (Python backend using pygls v2)

完整移植自 TypeScript 版本，使用 pygls v2 API。
"""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from lsprotocol import types
from pygls.lsp.server import LanguageServer
from pygls.workspace import TextDocument

# ============================================================
# 常量定义
# ============================================================

RECOGNIZED_EXTENSION = ".cutile.py"
CACHE_DIR_NAME = ".cutile-typeviz"
# 服务器脚本所在目录（server_py 目录）
SERVER_DIR = Path(__file__).parent
# cutile 源码目录（与 server_py 同级）
CUTILE_SRC_PATH = SERVER_DIR.parent / "cutile"
ASSEMBLE_SCRIPT_PATH = CUTILE_SRC_PATH / "typecheck" / "assemble.py"
OUTPUT_PATH = Path.home() / CACHE_DIR_NAME / "main.py"
TYPECHECK_INFO_PATH = Path.home() / CACHE_DIR_NAME / "typecheck.json"


# ============================================================
# 类型定义
# ============================================================


class Hint:
    """Python 脚本输出的 hint 结果类型"""

    def __init__(self, line: int, col_start: int, col_end: int, ty: str):
        self.line = line
        self.col_start = col_start
        self.col_end = col_end
        self.ty = ty

    @classmethod
    def from_dict(cls, data: dict) -> "Hint":
        return cls(line=data["line"], col_start=data["col_start"], col_end=data["col_end"], ty=data["ty"])


class TileErrorInfo:
    """TileError 错误信息接口"""

    def __init__(
        self,
        message: str,
        line: int,
        col: int,
        last_line: Optional[int] = None,
        end_col: Optional[int] = None,
        filename: Optional[str] = None,
    ):
        self.message = message
        self.line = line
        self.col = col
        self.last_line = last_line
        self.end_col = end_col
        self.filename = filename

    @classmethod
    def from_dict(cls, data: dict) -> "TileErrorInfo":
        return cls(
            message=data["message"],
            line=data["line"],
            col=data["col"],
            last_line=data.get("last_line"),
            end_col=data.get("end_col"),
            filename=data.get("filename"),
        )


class PythonResult:
    """新的结果格式接口"""

    def __init__(self, success: bool, content: Any):
        self.success = success
        self.content = content

    @classmethod
    def from_dict(cls, data: dict) -> "PythonResult":
        return cls(success=data["success"], content=data["content"])


# ============================================================
# 日志工具函数
# ============================================================


def get_timestamp() -> str:
    """获取带毫秒的时间戳"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"


def log(message: str, *args: Any) -> None:
    """日志输出 - 带时间戳"""
    print(f"[{get_timestamp()}] {message}", *args, file=sys.stderr)


def log_error(message: str, *args: Any) -> None:
    """错误日志输出 - 带时间戳"""
    print(f"[{get_timestamp()}] ERROR: {message}", *args, file=sys.stderr)


# ============================================================
# URI 转文件路径
# ============================================================


def uri_to_fs_path(uri: str) -> str:
    """将 file:// URI 转换为文件系统路径"""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return ""
    # 解码 URL 编码的路径
    path = unquote(parsed.path)
    # Windows 路径处理（/C:/path -> C:/path）
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return path


# ============================================================
# cuTile LSP Server
# ============================================================


class CuTileLSPServer(LanguageServer):
    """cuTile typeviz LSP 服务器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Python 解释器路径（从客户端获取，如果未设置则不提供 hints）
        self.python_executable: Optional[str] = None

        # 缓存：存储每个文件的 inlay hints
        self.hints_cache: Dict[str, List[Hint]] = {}

        # 缓存：存储每个文件的诊断信息
        self.diagnostics_cache: Dict[str, List[types.Diagnostic]] = {}

        # 记录正在运行的 Python 任务，避免重复启动
        self.running_tasks: Dict[str, bool] = {}

        # 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 锁，用于线程安全
        self._lock = threading.Lock()


# 创建服务器实例
server = CuTileLSPServer(name="cutile-typeviz", version="1.0.0")


# ============================================================
# 诊断创建辅助函数
# ============================================================


def create_diagnostics_from_tile_error(
    error_info: TileErrorInfo, document: TextDocument
) -> List[types.Diagnostic]:
    """从 TileError 信息创建诊断信息"""
    diagnostics: List[types.Diagnostic] = []

    diagnostic = types.Diagnostic(
        severity=types.DiagnosticSeverity.Error,
        range=create_range_from_error_info(error_info, document),
        message=error_info.message,
        source="cuTile-typeviz",
    )

    diagnostics.append(diagnostic)
    return diagnostics


def create_file_level_error_diagnostic(message: str) -> types.Diagnostic:
    """
    创建文件级别的错误诊断（当 Python 执行失败时）
    错误会显示在文件第一行
    """
    return types.Diagnostic(
        severity=types.DiagnosticSeverity.Error,
        # 显示在整个文件（第一行到最大行）
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=2147483647, character=2147483647),
        ),
        message=message,
        source="cuTile-typeviz",
    )


def create_range_from_error_info(error_info: TileErrorInfo, document: TextDocument) -> types.Range:
    """从错误信息创建范围"""
    line = max(0, error_info.line - 1)  # 转换为 0-based 索引
    col = max(0, error_info.col)  # col 需要的是 1-based 索引

    end_line = line
    end_col = col

    # 如果有结束位置信息，使用它
    if error_info.last_line is not None and error_info.end_col is not None:
        end_line = max(0, error_info.last_line - 1)  # 转换为 0-based 索引
        end_col = max(0, error_info.end_col)  # col 需要的是 1-based 索引
    else:
        # 如果没有结束位置，使用行的末尾
        try:
            lines = document.source.split("\n")
            if line < len(lines):
                end_col = len(lines[line])
        except Exception:
            end_col = col

    return types.Range(
        start=types.Position(line=line, character=col), end=types.Position(line=end_line, character=end_col)
    )


# ============================================================
# Python 类型检查调用函数
# ============================================================


def call_python_cutile_typecheck_sync(
    text: str, script_path: str, python_executable: Optional[str]
) -> Optional[PythonResult]:
    """
    调用 Python 脚本执行 cuTile 类型检查（同步版本）
    如果 Python 执行失败，抛出错误

    Args:
        text: 要分析的文本内容
        script_path: 要运行的 Python 脚本路径（正在监控的文件）
        python_executable: Python 解释器路径

    Returns:
        Python 脚本的 JSON 输出结果，如果没有 Python 解释器则返回 None
    """
    # 检查是否有 Python 解释器
    if not python_executable:
        log("No Python interpreter configured, skipping typecheck")
        return None

    env = os.environ.copy()
    env["PYTHONPATH"] = str(CUTILE_SRC_PATH)

    # 第一步：执行 assemble.py 脚本处理输入
    subprocess.run(
        [python_executable, str(ASSEMBLE_SCRIPT_PATH), "-f", script_path],
        input=text,
        encoding="utf-8",
        env=env,
        timeout=30,
        check=True,
        capture_output=True,
    )

    # 第二步：执行 OUTPUT_PATH 文件获取最终结果
    subprocess.run(
        [python_executable, str(OUTPUT_PATH)],
        encoding="utf-8",
        env=env,
        timeout=30,
        check=True,
        capture_output=True,
    )

    # 第三步：读取 TYPECHECK_INFO_PATH 文件获取结果
    with open(TYPECHECK_INFO_PATH, "r", encoding="utf-8") as f:
        typecheck_result = f.read()

    json_result = json.loads(typecheck_result)
    return PythonResult.from_dict(json_result)


def call_python_cutile_typecheck_async(text: str, script_path: str, uri: str) -> None:
    """
    调用 Python 脚本执行 cuTile 类型检查（异步版本）
    完成后更新缓存并触发刷新

    Args:
        text: 要分析的文本内容
        script_path: 要运行的 Python 脚本路径（正在监控的文件）
        uri: 文档 URI，用作缓存键
    """
    # 检查是否有 Python 解释器
    if not server.python_executable:
        log("No Python interpreter configured, skipping typecheck")
        # 清空缓存和诊断
        with server._lock:
            server.hints_cache[uri] = []
            server.diagnostics_cache[uri] = []
        server.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=[]))
        return

    # 如果该文件已有正在运行的任务，不重复启动
    with server._lock:
        if server.running_tasks.get(uri, False):
            log(f"Python task already running for {uri}, skipping")
            return
        server.running_tasks[uri] = True

    def run_typecheck():
        total_start_time = datetime.now()
        log(f"Starting async Python task for {uri}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(CUTILE_SRC_PATH)

        try:
            # 第一步：异步执行 assemble.py 脚本处理输入
            assemble_start_time = datetime.now()

            result = subprocess.run(
                [server.python_executable, str(ASSEMBLE_SCRIPT_PATH), "-f", script_path],
                input=text,
                encoding="utf-8",
                env=env,
                timeout=30,
                capture_output=True,
            )

            assemble_elapsed = (datetime.now() - assemble_start_time).total_seconds() * 1000

            if result.returncode != 0:
                log_error(f"Assemble script failed (took {assemble_elapsed:.0f}ms): {result.stderr}")
                with server._lock:
                    server.running_tasks[uri] = False

                # 报告错误给用户
                error_message = (
                    f"Assemble failed: {result.stderr.strip()}"
                    if result.stderr
                    else f"Assemble failed with code {result.returncode}"
                )
                with server._lock:
                    server.hints_cache[uri] = []
                    server.diagnostics_cache[uri] = [create_file_level_error_diagnostic(error_message)]
                server.text_document_publish_diagnostics(
                    types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
                )
                return

            log(f"Step 1 - Assemble script completed in {assemble_elapsed:.0f}ms")

            # 第二步：异步执行 OUTPUT_PATH 文件获取最终结果
            output_start_time = datetime.now()

            result2 = subprocess.run(
                [server.python_executable, str(OUTPUT_PATH)],
                encoding="utf-8",
                env=env,
                timeout=30,
                capture_output=True,
            )

            output_elapsed = (datetime.now() - output_start_time).total_seconds() * 1000

            with server._lock:
                server.running_tasks[uri] = False

            if result2.returncode != 0:
                log_error(f"Output script failed (took {output_elapsed:.0f}ms): {result2.stderr}")

                # 报告错误给用户
                error_message = (
                    f"Typecheck failed: {result2.stderr.strip()}"
                    if result2.stderr
                    else f"Typecheck failed with code {result2.returncode}"
                )
                with server._lock:
                    server.hints_cache[uri] = []
                    server.diagnostics_cache[uri] = [create_file_level_error_diagnostic(error_message)]
                server.text_document_publish_diagnostics(
                    types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
                )
                return

            log(f"Step 2 - Output script completed in {output_elapsed:.0f}ms")

            try:
                # 第三步：读取 TYPECHECK_INFO_PATH 文件获取结果
                read_start_time = datetime.now()
                with open(TYPECHECK_INFO_PATH, "r", encoding="utf-8") as f:
                    typecheck_result = f.read()
                json_result = PythonResult.from_dict(json.loads(typecheck_result))
                read_elapsed = (datetime.now() - read_start_time).total_seconds() * 1000

                log(f"Step 3 - Read and parse typecheck result completed in {read_elapsed:.0f}ms")

                # 处理结果
                if json_result.success and isinstance(json_result.content, list):
                    # 成功情况：更新 inlay hints 缓存，清空诊断缓存
                    hints = [Hint.from_dict(h) for h in json_result.content]
                    with server._lock:
                        server.hints_cache[uri] = hints
                        server.diagnostics_cache[uri] = []

                    total_elapsed = (datetime.now() - total_start_time).total_seconds() * 1000
                    log(
                        f"Cache updated for {uri} with {len(hints)} hints (total time: {total_elapsed:.0f}ms)"
                    )

                elif not json_result.success and isinstance(json_result.content, dict):
                    # 失败情况：清空 inlay hints 缓存，设置诊断信息
                    with server._lock:
                        server.hints_cache[uri] = []

                    error_info = TileErrorInfo.from_dict(json_result.content)
                    # 获取当前文档以创建诊断范围
                    current_document = server.workspace.get_text_document(uri)
                    diagnostics = (
                        create_diagnostics_from_tile_error(error_info, current_document)
                        if current_document
                        else []
                    )
                    with server._lock:
                        server.diagnostics_cache[uri] = diagnostics

                    log(f"Typecheck failed for {uri}: {error_info.message}")

                else:
                    # 未知结果格式：清空缓存
                    with server._lock:
                        server.hints_cache[uri] = []
                        server.diagnostics_cache[uri] = []
                    log(f"Unknown result format for {uri}: {json_result.content}")

                # 触发 inlay hints 刷新和诊断刷新
                try:
                    server.workspace_inlay_hint_refresh(None)
                except Exception as e:
                    log(f"Failed to refresh inlay hints: {e}")

                server.text_document_publish_diagnostics(
                    types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
                )

            except Exception as parse_error:
                log_error(f"Failed to parse typecheck result: {parse_error}")

                # 报告错误给用户
                error_message = f"Failed to parse typecheck result: {parse_error}"
                with server._lock:
                    server.hints_cache[uri] = []
                    server.diagnostics_cache[uri] = [create_file_level_error_diagnostic(error_message)]
                server.text_document_publish_diagnostics(
                    types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
                )

        except subprocess.TimeoutExpired:
            log_error(f"Python task timed out for {uri}")
            with server._lock:
                server.running_tasks[uri] = False
                server.hints_cache[uri] = []
                server.diagnostics_cache[uri] = [
                    create_file_level_error_diagnostic("Typecheck timed out (30s)")
                ]
            server.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
            )

        except Exception as e:
            log_error(f"Unexpected error in Python task for {uri}: {e}")
            with server._lock:
                server.running_tasks[uri] = False
                server.hints_cache[uri] = []
                server.diagnostics_cache[uri] = [create_file_level_error_diagnostic(f"Unexpected error: {e}")]
            server.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(uri=uri, diagnostics=server.diagnostics_cache.get(uri, []))
            )

    # 在线程池中执行
    server.executor.submit(run_typecheck)


# ============================================================
# 文档变更处理
# ============================================================


def handle_document_change(uri: str, event_name: str) -> None:
    """
    处理文档变更事件的通用函数

    Args:
        uri: 文档 URI
        event_name: 事件名称（用于日志）
    """
    file_path = uri_to_fs_path(uri)

    # 只为 RECOGNIZED_EXTENSION 扩展名的文件触发刷新
    if not file_path.endswith(RECOGNIZED_EXTENSION):
        return

    log(f"[Event: {event_name}] Triggering Python task: {uri}")

    document = server.workspace.get_text_document(uri)
    if document:
        text = document.source
        call_python_cutile_typecheck_async(text, file_path, uri)


# ============================================================
# LSP 事件处理器
# ============================================================


@server.feature(types.INITIALIZE)
def on_initialize(params: types.InitializeParams) -> types.InitializeResult:
    """初始化处理"""
    log("LSP Server initializing...")

    # 从初始化参数中获取 Python 解释器路径
    init_options = params.initialization_options
    if init_options and isinstance(init_options, dict) and init_options.get("pythonPath"):
        server.python_executable = init_options["pythonPath"]
        log(f"Python interpreter set to: {server.python_executable}")
    else:
        log("No Python interpreter provided, hints will be disabled")

    return types.InitializeResult(
        capabilities=types.ServerCapabilities(
            # 文档同步方式：完整同步
            text_document_sync=types.TextDocumentSyncOptions(
                open_close=True,
                change=types.TextDocumentSyncKind.Full,
                save=types.SaveOptions(include_text=True),
            ),
            # 启用 Inlay Hint 功能
            inlay_hint_provider=True,
            # 启用诊断功能
            diagnostic_provider=types.DiagnosticOptions(
                identifier="cutile-typeviz", inter_file_dependencies=False, workspace_diagnostics=False
            ),
        )
    )


@server.feature(types.INITIALIZED)
def on_initialized(params: types.InitializedParams) -> None:
    """初始化完成"""
    log("LSP Server initialized")


# 监听 Python 解释器路径变化的自定义通知
@server.feature("cutile/pythonPathChanged")
def on_python_path_changed(params: Any) -> None:
    """处理 Python 解释器路径变化"""
    old_path = server.python_executable

    if isinstance(params, dict):
        server.python_executable = params.get("pythonPath")
    else:
        server.python_executable = None

    log(f"Python interpreter changed: {old_path} -> {server.python_executable}")

    # 如果有新的解释器，触发所有打开文档的刷新
    if server.python_executable:
        for uri, doc in server.workspace.text_documents.items():
            file_path = uri_to_fs_path(uri)
            if file_path.endswith(RECOGNIZED_EXTENSION):
                log(f"Refreshing hints for {uri} due to interpreter change")
                call_python_cutile_typecheck_async(doc.source, file_path, uri)
    else:
        # 如果解释器被清除，清空所有缓存
        with server._lock:
            server.hints_cache.clear()
            server.diagnostics_cache.clear()

        for uri in server.workspace.text_documents.keys():
            server.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=[]))

        try:
            server.workspace_inlay_hint_refresh(None)
        except Exception as e:
            log(f"Failed to refresh inlay hints: {e}")


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def on_inlay_hint(params: types.InlayHintParams) -> List[types.InlayHint]:
    """Inlay Hint 请求处理"""
    uri = params.text_document.uri

    document = server.workspace.get_text_document(uri)
    if not document:
        return []

    # 将 URI 转换为文件路径（这就是正在监控的文件）
    file_path = uri_to_fs_path(uri)

    # 只为 RECOGNIZED_EXTENSION 扩展名的文件提供 Inlay Hints
    if not file_path.endswith(RECOGNIZED_EXTENSION):
        return []

    # 只从缓存获取结果，不触发 Python 任务（Python 任务只在保存和内容变更时触发）
    with server._lock:
        cached_result = server.hints_cache.get(uri)

    # 如果缓存中没有结果或者结果不是列表（可能是错误信息），返回空数组
    if not cached_result or not isinstance(cached_result, list):
        return []

    python_result: List[Hint] = cached_result

    # 获取请求的范围
    start_line = params.range.start.line
    end_line = params.range.end.line

    # 按行分组过滤结果
    line_groups: Dict[int, List[Hint]] = {}

    # 首先按行分组
    for item in python_result:
        if item.line < start_line or item.line > end_line:
            continue

        if item.line not in line_groups:
            line_groups[item.line] = []
        line_groups[item.line].append(item)

    filtered_result: List[Hint] = []

    # 检查每行的 hints
    for line, hints_in_line in line_groups.items():
        if len(hints_in_line) >= 2:
            # 如果有 2 个或更多 hints，检查它们的 ty 和起始列是否都相同
            first_ty = hints_in_line[0].ty
            first_col_start = hints_in_line[0].col_start
            all_same = True

            for i in range(1, len(hints_in_line)):
                if hints_in_line[i].ty != first_ty or hints_in_line[i].col_start != first_col_start:
                    all_same = False
                    break

            # 只有当所有 hints 具有相同的 ty 和起始列时才保留一个
            if all_same:
                filtered_result.append(hints_in_line[0])
        else:
            # 只有一个 hint，直接保留
            filtered_result.extend(hints_in_line)

    hints: List[types.InlayHint] = []

    # 遍历过滤后的结果创建 Inlay Hints
    for item in filtered_result:
        # 创建带有来自 Python 的类型信息的 Inlay Hint
        hint = types.InlayHint(
            # 位置：在指定的列
            position=types.Position(line=item.line - 1, character=item.col_start),
            # 显示内容：类型信息
            label=f"{item.ty}",
            # 类型：设为 Type
            kind=types.InlayHintKind.Type,
            # 在右侧显示（paddingRight 在右侧添加间距）
            padding_right=True,
        )

        hints.append(hint)

    return hints


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def on_did_save(params: types.DidSaveTextDocumentParams) -> None:
    """监听文档保存，触发 Inlay Hint 刷新"""
    handle_document_change(params.text_document.uri, "onDidSave")


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def on_did_change(params: types.DidChangeTextDocumentParams) -> None:
    """监听文档变化，触发 Inlay Hint 刷新"""
    handle_document_change(params.text_document.uri, "onDidChangeContent")


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def on_did_open(params: types.DidOpenTextDocumentParams) -> None:
    """监听文档打开"""
    handle_document_change(params.text_document.uri, "onDidOpen")


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def on_did_close(params: types.DidCloseTextDocumentParams) -> None:
    """监听文档关闭，清理诊断信息"""
    uri = params.text_document.uri

    # 清理缓存
    with server._lock:
        if uri in server.hints_cache:
            del server.hints_cache[uri]
        if uri in server.diagnostics_cache:
            del server.diagnostics_cache[uri]

    # 发送空诊断信息以清除显示
    server.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=[]))


# ============================================================
# 主入口
# ============================================================


def main():
    """主入口函数"""
    log("cuTile typeviz Server started (Python backend)")
    server.start_io()


if __name__ == "__main__":
    main()
