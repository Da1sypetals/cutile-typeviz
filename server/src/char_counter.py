#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSP Char Counter - Python 核心计算模块
计算每行出现最多的字符的个数，输出 JSON 格式
"""

import sys
import json
from collections import Counter


def get_max_char_count(line: str) -> int:
    """
    计算字符串中出现最多的字符的个数

    Args:
        line: 输入的字符串行

    Returns:
        出现最多的字符的个数
    """
    if not line:
        return 0

    # 使用 Counter 统计每个字符出现的次数
    char_counts = Counter(line)

    # 返回最大值
    return max(char_counts.values())


def main():
    """
    主函数：从 stdin 读取文本，计算每行的最大字符出现次数
    输出 JSON 格式：
    {
        "source": "python",
        "version": "1.0.0",
        "results": [
            {"line": 0, "maxCount": 5, "marker": "🐍"},
            {"line": 1, "maxCount": 3, "marker": "🐍"},
            ...
        ]
    }
    """
    try:
        # 从 stdin 读取全部输入
        text = sys.stdin.read()
        lines = text.split("\n")

        results = []
        for line_index, line in enumerate(lines):
            # 跳过空行
            if not line.strip():
                continue

            max_count = get_max_char_count(line)
            results.append(
                {
                    "line": line_index,
                    "maxCount": max_count,
                    "marker": "🐍",  # Python 蛇标记，用于区分
                }
            )

        # 输出 JSON 结果
        output = {"source": "python", "version": "1.0.0", "results": results}

        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        # 发生错误时，输出错误信息并以非零状态码退出（崩溃）
        error_output = {"error": True, "message": str(e), "type": type(e).__name__}
        print(json.dumps(error_output, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
