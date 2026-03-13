"""
MCP 异步调用模块

使用线程池实现并发调用 MCP 工具，不阻塞 FastAPI 事件循环。

核心原理：
- 事件循环：单线程内管理所有并发请求
- 线程池：将同步的 MCP 调用放到线程池执行
- 结果：既不阻塞事件循环，又能并发处理多个请求

使用方法：
    from app.mcp.async_pool import call_tool_async

    # 在 async 函数中调用
    result = await call_tool_async("symptom_associations", {"symptoms": ["头痛"]})
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from app.core.logging import logger

_thread_pool: Optional[ThreadPoolExecutor] = None


def get_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """获取线程池单例"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"MCP 线程池已初始化，大小: {max_workers}")
    return _thread_pool


async def call_tool_async(tool_name: str, arguments: dict) -> str:
    """
    异步调用 MCP 工具

    原理：
    1. 调用 call_tool_sync 在线程池中执行
    2. 线程池中的同步操作不会阻塞主事件循环
    3. 其他请求可以在等待期间被处理
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        get_thread_pool(), lambda: call_tool_sync(tool_name, arguments)
    )


def call_tool_sync(tool_name: str, arguments: dict) -> str:
    """同步调用 MCP 工具（在线程池中执行）"""
    import subprocess
    import json

    try:
        proc = subprocess.Popen(
            ["python", "-m", "app.mcp.patient_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
        )

        stdout, _ = proc.communicate(input=request, timeout=30)
        return stdout
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return json.dumps({"error": str(e)})
