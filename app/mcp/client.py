import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.logging import logger


class MCPClient:
    def __init__(
        self, server_command: str = "python", server_args: Optional[list] = None
    ):
        if server_args is None:
            server_args = ["-m", "app.mcp.patient_server"]
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args,
        )
        self._session: Optional[ClientSession] = None

    @asynccontextmanager
    async def _get_session(self):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        async def _call():
            async with self._get_session() as session:
                result = await session.call_tool(tool_name, arguments)
                if result.content:
                    return result.content[0].text
                return str(result)

        return asyncio.run(_call())

    def list_tools(self) -> list:
        async def _list():
            async with self._get_session() as session:
                tools = await session.list_tools()
                return tools.tools

        return asyncio.run(_list())


def get_patient_history_mcp(patient_name: str) -> str:
    try:
        client = MCPClient()
        return client.call_tool("get_patient_history", {"patient_name": patient_name})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def get_patient_by_id_mcp(patient_id: str) -> str:
    try:
        client = MCPClient()
        return client.call_tool("get_patient_by_id", {"patient_id": patient_id})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'
