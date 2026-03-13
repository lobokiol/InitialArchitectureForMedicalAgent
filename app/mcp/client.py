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


def symptom_associations_mcp(symptoms: list) -> str:
    try:
        client = MCPClient()
        return client.call_tool("symptom_associations", {"symptoms": symptoms})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def department_by_symptom_mcp(symptom: str) -> str:
    try:
        client = MCPClient()
        return client.call_tool("department_by_symptom", {"symptom": symptom})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def emergency_check_mcp(symptoms: list) -> str:
    try:
        client = MCPClient()
        return client.call_tool("emergency_check", {"symptoms": symptoms})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def symptom_search_mcp(keyword: str) -> str:
    try:
        client = MCPClient()
        return client.call_tool("symptom_search", {"keyword": keyword})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def full_symptom_info_mcp(symptom: str) -> str:
    try:
        client = MCPClient()
        return client.call_tool("full_symptom_info", {"symptom": symptom})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def hybrid_retrieval_mcp(query_text: str, known_symptoms: list = None) -> str:
    try:
        client = MCPClient()
        return client.call_tool(
            "hybrid_retrieval",
            {"query_text": query_text, "known_symptoms": known_symptoms or []},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def discriminative_symptoms_mcp(
    known_symptoms: list, candidate_symptoms: list, limit: int = 5
) -> str:
    try:
        client = MCPClient()
        return client.call_tool(
            "discriminative_symptoms",
            {
                "known_symptoms": known_symptoms,
                "candidate_symptoms": candidate_symptoms,
                "limit": limit,
            },
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'
