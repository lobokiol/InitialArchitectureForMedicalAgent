import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

# 禁用 MCP client 日志，避免干扰
logging.basicConfig(level=logging.CRITICAL)

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


def infer_department_mcp(symptoms: list, top_k: int = 3) -> str:
    """
    基于多症状推理推荐科室（带置信度）
    """
    try:
        client = MCPClient()
        return client.call_tool(
            "infer_department",
            {"symptoms": symptoms, "top_k": top_k},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def get_symptom_dept_probability_mcp(symptom: str) -> str:
    """
    获取单个症状的科室概率分布
    """
    try:
        client = MCPClient()
        return client.call_tool(
            "get_symptom_dept_probability",
            {"symptom": symptom},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def get_possible_diseases_mcp(symptoms: list, limit: int = 10) -> str:
    """
    根据症状查询可能的疾病
    """
    try:
        client = MCPClient()
        return client.call_tool(
            "get_possible_diseases",
            {"symptoms": symptoms, "limit": limit},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def calculate_confidence_mcp(symptoms: list) -> str:
    """
    计算多症状推理的置信度
    """
    try:
        client = MCPClient()
        return client.call_tool(
            "calculate_confidence",
            {"symptoms": symptoms},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def semantic_match_symptoms_mcp(
    query_text: str, top_k: int = 5, threshold: float = 0.5
) -> str:
    """Neo4j 向量语义匹配"""
    try:
        client = MCPClient()
        return client.call_tool(
            "semantic_match_symptoms",
            {"query_text": query_text, "top_k": top_k, "threshold": threshold},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def pg_get_patient_by_name_mcp(name: str) -> str:
    """PostgreSQL - 根据姓名查询患者"""
    try:
        client = MCPClient()
        return client.call_tool("pg_get_patient_by_name", {"name": name})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def pg_get_patient_by_id_mcp(patient_id: str) -> str:
    """PostgreSQL - 根据ID查询患者"""
    try:
        client = MCPClient()
        return client.call_tool("pg_get_patient_by_id", {"patient_id": patient_id})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def pg_get_patient_history_mcp(patient_id: str, limit: int = 20) -> str:
    """PostgreSQL - 查询患者就诊历史"""
    try:
        client = MCPClient()
        return client.call_tool(
            "pg_get_patient_history",
            {"patient_id": patient_id, "limit": limit},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def pg_search_patients_mcp(keyword: str, limit: int = 10) -> str:
    """PostgreSQL - 搜索患者"""
    try:
        client = MCPClient()
        return client.call_tool(
            "pg_search_patients", {"keyword": keyword, "limit": limit}
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def milvus_search_mcp(query: str, top_k: int = 15) -> str:
    """Milvus 向量检索"""
    try:
        client = MCPClient()
        return client.call_tool("milvus_search", {"query": query, "top_k": top_k})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def es_search_mcp(query: str, size: int = 50) -> str:
    """Elasticsearch 检索"""
    try:
        client = MCPClient()
        return client.call_tool("es_search", {"query": query, "size": size})
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'


def kg_rag_fusion_mcp(symptoms: list, user_query: str = "", top_k: int = 3) -> str:
    """KG + RAG 综合推理"""
    try:
        client = MCPClient()
        return client.call_tool(
            "kg_rag_fusion",
            {"symptoms": symptoms, "user_query": user_query, "top_k": top_k},
        )
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
        return f'{{"error": "MCP调用失败: {str(e)}"}}'
