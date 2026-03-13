from app.core.llm import get_lightweight_llm
from typing import Any, Dict, List
from app.core.logging import logger
from app.domain.models import AppState, RetrievedDoc
from app.mcp.client import (
    get_patient_history_mcp,
    get_patient_by_id_mcp,
    symptom_associations_mcp,
    department_by_symptom_mcp,
    emergency_check_mcp,
    symptom_search_mcp,
    full_symptom_info_mcp,
    hybrid_retrieval_mcp,
    discriminative_symptoms_mcp,
)
from langchain_core.tools import StructuredTool


def _create_tools() -> List[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=lambda symptoms: symptom_associations_mcp(symptoms),
            name="symptom_associations",
            description="批量查询症状的伴随症状、推荐科室和危急检查。当需要了解多个症状的相关信息时调用此工具。",
        ),
        StructuredTool.from_function(
            func=lambda symptom: department_by_symptom_mcp(symptom),
            name="department_by_symptom",
            description="根据症状查询推荐科室。当用户询问应该挂什么科时调用此工具。",
        ),
        StructuredTool.from_function(
            func=lambda symptoms: emergency_check_mcp(symptoms),
            name="emergency_check",
            description="检查危急症状。当用户描述的症状可能危急时调用此工具。",
        ),
        StructuredTool.from_function(
            func=lambda keyword: symptom_search_mcp(keyword),
            name="symptom_search",
            description="根据关键词搜索症状。当需要匹配用户描述的症状到标准症状名称时调用。",
        ),
        StructuredTool.from_function(
            func=lambda symptom: full_symptom_info_mcp(symptom),
            name="full_symptom_info",
            description="获取症状的完整信息（科室推荐+伴随症状）。",
        ),
        StructuredTool.from_function(
            func=lambda query_text, known_symptoms=None: hybrid_retrieval_mcp(
                query_text, known_symptoms or []
            ),
            name="hybrid_retrieval",
            description="混合检索（向量搜索+图推理）。当需要结合语义搜索和知识图谱推理时调用。",
        ),
        StructuredTool.from_function(
            func=lambda known_symptoms, candidate_symptoms, limit=5: (
                discriminative_symptoms_mcp(known_symptoms, candidate_symptoms, limit)
            ),
            name="discriminative_symptoms",
            description="获取判别性症状，用于动态生成追问问题。",
        ),
    ]


TOOLS = _create_tools()

llm_with_tools = get_lightweight_llm().bind_tools(TOOLS)


def _execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "get_patient_history":
        return get_patient_history_mcp(arguments.get("patient_name", ""))
    elif tool_name == "get_patient_by_id":
        return get_patient_by_id_mcp(arguments.get("patient_id", ""))
    elif tool_name == "symptom_associations":
        return symptom_associations_mcp(arguments.get("symptoms", []))
    elif tool_name == "department_by_symptom":
        return department_by_symptom_mcp(arguments.get("symptom", ""))
    elif tool_name == "emergency_check":
        return emergency_check_mcp(arguments.get("symptoms", []))
    elif tool_name == "symptom_search":
        return symptom_search_mcp(arguments.get("keyword", ""))
    elif tool_name == "full_symptom_info":
        return full_symptom_info_mcp(arguments.get("symptom", ""))
    elif tool_name == "hybrid_retrieval":
        return hybrid_retrieval_mcp(
            arguments.get("query_text", ""), arguments.get("known_symptoms", [])
        )
    elif tool_name == "discriminative_symptoms":
        return discriminative_symptoms_mcp(
            arguments.get("known_symptoms", []),
            arguments.get("candidate_symptoms", []),
            arguments.get("limit", 5),
        )
    return ""


MAX_PASSWORD_RETRIES = 2


def tool_calling_node(state: AppState) -> Dict[str, Any]:
    """工具调用节点 - 判断是否需要调用工具并执行(MCP方式)"""
    logger.info(">>> Enter node: tool_calling")

    # 直接执行 MCP 查询（移除密码验证）
    logger.info("tool_calling: 执行查询")

    query = state.messages[-1].content

    result = llm_with_tools.invoke(query)

    if not result.tool_calls:
        logger.info("tool_calling: 无需调用工具")
        return {
            "need_tool_call": False,
            "need_password_input": False,
        }

    logger.info("tool_calling: 需要调用工具 %s", result.tool_calls)

    tool_result_str = ""
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call.get("args", {})
        result_str = _execute_mcp_tool(tool_name, arguments)
        tool_result_str += result_str + "\n"

    return {
        "need_tool_call": True,
        "need_password_input": False,
        "medical_docs": [
            RetrievedDoc(
                id="tool_call",
                source="tool",
                title="患者病例查询结果",
                content=tool_result_str,
                score=1.0,
            )
        ],
    }
