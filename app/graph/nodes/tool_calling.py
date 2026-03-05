from app.core.llm import get_chat_llm
from typing import Any, Dict
from app.core.logging import logger
from app.domain.models import AppState, RetrievedDoc
# from app.mcp.client import get_patient_history_mcp, get_patient_by_id_mcp

# 暂时禁用 MCP 工具调用
TOOLS = []

llm_with_tools = get_chat_llm().bind_tools(TOOLS)


def _execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "get_patient_history":
        return get_patient_history_mcp(arguments.get("patient_name", ""))
    elif tool_name == "get_patient_by_id":
        return get_patient_by_id_mcp(arguments.get("patient_id", ""))
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
