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
    logger.info("tool_calling: password_verified=%s", state.password_verified)

    # 检查密码验证状态（每次查询都需要验证）
    # 如果 password_verified=True（验证成功后的请求），则跳过密码检查
    if not state.password_verified:
        logger.info("tool_calling: 需要密码验证")
        return {
            "need_password_input": True,
            "password_prompt": "查看病例需要密码验证，请输入密码（888）",
            "password_retry_count": state.password_retry_count,
        }

    # 密码已验证，执行 MCP 查询
    logger.info("tool_calling: 密码已验证，执行查询")

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
