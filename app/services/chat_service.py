from typing import Optional, Dict, Any, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.core.logging import logger
from app.domain.models import AppState, IntentResult, RetrievedDoc
from app.graph.builder import build_app
from app.infra.redis_client import checkpointer
from app.sessions.manager import SessionManager
import os
from dotenv import load_dotenv


def _ensure_no_proxy_hosts() -> None:
    hosts = {"localhost", "127.0.0.1"}
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key)
        if existing:
            values = {part.strip() for part in existing.split(",") if part.strip()}
            if hosts.issubset(values):
                continue
            values.update(hosts)
            os.environ[key] = ",".join(sorted(values))
        else:
            os.environ[key] = ",".join(sorted(hosts))


_ensure_no_proxy_hosts()
load_dotenv(override=False)

langfuse_handler = None

_app = build_app(checkpointer)
_session_manager = SessionManager()


def _extract_reply(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                return str(content[0]) if content else ""
            elif not isinstance(content, str):
                return str(content) if content else ""
            else:
                return content
    last_message_content = messages[-1].content
    if isinstance(last_message_content, list):
        return str(last_message_content[0]) if last_message_content else ""
    elif not isinstance(last_message_content, str):
        return str(last_message_content) if last_message_content else ""
    else:
        return last_message_content


def _ensure_thread(user_id: str, thread_id: Optional[str]) -> str:
    if thread_id:
        _session_manager.set_current_thread(user_id, thread_id)
        return thread_id
    existing = _session_manager.get_current_thread(user_id)
    if existing:
        return existing
    return _session_manager.create_thread(user_id, title="默认对话")


def chat_once(
    user_id: str,
    thread_id: Optional[str],
    message: str,
    password_verified: bool = True,
) -> Dict[str, Any]:
    logger.info(
        "chat_once called (user_id=%s, thread_id=%s, password_verified=%s)",
        user_id,
        thread_id,
        password_verified,
    )

    thread_id = _ensure_thread(user_id, thread_id)

    inputs = {
        "messages": [HumanMessage(content=message)],
        "password_verified": password_verified,
    }
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
    }

    state = _app.invoke(inputs, config=config)
    if isinstance(state, dict):
        state = AppState(**state)

    logger.info(
        "chat_once state after invoke: password_verified=%s, need_password_input=%s",
        state.password_verified,
        state.need_password_input,
    )

    if state.need_password_input:
        return {
            "user_id": user_id,
            "thread_id": thread_id,
            "reply": "",
            "need_password_input": True,
            "password_prompt": state.password_prompt,
            "password_retry_count": state.password_retry_count,
            "intent_result": None,
            "used_docs": {"medical": [], "process": []},
        }

    reply = _extract_reply(state.messages)

    _session_manager.touch_thread(thread_id)

    def _dump_docs(docs: list[RetrievedDoc]):
        return [d.model_dump() for d in docs]

    intent_dict = (
        state.intent_result.model_dump()
        if isinstance(state.intent_result, IntentResult)
        else None
    )
    diagnosis_info = None
    if hasattr(state, "diagnosis_slots") and state.diagnosis_slots:
        diagnosis_info = {
            "type": state.diagnosis_type
            if hasattr(state, "diagnosis_type")
            else "in_progress",
            "completed": state.diagnosis_completed
            if hasattr(state, "diagnosis_completed")
            else False,
            "terminated": state.diagnosis_terminated
            if hasattr(state, "diagnosis_terminated")
            else False,
            "termination_reason": state.diagnosis_termination_reason
            if hasattr(state, "diagnosis_termination_reason")
            else None,
            "slots": state.diagnosis_slots.to_dict() if state.diagnosis_slots else {},
            "risk_level": state.diagnosis_risk_level
            if hasattr(state, "diagnosis_risk_level")
            else "none",
            "risk_signals": state.diagnosis_risk_signals
            if hasattr(state, "diagnosis_risk_signals")
            else [],
            "question_count": state.diagnosis_question_count
            if hasattr(state, "diagnosis_question_count")
            else 0,
            "missing_slots": state.diagnosis_missing_slots
            if hasattr(state, "diagnosis_missing_slots")
            else [],
        }

    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "reply": reply,
        "intent_result": intent_dict,
        "used_docs": {
            "medical": _dump_docs(state.medical_docs),
            "process": _dump_docs(state.process_docs),
        },
        "diagnosis": diagnosis_info,
    }


async def chat_stream(
    user_id: str,
    thread_id: Optional[str],
    message: str,
    password_verified: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """流式版本的对话服务"""
    logger.info(
        "chat_stream called (user_id=%s, thread_id=%s, password_verified=%s)",
        user_id,
        thread_id,
        password_verified,
    )

    thread_id = _ensure_thread(user_id, thread_id)

    inputs = {
        "messages": [HumanMessage(content=message)],
        "password_verified": password_verified,
    }
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
    }

    # 发送开始标记
    yield {
        "type": "start",
        "thread_id": thread_id,
    }

    try:
        # 使用 LangGraph 的流式API
        async for event in _app.astream_events(inputs, config=config, version="v1"):
            kind = event.get("event")

            # 处理不同的事件类型
            if kind == "on_chat_model_stream":
                # LLM 流式输出 token
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content"):
                    token = chunk.content
                    if token:
                        yield {
                            "type": "token",
                            "token": token,
                        }

            elif kind == "on_chain_end":
                # 节点执行完成，可以返回中间状态
                output = event.get("data", {}).get("output")
                if output:
                    yield {
                        "type": "node_complete",
                        "node": event.get("name"),
                        "metadata": output,
                    }

        # 获取最终状态并返回完整响应
        final_state = _app.invoke(inputs, config=config)
        if isinstance(final_state, dict):
            final_state = AppState(**final_state)

        # 提取回复和诊断信息
        reply = _extract_reply(final_state.messages)

        diagnosis_info = None
        if hasattr(final_state, "diagnosis_slots") and final_state.diagnosis_slots:
            diagnosis_info = {
                "type": getattr(final_state, "diagnosis_type", "in_progress"),
                "completed": getattr(final_state, "diagnosis_completed", False),
                "terminated": getattr(final_state, "diagnosis_terminated", False),
                "termination_reason": getattr(
                    final_state, "diagnosis_termination_reason", None
                ),
                "slots": final_state.diagnosis_slots.to_dict()
                if final_state.diagnosis_slots
                else {},
                "risk_level": getattr(final_state, "diagnosis_risk_level", "none"),
                "risk_signals": getattr(final_state, "diagnosis_risk_signals", []),
                "question_count": getattr(final_state, "diagnosis_question_count", 0),
                "missing_slots": getattr(final_state, "diagnosis_missing_slots", []),
            }

        # 返回完整的响应对象
        yield {
            "type": "complete",
            "user_id": user_id,
            "thread_id": thread_id,
            "reply": reply,
            "diagnosis": diagnosis_info,
        }

        # 触摸会话保持活跃
        _session_manager.touch_thread(thread_id)

    except Exception as e:
        logger.exception("chat_stream 发生错误")
        yield {
            "type": "error",
            "message": str(e),
        }


def get_session_manager() -> SessionManager:
    return _session_manager
