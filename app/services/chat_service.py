from typing import Optional, Dict, Any

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

# 从 config 导入 langfuse 配置
from app.core.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

logger.info(
    "Langfuse env detected (public=%s, secret=%s, base_url=%s)",
    bool(LANGFUSE_PUBLIC_KEY),
    bool(LANGFUSE_SECRET_KEY),
    LANGFUSE_BASE_URL,
)

# 初始化 langfuse 客户端
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_BASE_URL,
)
langfuse_handler = CallbackHandler()

_app = build_app(checkpointer)
_session_manager = SessionManager()


def _extract_reply(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return messages[-1].content if messages else ""


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
    password_verified: bool = False,
) -> Dict[str, Any]:
    """
    Synchronous entry for chat; backend LangGraph uses synchronous invoke.
    """
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
        "callbacks": [langfuse_handler],
    }

    state = _app.invoke(inputs, config=config)
    if isinstance(state, dict):
        state = AppState(**state)

    logger.info(
        "chat_once state after invoke: password_verified=%s, need_password_input=%s",
        state.password_verified,
        state.need_password_input,
    )

    # 检查是否需要密码验证
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

    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "reply": reply,
        "intent_result": intent_dict,
        "used_docs": {
            "medical": _dump_docs(state.medical_docs),
            "process": _dump_docs(state.process_docs),
        },
    }


def get_session_manager() -> SessionManager:
    return _session_manager
