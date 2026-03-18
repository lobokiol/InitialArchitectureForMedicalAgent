from typing import Optional, List, AsyncGenerator

import asyncio
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.domain.models import IntentResult, RetrievedDoc, User
from app.middleware.auth import get_current_user_optional
from app.services import chat_service


class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    message: str
    password_verified: bool = True


class VerifyPasswordRequest(BaseModel):
    user_id: str
    thread_id: Optional[str] = None
    password: str
    retry_count: int = 0


class VerifyPasswordResponse(BaseModel):
    success: bool
    message: str
    retry_count: int = 0
    locked: bool = False


class PasswordRequiredResponse(BaseModel):
    need_password_input: bool = True
    password_prompt: str


class UsedDocs(BaseModel):
    medical: List[RetrievedDoc] = []
    process: List[RetrievedDoc] = []


class DiagnosisInfo(BaseModel):
    type: str = "in_progress"
    completed: bool = False
    terminated: bool = False
    termination_reason: Optional[str] = None
    slots: dict = {}
    risk_level: str = "none"
    risk_signals: List[str] = []
    question_count: int = 0
    missing_slots: List[str] = []


class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    reply: str = ""
    intent_result: Optional[IntentResult] = None
    used_docs: UsedDocs = UsedDocs(medical=[], process=[])
    need_password_input: bool = False
    password_prompt: str = ""
    password_retry_count: int = 0
    diagnosis: Optional[DiagnosisInfo] = None


router = APIRouter(prefix="/chat", tags=["chat"])

PASSWORD = "888"
MAX_RETRIES = 2


@router.post("/verify-password", response_model=VerifyPasswordResponse)
async def verify_password(body: VerifyPasswordRequest):
    """验证病例查询密码"""
    if body.retry_count >= MAX_RETRIES:
        return VerifyPasswordResponse(
            success=False,
            message="密码错误次数过多，查询已结束",
            retry_count=body.retry_count,
            locked=True,
        )

    if body.password == PASSWORD:
        return VerifyPasswordResponse(
            success=True,
            message="验证成功",
            retry_count=0,
        )
    else:
        return VerifyPasswordResponse(
            success=False,
            message=f"密码错误，剩余尝试次数: {MAX_RETRIES - body.retry_count - 1}",
            retry_count=body.retry_count + 1,
        )


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    body: ChatRequest, current_user: Optional[User] = Depends(get_current_user_optional)
):
    user_id: str = body.user_id or (
        current_user.user_id if current_user else "anonymous"
    )
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        chat_service.chat_once,
        user_id,
        body.thread_id,
        body.message,
        body.password_verified,
    )
    return result


@router.post("/stream")
async def chat_stream_endpoint(
    body: ChatRequest, current_user: Optional[User] = Depends(get_current_user_optional)
):
    """流式输出接口 - Server-Sent Events"""
    user_id: str = body.user_id or (
        current_user.user_id if current_user else "anonymous"
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """生成 SSE 格式的事件流"""
        try:
            async for chunk in chat_service.chat_stream(
                user_id,
                body.thread_id,
                body.message,
                body.password_verified,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {"error": str(e), "type": "error"}
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
