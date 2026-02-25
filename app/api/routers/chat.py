from typing import Optional, List

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

from app.domain.models import IntentResult, RetrievedDoc
from app.services import chat_service


class ChatRequest(BaseModel):
    user_id: str
    thread_id: Optional[str] = None
    message: str
    password_verified: bool = False


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


class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    reply: str = ""
    intent_result: Optional[IntentResult] = None
    used_docs: UsedDocs = UsedDocs(medical=[], process=[])
    need_password_input: bool = False
    password_prompt: str = ""
    password_retry_count: int = 0


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
async def chat_endpoint(body: ChatRequest):
    # Run sync chat_once in thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        chat_service.chat_once,
        body.user_id,
        body.thread_id,
        body.message,
        body.password_verified,
    )
    return result
