"""
认证路由
提供用户登录、注册、Token刷新、微信登录等功能
"""

import os
import uuid
import httpx
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from app.core.jwt_utils import get_jwt_handler, create_tokens
from app.infra.postgres_client import get_user_client
from app.domain.models import (
    User,
    UserLoginRequest,
    UserLoginResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
)
from app.middleware.auth import get_current_user


router = APIRouter(prefix="/auth", tags=["auth"])


class WeChatLoginRequest(BaseModel):
    """微信登录请求"""

    code: str


class WeChatLoginResponse(BaseModel):
    """微信登录响应"""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: User
    is_new_user: bool = False


class PhoneLoginRequest(BaseModel):
    """手机号登录请求"""

    phone: str
    verify_code: Optional[str] = None  # 短信验证码（可选）


class RegisterRequest(BaseModel):
    """注册请求"""

    phone: str
    verify_code: Optional[str] = None
    nickname: Optional[str] = None
    password: Optional[str] = None


async def get_wechat_openid(code: str) -> tuple[str, str]:
    """调用微信服务器获取openid和session_key"""
    app_id = os.getenv("WECHAT_APP_ID", "")
    app_secret = os.getenv("WECHAT_APP_SECRET", "")

    if not app_id or not app_secret:
        raise HTTPException(status_code=500, detail="WeChat not configured")

    url = "https://api.weixin.qq.com/sns/jscode2session"
    params = {
        "appid": app_id,
        "secret": app_secret,
        "js_code": code,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        data = response.json()

        if "errcode" in data and data["errcode"] != 0:
            raise HTTPException(
                status_code=400,
                detail=f"WeChat API error: {data.get('errmsg', 'unknown error')}",
            )

        return data.get("openid", ""), data.get("session_key", "")


@router.post("/wechat/login", response_model=WeChatLoginResponse)
async def wechat_login(request: WeChatLoginRequest):
    """微信小程序登录"""
    try:
        openid, session_key = await get_wechat_openid(request.code)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to get WeChat session: {str(e)}"
        )

    if not openid:
        raise HTTPException(status_code=400, detail="Failed to get openid from WeChat")

    user_client = get_user_client()
    user_data = user_client.get_user_by_openid(openid)

    is_new_user = False
    if not user_data:
        user_id = f"wx_{uuid.uuid4().hex[:12]}"
        user_client.create_user(
            user_id=user_id, openid=openid, nickname=f"User_{user_id[-4:]}"
        )
        user_data = user_client.get_user_by_user_id(user_id)
        is_new_user = True
    else:
        user_client.update_last_login(user_data["user_id"])
        user_data = user_client.get_user_by_user_id(user_data["user_id"])

    if not user_data:
        raise HTTPException(status_code=500, detail="Failed to create or retrieve user")

    tokens = create_tokens(user_data["user_id"], openid=openid)

    return WeChatLoginResponse(
        **tokens, user=User(**user_data), is_new_user=is_new_user
    )


@router.post("/login", response_model=UserLoginResponse)
async def login(request: UserLoginRequest):
    """手机号/密码登录"""
    user_client = get_user_client()

    if request.phone:
        user_data = user_client.get_user_by_phone(request.phone)
    else:
        raise HTTPException(status_code=400, detail="Phone number required")

    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    user_client.update_last_login(user_data["user_id"])
    user_data = user_client.get_user_by_user_id(user_data["user_id"])

    tokens = create_tokens(user_data["user_id"])

    return UserLoginResponse(**tokens, user=User(**user_data))


@router.post("/register", response_model=UserLoginResponse)
async def register(request: RegisterRequest):
    """用户注册"""
    user_client = get_user_client()

    existing = user_client.get_user_by_phone(request.phone)
    if existing:
        raise HTTPException(status_code=400, detail="Phone already registered")

    user_id = f"user_{uuid.uuid4().hex[:12]}"
    user_client.create_user(
        user_id=user_id,
        phone=request.phone,
        nickname=request.nickname or f"User_{user_id[-4:]}",
    )

    user_data = user_client.get_user_by_user_id(user_id)
    if not user_data:
        raise HTTPException(status_code=500, detail="Failed to create user")

    tokens = create_tokens(user_id)

    return UserLoginResponse(**tokens, user=User(**user_data))


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(request: TokenRefreshRequest):
    """刷新访问令牌"""
    jwt_handler = get_jwt_handler()
    result = jwt_handler.refresh_access_token(request.refresh_token)

    if not result:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return TokenRefreshResponse(
        access_token=result["access_token"], expires_in=result["expires_in"]
    )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """退出登录（可扩展为加入黑名单）"""
    return {"message": "Logged out successfully"}
