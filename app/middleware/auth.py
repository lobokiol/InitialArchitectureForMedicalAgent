"""
认证中间件
提供 JWT Token 验证和用户认证功能
"""

import os
from typing import Optional, Callable
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.jwt_utils import get_jwt_handler, TokenPayload
from app.infra.postgres_client import get_user_client
from app.domain.models import User


security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""

    PUBLIC_PATHS = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v1/auth/wechat/login",
        "/api/v1/wechat/callback",
    ]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if any(path.startswith(public) for public in self.PUBLIC_PATHS):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header"},
            )

        token = auth_header.replace("Bearer ", "")
        jwt_handler = get_jwt_handler()
        payload = jwt_handler.verify_token(token)

        if not payload:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or expired token"}
            )

        user_client = get_user_client()
        user_data = user_client.get_user_by_user_id(payload.user_id)

        if not user_data:
            return JSONResponse(status_code=401, content={"detail": "User not found"})

        request.state.user = User(**user_data)
        request.state.user_id = payload.user_id

        return await call_next(request)


async def get_current_user(request: Request) -> User:
    """获取当前登录用户（依赖注入）"""
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user


async def get_current_user_optional(request: Request) -> Optional[User]:
    """获取当前登录用户（可选）"""
    if hasattr(request.state, "user"):
        return request.state.user
    return None


def require_roles(*roles: str):
    """角色权限检查装饰器"""

    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=403, detail=f"Requires one of roles: {', '.join(roles)}"
            )
        return user

    return role_checker
