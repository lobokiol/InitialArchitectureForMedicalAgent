"""
JWT Token 工具模块
提供 JWT Token 生成和验证功能
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from pydantic import BaseModel


class TokenPayload(BaseModel):
    """JWT Token 载荷"""

    user_id: str
    openid: Optional[str] = None
    phone: Optional[str] = None
    role: str = "patient"
    type: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None


class JWTHandler:
    """JWT Token 处理器"""

    def __init__(self):
        self.secret_key = os.getenv(
            "JWT_SECRET_KEY", "hospital-guidance-secret-key-2024"
        )
        self.algorithm = "HS256"
        self.access_token_expire = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60)
        )  # 60分钟
        self.refresh_token_expire = int(
            os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7)
        )  # 7天

    def create_access_token(self, user_id: str, **kwargs) -> str:
        """创建访问令牌"""
        payload = {
            "user_id": user_id,
            "type": "access",
            "exp": int(time.time()) + self.access_token_expire * 60,
            "iat": int(time.time()),
            **kwargs,
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """创建刷新令牌"""
        payload = {
            "user_id": user_id,
            "type": "refresh",
            "exp": int(time.time()) + self.refresh_token_expire * 24 * 60 * 60,
            "iat": int(time.time()),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_token_pair(self, user_id: str, **extra_data) -> Dict[str, str]:
        """创建令牌对"""
        extra_data.pop("type", None)
        extra_data.pop("exp", None)
        extra_data.pop("iat", None)

        return {
            "access_token": self.create_access_token(user_id, **extra_data),
            "refresh_token": self.create_refresh_token(user_id),
            "token_type": "Bearer",
            "expires_in": self.access_token_expire * 60,
        }

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """验证Token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """解码Token（不验证）"""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception:
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """刷新访问令牌"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.type != "refresh":
            return None

        return self.create_token_pair(payload.user_id)


# 全局单例
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """获取JWT处理器单例"""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler


def create_tokens(user_id: str, **extra_data) -> Dict[str, str]:
    """创建令牌对的便捷函数"""
    return get_jwt_handler().create_token_pair(user_id, **extra_data)


def verify_token(token: str) -> Optional[TokenPayload]:
    """验证Token的便捷函数"""
    return get_jwt_handler().verify_token(token)
