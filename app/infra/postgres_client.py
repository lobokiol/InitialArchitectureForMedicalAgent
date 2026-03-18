from typing import List, Optional, Dict, Any
import json
from app.core import config
from app.core.logging import logger

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning(
        "SQLAlchemy not available, PostgreSQL patient database will use mock mode"
    )


class PostgresPatientClient:
    def __init__(self):
        self._engine = None
        self._Session = None
        if SQLALCHEMY_AVAILABLE:
            try:
                self._engine = create_engine(config.POSTGRES_URI)
                self._Session = sessionmaker(bind=self._engine)
                logger.info("PostgreSQL patient client initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}")

    def get_patient_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not self._Session:
            return None
        try:
            with self._Session() as session:
                result = session.execute(
                    text("SELECT * FROM patients WHERE name = :name LIMIT 1"),
                    {"name": name},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.warning(f"Query patient by name failed: {e}")
        return None

    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if not self._Session:
            return None
        try:
            with self._Session() as session:
                result = session.execute(
                    text("SELECT * FROM patients WHERE id = :id LIMIT 1"),
                    {"id": patient_id},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.warning(f"Query patient by id failed: {e}")
        return None

    def get_patient_history(
        self, patient_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        if not self._Session:
            return []
        try:
            with self._Session() as session:
                result = session.execute(
                    text("""
                        SELECT * FROM medical_records 
                        WHERE patient_id = :patient_id 
                        ORDER BY visit_date DESC 
                        LIMIT :limit
                    """),
                    {"patient_id": patient_id, "limit": limit},
                )
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Query patient history failed: {e}")
            return []

    def search_patients(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._Session:
            return []
        try:
            with self._Session() as session:
                result = session.execute(
                    text("""
                        SELECT * FROM patients 
                        WHERE name ILIKE :keyword OR phone ILIKE :keyword
                        LIMIT :limit
                    """),
                    {"keyword": f"%{keyword}%", "limit": limit},
                )
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Search patients failed: {e}")
            return []


_patient_client: Optional[PostgresPatientClient] = None


def get_patient_client() -> PostgresPatientClient:
    global _patient_client
    if _patient_client is None:
        _patient_client = PostgresPatientClient()
    return _patient_client


class PostgresUserClient:
    """PostgreSQL 用户管理客户端"""

    def __init__(self):
        self._engine = None
        self._Session = None
        if SQLALCHEMY_AVAILABLE:
            try:
                self._engine = create_engine(config.POSTGRES_URI)
                self._Session = sessionmaker(bind=self._engine)
                logger.info("PostgreSQL user client initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}")

    def create_user(
        self,
        user_id: str,
        openid: str = None,
        phone: str = None,
        nickname: str = None,
        role: str = "patient",
    ) -> bool:
        """创建新用户"""
        if not self._Session:
            return False
        try:
            with self._Session() as session:
                session.execute(
                    text("""
                        INSERT INTO users (user_id, openid, phone, nickname, role, created_at, updated_at, is_active)
                        VALUES (:user_id, :openid, :phone, :nickname, :role, NOW(), NOW(), TRUE)
                        ON CONFLICT (user_id) DO NOTHING
                    """),
                    {
                        "user_id": user_id,
                        "openid": openid,
                        "phone": phone,
                        "nickname": nickname,
                        "role": role,
                    },
                )
                session.commit()
                return True
        except Exception as e:
            logger.warning(f"Create user failed: {e}")
            return False

    def get_user_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据user_id获取用户"""
        if not self._Session:
            return None
        try:
            with self._Session() as session:
                result = session.execute(
                    text(
                        "SELECT * FROM users WHERE user_id = :user_id AND is_active = TRUE LIMIT 1"
                    ),
                    {"user_id": user_id},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.warning(f"Query user by user_id failed: {e}")
        return None

    def get_user_by_openid(self, openid: str) -> Optional[Dict[str, Any]]:
        """根据openid获取用户（微信登录用）"""
        if not self._Session:
            return None
        try:
            with self._Session() as session:
                result = session.execute(
                    text(
                        "SELECT * FROM users WHERE openid = :openid AND is_active = TRUE LIMIT 1"
                    ),
                    {"openid": openid},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.warning(f"Query user by openid failed: {e}")
        return None

    def get_user_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """根据手机号获取用户"""
        if not self._Session:
            return None
        try:
            with self._Session() as session:
                result = session.execute(
                    text(
                        "SELECT * FROM users WHERE phone = :phone AND is_active = TRUE LIMIT 1"
                    ),
                    {"phone": phone},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.warning(f"Query user by phone failed: {e}")
        return None

    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户信息"""
        if not self._Session:
            return False
        try:
            set_clauses = []
            params = {"user_id": user_id}
            for key, value in kwargs.items():
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
            set_clauses.append("updated_at = NOW()")

            with self._Session() as session:
                session.execute(
                    text(
                        f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = :user_id"
                    ),
                    params,
                )
                session.commit()
                return True
        except Exception as e:
            logger.warning(f"Update user failed: {e}")
            return False

    def update_last_login(self, user_id: str) -> bool:
        """更新最后登录时间"""
        return self.update_user(user_id, last_login_at="NOW()")

    def deactivate_user(self, user_id: str) -> bool:
        """禁用用户"""
        return self.update_user(user_id, is_active=False)

    def activate_user(self, user_id: str) -> bool:
        """启用用户"""
        return self.update_user(user_id, is_active=True)


_user_client: Optional[PostgresUserClient] = None


def get_user_client() -> PostgresUserClient:
    global _user_client
    if _user_client is None:
        _user_client = PostgresUserClient()
    return _user_client
