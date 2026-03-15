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
