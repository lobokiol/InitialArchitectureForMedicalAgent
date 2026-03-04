from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DiagnosisSlot(str, Enum):
    CHIEF_COMPLAINT = "chief_complaint"
    SYMPTOMS = "symptoms"
    DURATION = "duration"
    SEVERITY = "severity"
    LOCATION = "location"
    TRIGGERS = "triggers"
    ACCOMPANYING = "accompanying_symptoms"
    MEDICAL_HISTORY = "medical_history"


REQUIRED_SLOTS = [DiagnosisSlot.CHIEF_COMPLAINT]
OPTIONAL_SLOTS = [
    DiagnosisSlot.SYMPTOMS,
    DiagnosisSlot.DURATION,
    DiagnosisSlot.SEVERITY,
    DiagnosisSlot.LOCATION,
    DiagnosisSlot.TRIGGERS,
    DiagnosisSlot.ACCOMPANYING,
    DiagnosisSlot.MEDICAL_HISTORY,
]

ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS


class DiagnosisSlots(BaseModel):
    chief_complaint: Optional[str] = ""
    symptoms: list[str] = []
    duration: str = ""
    severity: str = ""
    location: str = ""
    triggers: list[str] = []
    accompanying_symptoms: list[str] = []
    medical_history: list[str] = []
    risk_signals: list[str] = []
    risk_warning_issued: bool = False

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_filled(self) -> dict:
        result = {}
        for field, value in self.model_dump().items():
            if value and (
                isinstance(value, list)
                and len(value) > 0
                or isinstance(value, str)
                and value
            ):
                result[field] = value
        return result

    def get_missing(self, include_optional: bool = True) -> list[str]:
        missing = []
        for slot in ALL_SLOTS:
            value = getattr(self, slot.value, None)
            if slot in REQUIRED_SLOTS:
                if not value or (isinstance(value, list) and len(value) == 0):
                    missing.append(slot.value)
            elif include_optional and slot not in REQUIRED_SLOTS:
                if not value or (isinstance(value, list) and len(value) == 0):
                    missing.append(slot.value)
        return missing

    def is_complete(self) -> bool:
        return len(self.get_missing()) == 0
