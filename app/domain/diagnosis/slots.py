from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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

MINIMUM_SLOTS = [
    DiagnosisSlot.SYMPTOMS,
    DiagnosisSlot.DURATION,
    DiagnosisSlot.LOCATION,
]

AUXILIARY_SLOTS = [
    DiagnosisSlot.SEVERITY,
    DiagnosisSlot.TRIGGERS,
    DiagnosisSlot.ACCOMPANYING,
    DiagnosisSlot.MEDICAL_HISTORY,
]

OPTIONAL_SLOTS = MINIMUM_SLOTS + AUXILIARY_SLOTS

ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS


class DiagnosisSlots(BaseModel):
    chief_complaint: str = ""

    symptoms: list[str] = []
    negative_symptoms: list[str] = Field(
        default_factory=list, description="排除的症状（用户明确说没有/不有的症状）"
    )
    uncertain_symptoms: list[str] = Field(
        default_factory=list, description="待确认的症状(KG校验失败)"
    )
    expanded_symptoms: list[str] = Field(
        default_factory=list, description="KG扩展的症状"
    )

    duration: str = ""
    severity: str = ""
    location: str = ""

    triggers: list[str] = []
    accompanying_symptoms: list[str] = []
    medical_history: list[str] = []

    risk_signals: list[str] = []
    risk_warning_issued: bool = False

    symptom_sources: dict = Field(
        default_factory=dict, description="症状来源: {symptom: source}"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="整体置信度"
    )
    validated: bool = Field(default=False, description="是否经过KG校验")

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

    def is_minimum_filled(self) -> bool:
        filled_count = 0
        for slot in MINIMUM_SLOTS:
            value = getattr(self, slot.value, None)
            if value and (
                isinstance(value, list)
                and len(value) > 0
                or isinstance(value, str)
                and value
            ):
                filled_count += 1
        return filled_count >= len(MINIMUM_SLOTS)

    def get_filled_count(self) -> dict:
        result = {"minimum": 0, "auxiliary": 0, "total": 0}
        for slot in MINIMUM_SLOTS:
            value = getattr(self, slot.value, None)
            if value and (
                isinstance(value, list)
                and len(value) > 0
                or isinstance(value, str)
                and value
            ):
                result["minimum"] += 1
        for slot in AUXILIARY_SLOTS:
            value = getattr(self, slot.value, None)
            if value and (
                isinstance(value, list)
                and len(value) > 0
                or isinstance(value, str)
                and value
            ):
                result["auxiliary"] += 1
        result["total"] = result["minimum"] + result["auxiliary"]
        return result

    def get_all_symptoms(self) -> list[str]:
        """获取所有症状，包括确认的和待确认的"""
        return self.symptoms + self.uncertain_symptoms + self.expanded_symptoms
