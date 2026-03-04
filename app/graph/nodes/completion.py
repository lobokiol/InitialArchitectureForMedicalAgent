from typing import Optional, Any
from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.core import config


def is_user_ending(text: Any) -> bool:
    """判断用户是否主动结束问诊"""
    if isinstance(text, list):
        text = str(text[0]) if text else ""
    elif not isinstance(text, str):
        text = str(text) if text else ""

    if not text:
        return False
    text_lower = text.lower()
    ending_keywords = ["结束", "好了", "不需要", "不用了", "结束问诊"]
    return any(kw in text_lower for kw in ending_keywords)


def completion_node(state: AppState) -> dict:
    """
    Agent 6: 结束判断
    判断问诊是否完成，决定流程走向
    """
    user_input = state.messages[-1].content
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    risk_level = (
        state.diagnosis_risk_level if hasattr(state, "diagnosis_risk_level") else "none"
    )
    question_count = (
        state.diagnosis_question_count
        if hasattr(state, "diagnosis_question_count")
        else 0
    )

    if risk_level == "critical":
        return {
            "diagnosis_completed": True,
            "diagnosis_terminated": True,
            "diagnosis_termination_reason": "critical_risk",
        }

    if is_user_ending(user_input):
        return {
            "diagnosis_completed": True,
            "diagnosis_terminated": False,
            "diagnosis_termination_reason": "user_ended",
        }

    slots = state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    if slots is None:
        slots = DiagnosisSlots()

    if slots.is_complete():
        return {
            "diagnosis_completed": True,
            "diagnosis_terminated": False,
            "diagnosis_termination_reason": "slots_filled",
        }

    if question_count >= config.DIAGNOSIS_MAX_QUESTIONS:
        return {
            "diagnosis_completed": True,
            "diagnosis_terminated": False,
            "diagnosis_termination_reason": "max_questions",
        }

    return {
        "diagnosis_completed": False,
        "diagnosis_terminated": False,
        "diagnosis_termination_reason": None,
    }
