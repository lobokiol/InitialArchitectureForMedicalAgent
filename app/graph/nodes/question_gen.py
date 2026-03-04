from typing import Optional
from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.questions import get_next_question, QUESTION_ORDER
from app.core import config


def question_gen_node(state: AppState) -> dict:
    """
    Agent 5: 追问生成
    根据缺失槽位生成下一个问题
    """
    slots = state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    if slots is None:
        slots = DiagnosisSlots()

    filled = slots.to_dict()
    missing = []

    for slot in QUESTION_ORDER:
        value = filled.get(slot)
        if slot in ["triggers", "accompanying_symptoms", "medical_history", "symptoms"]:
            if not value or (isinstance(value, list) and len(value) == 0):
                missing.append(slot)
        else:
            if not value:
                missing.append(slot)

    next_question = get_next_question(missing)
    question_count = (
        state.diagnosis_question_count
        if hasattr(state, "diagnosis_question_count")
        else 0
    ) + 1

    return {
        "diagnosis_next_question": next_question,
        "diagnosis_question_count": question_count,
        "diagnosis_missing_slots": missing,
    }
