from typing import Optional, List
from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.questions import get_next_question, QUESTION_ORDER
from app.core import config


def question_gen_node(
    state: AppState,
    associated_symptoms: Optional[List[str]] = None,
    recommended_departments: Optional[List[str]] = None,
) -> dict:
    """
    Agent 4: 追问生成
    根据缺失槽位和知识图谱返回的伴随症状生成个性化追问

    Args:
        state: 当前诊断状态
        associated_symptoms: 知识图谱返回的常见伴随症状
        recommended_departments: 知识图谱返回的推荐科室
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

    if associated_symptoms is None:
        associated_symptoms = getattr(state, "diagnosis_associated_symptoms", [])

    next_question = get_next_question(missing, associated_symptoms=associated_symptoms)
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
