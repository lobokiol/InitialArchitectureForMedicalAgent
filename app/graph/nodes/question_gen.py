from typing import Optional, List
from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.questions import get_next_question, QUESTION_ORDER
from app.core import config


# 是否启用 LLM 追问
USE_LLM_QUESTION = True


def question_gen_node(
    state: AppState,
    associated_symptoms: Optional[List[str]] = None,
    recommended_departments: Optional[List[str]] = None,
) -> dict:
    """
    Agent 4: 追问生成
    根据缺失槽位和知识图谱返回的伴随症状生成个性化追问

    优先使用 LLM 生成，fallback 到模板

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

    if recommended_departments is None:
        recommended_departments = getattr(
            state, "diagnosis_recommended_departments", []
        )

    # 获取对话历史
    conversation_history = ""
    if hasattr(state, "messages") and state.messages:
        history = []
        for msg in state.messages[-4:]:  # 最近4条
            role = "用户" if hasattr(msg, "type") and msg.type == "human" else "医生"
            content = msg.content if hasattr(msg, "content") else str(msg)
            if content:
                history.append(f"{role}: {content}")
        conversation_history = "\n".join(history)

    # 优先使用 LLM 生成追问
    if USE_LLM_QUESTION and filled.get("symptoms"):
        try:
            from app.domain.diagnosis.llm_question_generator import (
                generate_question_with_llm,
            )

            llm_question = generate_question_with_llm(
                symptoms=filled.get("symptoms", []),
                missing_slots=missing,
                associated_symptoms=associated_symptoms,
                recommended_departments=recommended_departments,
                conversation_history=conversation_history,
            )

            # 如果 LLM 生成了有效追问
            if llm_question and len(llm_question) > 0:
                question_count = (
                    state.diagnosis_question_count
                    if hasattr(state, "diagnosis_question_count")
                    else 0
                ) + 1

                return {
                    "diagnosis_next_question": llm_question,
                    "diagnosis_question_count": question_count,
                    "diagnosis_missing_slots": missing,
                }
        except Exception as e:
            print(f"LLM 追问生成失败，使用模板: {e}")

    # Fallback 到模板追问
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
