from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.risk import (
    is_critical as check_is_critical,
    check_risks_with_kg,
    generate_emergency_warning,
)
from app.domain.diagnosis.questions import get_emergency_warning, get_completion_message
from app.graph.nodes.risk_check import risk_check_node
from app.graph.nodes.completion import completion_node
from app.graph.nodes.question_gen import question_gen_node
from app.graph.nodes.normalize import normalize_text
from langchain_core.messages import AIMessage
from app.mcp.client import symptom_associations_mcp, department_by_symptom_mcp
import json


def fill_slots_with_input(
    user_input: str, current_slots: DiagnosisSlots
) -> DiagnosisSlots:
    """使用标准化后的输入填充槽位（LLM增强版）"""
    from app.domain.diagnosis.filler import fill_slots

    if not user_input or not user_input.strip():
        return current_slots

    return fill_slots(user_input, current_slots)


def diagnosis_node(state: AppState) -> dict:
    """
    主编排器：协调各Agent节点完成问诊
    流程: normalize -> slot_fill -> knowledge_graph -> risk_check -> (emergency/completion/question_gen)
    """
    from app.graph.nodes.slot_fill import slot_fill_node

    user_input = state.messages[-1].content
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    if not user_input:
        return {"messages": [AIMessage(content="请描述您的症状")]}

    normalized_input = normalize_text(user_input)

    existing_slots = (
        state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    )
    if existing_slots is None:
        existing_slots = DiagnosisSlots()

    filled_slots = fill_slots_with_input(normalized_input, existing_slots)
    state.diagnosis_slots = filled_slots

    associated_symptoms = []
    recommended_departments = []
    emergency_rules = []
    if filled_slots.symptoms:
        # 使用 MCP 调用知识图谱工具
        mcp_result = symptom_associations_mcp(filled_slots.symptoms)
        try:
            kg_result = json.loads(mcp_result)
        except json.JSONDecodeError:
            kg_result = {
                "associated_symptoms": [],
                "recommended_departments": [],
                "emergency_rules": [],
                "error": mcp_result,
            }
        associated_symptoms = kg_result.get("associated_symptoms", [])
        recommended_departments = kg_result.get("recommended_departments", [])
        emergency_rules = kg_result.get("emergency_rules", [])

        # 如果知识图谱检测到危急规则，优先处理
        if emergency_rules:
            warning = generate_emergency_warning(emergency_rules)
            return {
                "messages": [AIMessage(content=warning)],
                "diagnosis_completed": True,
                "diagnosis_terminated": True,
                "diagnosis_type": "emergency",
                "diagnosis_slots": state.diagnosis_slots,
                "diagnosis_recommended_departments": recommended_departments,
                "diagnosis_associated_symptoms": associated_symptoms,
                "diagnosis_emergency_rules": emergency_rules,
            }

        # 使用知识图谱增强风险检查
        normalized_symptoms = kg_result.get(
            "normalized_symptoms", filled_slots.symptoms
        )
        kg_risk_result = check_risks_with_kg(normalized_symptoms, user_input)
        state.diagnosis_risk_level = kg_risk_result.get("risk_level", "none")
        state.diagnosis_risk_signals = kg_risk_result.get("risk_signals", [])

        # 如果检测到高风险，也返回警告
        if state.diagnosis_risk_level in ["critical", "high"]:
            warning = generate_emergency_warning(
                kg_risk_result.get("emergency_rules", [])
            )
            return {
                "messages": [AIMessage(content=warning)],
                "diagnosis_completed": True,
                "diagnosis_terminated": True,
                "diagnosis_type": "emergency",
                "diagnosis_slots": state.diagnosis_slots,
                "diagnosis_recommended_departments": recommended_departments,
            }

        # 继续原有的风险检查
        risk_result = risk_check_node(state)
        state.diagnosis_risk_level = risk_result.get("diagnosis_risk_level", "none")
        state.diagnosis_risk_signals = risk_result.get("diagnosis_risk_signals", [])
        state.diagnosis_slots = risk_result.get(
            "diagnosis_slots", state.diagnosis_slots
        )
    else:
        risk_result = risk_check_node(state)
        state.diagnosis_risk_level = risk_result.get("diagnosis_risk_level", "none")
        state.diagnosis_risk_signals = risk_result.get("diagnosis_risk_signals", [])
        state.diagnosis_slots = risk_result.get(
            "diagnosis_slots", state.diagnosis_slots
        )

    completion_result = completion_node(state)
    state.diagnosis_completed = completion_result.get("diagnosis_completed", False)
    state.diagnosis_terminated = completion_result.get("diagnosis_terminated", False)
    state.diagnosis_termination_reason = completion_result.get(
        "diagnosis_termination_reason"
    )

    if state.diagnosis_risk_level == "critical":
        warning = get_emergency_warning(state.diagnosis_risk_signals)
        return {
            "messages": [AIMessage(content=warning)],
            "diagnosis_completed": True,
            "diagnosis_terminated": True,
            "diagnosis_type": "emergency",
            "diagnosis_slots": state.diagnosis_slots,
        }

    if state.diagnosis_completed:
        completion_msg = get_completion_message()
        return {
            "messages": [AIMessage(content=completion_msg)],
            "diagnosis_completed": True,
            "diagnosis_type": "complete",
            "diagnosis_slots": state.diagnosis_slots,
        }

    question_result = question_gen_node(
        state,
        associated_symptoms=associated_symptoms,
        recommended_departments=recommended_departments,
    )
    state.diagnosis_next_question = question_result.get("diagnosis_next_question", "")
    state.diagnosis_question_count = question_result.get("diagnosis_question_count", 1)
    state.diagnosis_missing_slots = question_result.get("diagnosis_missing_slots", [])

    return {
        "messages": [AIMessage(content=state.diagnosis_next_question)],
        "diagnosis_completed": False,
        "diagnosis_type": "in_progress",
        "diagnosis_slots": state.diagnosis_slots,
        "diagnosis_question_count": state.diagnosis_question_count,
        "diagnosis_missing_slots": state.diagnosis_missing_slots,
    }
