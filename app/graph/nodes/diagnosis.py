from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.risk import is_critical as check_is_critical
from app.domain.diagnosis.questions import get_emergency_warning, get_completion_message
from app.graph.nodes.risk_check import risk_check_node
from app.graph.nodes.completion import completion_node
from app.graph.nodes.question_gen import question_gen_node
from langchain_core.messages import AIMessage


def diagnosis_node(state: AppState) -> dict:
    """
    主编排器：协调各Agent节点完成问诊
    流程: slot_fill -> normalize -> risk_check -> (emergency/completion/question_gen)
    """
    from app.graph.nodes.slot_fill import slot_fill_node
    from app.graph.nodes.normalize import normalize_node

    user_input = state.messages[-1].content
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    if not user_input:
        return {"messages": [AIMessage(content="请描述您的症状")]}

    slot_result = slot_fill_node(state)
    state.diagnosis_slots = slot_result.get("diagnosis_slots")

    normalize_result = normalize_node(state)
    state.diagnosis_slots = normalize_result.get(
        "diagnosis_slots", state.diagnosis_slots
    )

    risk_result = risk_check_node(state)
    state.diagnosis_risk_level = risk_result.get("diagnosis_risk_level", "none")
    state.diagnosis_risk_signals = risk_result.get("diagnosis_risk_signals", [])
    state.diagnosis_slots = risk_result.get("diagnosis_slots", state.diagnosis_slots)

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

    question_result = question_gen_node(state)
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
