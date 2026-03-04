from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.filler import fill_slots


def slot_fill_node(state: AppState) -> dict:
    """
    Agent 2: 槽位填充
    从用户输入中提取并填充槽位信息
    """
    user_input = state.messages[-1].content
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    existing_slots = (
        state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    )
    if existing_slots is None:
        existing_slots = DiagnosisSlots()

    filled_slots = fill_slots(user_input, existing_slots)

    return {
        "diagnosis_slots": filled_slots,
    }
