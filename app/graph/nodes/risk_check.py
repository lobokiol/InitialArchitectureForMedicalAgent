from typing import Optional
from app.domain.models import AppState
from app.domain.diagnosis.risk import detect_risks, is_critical as check_is_critical
from app.domain.diagnosis.risk import CRITICAL_RISKS, WARNING_RISKS


def risk_check_node(state: AppState) -> dict:
    """
    Agent 4: 风险评估
    检测用户输入中的危险信号，返回风险等级（排除否定症状）
    """
    user_input = state.messages[-1].content

    # 获取否定症状
    slots = state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    negative_symptoms = (
        slots.negative_symptoms
        if slots and hasattr(slots, "negative_symptoms")
        else None
    )

    # 检测风险时排除否定症状
    detected_risks = detect_risks(user_input, negative_symptoms)

    has_critical = check_is_critical(detected_risks)
    risk_level = (
        "critical" if has_critical else ("warning" if detected_risks else "none")
    )

    if slots and detected_risks:
        slots.risk_signals = list(set(slots.risk_signals + detected_risks))
        slots.risk_warning_issued = has_critical

    return {
        "diagnosis_risk_level": risk_level,
        "diagnosis_risk_signals": detected_risks,
        "diagnosis_slots": slots,
    }
