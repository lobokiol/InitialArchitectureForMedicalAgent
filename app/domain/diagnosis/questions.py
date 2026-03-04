from typing import Optional


QUESTION_TEMPLATES = {
    "duration": "这个症状持续多长时间了？",
    "severity": "疼痛程度如何？0-10分大概几分？",
    "location": "具体是哪个部位不舒服？",
    "triggers": "有没有什么情况下会加重或缓解？",
    "accompanying_symptoms": "有没有伴随其他症状？比如发烧、恶心、呕吐等",
    "medical_history": "以前有过类似症状吗？有什么病史？",
    "symptoms": "还有哪些不舒服的症状？",
}

QUESTION_ORDER = [
    "duration",
    "severity",
    "location",
    "triggers",
    "accompanying_symptoms",
    "medical_history",
]


def get_next_question(missing_slots: list[str]) -> str:
    """根据缺失槽位生成下一个问题"""
    if not missing_slots:
        return "感谢您的配合，问诊信息已收集完整。"

    for slot in QUESTION_ORDER:
        if slot in missing_slots:
            return QUESTION_TEMPLATES.get(slot, "还有其他不舒服的吗？")

    return QUESTION_TEMPLATES.get(missing_slots[0], "还有其他不舒服的吗？")


def get_emergency_warning(risk_signals: list[str]) -> str:
    """生成危险信号警告"""
    signal_str = "、".join(risk_signals)
    return f"⚠️ 检测到危险信号：{signal_str}，建议立即挂急诊就医！"


def get_completion_message() -> str:
    return "问诊信息已收集完整，正在为您匹配科室..."
