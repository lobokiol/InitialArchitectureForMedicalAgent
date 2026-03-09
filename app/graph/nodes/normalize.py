from typing import Optional

from app.domain.models import AppState


SYMPTOM_NORMALIZE_MAP = {
    "肚子疼": "腹痛",
    "胃疼": "胃痛",
    "胸口疼": "胸痛",
    "心口疼": "胸痛",
    "头疼": "头痛",
    "头昏": "头晕",
    "拉肚子": "腹泻",
    "跑肚": "腹泻",
    "感冒": "上呼吸道感染",
    "咳嗽": "咳嗽",
    "发烧": "发热",
    "体温高": "发热",
    "恶心": "恶心",
    "想吐": "恶心",
    "呕吐": "呕吐",
    "喘不上气": "呼吸困难",
    "气短": "呼吸困难",
    "没力气": "乏力",
    "疲乏": "乏力",
    "睡不着": "失眠",
    "失眠": "失眠",
    "胃胀": "腹胀",
    "腹胀": "腹胀",
    "腰疼": "腰痛",
    "背疼": "背痛",
    "腿疼": "腿痛",
    "脚肿": "下肢水肿",
    "手麻": "手足麻木",
    "脚麻": "手足麻木",
}


def normalize_text(text: str) -> str:
    """对用户输入的整句进行症状标准化"""
    result = text
    for colloquial, standard in SYMPTOM_NORMALIZE_MAP.items():
        if colloquial in result:
            result = result.replace(colloquial, standard)
    return result


def normalize_symptom(symptom: str) -> str:
    """将口语症状映射为标准医学术语"""
    return SYMPTOM_NORMALIZE_MAP.get(symptom, symptom)


def normalize_node(state: AppState) -> dict:
    """
    Agent 3: 语义对齐
    将用户口语描述映射为标准医学术语
    """
    if not hasattr(state, "diagnosis_slots") or state.diagnosis_slots is None:
        return {}

    slots = state.diagnosis_slots

    normalized_symptoms = []
    for symptom in slots.symptoms:
        normalized_symptoms.append(normalize_symptom(symptom))
    slots.symptoms = normalized_symptoms

    normalized_accompanying = []
    for symptom in slots.accompanying_symptoms:
        normalized_accompanying.append(normalize_symptom(symptom))
    slots.accompanying_symptoms = normalized_accompanying

    return {
        "diagnosis_slots": slots,
    }
