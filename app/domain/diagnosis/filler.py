from typing import Optional, Any
from app.domain.diagnosis.slots import DiagnosisSlots


def fill_slots(
    user_input: Any, current_slots: Optional[DiagnosisSlots] = None
) -> DiagnosisSlots:
    """从用户输入填充槽位"""
    if current_slots is None:
        current_slots = DiagnosisSlots()

    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    if not user_input or not user_input.strip():
        return current_slots

    user_input = user_input.strip()

    if not current_slots.chief_complaint:
        current_slots.chief_complaint = user_input

    filled = current_slots.to_dict()

    fill_from_text(user_input, filled)

    return DiagnosisSlots(**filled)


def fill_from_text(text: str, slots: dict) -> dict:
    """从文本中提取槽位信息（简单规则匹配，后续可接入LLM增强）"""
    text = text.lower()

    duration_keywords = {
        "一分钟": "1分钟",
        "几分钟": "几分钟",
        "十分钟": "10分钟",
        "半小时": "30分钟",
        "一小时": "1小时",
        "一天": "1天",
        "两天": "2天",
        "三天": "3天",
        "四天": "4天",
        "五天": "5天",
        "一周": "1周",
        "两周": "2周",
        "一个月": "1个月",
        "三个月": "3个月",
        "半年": "6个月",
        "一年": "1年",
    }
    for kw, val in duration_keywords.items():
        if kw in text:
            slots["duration"] = val
            break

    import re

    match = re.search(r"(\d+)\s*天", text)
    if match:
        days = int(match.group(1))
        if days <= 5:
            slots["duration"] = f"{days}天"
        elif days < 30:
            slots["duration"] = f"{days}天"
        elif days < 365:
            months = days // 30
            slots["duration"] = f"{months}个月"
        else:
            years = days // 365
            slots["duration"] = f"{years}年"

    match = re.search(r"(\d+)\s*周", text)
    if match:
        weeks = int(match.group(1))
        if weeks < 4:
            slots["duration"] = f"{weeks}周"
        else:
            slots["duration"] = f"{weeks // 4}个月"

    match = re.search(r"(\d+)\s*[个]?\s*月", text)
    if match:
        slots["duration"] = match.group(1) + "个月"

    severity_keywords = {
        "不疼": "0",
        "轻微": "1-2",
        "轻度": "2-3",
        "中等": "4-5",
        "中度": "4-5",
        "较重": "6-7",
        "严重": "8-9",
        "剧痛": "10",
    }
    for kw, val in severity_keywords.items():
        if kw in text:
            slots["severity"] = val
            break

    if "分" in text and any(str(i) in text for i in range(11)):
        import re

        match = re.search(r"(\d+)\s*分", text)
        if match:
            score = int(match.group(1))
            if score <= 2:
                slots["severity"] = "1-2"
            elif score <= 5:
                slots["severity"] = "4-5"
            elif score <= 7:
                slots["severity"] = "6-7"
            else:
                slots["severity"] = "8-10"

    location_keywords = {
        "头疼": "头部",
        "头痛": "头部",
        "头晕": "头部",
        "肚子疼": "腹部",
        "腹痛": "腹部",
        "胃疼": "腹部",
        "胸口疼": "胸部",
        "胸痛": "胸部",
        "心口疼": "胸部",
        "背疼": "背部",
        "腰痛": "腰部",
        "腿疼": "腿部",
        "手麻": "手部",
        "脚麻": "脚部",
    }
    for kw, val in location_keywords.items():
        if kw in text:
            if val not in slots.get("location", ""):
                slots["location"] = val
            if val not in slots.get("symptoms", []):
                slots["symptoms"] = slots.get("symptoms", []) + [kw]
            break

    accompanying_keywords = {
        "发烧": "发热",
        "发热": "发热",
        "体温高": "发热",
        "恶心": "恶心",
        "想吐": "恶心",
        "呕吐": "呕吐",
        "拉肚子": "腹泻",
        "腹泻": "腹泻",
        "便秘": "便秘",
        "咳嗽": "咳嗽",
        "鼻涕": "鼻涕",
        "嗓子疼": "咽痛",
    }
    for kw, val in accompanying_keywords.items():
        if kw in text and val not in slots.get("accompanying_symptoms", []):
            slots["accompanying_symptoms"] = slots.get("accompanying_symptoms", []) + [
                val
            ]

    medical_keywords = {
        "高血压": "高血压",
        "糖尿病": "糖尿病",
        "心脏病": "心脏病",
        "胃病": "胃病",
        "肝病": "肝病",
        "肾病": "肾病",
        "手术": "手术史",
        "住院": "住院史",
    }
    for kw, val in medical_keywords.items():
        if kw in text and val not in slots.get("medical_history", []):
            slots["medical_history"] = slots.get("medical_history", []) + [val]

    trigger_keywords = {
        "吃饭": "进食",
        "吃完饭": "进食",
        "空腹": "空腹",
        "劳累": "劳累",
        "运动": "运动",
        "睡觉": "睡眠",
        "生气": "情绪",
        "焦虑": "情绪",
        "紧张": "情绪",
    }
    for kw, val in trigger_keywords.items():
        if kw in text and val not in slots.get("triggers", []):
            slots["triggers"] = slots.get("triggers", []) + [val]

    no_keywords = ["没有", "无", "没有的", "没有任何", "没啥"]
    if any(kw in text for kw in no_keywords):
        if "medical_history" not in slots or not slots.get("medical_history"):
            slots["medical_history"] = ["无"]
        if "triggers" not in slots or not slots.get("triggers"):
            slots["triggers"] = ["无"]
        if "accompanying_symptoms" not in slots or not slots.get(
            "accompanying_symptoms"
        ):
            slots["accompanying_symptoms"] = ["无"]

    location_keywords = {
        "左上腹": "左上腹",
        "右上腹": "右上腹",
        "左下腹": "左下腹",
        "右下腹": "右下腹",
        "肚脐": "肚脐周围",
    }
    for kw, val in location_keywords.items():
        if kw in text:
            slots["location"] = val
            break

    return slots
