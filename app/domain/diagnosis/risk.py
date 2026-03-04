from typing import Any


CRITICAL_RISKS = {
    "胸痛",
    "胸口疼",
    "胸口痛",
    "呼吸困难",
    "气短",
    "喘不上气",
    "大出血",
    "意识丧失",
    "失去意识",
    "剧烈头痛",
    "头疼",
    "头痛",
    "高热不退",
    "持续高热",
    "持续呕吐",
    "呕吐不止",
    "昏迷",
    "晕倒",
    "呕血",
    "咳血",
    "便血",
    "黑便",
    "阴道大出血",
    "大出血",
    "休克",
    "中风",
    "脑卒中",
    "心肌梗死",
    "心梗",
    "脑出血",
    "主动脉夹层",
}

WARNING_RISKS = {
    "阴道出血",
    "非经期出血",
    "体重下降",
    "体重骤降",
    "瘦了",
    "夜间盗汗",
    "盗汗",
    "持续疼痛",
    "一直疼",
    "吞咽困难",
    "咽不下",
    "下肢肿胀",
    "腿肿",
    "脚肿",
    "持续低热",
    "低烧",
    "乏力",
    "没力气",
    "疲倦",
    "心悸",
    "心跳快",
    "心跳过速",
    "头晕",
    "头昏",
    "恶心",
    "想吐",
    "腹泻",
    "拉肚子",
}


def detect_risks(text: Any) -> list[str]:
    """检测文本中的危险信号"""
    if isinstance(text, list):
        text = str(text[0]) if text else ""
    elif not isinstance(text, str):
        text = str(text) if text else ""

    if not text:
        return []

    text_lower = text.lower()
    detected = []

    for risk in CRITICAL_RISKS:
        if risk in text_lower:
            detected.append(risk)

    for risk in WARNING_RISKS:
        if risk in text_lower:
            detected.append(risk)

    return list(set(detected))


def is_critical(risk_signals: list[str]) -> bool:
    """判断是否为危急信号"""
    return any(r in CRITICAL_RISKS for r in risk_signals)
