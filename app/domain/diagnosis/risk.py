from typing import Any, List, Dict, Optional


CRITICAL_RISKS = {
    # 危急生命体征
    "呼吸困难",
    "气短",
    "喘不上气",
    "大出血",
    "呕血",
    "咳血",
    "便血",
    "阴道大出血",
    "休克",
    "意识丧失",
    "失去意识",
    "昏迷",
    "晕倒",
    # 神经系统危急
    "剧烈头痛",
    "脑卒中",
    "中风",
    "脑出血",
    # 心血管危急
    "心肌梗死",
    "心梗",
    "主动脉夹层",
    # 持续症状
    "高热不退",
    "持续高热",
    "持续呕吐",
    "呕吐不止",
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
    "胸痛",
    "胸口痛",
    "胸口疼",
}


def detect_risks(text: Any, negative_symptoms: list = None) -> list[str]:
    """
    检测文本中的危险信号

    Args:
        text: 用户输入文本
        negative_symptoms: 否定症状列表（用于排除）
    """
    if isinstance(text, list):
        text = str(text[0]) if text else ""
    elif not isinstance(text, str):
        text = str(text) if text else ""

    if not text:
        return []

    # 如果有否定症状，先移除否定部分，只保留肯定描述
    if negative_symptoms:
        text_to_check = text
        for neg_word in [
            "不",
            "没有",
            "没",
            "不是",
            "无",
            "非",
            "不会",
            "未曾",
            "从不",
        ]:
            if neg_word in text_to_check:
                idx = text_to_check.find(neg_word)
                text_to_check = text_to_check[:idx].strip()

        # 移除"但/而且"等转折词后的内容
        for conj in ["但", "但是", "而且", "不过", "只是"]:
            if conj in text_to_check:
                idx = text_to_check.find(conj)
                text_to_check = text_to_check[:idx].strip()

        text_lower = text_to_check.lower()
    else:
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


def check_emergency_rules(symptoms: List[str]) -> List[Dict]:
    """
    使用知识图谱检查危急规则

    Args:
        symptoms: 症状列表

    Returns:
        匹配的危急规则列表
    """
    if not symptoms:
        return []

    try:
        from app.tools.knowledge_graph_tool import check_emergency

        return check_emergency(symptoms)
    except Exception as e:
        print(f"知识图谱危急规则检查失败: {e}")
        return []


def generate_emergency_warning(emergency_rules: List[Dict]) -> str:
    """
    根据危急规则生成警告消息

    Args:
        emergency_rules: 危急规则列表

    Returns:
        警告消息
    """
    if not emergency_rules:
        return ""

    rule = emergency_rules[0]  # 取优先级最高的
    warnings = [
        f"⚠️ 检测到危急症状：{rule.get('name', '')}",
        f"可能情况：{rule.get('description', '未知')}",
        f"建议：立即前往{rule.get('action', '急诊')}就诊！",
        "",
        "如有胸痛、出汗、呼吸困难等症状，请立即拨打120！",
    ]

    return "\n".join(warnings)


def check_risks_with_kg(
    symptoms: List[str], text: str = "", negative_symptoms: List[str] = None
) -> Dict[str, Any]:
    """
    使用知识图谱检查风险

    Args:
        symptoms: 识别的症状列表
        text: 原始文本（用于本地字典匹配）
        negative_symptoms: 否定症状列表

    Returns:
        包含风险等级、信号、危急规则的字典
    """
    # 1. 本地字典检测（排除否定症状）
    local_risks = detect_risks(text, negative_symptoms) if text else []
    is_crit = is_critical(local_risks)

    # 2. 知识图谱危急规则检测
    emergency_rules = check_emergency_rules(symptoms)
    has_emergency = len(emergency_rules) > 0

    # 3. 综合判断风险等级
    if has_emergency:
        risk_level = "critical"
    elif is_crit:
        risk_level = "high"
    elif local_risks:
        risk_level = "medium"
    else:
        risk_level = "none"

    return {
        "risk_level": risk_level,
        "risk_signals": local_risks,
        "emergency_rules": emergency_rules,
        "is_critical": has_emergency or is_crit,
    }
