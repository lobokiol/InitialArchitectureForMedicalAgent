"""
症状填充器 - 四层架构整合

Layer 1: 词典映射 (symptom_dict.py)
Layer 2: LLM抽取 (llm_extractor.py)
Layer 3: 结构化Slot (本文件)
Layer 4: 知识图谱校验 (kg_validator.py)

流程:
    用户输入 → 词典匹配 → LLM提取 → 合并 → KG校验 → 最终Slots
"""

from typing import Optional, Any, Set
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.symptom_dict import match_symptoms_from_text, get_symptom_dict


USE_LLM_EXTRACTOR = True


def fill_slots(
    user_input: Any, current_slots: Optional[DiagnosisSlots] = None
) -> DiagnosisSlots:
    """从用户输入填充槽位（四层架构）"""
    if current_slots is None:
        current_slots = DiagnosisSlots()

    # 统一转为字符串
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    if not user_input or not user_input.strip():
        return current_slots

    user_input = user_input.strip()

    # 记录主诉
    if not current_slots.chief_complaint:
        current_slots.chief_complaint = user_input

    # ============ Layer 1: 词典映射 ============
    dict_symptoms = _layer1_dict_match(user_input)

    # ============ Layer 2: LLM抽取 ============
    llm_result = _layer2_llm_extract(user_input)

    # ============ Layer 3: 合并到Slot ============
    slots = _layer3_merge_to_slots(
        user_input=user_input,
        dict_symptoms=dict_symptoms,
        llm_result=llm_result,
        existing_slots=current_slots,
    )

    # ============ Layer 4: 知识图谱校验 ============
    slots = _layer4_kg_validate(slots, user_input)

    return slots


def _layer1_dict_match(user_input: str) -> dict:
    """
    Layer 1: 词典映射
    基于 keywords 的快速匹配
    """
    try:
        dict_matcher = get_symptom_dict()
        matches = dict_matcher.match(user_input)

        # 提取唯一症状
        symptoms = list(set(m["symptom"] for m in matches))

        return {
            "symptoms": symptoms,
            "sources": {s: "dict" for s in symptoms},
            "raw_matches": matches,
        }
    except Exception as e:
        print(f"Layer 1 (词典映射) 失败: {e}")
        return {"symptoms": [], "sources": {}, "raw_matches": []}


def _layer2_llm_extract(user_input: str) -> dict:
    """
    Layer 2: LLM抽取
    使用 Qwen Turbo 进行语义提取
    """
    if not USE_LLM_EXTRACTOR:
        return {"symptoms": [], "sources": {}}

    try:
        from app.domain.diagnosis.llm_extractor import extract_symptoms_with_llm

        result = extract_symptoms_with_llm(user_input)

        if not result:
            return {"symptoms": [], "sources": {}}

        # 获取原始症状和标准化症状
        raw_symptoms = result.get("symptoms", [])
        std_symptoms = result.get("standardized_symptoms", [])

        # 优先使用标准化症状，如果没有则用原始症状
        symptoms = std_symptoms if std_symptoms else raw_symptoms

        # 记录来源
        sources = {}
        for s in std_symptoms:
            sources[s] = "llm_standardized"
        for s in raw_symptoms:
            if s not in sources:
                sources[s] = "llm_raw"

        return {
            "symptoms": symptoms,
            "raw_symptoms": raw_symptoms,
            "standardized_symptoms": std_symptoms,
            "sources": sources,
            "full_result": result,
        }

    except Exception as e:
        print(f"Layer 2 (LLM抽取) 失败: {e}")
        return {"symptoms": [], "sources": {}}


def _layer3_merge_to_slots(
    user_input: str,
    dict_symptoms: dict,
    llm_result: dict,
    existing_slots: DiagnosisSlots,
) -> DiagnosisSlots:
    """
    Layer 3: 合并到结构化Slot
    合并词典和LLM的结果
    """
    # 收集所有症状（去重）
    all_symptoms: Set[str] = set()
    sources = {}

    # 添加词典匹配的症状
    for s in dict_symptoms.get("symptoms", []):
        all_symptoms.add(s)
        sources[s] = "dict"

    # 添加LLM提取的症状（优先）
    for s in llm_result.get("symptoms", []):
        all_symptoms.add(s)
        # 如果已存在，升级来源
        if s in sources:
            sources[s] = "llm_priority"
        else:
            sources[s] = "llm"

    # 转换为列表
    final_symptoms = list(all_symptoms)

    # 构建slots
    llm_full = llm_result.get("full_result", {})

    slots_dict = existing_slots.to_dict()
    slots_dict["symptoms"] = final_symptoms
    slots_dict["symptom_sources"] = sources

    # LLM其他字段
    if llm_full.get("location"):
        slots_dict["location"] = llm_full["location"]
    if llm_full.get("duration"):
        slots_dict["duration"] = llm_full["duration"]
    if llm_full.get("severity"):
        slots_dict["severity"] = llm_full["severity"]
    if llm_full.get("triggers"):
        slots_dict["triggers"] = llm_full["triggers"]
    if llm_full.get("accompanying_symptoms"):
        slots_dict["accompanying_symptoms"] = llm_full["accompanying_symptoms"]
    if llm_full.get("medical_history"):
        slots_dict["medical_history"] = llm_full["medical_history"]
    if llm_full.get("is_emergency"):
        slots_dict["risk_warning_issued"] = True

    return DiagnosisSlots(**slots_dict)


def _layer4_kg_validate(
    slots: DiagnosisSlots,
    user_input: str,
) -> DiagnosisSlots:
    """
    Layer 4: 知识图谱校验
    验证症状、扩展、标记不确定项
    """
    try:
        from app.domain.diagnosis.kg_validator import get_kg_validator

        validator = get_kg_validator()

        # 合并所有已知症状进行校验
        all_symptoms = (
            slots.symptoms + slots.uncertain_symptoms + slots.expanded_symptoms
        )

        # 如果没有症状，直接返回
        if not all_symptoms:
            slots.confidence_score = 0.3
            slots.validated = False
            return slots

        # 校验
        result = validator.validate(
            symptoms=all_symptoms,
            user_input=user_input,
            symptom_sources=slots.symptom_sources,
        )

        # 更新slots
        slots.symptoms = result.valid_symptoms
        slots.uncertain_symptoms = result.uncertain_symptoms
        slots.expanded_symptoms = result.expanded_symptoms
        slots.symptom_sources = result.symptom_sources
        slots.confidence_score = result.confidence_score
        slots.validated = True

        return slots

    except Exception as e:
        print(f"Layer 4 (KG校验) 失败: {e}")
        slots.confidence_score = 0.5
        slots.validated = False
        return slots


def fill_from_text_with_llm(text: str) -> Optional[dict]:
    """兼容旧接口"""
    slots = fill_slots(text)
    return slots.to_dict()


def fill_from_text(text: str, slots: dict) -> dict:
    """兼容旧接口（简单规则匹配）"""
    filled = slots.copy()

    # 简单的关键词匹配
    duration_keywords = {
        "一分钟": "1分钟",
        "几分钟": "几分钟",
        "十分钟": "10分钟",
        "半小时": "30分钟",
        "一小时": "1小时",
        "一天": "1天",
        "两天": "2天",
        "三天": "3天",
        "一周": "1周",
        "一个月": "1个月",
    }
    for kw, val in duration_keywords.items():
        if kw in text:
            filled["duration"] = val
            break

    severity_keywords = {
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
            filled["severity"] = val
            break

    return filled
