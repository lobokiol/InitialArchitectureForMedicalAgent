"""
症状填充器 - 四层架构整合

Layer 0: Neo4j CM3KG 匹配 (新增，支持否定症状)
Layer 1: 词典映射 (symptom_dict.py)
Layer 2: LLM抽取 (llm_extractor.py)
Layer 3: 结构化Slot (本文件)
Layer 4: 知识图谱校验 (kg_validator.py)

流程:
    用户输入 → 否定词检测 → Neo4j匹配 → 词典匹配 → LLM提取 → 合并 → KG校验 → 最终Slots
"""

import re
from typing import Optional, Any, Set, List, Tuple
from app.domain.diagnosis.slots import DiagnosisSlots
from app.domain.diagnosis.symptom_dict import match_symptoms_from_text, get_symptom_dict


USE_LLM_EXTRACTOR = True

# 否定词列表
NEGATIVE_WORDS = [
    "不",
    "没有",
    "没",
    "不是",
    "无",
    "非",
    "未",
    "不会出现",
    "不会出现",
    "不会",
    "未曾",
    "从未",
    "无任何",
    "没有任何",
    "毫无",
    "不存在",
]

# 否定模式（检测否定词+症状的组合）
NEGATIVE_PATTERNS = [
    r"(不|没有|没|不是|无|非|未|不会|未曾|从不)(.+?)(症状?|问题|感觉)?$",
    r"(.+?)(没有|不含|不包括|排除)(.+?)$",
]


def detect_negative_symptoms(user_input: str) -> Tuple[List[str], List[str]]:
    """
    检测用户输入中的否定症状

    Args:
        user_input: 用户输入文本

    Returns:
        (肯定症状列表, 否定症状列表)
    """
    negative_symptoms = []
    positive_text = user_input

    # 定义否定词模式
    negation_patterns = [
        r"不\s*(发热|发烧|咳嗽|头痛|肚子疼|恶心|呕吐|腹泻|便秘|胸闷|气短|头晕|眼花)",
        r"没\s*(有)?\s*(发热|发烧|咳嗽|头痛|肚子疼|恶心|呕吐|腹泻|便秘|胸闷|气短|头晕|眼花)",
        r"(没有|不是|无)\s*(发热|发烧|咳嗽|头痛|肚子疼|恶心|呕吐|腹泻|便秘|胸闷|气短|头晕|眼花)",
    ]

    # 检测否定模式
    for pattern in negation_patterns:
        matches = re.finditer(pattern, user_input)
        for match in matches:
            # 提取否定症状
            negated_symptom = match.group()
            # 清理
            negated_symptom = (
                negated_symptom.replace("不", "")
                .replace("没有", "")
                .replace("不是", "")
                .replace("无", "")
                .replace("有", "")
                .strip()
            )
            if negated_symptom and negated_symptom not in negative_symptoms:
                negative_symptoms.append(negated_symptom)

    # 移除否定部分，保留肯定部分
    positive_text = user_input
    for neg in [
        "不发热",
        "不发烧",
        "不咳嗽",
        "不头痛",
        "不肚子疼",
        "不恶心",
        "不呕吐",
        "没有发热",
        "没有发烧",
        "没有咳嗽",
        "没有头痛",
        "没有恶心",
        "没有呕吐",
        "不是发热",
        "不是发烧",
        "不是咳嗽",
        "不是头痛",
        "不是恶心",
        "不是呕吐",
    ]:
        if neg in positive_text:
            idx = positive_text.find(neg)
            positive_text = positive_text[:idx].strip()

    # 移除"但/而且"等转折词后的内容（可能包含否定）
    for conj in ["但", "但是", "而且", "不过"]:
        if conj in positive_text:
            idx = positive_text.find(conj)
            positive_text = positive_text[:idx].strip()

    return [], negative_symptoms  # 肯定症状由后续匹配补充


def _clean_symptom(text: str) -> str:
    """清理症状文本"""
    # 移除常见修饰词
    removals = ["感觉", "有些", "有点", "稍微", "轻微", "有一点", "有一些"]
    for r in removals:
        text = text.replace(r, "")

    return text.strip()


def _layer0_neo4j_match(user_input: str) -> dict:
    """
    Layer 0: Neo4j CM3KG 向量语义匹配（支持否定症状）
    使用 embedding 向量语义对齐，将主诉映射至标准医学术语
    自动区分肯定症状和否定症状
    """
    # 先检测否定症状
    _, negative_symptoms = detect_negative_symptoms(user_input)

    # 移除否定部分，保留肯定部分进行匹配
    positive_input = user_input
    for neg_word in NEGATIVE_WORDS:
        if neg_word in positive_input:
            idx = positive_input.find(neg_word)
            positive_input = positive_input[:idx].strip()

    # 如果移除否定词后为空，用原始输入
    if not positive_input:
        positive_input = user_input

    try:
        from app.infra.neo4j_client import get_neo4j_client

        client = get_neo4j_client()

        if client and client._driver:
            # 向量语义搜索 (阈值 0.7 平衡精度和召回)
            vector_matches = client.semantic_match_symptoms(
                positive_input, top_k=5, threshold=0.7
            )

            if vector_matches:
                # 过滤掉否定症状
                filtered_matches = []
                for m in vector_matches:
                    symptom_name = m.get("name", "")
                    # 排除与否定症状相似的症状
                    is_negative = False
                    for neg_sym in negative_symptoms:
                        if neg_sym in symptom_name or symptom_name in neg_sym:
                            is_negative = True
                            break

                    if not is_negative:
                        filtered_matches.append(m)

                # 同时记录否定症状
                negative_matched = []
                if negative_symptoms and client and client._driver:
                    for neg_sym in negative_symptoms:
                        neg_matches = client.semantic_match_symptoms(
                            neg_sym, top_k=3, threshold=0.6
                        )
                        negative_matched.extend(neg_matches)

                symptoms = [
                    m.get("name") for m in filtered_matches[:3] if m.get("name")
                ]

                return {
                    "symptoms": symptoms,
                    "negative_symptoms": [m.get("name") for m in negative_matched[:3]],
                    "sources": {s["name"]: "semantic" for s in filtered_matches[:3]},
                    "raw_matches": filtered_matches[:3],
                    "has_negation": len(negative_symptoms) > 0,
                }
    except Exception as e:
        print(f"Layer 0 (Neo4j语义匹配) 失败: {e}")

    return {
        "symptoms": [],
        "negative_symptoms": [],
        "sources": {},
        "raw_matches": [],
        "has_negation": False,
    }


def fill_slots(
    user_input: Any, current_slots: Optional[DiagnosisSlots] = None
) -> DiagnosisSlots:
    """从用户输入填充槽位（方案A：LLM提原词 → Neo4j转标准术语）"""
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

    # ============ Layer 1: LLM 提取原始症状 ============
    llm_result = _layer2_llm_extract(user_input)
    raw_symptoms = llm_result.get("symptoms", [])
    raw_negative = llm_result.get("negative_symptoms", [])

    # ============ Layer 2: Neo4j 俗语转换（原词 → 标准术语） ============
    standardized_symptoms = []
    symptom_mapping = {}  # 记录映射关系: {"头壳痛": "头痛"}
    for raw in raw_symptoms:
        std = _neo4j_convert_symptom(raw)
        if std:
            standardized_symptoms.append(std)
            symptom_mapping[raw] = std
        else:
            # Neo4j 转换失败，保留原词
            standardized_symptoms.append(raw)
            symptom_mapping[raw] = raw

    # 否定症状也做标准化
    standardized_negative = []
    for raw_neg in raw_negative:
        std_neg = _neo4j_convert_symptom(raw_neg)
        if std_neg:
            standardized_negative.append(std_neg)
        else:
            standardized_negative.append(raw_neg)

    # 从症状中移除被否定的症状
    final_symptoms = [
        s for s in standardized_symptoms if s not in standardized_negative
    ]

    # 合并历史症状（保留之前轮次已确认的症状）
    existing_symptoms = current_slots.symptoms if current_slots else []
    merged_symptoms = list(dict.fromkeys(existing_symptoms + final_symptoms))

    # 从合并后的症状中移除本轮新发现的否定症状
    merged_symptoms = [s for s in merged_symptoms if s not in standardized_negative]

    # 构建 slots
    slots_dict = current_slots.to_dict()
    slots_dict["symptoms"] = merged_symptoms
    slots_dict["negative_symptoms"] = standardized_negative

    # 合并症状来源：保留历史来源，新增本轮来源
    merged_sources = dict(current_slots.symptom_sources if current_slots else {})
    merged_sources.update({s: "llm+neo4j" for s in final_symptoms})
    # 移除被否定的症状的来源
    for neg in standardized_negative:
        merged_sources.pop(neg, None)
    slots_dict["symptom_sources"] = merged_sources

    # LLM 其他字段
    llm_full = llm_result.get("full_result", {})
    if llm_full.get("location"):
        slots_dict["location"] = llm_full["location"]
    if llm_full.get("duration"):
        slots_dict["duration"] = llm_full["duration"]
    if llm_full.get("severity"):
        slots_dict["severity"] = llm_full["severity"]
    if llm_full.get("triggers"):
        slots_dict["triggers"] = llm_full["triggers"]

    slots = DiagnosisSlots(**slots_dict)

    # ============ Layer 4: 知识图谱校验 ============
    slots = _layer4_kg_validate(slots, user_input)

    return slots


def _neo4j_convert_symptom(raw_symptom: str) -> str:
    """
    Neo4j 俗语转换：将用户原词转换为标准医学术语
    例如: "头壳痛" → "头痛", "肚子疼" → "腹痛"
    """
    if not raw_symptom:
        return ""

    try:
        from app.infra.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        if client and client._driver:
            # 先尝试向量语义匹配
            matches = client.semantic_match_symptoms(
                raw_symptom, top_k=1, threshold=0.7
            )
            if matches:
                matched_name = matches[0].get("name", raw_symptom)
                # 如果匹配到的症状和原词差异较大，说明是俗语转换成功
                if matched_name != raw_symptom:
                    return matched_name
                # 如果完全匹配，也返回
                return matched_name

            # 向量搜索无结果时，尝试关键词匹配
            keyword_matches = client.query_symptoms_by_keyword(raw_symptom)
            if keyword_matches:
                return keyword_matches[0]
    except Exception as e:
        print(f"Neo4j 俗语转换失败 ({raw_symptom}): {e}")

    return raw_symptom


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
    Layer 1: LLM 提取原始症状（保留用户原词，用于审计追溯）
    使用 Qwen Turbo 进行语义提取
    """
    if not USE_LLM_EXTRACTOR:
        return {"symptoms": [], "sources": {}}

    try:
        from app.domain.diagnosis.llm_extractor import extract_symptoms_with_llm

        result = extract_symptoms_with_llm(user_input)

        if not result:
            return {"symptoms": [], "sources": {}}

        # 获取原始症状（LLM 现在只提取原词，不做标准化）
        raw_symptoms = result.get("symptoms", [])
        negative_symptoms = result.get("negative_symptoms", [])

        return {
            "symptoms": raw_symptoms,
            "negative_symptoms": negative_symptoms,
            "sources": {s: "llm_raw" for s in raw_symptoms},
            "full_result": result,
        }

    except Exception as e:
        print(f"Layer 1 (LLM提取原词) 失败: {e}")
        return {"symptoms": [], "negative_symptoms": [], "sources": {}}


def _layer3_merge_to_slots(
    user_input: str,
    dict_symptoms: dict,
    llm_result: dict,
    existing_slots: DiagnosisSlots,
    neo4j_symptoms: dict = None,
) -> DiagnosisSlots:
    """
    Layer 3: 合并到结构化Slot
    合并 Neo4j、词典和LLM的结果
    优先级: LLM > 词典 > Neo4j
    """
    if neo4j_symptoms is None:
        neo4j_symptoms = {"symptoms": [], "sources": {}}

    # 收集所有症状（去重）
    all_symptoms: Set[str] = set()
    sources = {}

    # 添加 Neo4j 匹配的症状（最低优先级）
    for s in neo4j_symptoms.get("symptoms", []):
        all_symptoms.add(s)
        sources[s] = "neo4j"

    # 添加词典匹配的症状
    for s in dict_symptoms.get("symptoms", []):
        all_symptoms.add(s)
        sources[s] = "dict"

    # 添加LLM提取的症状（最高优先级）
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

    # 处理否定症状
    negative = llm_full.get("negative_symptoms", [])
    existing_negative = existing_slots.negative_symptoms if existing_slots else []
    all_negative = list(set(existing_negative + negative))

    # 合并历史症状（保留之前轮次已确认的症状）
    existing_symptoms = existing_slots.symptoms if existing_slots else []
    merged_symptoms = list(dict.fromkeys(existing_symptoms + list(final_symptoms)))

    # 从合并后的症状中移除被否定的症状（重要！）
    merged_symptoms = [s for s in merged_symptoms if s not in all_negative]

    slots_dict = existing_slots.to_dict()
    slots_dict["symptoms"] = merged_symptoms

    # 合并症状来源：保留历史来源，新增本轮来源
    merged_sources = dict(existing_slots.symptom_sources if existing_slots else {})
    merged_sources.update({k: v for k, v in sources.items() if k not in all_negative})
    slots_dict["symptom_sources"] = merged_sources
    slots_dict["negative_symptoms"] = all_negative

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
