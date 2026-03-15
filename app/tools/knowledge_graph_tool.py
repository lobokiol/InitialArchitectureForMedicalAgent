"""
知识图谱 Tool - 用于症状关联查询和科室推荐

支持双模式：
1. Neo4j 图数据库（优先）
2. 本地字典（fallback）

在 LangGraph 中以 Tool 形式集成，供 diagnosis 节点调用：
- query_symptom_associations: 查询某症状常伴随的其他症状（用于追问）
- query_department: 查询症状对应的推荐科室（用于分诊）
- check_emergency: 检查危急症状
"""

from typing import List, Dict, Any, Optional

# 本地字典模式（fallback）
SYMPTOM_ASSOCIATIONS = {
    "腹痛": ["恶心", "腹胀", "腹泻", "发热", "呕吐"],
    "胃痛": ["恶心", "反酸", "腹胀", "呕血", "黑便"],
    "胸痛": ["呼吸困难", "心悸", "胸闷", "出汗", "恶心"],
    "头痛": ["头晕", "恶心", "发热", "视力模糊", "嗜睡"],
    "咳嗽": ["发热", "咳痰", "胸闷", "气促", "咽痛"],
    "发热": ["头痛", "乏力", "寒战", "出汗", "咽痛"],
    "腹泻": ["腹痛", "腹胀", "恶心", "发热", "脱水"],
    "恶心": ["呕吐", "腹痛", "腹胀", "头晕", "乏力"],
    "呕吐": ["腹痛", "腹胀", "发热", "头痛", "脱水"],
    "呼吸困难": ["胸闷", "心悸", "咳嗽", "发热", "出汗"],
    "头晕": ["头痛", "恶心", "乏力", "视力模糊", "耳鸣"],
    "腰痛": ["下肢麻木", "发热", "尿频", "尿急", "血尿"],
}

DEPARTMENT_RECOMMEND = {
    "腹痛": "消化内科",
    "胃痛": "消化内科",
    "胸痛": "心血管内科",
    "头痛": "神经内科",
    "咳嗽": "呼吸内科",
    "发热": "发热门诊",
    "腹泻": "消化内科",
    "恶心": "消化内科",
    "呕吐": "消化内科",
    "呼吸困难": "呼吸内科",
    "头晕": "神经内科",
    "腰痛": "泌尿外科",
    "胸闷": "心血管内科",
    "心悸": "心血管内科",
    "乏力": "内分泌科",
    "皮疹": "皮肤科",
    "失眠": "神经内科",
    "水肿": "肾内科",
}

# Neo4j 客户端（延迟初始化）
_neo4j_client = None


def get_neo4j_client():
    """获取 Neo4j 客户端"""
    global _neo4j_client
    if _neo4j_client is None:
        try:
            from app.infra.neo4j_client import get_neo4j_client as _get_client

            _neo4j_client = _get_client()
        except Exception as e:
            print(f"Neo4j 客户端初始化失败: {e}")
            _neo4j_client = None
    return _neo4j_client


def is_neo4j_available() -> bool:
    """检查 Neo4j 是否可用"""
    client = get_neo4j_client()
    return client is not None and client._driver is not None


def query_symptom_associations(symptom: str) -> List[str]:
    """
    查询某症状常伴随的其他症状（用于追问）

    Args:
        symptom: 标准化后的症状名称

    Returns:
        伴随症状列表
    """
    # 优先使用 Neo4j
    if is_neo4j_available():
        try:
            result = get_neo4j_client().query_associated_symptoms(symptom)
            if result:
                return [r["name"] for r in result]
        except Exception as e:
            print(f"Neo4j 查询失败，使用本地字典: {e}")

    # Fallback 到本地字典
    return SYMPTOM_ASSOCIATIONS.get(symptom, [])


def query_department(symptom: str) -> Optional[str]:
    """
    查询症状对应的推荐科室

    Args:
        symptom: 标准化后的症状名称

    Returns:
        推荐科室名称
    """
    # 优先使用 Neo4j
    if is_neo4j_available():
        try:
            # 尝试精确匹配
            result = get_neo4j_client().query_departments_by_symptom(symptom)
            if result:
                return result[0]["name"]

            # 如果精确匹配失败，尝试关键词搜索
            matched_symptoms = get_neo4j_client().query_symptoms_by_keyword(symptom)
            if matched_symptoms:
                # 使用第一个匹配结果
                result = get_neo4j_client().query_departments_by_symptom(
                    matched_symptoms[0]
                )
                if result:
                    return result[0]["name"]
        except Exception as e:
            print(f"Neo4j 查询失败，使用本地字典: {e}")

    # Fallback 到本地字典
    return DEPARTMENT_RECOMMEND.get(symptom)


def query_hybrid_retrieval(
    query_text: str, known_symptoms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    混合检索: 向量搜索 + 图推理

    Args:
        query_text: 用户描述文本
        known_symptoms: 已知的症状列表

    Returns:
        混合检索结果
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().hybrid_retrieval(query_text, known_symptoms)
        except Exception as e:
            print(f"混合检索失败: {e}")

    return {
        "recommended_departments": [],
        "expanded_symptoms": [],
        "possible_diseases": [],
    }


def get_discriminative_symptoms(
    known_symptoms: List[str], candidate_symptoms: List[str], limit: int = 5
) -> List[Dict[str, Any]]:
    """
    获取判别性症状 - 用于动态生成追问

    Args:
        known_symptoms: 已知的症状
        candidate_symptoms: 候选症状列表
        limit: 返回数量

    Returns:
        判别性症状列表
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().get_discriminative_symptoms(
                known_symptoms, candidate_symptoms, limit
            )
        except Exception as e:
            print(f"获取判别性症状失败: {e}")

    return []


def query_symptom_associations_with_context(symptoms: List[str]) -> Dict[str, Any]:
    """
    批量查询多个症状的关联信息（供 Tool 调用）

    Args:
        symptoms: 症状列表

    Returns:
        包含 associated_symptoms 和 recommended_departments 的字典
    """
    all_associated = set()
    departments = set()
    normalized_symptoms = set()  # 标准化后的症状

    for symptom in symptoms:
        # 查询伴随症状
        associated = query_symptom_associations(symptom)
        all_associated.update(associated)

        # 查询科室
        dept = query_department(symptom)
        if dept:
            departments.add(dept)

        # 收集标准化后的症状（用于危急检查）
        if is_neo4j_available():
            try:
                matched = get_neo4j_client().query_symptoms_by_keyword(symptom)
                normalized_symptoms.update(matched)
            except Exception:
                pass

    # 如果没有标准化症状，使用原始症状
    if not normalized_symptoms:
        normalized_symptoms = set(symptoms)

    # 检查危急症状（使用标准化后的症状）
    emergency_rules = []
    if is_neo4j_available():
        try:
            emergency_rules = get_neo4j_client().check_emergency(
                list(normalized_symptoms)
            )
        except Exception as e:
            print(f"危急检查失败: {e}")

    return {
        "symptoms": symptoms,
        "normalized_symptoms": list(normalized_symptoms),
        "associated_symptoms": list(all_associated),
        "recommended_departments": list(departments),
        "emergency_rules": emergency_rules,
    }


def query_symptoms_by_keyword(keyword: str) -> List[str]:
    """
    根据关键词搜索症状

    Args:
        keyword: 关键词

    Returns:
        症状列表
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().query_symptoms_by_keyword(keyword)
        except Exception as e:
            print(f"关键词搜索失败: {e}")

    # 简单的本地搜索
    results = []
    for symptom in SYMPTOM_ASSOCIATIONS.keys():
        if keyword in symptom:
            results.append(symptom)
    return results


def check_emergency(symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    检查危急症状

    Args:
        symptoms: 症状列表

    Returns:
        匹配的危急规则列表
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().check_emergency(symptoms)
        except Exception as e:
            print(f"危急检查失败: {e}")

    return []


def get_full_symptom_info(symptom: str) -> Dict[str, Any]:
    """
    获取症状的完整信息

    Args:
        symptom: 症状名称

    Returns:
        包含科室推荐和伴随症状的字典
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().get_full_symptom_info(symptom)
        except Exception as e:
            print(f"获取完整信息失败: {e}")

    # Fallback 到本地字典
    return {
        "symptom": symptom,
        "recommended_departments": [DEPARTMENT_RECOMMEND.get(symptom, "未知")],
        "associated_symptoms": SYMPTOM_ASSOCIATIONS.get(symptom, []),
    }


def infer_department(symptoms: List[str], top_k: int = 3) -> Dict[str, Any]:
    """
    基于多症状推理推荐科室（带置信度）

    Args:
        symptoms: 症状列表
        top_k: 返回前 k 个科室

    Returns:
        科室推荐结果，包含置信度
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().infer_department(symptoms, top_k)
        except Exception as e:
            print(f"科室推理失败: {e}")

    return {
        "departments": [],
        "confidence": {"overall_confidence": 0.0},
        "error": "Neo4j 不可用",
    }


def get_possible_diseases(symptoms: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    根据症状查询可能的疾病

    Args:
        symptoms: 症状列表
        limit: 返回数量

    Returns:
        疾病列表
    """
    if is_neo4j_available():
        try:
            return get_neo4j_client().get_diseases_by_symptoms(symptoms, limit)
        except Exception as e:
            print(f"查询疾病失败: {e}")

    return []


def create_knowledge_graph_tools():
    """
    创建 LangChain Tool 列表

    Returns:
        Tool 列表
    """
    from langchain_core.tools import StructuredTool

    symptom_associations_tool = StructuredTool.from_function(
        func=query_symptom_associations_with_context,
        name="symptom_associations",
        description="查询症状常伴随的其他症状，用于生成个性化追问。当需要了解某症状常见伴随症状时调用此工具。",
    )

    return [symptom_associations_tool]


# 模块加载时打印状态
print(
    f"知识图谱工具初始化: Neo4j={'可用' if is_neo4j_available() else '不可用（使用本地字典）'}"
)
