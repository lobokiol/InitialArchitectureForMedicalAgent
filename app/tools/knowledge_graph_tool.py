"""
知识图谱 Tool - 用于症状关联查询和科室推荐

在 LangGraph 中以 Tool 形式集成，供 diagnosis 节点调用：
- query_symptom_associations: 查询某症状常伴随的其他症状（用于追问）
- query_department: 查询症状对应的推荐科室（用于分诊）
"""

from typing import List, Dict, Any, Optional


# 简化版知识图谱数据（企业级会从 Neo4j 导入）
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


def query_symptom_associations(symptom: str) -> List[str]:
    """
    查询某症状常伴随的其他症状（用于追问）

    Args:
        symptom: 标准化后的症状名称

    Returns:
        伴随症状列表
    """
    return SYMPTOM_ASSOCIATIONS.get(symptom, [])


def query_department(symptom: str) -> Optional[str]:
    """
    查询症状对应的推荐科室

    Args:
        symptom: 标准化后的症状名称

    Returns:
        推荐科室名称
    """
    return DEPARTMENT_RECOMMEND.get(symptom)


def query_symptom_associations_with_context(symptoms: List[str]) -> Dict[str, Any]:
    """
    批量查询多个症状的关联信息（供 Tool 调用）

    Args:
        symptoms: 症状列表

    Returns:
        包含 associated_symptoms 和 recommended_department 的字典
    """
    all_associated = set()
    departments = set()

    for symptom in symptoms:
        associated = query_symptom_associations(symptom)
        all_associated.update(associated)

        dept = query_department(symptom)
        if dept:
            departments.add(dept)

    return {
        "symptoms": symptoms,
        "associated_symptoms": list(all_associated),
        "recommended_departments": list(departments),
    }


def create_knowledge_graph_tools():
    """
    创建 LangChain Tool 列表（企业级实现会连接 Neo4j）

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
