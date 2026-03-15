from mcp.server.fastmcp import FastMCP
import json
import os
import logging
from typing import List, Optional

# 禁用日志输出，避免干扰 MCP 协议
logging.basicConfig(level=logging.CRITICAL)
for logger_name in logging.Logger.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("HospitalTools")

from app.tools.knowledge_graph_tool import (
    query_symptom_associations,
    query_department,
    check_emergency,
    query_symptom_associations_with_context,
    query_symptoms_by_keyword,
    get_full_symptom_info,
    query_hybrid_retrieval,
    get_discriminative_symptoms,
)
from app.infra.neo4j_client import get_neo4j_client
from app.infra.postgres_client import get_patient_client

MOCK_PATIENTS = {
    "001": {
        "name": "张三",
        "age": 45,
        "gender": "男",
        "phone": "13800138001",
        "records": [
            {
                "record_id": "R001",
                "visit_date": "2024-01-15",
                "diagnosis": "高血压",
                "treatment": "降压药治疗",
                "doctor": "李医生",
            },
            {
                "record_id": "R002",
                "visit_date": "2024-02-20",
                "diagnosis": "糖尿病",
                "treatment": "胰岛素治疗",
                "doctor": "王医生",
            },
        ],
    },
}


@mcp.tool()
def get_patient_history(patient_name: str) -> str:
    for pid, patient in MOCK_PATIENTS.items():
        if patient["name"] == patient_name:
            return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者 {patient_name} 的记录"})


@mcp.tool()
def get_patient_by_id(patient_id: str) -> str:
    patient = MOCK_PATIENTS.get(patient_id)
    if patient:
        return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者ID {patient_id}"})


@mcp.tool()
def symptom_associations(symptoms: List[str]) -> str:
    result = query_symptom_associations_with_context(symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def department_by_symptom(symptom: str) -> str:
    dept = query_department(symptom)
    return json.dumps({"department": dept}, ensure_ascii=False)


@mcp.tool()
def emergency_check(symptoms: List[str]) -> str:
    result = check_emergency(symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def symptom_search(keyword: str) -> str:
    result = query_symptoms_by_keyword(keyword)
    return json.dumps({"symptoms": result}, ensure_ascii=False)


@mcp.tool()
def full_symptom_info(symptom: str) -> str:
    result = get_full_symptom_info(symptom)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def hybrid_retrieval(
    query_text: str, known_symptoms: Optional[List[str]] = None
) -> str:
    result = query_hybrid_retrieval(query_text, known_symptoms)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def discriminative_symptoms(
    known_symptoms: List[str], candidate_symptoms: List[str], limit: int = 5
) -> str:
    result = get_discriminative_symptoms(known_symptoms, candidate_symptoms, limit)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def infer_department(symptoms: List[str], top_k: int = 3) -> str:
    """
    基于多症状推理推荐科室（带置信度）

    Args:
        symptoms: 症状列表
        top_k: 返回前 k 个科室

    Returns:
        科室推荐结果，包含置信度
    """
    client = get_neo4j_client()
    if client:
        result = client.infer_department(symptoms, top_k)
    else:
        result = {"error": "Neo4j 不可用"}
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_symptom_dept_probability(symptom: str) -> str:
    """
    获取单个症状的科室概率分布

    Args:
        symptom: 症状名称

    Returns:
        科室概率分布 JSON
    """
    client = get_neo4j_client()
    if client:
        result = client.get_symptom_dept_probs(symptom)
    else:
        result = {"error": "Neo4j 不可用"}
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_possible_diseases(symptoms: List[str], limit: int = 10) -> str:
    """
    根据症状查询可能的疾病

    Args:
        symptoms: 症状列表
        limit: 返回数量

    Returns:
        可能的疾病列表
    """
    client = get_neo4j_client()
    if client:
        result = client.get_diseases_by_symptoms(symptoms, limit)
    else:
        result = {"error": "Neo4j 不可用"}
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def calculate_confidence(symptoms: List[str]) -> str:
    """
    计算多症状推理的置信度

    Args:
        symptoms: 症状列表

    Returns:
        置信度指标
    """
    client = get_neo4j_client()
    if client:
        # 获取每个症状的概率
        symptom_probs = []
        for s in symptoms:
            probs = client.get_symptom_dept_probs(s)
            if probs:
                symptom_probs.append(probs)

        confidence = client.calculate_confidence(symptom_probs, len(symptoms))
    else:
        confidence = {"error": "Neo4j 不可用"}
    return json.dumps(confidence, ensure_ascii=False)


@mcp.tool()
def semantic_match_symptoms(
    query_text: str, top_k: int = 5, threshold: float = 0.5
) -> str:
    """
    Neo4j 向量语义匹配 - 将用户描述匹配到标准症状名称

    Args:
        query_text: 用户描述或查询文本
        top_k: 返回前 k 个结果
        threshold: 相似度阈值

    Returns:
        匹配的症状列表
    """
    client = get_neo4j_client()
    if client:
        result = client.semantic_match_symptoms(query_text, top_k, threshold)
    else:
        result = {"error": "Neo4j 不可用"}
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def pg_get_patient_by_name(name: str) -> str:
    """
    PostgreSQL - 根据姓名查询患者信息

    Args:
        name: 患者姓名

    Returns:
        患者信息 JSON
    """
    client = get_patient_client()
    patient = client.get_patient_by_name(name)
    if patient:
        return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者 {name}"})


@mcp.tool()
def pg_get_patient_by_id(patient_id: str) -> str:
    """
    PostgreSQL - 根据ID查询患者信息

    Args:
        patient_id: 患者ID

    Returns:
        患者信息 JSON
    """
    client = get_patient_client()
    patient = client.get_patient_by_id(patient_id)
    if patient:
        return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者ID {patient_id}"})


@mcp.tool()
def pg_get_patient_history(patient_id: str, limit: int = 20) -> str:
    """
    PostgreSQL - 查询患者就诊历史

    Args:
        patient_id: 患者ID
        limit: 返回记录数

    Returns:
        就诊历史列表
    """
    client = get_patient_client()
    history = client.get_patient_history(patient_id, limit)
    return json.dumps(history, ensure_ascii=False)


@mcp.tool()
def pg_search_patients(keyword: str, limit: int = 10) -> str:
    """
    PostgreSQL - 搜索患者（按姓名或电话）

    Args:
        keyword: 搜索关键词
        limit: 返回数量

    Returns:
        患者列表
    """
    client = get_patient_client()
    patients = client.search_patients(keyword, limit)
    return json.dumps(patients, ensure_ascii=False)


@mcp.tool()
def milvus_search(query: str, top_k: int = 15) -> str:
    """
    Milvus 向量检索 - 从病历库中检索相关文档

    Args:
        query: 查询文本
        top_k: 返回数量

    Returns:
        检索结果列表
    """
    from app.infra.milvus_client import search_medical_docs

    try:
        docs = search_medical_docs(query)
        result = [
            {"id": d.id, "title": d.title, "content": d.content, "score": d.score}
            for d in docs
        ]
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Milvus 检索失败: {str(e)}"})


@mcp.tool()
def es_search(query: str, size: int = 50) -> str:
    """
    Elasticsearch 检索 - 从指南库中检索相关文档

    Args:
        query: 查询文本
        size: 返回数量

    Returns:
        检索结果列表
    """
    from app.infra.es_client import search_rag_es

    try:
        docs = search_rag_es(query, size)
        result = [
            {"id": d.id, "title": d.title, "content": d.content, "score": d.score}
            for d in docs
        ]
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"ES 检索失败: {str(e)}"})


@mcp.tool()
def kg_rag_fusion(symptoms: List[str], user_query: str = "", top_k: int = 3) -> str:
    """
    KG + RAG 综合推理 - 融合知识图谱和 RAG 的科室推荐

    Args:
        symptoms: 症状列表
        user_query: 用户原始问题
        top_k: 返回前 k 个科室

    Returns:
        综合推理结果
    """
    from app.graph.nodes.kg_rag_fusion import diagnose_with_kg_rag

    try:
        result = diagnose_with_kg_rag(symptoms, user_query, top_k)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"KG+RAG 推理失败: {str(e)}"})


if __name__ == "__main__":
    import sys

    if "--sse" in sys.argv:
        print("Starting MCP Server in SSE mode on port 8001")
        mcp.run(transport="sse")
    else:
        # 默认使用 stdio 模式
        mcp.run()
