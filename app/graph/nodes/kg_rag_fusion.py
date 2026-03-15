"""
KG + RAG 综合推理模块

功能：
1. KG 推理: 症状 → 科室概率
2. RAG 检索: 相关文档 + 科室信息
3. 综合评分: 融合 KG 和 RAG 的结果
"""

from typing import List, Dict, Any, Optional
from app.core.logging import logger


def kg_rag_fusion(
    symptoms: List[str],
    kg_result: Optional[Dict[str, Any]],
    rag_docs: List[Any],
    user_query: str = "",
    kg_weight: float = 0.6,
    rag_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    KG + RAG 综合评分

    Args:
        symptoms: 症状列表
        kg_result: KG 推理结果 (包含 departments)
        rag_docs: RAG 检索结果 (文档列表)
        user_query: 用户原始问题
        kg_weight: KG 权重
        rag_weight: RAG 权重

    Returns:
        综合推理结果
    """
    if not symptoms:
        return {
            "departments": [],
            "confidence": {"overall_confidence": 0.0, "reason": "无症状"},
            "sources": {"kg": 0, "rag": 0},
        }

    # 1. 提取 KG 科室评分
    kg_depts = {}
    if kg_result and kg_result.get("departments"):
        for dept in kg_result["departments"]:
            name = dept.get("name", "")
            prob = dept.get("probability", 0)
            kg_depts[name] = prob

    # 2. 从 RAG 文档中提取科室信息
    rag_depts = _extract_depts_from_rag(rag_docs)

    # 3. 合并评分
    all_depts = set(kg_depts.keys()) | set(rag_depts.keys())

    final_scores = {}
    for dept in all_depts:
        kg_score = kg_depts.get(dept, 0)
        rag_score = rag_depts.get(dept, 0)

        # 加权综合评分
        combined = kg_score * kg_weight + rag_score * rag_weight
        final_scores[dept] = combined

    # 4. 排序并返回结果
    sorted_depts = sorted(final_scores.items(), key=lambda x: -x[1])

    # 5. 计算综合置信度
    confidence = _calculate_fusion_confidence(
        kg_depts=kg_depts,
        rag_depts=rag_depts,
        kg_weight=kg_weight,
        rag_weight=rag_weight,
    )

    return {
        "departments": [
            {"name": name, "score": round(score, 3), "probability": round(score, 3)}
            for name, score in sorted_depts[:5]
        ],
        "confidence": confidence,
        "sources": {
            "kg": len(kg_depts),
            "rag": len(rag_depts),
            "total": len(all_depts),
        },
        "kg_depts": kg_depts,
        "rag_depts": rag_depts,
    }


def _extract_depts_from_rag(rag_docs: List[Any]) -> Dict[str, float]:
    """
    从 RAG 检索结果中提取科室信息

    方法:
    1. 关键词匹配科室名称
    2. 从 Neo4j 查找疾病对应的科室
    """
    dept_keywords = [
        "消化内科",
        "心内科",
        "神经内科",
        "呼吸内科",
        "内分泌科",
        "血液科",
        "肾内科",
        "免疫科",
        "感染科",
        "急诊科",
        "普外科",
        "骨科",
        "神经外科",
        "心胸外科",
        "泌尿外科",
        "烧伤科",
        "妇科",
        "产科",
        "儿科",
        "小儿内科",
        "眼科",
        "耳鼻喉科",
        "口腔科",
        "皮肤科",
        "中医科",
        "肿瘤科",
        "康复科",
        "老年科",
        "风湿科",
        "麻醉科",
    ]

    common_diseases = [
        "高血压",
        "糖尿病",
        "感冒",
        "发烧",
        "肺炎",
        "支气管炎",
        "胃炎",
        "肠炎",
        "肝炎",
        "冠心病",
        "心肌炎",
        "心律失常",
        "脑卒中",
        "脑梗",
        "癫痫",
        "帕金森",
        "阿尔茨海默",
        "关节炎",
        "骨质疏松",
        "腰椎间盘突出",
        "颈椎病",
        "皮炎",
        "湿疹",
        "荨麻疹",
        "银屑病",
        "白内障",
        "青光眼",
        "近视",
        "远视",
        "鼻炎",
        "咽炎",
        "喉炎",
        "中耳炎",
        "子宫肌瘤",
        "卵巢囊肿",
        "月经不调",
        "痛经",
        "前列腺增生",
        "肾结石",
        "尿路感染",
        "贫血",
        "白血病",
        "淋巴瘤",
        "肺癌",
        "胃癌",
        "肝癌",
        "肠癌",
        "乳腺癌",
    ]

    dept_counts = {}
    disease_to_dept = {}
    neo4j_client = None
    neo4j_available = False

    try:
        from app.infra.neo4j_client import get_neo4j_client

        neo4j_client = get_neo4j_client()
        neo4j_available = True
    except Exception:
        pass

    for doc in rag_docs:
        content = ""
        if hasattr(doc, "content"):
            content = doc.content
        elif isinstance(doc, dict):
            content = doc.get("content", "")

        if not content:
            continue

        for dept in dept_keywords:
            if dept in content:
                dept_counts[dept] = dept_counts.get(dept, 0) + 1

        if neo4j_available:
            for disease in common_diseases:
                if disease in content and disease not in disease_to_dept:
                    try:
                        depts = neo4j_client.query_department_by_disease(disease)
                        for d in depts[:2]:
                            dept_counts[d] = dept_counts.get(d, 0) + 0.5
                        disease_to_dept[disease] = depts
                    except Exception:
                        pass

    if not dept_counts:
        return {}

    max_count = max(dept_counts.values())
    normalized = {dept: count / max_count for dept, count in dept_counts.items()}

    return normalized


def _calculate_fusion_confidence(
    kg_depts: Dict[str, float],
    rag_depts: Dict[str, float],
    kg_weight: float,
    rag_weight: float,
) -> Dict[str, Any]:
    """
    计算综合置信度
    """
    # KG 置信度
    kg_conf = 0.0
    if kg_depts:
        top_kg = max(kg_depts.values())
        # 一致性
        consistency = len([v for v in kg_depts.values() if v > 0.3]) / len(kg_depts)
        kg_conf = top_kg * 0.7 + consistency * 0.3

    # RAG 置信度
    rag_conf = 0.0
    if rag_depts:
        top_rag = max(rag_depts.values())
        # 覆盖度
        coverage = min(len(rag_depts) / 3, 1.0)
        rag_conf = top_rag * 0.6 + coverage * 0.4

    # 综合置信度
    overall = kg_conf * kg_weight + rag_conf * rag_weight

    return {
        "kg_confidence": round(kg_conf, 3),
        "rag_confidence": round(rag_conf, 3),
        "overall_confidence": round(overall, 3),
        "reason": _get_confidence_reason(kg_depts, rag_depts),
    }


def _get_confidence_reason(kg_depts: Dict, rag_depts: Dict) -> str:
    """根据数据来源给出置信度原因"""
    kg_count = len(kg_depts)
    rag_count = len(rag_depts)

    if kg_count > 0 and rag_count > 0:
        return "KG + RAG 综合"
    elif kg_count > 0:
        return "仅 KG 推理"
    elif rag_count > 0:
        return "仅 RAG 检索"
    else:
        return "无数据"


def diagnose_with_kg_rag(
    symptoms: List[str],
    user_query: str = "",
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    完整的 KG + RAG 诊断流程（并行执行）

    Args:
        symptoms: 症状列表
        user_query: 用户原始问题
        top_k: 返回前 k 个科室

    Returns:
        综合诊断结果
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from app.infra.neo4j_client import get_neo4j_client
    from app.graph.nodes.milvus_rag import rrf_fusion, rerank_with_qwen
    from app.infra.milvus_client import search_medical_docs
    from app.infra.es_client import search_rag_es
    from app.core import config

    kg_result = None
    rag_docs = []

    def run_kg():
        try:
            client = get_neo4j_client()
            if client:
                return client.infer_department(symptoms, top_k=top_k)
        except Exception as e:
            logger.warning(f"KG 推理失败: {e}")
        return None

    def run_rag():
        try:
            if not user_query:
                return []
            es_docs = search_rag_es(user_query, size=config.RETRIEVAL_K)
            milvus_docs = search_medical_docs(user_query)
            fused = rrf_fusion(es_docs, milvus_docs, k=config.RRF_K)
            return rerank_with_qwen(user_query, fused, top_n=config.RERANK_TOP_N)
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return []

    # 并行执行 KG 和 RAG
    with ThreadPoolExecutor(max_workers=2) as executor:
        kg_future = executor.submit(run_kg)
        rag_future = executor.submit(run_rag)

        try:
            kg_result = kg_future.result(timeout=10)
            logger.info(
                f"KG 推理完成: {kg_result.get('departments', [])[:3] if kg_result else []}"
            )
        except Exception as e:
            logger.warning(f"KG 任务异常: {e}")
            kg_result = None

        try:
            rag_docs = rag_future.result(timeout=10)
            logger.info(f"RAG 检索完成: {len(rag_docs)} 条")
        except Exception as e:
            logger.warning(f"RAG 任务异常: {e}")
            rag_docs = []

    # 融合评分（带容错）
    fusion_result = kg_rag_fusion(
        symptoms=symptoms,
        kg_result=kg_result,
        rag_docs=rag_docs,
        user_query=user_query,
        kg_weight=0.6,
        rag_weight=0.4,
    )

    # 兜底处理：两者都为空
    if not fusion_result.get("departments"):
        fusion_result["fallback"] = True
        fusion_result["fallback_message"] = (
            "无法确定科室，建议您到医院预检台咨询或线下就医"
        )
        logger.warning("KG+RAG 均无结果，返回兜底提示")

    return fusion_result
