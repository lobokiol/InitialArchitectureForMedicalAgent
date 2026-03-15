"""
科室推理节点 - 基于多症状的置信度推理

功能：
1. 收集当前已识别的症状
2. 调用 Neo4j 进行科室推理
3. 计算置信度
4. 判断是否需要追问
"""

import json
from typing import Dict, Any, List

from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.models import AppState
from app.infra.neo4j_client import get_neo4j_client, Neo4jClient


def department_infer_node(state: AppState) -> dict:
    """
    科室推理节点

    根据已收集的症状，推理推荐科室并计算置信度

    Returns:
        - department_inference: 科室推理结果
        - confidence: 置信度
        - need_more_info: 是否需要追问
    """
    logger.info(">>> Enter node: department_infer")

    # 获取当前已收集的症状
    symptoms = []
    if hasattr(state, "diagnosis_slots") and state.diagnosis_slots:
        symptoms = state.diagnosis_slots.symptoms or []

    if not symptoms:
        logger.info("department_infer: no symptoms collected yet")
        return {
            "department_inference": None,
            "confidence": {"overall_confidence": 0.0},
            "need_more_info": True,
        }

    logger.info(f"department_infer: symptoms={symptoms}")

    # 调用 Neo4j 进行推理
    client = get_neo4j_client()

    if client is None:
        logger.warning("department_infer: Neo4j not available")
        return {
            "department_inference": {"departments": []},
            "confidence": {"overall_confidence": 0.0, "error": "Neo4j 不可用"},
            "need_more_info": True,
        }

    try:
        # 推理科室
        result = client.infer_department(symptoms, top_k=3)

        confidence = result.get("confidence", {})
        overall_conf = confidence.get("overall_confidence", 0.0)

        # 判断是否需要追问
        need_more_info = overall_conf < 0.7

        logger.info(
            f"department_infer: departments={result.get('departments')}, "
            f"confidence={overall_conf}, need_more_info={need_more_info}"
        )

        return {
            "department_inference": result,
            "confidence": confidence,
            "need_more_info": need_more_info,
        }

    except Exception as e:
        logger.exception("department_infer: 推理失败")
        return {
            "department_inference": {"departments": []},
            "confidence": {"overall_confidence": 0.0, "error": str(e)},
            "need_more_info": True,
        }


def generate_inference_message(state: AppState) -> dict:
    """
    生成科室推理结果的消息

    根据置信度决定：
    - 高置信度 (>0.8): 直接输出科室推荐
    - 中置信度 (0.5-0.8): 输出推荐 + 建议追问
    - 低置信度 (<0.5): 建议继续补充症状
    """
    inference = getattr(state, "department_inference", None)
    confidence = getattr(state, "confidence", {})

    if not inference or not inference.get("departments"):
        return {"messages": [AIMessage(content="请继续描述您的症状")]}

    departments = inference.get("departments", [])
    overall_conf = confidence.get("overall_confidence", 0.0)

    # 构建消息
    lines = []
    lines.append("根据您描述的症状，我的分析结果是：")
    lines.append("")

    # 科室推荐
    for i, dept in enumerate(departments, 1):
        prob_pct = dept.get("probability", 0) * 100
        lines.append(f"{i}. {dept['name']} (置信度: {prob_pct:.0f}%)")

    lines.append("")

    # 置信度说明
    if overall_conf >= 0.8:
        lines.append("✅ 置信度较高，建议您选择上述科室就诊。")
    elif overall_conf >= 0.5:
        lines.append(
            "⚠️ 置信度中等，您可以参考以上建议。如果有更多症状描述，可能更准确。"
        )
    else:
        lines.append("❌ 置信度较低，建议您补充更多症状信息，以便更准确地推荐科室。")

    # 添加可能的疾病信息（如果有）
    if hasattr(state, "diagnosis_slots") and state.diagnosis_slots:
        client = get_neo4j_client()
        if client:
            try:
                diseases = client.get_diseases_by_symptoms(
                    state.diagnosis_slots.symptoms or [], limit=3
                )
                if diseases:
                    lines.append("")
                    lines.append("可能相关的疾病：")
                    for d in diseases:
                        lines.append(
                            f"  • {d['disease']} (对应科室: {d['department']})"
                        )
            except Exception as e:
                logger.warning(f"获取疾病信息失败: {e}")

    message = "\n".join(lines)
    return {"messages": [AIMessage(content=message)]}
