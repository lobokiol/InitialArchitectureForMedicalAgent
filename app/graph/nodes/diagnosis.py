import logging

from app.domain.models import AppState
from app.domain.diagnosis.slots import DiagnosisSlots

logger = logging.getLogger(__name__)
from app.domain.diagnosis.risk import (
    is_critical as check_is_critical,
    check_risks_with_kg,
    generate_emergency_warning,
)
from app.domain.diagnosis.questions import get_emergency_warning, get_completion_message
from app.graph.nodes.risk_check import risk_check_node
from app.graph.nodes.completion import completion_node
from app.graph.nodes.question_gen import question_gen_node
from app.graph.nodes.normalize import normalize_text
from langchain_core.messages import AIMessage
from app.mcp.client import symptom_associations_mcp, department_by_symptom_mcp
from app.infra.neo4j_client import get_neo4j_client
import json
import random

CONFIDENT_THRESHOLD = 0.65


def fill_slots_with_input(
    user_input: str, current_slots: DiagnosisSlots
) -> DiagnosisSlots:
    """使用标准化后的输入填充槽位（LLM增强版）"""
    from app.domain.diagnosis.filler import fill_slots

    if not user_input or not user_input.strip():
        return current_slots

    return fill_slots(user_input, current_slots)


def diagnosis_node(state: AppState) -> dict:
    """
    主编排器：协调各Agent节点完成问诊
    流程: normalize -> slot_fill -> knowledge_graph -> risk_check -> (emergency/completion/question_gen)
    """
    from app.graph.nodes.slot_fill import slot_fill_node

    user_input = state.messages[-1].content
    if isinstance(user_input, list):
        user_input = str(user_input[0]) if user_input else ""
    elif not isinstance(user_input, str):
        user_input = str(user_input) if user_input else ""

    if not user_input:
        return {"messages": [AIMessage(content="请描述您的症状")]}

    normalized_input = normalize_text(user_input)

    existing_slots = (
        state.diagnosis_slots if hasattr(state, "diagnosis_slots") else None
    )
    if existing_slots is None:
        existing_slots = DiagnosisSlots()

    filled_slots = fill_slots_with_input(normalized_input, existing_slots)
    state.diagnosis_slots = filled_slots

    associated_symptoms = []
    recommended_departments = []
    emergency_rules = []
    if filled_slots.symptoms:
        # 使用 MCP 调用知识图谱工具
        mcp_result = symptom_associations_mcp(filled_slots.symptoms)
        try:
            kg_result = json.loads(mcp_result)
        except json.JSONDecodeError:
            kg_result = {
                "associated_symptoms": [],
                "recommended_departments": [],
                "emergency_rules": [],
                "error": mcp_result,
            }
        associated_symptoms = kg_result.get("associated_symptoms", [])
        recommended_departments = kg_result.get("recommended_departments", [])
        emergency_rules = kg_result.get("emergency_rules", [])

        # 如果知识图谱检测到危急规则，优先处理
        if emergency_rules:
            warning = generate_emergency_warning(emergency_rules)
            return {
                "messages": [AIMessage(content=warning)],
                "diagnosis_completed": True,
                "diagnosis_terminated": True,
                "diagnosis_type": "emergency",
                "diagnosis_slots": state.diagnosis_slots,
                "diagnosis_recommended_departments": recommended_departments,
                "diagnosis_associated_symptoms": associated_symptoms,
                "diagnosis_emergency_rules": emergency_rules,
            }

        # 使用知识图谱增强风险检查
        normalized_symptoms = kg_result.get(
            "normalized_symptoms", filled_slots.symptoms
        )
        kg_risk_result = check_risks_with_kg(normalized_symptoms, user_input)
        state.diagnosis_risk_level = kg_risk_result.get("risk_level", "none")
        state.diagnosis_risk_signals = kg_risk_result.get("risk_signals", [])

        # 如果检测到高风险，也返回警告
        if state.diagnosis_risk_level in ["critical", "high"]:
            warning = generate_emergency_warning(
                kg_risk_result.get("emergency_rules", [])
            )
            return {
                "messages": [AIMessage(content=warning)],
                "diagnosis_completed": True,
                "diagnosis_terminated": True,
                "diagnosis_type": "emergency",
                "diagnosis_slots": state.diagnosis_slots,
                "diagnosis_recommended_departments": recommended_departments,
            }

        # 继续原有的风险检查
        risk_result = risk_check_node(state)
        state.diagnosis_risk_level = risk_result.get("diagnosis_risk_level", "none")
        state.diagnosis_risk_signals = risk_result.get("diagnosis_risk_signals", [])
        state.diagnosis_slots = risk_result.get(
            "diagnosis_slots", state.diagnosis_slots
        )
    else:
        risk_result = risk_check_node(state)
        state.diagnosis_risk_level = risk_result.get("diagnosis_risk_level", "none")
        state.diagnosis_risk_signals = risk_result.get("diagnosis_risk_signals", [])
        state.diagnosis_slots = risk_result.get(
            "diagnosis_slots", state.diagnosis_slots
        )

    completion_result = completion_node(state)
    state.diagnosis_completed = completion_result.get("diagnosis_completed", False)
    state.diagnosis_terminated = completion_result.get("diagnosis_terminated", False)
    state.diagnosis_termination_reason = completion_result.get(
        "diagnosis_termination_reason"
    )

    # 危急情况
    if state.diagnosis_risk_level == "critical":
        warning = get_emergency_warning(state.diagnosis_risk_signals)
        return {
            "messages": [AIMessage(content=warning)],
            "diagnosis_completed": True,
            "diagnosis_terminated": True,
            "diagnosis_type": "emergency",
            "diagnosis_slots": state.diagnosis_slots,
        }

    # 诊断完成：KG + RAG 综合评估
    if state.diagnosis_completed:
        symptoms = filled_slots.symptoms if filled_slots.symptoms else []

        # KG + RAG 综合推理
        fusion_result = None

        if symptoms:
            try:
                from app.graph.nodes.kg_rag_fusion import diagnose_with_kg_rag

                # 调用 KG + RAG 综合推理
                fusion_result = diagnose_with_kg_rag(
                    symptoms=symptoms,
                    user_query=user_input,
                    top_k=3,
                )

                logger.info(f"KG+RAG 综合推理: {fusion_result.get('departments', [])}")
                logger.info(f"数据来源: {fusion_result.get('sources', {})}")

            except Exception as e:
                logger.warning(f"KG+RAG 推理失败: {e}")
                # 降级到纯 KG
                try:
                    client = get_neo4j_client()
                    if client:
                        fusion_result = client.infer_department(symptoms, top_k=3)
                except:
                    pass

        # 提取结果
        if fusion_result and fusion_result.get("departments"):
            final_departments = fusion_result["departments"]
            confidence_result = fusion_result.get("confidence", {})
            overall_conf = (
                confidence_result.get("overall_confidence", 0)
                if confidence_result
                else 0
            )
            sources_info = fusion_result.get("sources", {})
        else:
            final_departments = []
            confidence_result = {"overall_confidence": 0}
            overall_conf = 0
            sources_info = {}
            fusion_result = None

        # 保存推理结果到状态
        state.department_inference = fusion_result
        state.confidence = confidence_result

        # 保存 RAG 检索结果到 state.medical_docs（供 answer_generate 使用）
        if fusion_result and fusion_result.get("rag_docs"):
            state.medical_docs = fusion_result["rag_docs"]
        elif fusion_result is None:
            # 确保即使 fusion_result 为空也初始化为空列表
            state.medical_docs = []

        # 置信度 >= 0.65 → 直接输出分诊建议
        if overall_conf >= CONFIDENT_THRESHOLD:
            lines = []
            lines.append("根据您描述的症状，我的分析结果是：")
            lines.append("")

            # 显示数据来源
            if sources_info.get("kg", 0) > 0 and sources_info.get("rag", 0) > 0:
                lines.append(f"📊 综合分析 (知识图谱 + 文档检索)")
            elif sources_info.get("kg", 0) > 0:
                lines.append(f"📊 知识图谱推理")
            elif sources_info.get("rag", 0) > 0:
                lines.append(f"📊 文档检索分析")
            lines.append("")

            if final_departments:
                for i, dept in enumerate(final_departments, 1):
                    prob_pct = dept.get("probability", 0) * 100
                    lines.append(f"{i}. {dept['name']} (置信度: {prob_pct:.0f}%)")
            else:
                lines.append("暂未匹配到明确科室，建议到医院导诊台咨询。")

            lines.append("")
            lines.append("✅ 置信度较高，建议您选择上述科室就诊。")

            completion_msg = "\n".join(lines)
            return {
                "messages": [AIMessage(content=completion_msg)],
                "diagnosis_completed": True,
                "diagnosis_type": "complete",
                "diagnosis_slots": state.diagnosis_slots,
                "need_more_info": False,
            }

        # 置信度 < 0.65 → 需要追问
        else:
            # 生成追问问题（使用 KG 结果）
            kg_result = {
                "departments": final_departments,
                "confidence": confidence_result,
            }
            questions = _generate_followup_questions(symptoms, kg_result, overall_conf)
            return {
                "messages": [AIMessage(content=questions)],
                "diagnosis_completed": False,
                "diagnosis_type": "in_progress",
                "diagnosis_slots": state.diagnosis_slots,
                "need_more_info": True,
                "department_inference": kg_result,
                "confidence": confidence_result,
            }

    # 诊断未完成：继续追问
    question_result = question_gen_node(
        state,
        associated_symptoms=associated_symptoms,
        recommended_departments=recommended_departments,
    )
    state.diagnosis_next_question = question_result.get("diagnosis_next_question", "")
    state.diagnosis_question_count = question_result.get("diagnosis_question_count", 1)
    state.diagnosis_missing_slots = question_result.get("diagnosis_missing_slots", [])

    return {
        "messages": [AIMessage(content=state.diagnosis_next_question)],
        "diagnosis_completed": False,
        "diagnosis_type": "in_progress",
        "diagnosis_slots": state.diagnosis_slots,
        "diagnosis_question_count": state.diagnosis_question_count,
        "diagnosis_missing_slots": state.diagnosis_missing_slots,
    }


def _generate_followup_questions(
    symptoms: list, kg_result: dict, overall_conf: float
) -> str:
    """生成追问问题 - 完全依赖 KG 数据"""
    lines = []

    # 从 KG 获取初步科室推荐
    if kg_result and kg_result.get("departments"):
        lines.append("根据您描述的症状，初步分析可能是：")
        for i, dept in enumerate(kg_result["departments"][:2], 1):
            prob_pct = dept.get("probability", 0) * 100
            lines.append(f"  {i}. {dept['name']} ({prob_pct:.0f}%)")

    lines.append("")
    lines.append(
        f"为了更准确判断（当前置信度 {overall_conf:.0%}），请问您还有以下症状吗？"
    )

    # 从 KG 查询每个症状的伴随症状，动态生成追问
    try:
        from app.infra.neo4j_client import get_neo4j_client

        client = get_neo4j_client()

        if client:
            # 收集所有可能的伴随症状（处理 dict 格式）
            all_associated = []

            for symptom in symptoms:
                # 查询该症状的伴随症状
                associated = client.query_associated_symptoms(symptom)
                # 提取症状名称
                for item in associated:
                    if isinstance(item, dict):
                        name = item.get("name")
                    else:
                        name = item
                    if name:
                        all_associated.append(name)

            # 去重
            all_associated = list(set(all_associated))

            # 排除已知的症状
            known_symptoms = set(symptoms)
            followup_symptoms = [s for s in all_associated if s not in known_symptoms]

            if followup_symptoms:
                # 随机选 2-3 个追问
                import random

                selected = random.sample(
                    followup_symptoms, min(3, len(followup_symptoms))
                )

                lines.append("")
                for s in selected:
                    lines.append(f"  ▸ 是否有 {s}?")
            else:
                lines.append("")
                lines.append("  ▸ 请补充更多症状信息")
        else:
            lines.append("")
            lines.append("  ▸ 请补充更多症状信息")

    except Exception as e:
        print(f"KG查询失败: {e}")
        lines.append("")
        lines.append("  ▸ 请补充更多症状信息")

    lines.append("")
    lines.append("您的回答可以帮助我更准确地为您推荐科室。")

    return "\n".join(lines)
