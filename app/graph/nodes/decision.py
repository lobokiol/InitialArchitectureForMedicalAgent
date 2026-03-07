from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.core.llm import get_lightweight_llm
from app.domain.models import AppState, IntentResult


DECISION_PROMPT = """
    你是一个医疗问答系统的"意图识别"模块。

    你必须只返回一个 json 对象（合法的 JSON），不要输出任何解释或多余文字。

    字段要求与 IntentResult 一致：
    - has_symptom: bool - 用户是否在描述身体症状/不适（如"头疼"、"肚子大"）
    - has_process: bool - 用户是否在询问如何做某事（如"怎么办"、"怎么治疗"、"如何减小"）
    - main_intent: "symptom" | "process" | "mixed" | "non_medical"
    - symptom_query: string 或 null - 从问题中提取的症状描述
    - process_query: string 或 null - 从问题中提取的操作请求（如"怎么减小"、"如何治疗""）
    - need_symptom_search: bool
    - need_process_search: bool
    - need_tool_call: bool - 用户是否在询问特定患者的病例信息（如"张三的病例"、"患者 001 的情况"）

    判断规则：
    - 如果用户问"怎么 XX"、"如何 XX"、"怎么办"，即使同时描述了症状，也应设置 has_process=True
    - 如果用户询问具体患者的名字或 ID 的病例（如"张三的病例"、"患者 001"），设置 need_tool_call=true
    - symptom_query 应提取症状本身（如"肚子大"）
    - process_query 应提取操作意图（如"怎么减小"）

    用户问题：{query}
"""

DIAGNOSIS_DECISION_PROMPT = """
    你是一个医疗问答系统的"意图识别"模块。

    你已经完成了多轮问诊，收集到以下诊断信息：
    {diagnosis_summary}

    请根据这些诊断信息判断下一步需要什么类型的帮助：
    - 是否需要查询就医流程（has_process）
    - 是否需要进一步症状检索（has_symptom）
    - 是否混合意图（既有症状又有流程询问）

    你必须只返回一个 json 对象（合法的 JSON），不要输出任何解释或多余文字。

    字段要求与 IntentResult 一致：
    - has_symptom: bool - 是否存在需要进一步检索的症状
    - has_process: bool - 是否需要查询就医流程
    - main_intent: "symptom" | "process" | "mixed" | "non_medical"
    - symptom_query: string 或 null - 主要症状描述
    - process_query: string 或 null - 流程相关的问题
    - need_symptom_search: bool
    - need_process_search: bool
    - need_tool_call: bool - 通常设为 false，因为已经完成了问诊
"""


def decision_node(state: AppState) -> dict:
    logger.info(">>> Enter node: decision")

    # 检查是否有诊断总结，如果有则使用专门的诊断后意图识别
    if hasattr(state, "diagnosis_summary") and state.diagnosis_summary:
        logger.info("decision_node: 检测到诊断总结，使用诊断后意图识别")
        return _decision_after_diagnosis(state)

    # 正常的用户输入意图识别
    user_query = state.messages[-1].content
    logger.info("decision_node user_query=%s", user_query)

    try:
        structured_llm = get_lightweight_llm().with_structured_output(IntentResult)
        intent = structured_llm.invoke(
            [HumanMessage(content=DECISION_PROMPT.format(query=user_query))]
        )
    except Exception:
        logger.exception("decision_node LLM 调用失败，使用兜底 intent_result")
        intent = IntentResult(
            has_symptom=False,
            has_process=False,
            main_intent="non_medical",
            symptom_query=None,
            process_query=None,
            need_symptom_search=False,
            need_process_search=False,
        )
        return {"intent_result": intent}

    intent.need_symptom_search = intent.has_symptom
    intent.need_process_search = intent.has_process

    # 自动判断是否需要查询患者病例（如果问题中提到具体患者姓名或 ID）
    user_query_lower = user_query.lower()
    if any(
        keyword in user_query_lower
        for keyword in ["病例", "病历", "患者", "的病史", "记录"]
    ):
        # 检查是否提到了具体患者（名字或 ID）
        import re

        if re.search(
            r"[\u4e00-\u9fa5]{2,4}(的病例 | 病历 | 病史 | 记录)|患者\s*\d{3,}|\d{3,}(患者 | 病例)",
            user_query,
        ):
            intent.need_tool_call = True

    logger.info("decision_node intent_result=%s", intent)
    return {"intent_result": intent}


def _decision_after_diagnosis(state: AppState) -> dict:
    """
    在多轮问诊完成后，基于诊断信息进行意图识别

    主要判断：
    1. 是否需要查询就医流程（如挂号、检查等）
    2. 是否需要进一步的症状检索
    3. 是否是混合意图
    """
    diagnosis_summary = state.diagnosis_summary

    try:
        structured_llm = get_lightweight_llm().with_structured_output(IntentResult)
        intent = structured_llm.invoke(
            [
                HumanMessage(
                    content=DIAGNOSIS_DECISION_PROMPT.format(
                        diagnosis_summary=diagnosis_summary
                    )
                )
            ]
        )

        # 诊断完成后，默认需要查询流程和症状
        if intent.main_intent == "mixed" or (intent.has_symptom and intent.has_process):
            intent.need_symptom_search = True
            intent.need_process_search = True
        elif intent.has_process:
            intent.need_process_search = True
        elif intent.has_symptom:
            intent.need_symptom_search = True

        # 诊断完成后通常不需要 tool_call
        intent.need_tool_call = False

    except Exception:
        logger.exception("decision_after_diagnosis LLM 调用失败，使用兜底策略")
        # 兜底策略：假设需要查询流程和症状
        intent = IntentResult(
            has_symptom=True,
            has_process=True,
            main_intent="mixed",
            symptom_query="已收集的症状",
            process_query="就诊流程",
            need_symptom_search=True,
            need_process_search=True,
            need_tool_call=False,
        )

    logger.info("decision_after_diagnosis intent_result=%s", intent)
    return {"intent_result": intent}
