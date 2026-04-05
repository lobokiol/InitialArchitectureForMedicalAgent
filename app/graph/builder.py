from langgraph.graph import StateGraph, START, END

from app.domain.models import AppState
from app.domain.routing import (
    route_after_decision,
    route_after_tool_calling,
    route_after_es,
    route_after_docs,
)
from app.graph.nodes.decision import decision_node
from app.graph.nodes.es_rag import es_rag_node
from app.graph.nodes.milvus_rag import milvus_rag_node
from app.graph.nodes.check_docs import check_docs_node
from app.graph.nodes.rewrite import rewrite_question
from app.graph.nodes.answer import answer_generate_node
from app.graph.nodes.trim_history import trim_history_node
from app.graph.nodes.tool_calling import tool_calling_node
from app.graph.nodes.diagnosis import diagnosis_node
from app.graph.nodes.diagnosis_router import diagnosis_router_node


def route_after_diagnosis_router(state: AppState) -> str:
    """诊断路由节点之后的路由

    根据 diagnosis 节点返回的结果决定下一步:
    - emergency: 危急情况
    - complete: 诊断完成 → 直接输出 (跳过RAG)
    - in_progress: 需要追问
    """
    # 检查危急
    if state.diagnosis_type == "emergency":
        return "emergency"

    # 检查是否需要追问
    need_more = getattr(state, "need_more_info", False)
    if need_more:
        return "in_progress"

    # 检查诊断是否完成
    if state.diagnosis_completed:
        # 如果 diagnosis_node 已经生成了完整的科室推荐消息，直接结束
        # 否则走 answer_generate 生成最终回复
        last_msg = state.messages[-1] if state.messages else None
        if last_msg and hasattr(last_msg, "content") and last_msg.content:
            # 检查最后一条消息是否包含科室推荐
            if any(
                kw in last_msg.content
                for kw in [
                    "我的分析结果是",
                    "建议您选择",
                    "知识图谱推理",
                    "文档检索分析",
                ]
            ):
                return "answer_generate"  # 走 answer_generate 来整合科室推荐

        return "answer_generate"

    return "answer_generate"


def build_graph() -> StateGraph:
    graph = StateGraph(AppState)

    graph.add_node("trim_history", trim_history_node)
    graph.add_node("decision", decision_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("diagnosis_router", diagnosis_router_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("es_rag", es_rag_node)
    graph.add_node("milvus_rag", milvus_rag_node)
    graph.add_node("check_docs", check_docs_node)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("answer_generate", answer_generate_node)
    graph.add_node("emergency", lambda state: state)
    graph.add_node("hil", lambda state: state)

    graph.add_edge(START, "trim_history")
    graph.add_edge("trim_history", "decision")

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "answer_generate": "answer_generate",
            "es_rag": "es_rag",
            "diagnosis": "diagnosis",
        },
    )

    graph.add_edge("diagnosis", "diagnosis_router")

    graph.add_conditional_edges(
        "diagnosis_router",
        route_after_diagnosis_router,
        {
            "emergency": "emergency",
            "milvus_rag": "milvus_rag",
            "in_progress": END,
            "answer_generate": "answer_generate",
        },
    )

    graph.add_edge("emergency", "hil")
    graph.add_edge("hil", END)

    graph.add_edge("es_rag", "diagnosis")

    graph.add_conditional_edges(
        "tool_calling",
        route_after_tool_calling,
        {
            "answer_generate": "answer_generate",
            "es_rag": "es_rag",
        },
    )

    graph.add_conditional_edges(
        "es_rag",
        route_after_es,
        {
            "milvus_rag": "milvus_rag",
            "check_docs": "check_docs",
        },
    )

    graph.add_edge("milvus_rag", "check_docs")

    graph.add_conditional_edges(
        "check_docs",
        route_after_docs,
        {
            "answer_generate": "answer_generate",
            "rewrite_question": "rewrite_question",
        },
    )

    graph.add_edge("rewrite_question", "es_rag")
    graph.add_edge("answer_generate", END)

    return graph


def build_app(checkpointer):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
