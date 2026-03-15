from typing import List, Dict, Any

from app.core import config
from app.core.logging import logger
from app.core.llm import get_embedding_model
from app.domain.models import AppState, RetrievedDoc
from app.infra.milvus_client import search_medical_docs
from app.infra.es_client import search_rag_es


def rrf_fusion(
    es_docs: List[RetrievedDoc], milvus_docs: List[RetrievedDoc], k: int = 60
) -> List[RetrievedDoc]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    将 ES 和 Milvus 的检索结果进行融合排序
    """
    doc_scores: Dict[str, float] = {}

    for rank, doc in enumerate(es_docs, 1):
        score = 1.0 / (k + rank)
        if doc.id in doc_scores:
            doc_scores[doc.id] += score
        else:
            doc_scores[doc.id] = score

    for rank, doc in enumerate(milvus_docs, 1):
        score = 1.0 / (k + rank)
        if doc.id in doc_scores:
            doc_scores[doc.id] += score
        else:
            doc_scores[doc.id] = score

    all_docs = {doc.id: doc for doc in es_docs + milvus_docs}
    fused = [(doc_id, rrf_score) for doc_id, rrf_score in doc_scores.items()]
    fused.sort(key=lambda x: -x[1])

    result = []
    for doc_id, rrf_score in fused:
        doc = all_docs[doc_id]
        doc.score = rrf_score
        result.append(doc)

    return result


def rerank_with_qwen(
    query: str, candidates: List[RetrievedDoc], top_n: int = 10
) -> List[RetrievedDoc]:
    """
    使用 Qwen3-Rerank API 对候选文档进行精排
    """
    if not candidates:
        return []

    import requests

    docs_text = [doc.content[:4000] for doc in candidates]

    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            headers={
                "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3-rerank",  # BGE—reranker v2
                "query": query,
                "documents": docs_text,
                "top_n": min(top_n, len(candidates)),
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        reranked = []
        if "results" in result:
            for item in result["results"]:
                idx = item["index"]
                reranked.append(candidates[idx])

        logger.info(f"Rerank 成功，返回 {len(reranked)} 条")
        return reranked

    except Exception as e:
        logger.warning(f"Rerank 失败: {e}，返回原始排序")
        return candidates[:top_n]


def milvus_rag_node(state: AppState) -> dict:
    """
    Milvus RAG 节点：
    1. 双路检索：ES (rag_es) + Milvus (medical_knowledge)
    2. RRF 融合
    3. Qwen3-Rerank 精排
    """
    logger.info(">>> Enter node: milvus_rag")
    ir = state.intent_result
    logger.info("milvus_rag_node intent_result=%s", ir)

    if not (ir and ir.has_symptom and ir.need_symptom_search and ir.symptom_query):
        logger.info("milvus_rag_node: no symptom search needed, skip Milvus")
        return {}

    query = ir.symptom_query.strip()
    if not query:
        logger.info("milvus_rag_node: symptom_query is empty, skip Milvus")
        return {}

    logger.info("milvus_rag_node: query=%s", query)

    retrieval_k = config.RETRIEVAL_K

    try:
        logger.info("milvus_rag_node: 双路检索开始 (k=%d)", retrieval_k)
        es_docs = search_rag_es(query, size=retrieval_k)
        logger.info(f"milvus_rag_node: ES 检索返回 {len(es_docs)} 条")
    except Exception:
        logger.exception("ES rag_es 查询失败")
        es_docs = []

    try:
        milvus_docs = search_medical_docs(query)
        logger.info(f"milvus_rag_node: Milvus 检索返回 {len(milvus_docs)} 条")
    except Exception:
        logger.exception("Milvus 查询失败")
        milvus_docs = []

    if not es_docs and not milvus_docs:
        logger.info("milvus_rag_node: 双路均无结果")
        return {}

    logger.info("milvus_rag_node: 执行 RRF 融合 (k=%d)", config.RRF_K)
    fused_docs = rrf_fusion(es_docs, milvus_docs, k=config.RRF_K)
    logger.info(f"milvus_rag_node: RRF 融合后 {len(fused_docs)} 条")

    logger.info("milvus_rag_node: 执行 Rerank 精排 (top_n=%d)", config.RERANK_TOP_N)
    reranked_docs = rerank_with_qwen(query, fused_docs, top_n=config.RERANK_TOP_N)
    logger.info(f"milvus_rag_node: Rerank 后 {len(reranked_docs)} 条")

    for i, doc in enumerate(reranked_docs[:5], 1):
        logger.info(
            "milvus_rag_node final doc[%d]: id=%s, score=%.4f",
            i,
            doc.id,
            doc.score or 0,
        )

    return {"medical_docs": reranked_docs}
