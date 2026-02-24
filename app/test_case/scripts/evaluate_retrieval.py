"""
RAG检索质量评测模块
评估ES流程检索和Milvus向量检索的质量
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.chat_service import chat_once


def load_test_data(data_path: str) -> Dict[str, List[Dict]]:
    """加载测试数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_recall_at_k(
    retrieved_docs: List[Dict], expected_keywords: List[str], k: int = 5
) -> float:
    """
    计算Recall@K

    Args:
        retrieved_docs: 检索返回的文档列表
        expected_keywords: 期望包含的关键词
        k: 取前k个结果

    Returns:
        Recall@K分数
    """
    if not expected_keywords or not retrieved_docs:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    retrieved_text = " ".join(
        [doc.get("content", "") + " " + doc.get("title", "") for doc in retrieved_k]
    ).lower()

    hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
    return hits / len(expected_keywords)


def calculate_precision_at_k(
    retrieved_docs: List[Dict], expected_keywords: List[str], k: int = 5
) -> float:
    """
    计算Precision@K

    Args:
        retrieved_docs: 检索返回的文档列表
        expected_keywords: 期望包含的关键词
        k: 取前k个结果

    Returns:
        Precision@K分数
    """
    if not retrieved_docs or not expected_keywords:
        return 0.0

    retrieved_k = retrieved_docs[:k]
    retrieved_text = " ".join(
        [doc.get("content", "") + " " + doc.get("title", "") for doc in retrieved_k]
    ).lower()

    hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
    return hits / k if k > 0 else 0.0


def calculate_mrr(retrieved_docs: List[Dict], expected_keywords: List[str]) -> float:
    """
    计算MRR (Mean Reciprocal Rank)

    第一个相关结果排名的倒数
    """
    if not expected_keywords or not retrieved_docs:
        return 0.0

    for i, doc in enumerate(retrieved_docs):
        doc_text = (doc.get("content", "") + " " + doc.get("title", "")).lower()
        if any(kw.lower() in doc_text for kw in expected_keywords):
            return 1.0 / (i + 1)

    return 0.0


def evaluate_retrieval(
    test_data: List[Dict], user_id: str = "test_user_retrieval"
) -> Dict[str, Any]:
    """
    评测RAG检索质量

    Args:
        test_data: 测试数据列表
        user_id: 测试用户ID

    Returns:
        评测结果字典
    """
    results = []
    k_values = [3, 5, 8, 10]

    milvus_recalls = {k: [] for k in k_values}
    milvus_precisions = {k: [] for k in k_values}
    es_recalls = {k: [] for k in k_values}
    es_precisions = {k: [] for k in k_values}

    mrr_scores = []

    print(f"\n{'=' * 60}")
    print("RAG检索质量评测开始")
    print(f"{'=' * 60}")
    print(f"测试样本数: {len(test_data)}")

    for i, item in enumerate(test_data):
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        doc_type = item.get("type", "symptom")

        try:
            response = chat_once(user_id=user_id, thread_id=None, message=question)

            used_docs = response.get("used_docs", {})

            if doc_type == "symptom":
                retrieved_docs = used_docs.get("medical", [])
            else:
                retrieved_docs = used_docs.get("process", [])

            if not retrieved_docs:
                retrieved_docs = used_docs.get("medical", []) + used_docs.get(
                    "process", []
                )

            mrr = calculate_mrr(retrieved_docs, expected_keywords)
            mrr_scores.append(mrr)

            for k in k_values:
                recall = calculate_recall_at_k(retrieved_docs, expected_keywords, k)
                precision = calculate_precision_at_k(
                    retrieved_docs, expected_keywords, k
                )

                if doc_type == "symptom":
                    milvus_recalls[k].append(recall)
                    milvus_precisions[k].append(precision)
                else:
                    es_recalls[k].append(recall)
                    es_precisions[k].append(precision)

            results.append(
                {
                    "id": item.get("id", i + 1),
                    "question": question,
                    "type": doc_type,
                    "expected_keywords": expected_keywords,
                    "retrieved_count": len(retrieved_docs),
                    "mrr": mrr,
                    "retrieved_docs": [
                        {
                            "title": doc.get("title", ""),
                            "content": doc.get("content", "")[:100],
                            "score": doc.get("score", 0),
                        }
                        for doc in retrieved_docs[:3]
                    ],
                }
            )

            status = "✓" if mrr > 0 else "✗"
            print(
                f"{status} [{i + 1}/{len(test_data)}] {doc_type}: {question[:25]}... | 检索到:{len(retrieved_docs)}条 | MRR:{mrr:.2f}"
            )

        except Exception as e:
            print(f"✗ [{i + 1}/{len(test_data)}] {question[:25]}... | 错误: {str(e)}")
            results.append(
                {
                    "id": item.get("id", i + 1),
                    "question": question,
                    "type": doc_type,
                    "error": str(e),
                }
            )
            mrr_scores.append(0)
            for k in k_values:
                if doc_type == "symptom":
                    milvus_recalls[k].append(0)
                    milvus_precisions[k].append(0)
                else:
                    es_recalls[k].append(0)
                    es_precisions[k].append(0)

    print(f"\n{'=' * 60}")
    print("评测结果汇总")
    print(f"{'=' * 60}")

    mrr_avg = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    print(f"\n整体MRR: {mrr_avg:.2%}")

    print(f"\n--- Milvus (症状检索) ---")
    milvus_summary = {}
    for k in k_values:
        recall_avg = (
            sum(milvus_recalls[k]) / len(milvus_recalls[k]) if milvus_recalls[k] else 0
        )
        precision_avg = (
            sum(milvus_precisions[k]) / len(milvus_precisions[k])
            if milvus_precisions[k]
            else 0
        )
        milvus_summary[k] = {"recall": recall_avg, "precision": precision_avg}
        print(f"Recall@{k}: {recall_avg:.2%} | Precision@{k}: {precision_avg:.2%}")

    print(f"\n--- ES (流程检索) ---")
    es_summary = {}
    for k in k_values:
        recall_avg = sum(es_recalls[k]) / len(es_recalls[k]) if es_recalls[k] else 0
        precision_avg = (
            sum(es_precisions[k]) / len(es_precisions[k]) if es_precisions[k] else 0
        )
        es_summary[k] = {"recall": recall_avg, "precision": precision_avg}
        print(f"Recall@{k}: {recall_avg:.2%} | Precision@{k}: {precision_avg:.2%}")

    symptom_count = sum(1 for r in results if r.get("type") == "symptom")
    process_count = sum(1 for r in results if r.get("type") == "process")
    print(f"\n测试样本分布: 症状类={symptom_count}, 流程类={process_count}")

    return {
        "mrr": mrr_avg,
        "k_values": k_values,
        "milvus": milvus_summary,
        "es": es_summary,
        "total": len(test_data),
        "symptom_count": symptom_count,
        "process_count": process_count,
        "details": results,
    }


def save_results(results: Dict[str, Any], output_path: str):
    """保存评测结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_path = data_dir / "test_data.json"
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "retrieval_evaluation_result.json"

    test_data = load_test_data(str(data_path))
    retrieval_test_data = test_data.get("retrieval_test_data", [])

    results = evaluate_retrieval(retrieval_test_data)
    save_results(results, str(output_path))

    print(f"\n{'=' * 60}")
    print("RAG检索质量评测完成")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
