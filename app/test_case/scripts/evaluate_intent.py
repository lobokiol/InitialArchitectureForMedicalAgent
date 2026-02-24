"""
意图识别评测模块
评估Agent对用户问题的意图分类能力
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.chat_service import chat_once


def load_test_data(data_path: str) -> Dict[str, List[Dict]]:
    """加载测试数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> Dict[str, float]:
    """手动计算分类指标"""
    metrics = {}

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0

    metrics["accuracy"] = accuracy

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_f1"] = f1

    total_precision = (
        sum(metrics.get(f"{l}_precision", 0) for l in labels) / len(labels)
        if labels
        else 0
    )
    total_recall = (
        sum(metrics.get(f"{l}_recall", 0) for l in labels) / len(labels)
        if labels
        else 0
    )
    metrics["precision"] = total_precision
    metrics["recall"] = total_recall
    metrics["f1"] = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )

    return metrics


def build_confusion_matrix(
    y_true: List[str], y_pred: List[str], labels: List[str]
) -> List[List[int]]:
    """构建混淆矩阵"""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cm = [[0] * len(labels) for _ in range(len(labels))]

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1

    return cm


def evaluate_intent_recognition(
    test_data: List[Dict], user_id: str = "test_user"
) -> Dict[str, Any]:
    """
    评测意图识别能力

    Args:
        test_data: 测试数据列表
        user_id: 测试用户ID

    Returns:
        评测结果字典
    """
    results = []
    correct = 0
    total = len(test_data)

    print(f"\n{'=' * 60}")
    print("意图识别评测开始")
    print(f"{'=' * 60}")
    print(f"测试样本数: {total}")

    for i, item in enumerate(test_data):
        question = item["question"]
        expected_intent = item["expected_intent"]

        try:
            response = chat_once(user_id=user_id, thread_id=None, message=question)

            intent_result = response.get("intent_result", {})
            predicted_intent = intent_result.get("main_intent", "unknown")

            is_correct = predicted_intent == expected_intent
            if is_correct:
                correct += 1

            results.append(
                {
                    "id": item.get("id", i + 1),
                    "question": question,
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "correct": is_correct,
                    "raw_intent": intent_result,
                }
            )

            status = "✓" if is_correct else "✗"
            print(
                f"{status} [{i + 1}/{total}] Q: {question[:30]}... | 预期: {expected_intent} | 实际: {predicted_intent}"
            )

        except Exception as e:
            print(f"✗ [{i + 1}/{total}] Q: {question[:30]}... | 错误: {str(e)}")
            results.append(
                {
                    "id": item.get("id", i + 1),
                    "question": question,
                    "expected": expected_intent,
                    "predicted": "ERROR",
                    "correct": False,
                    "error": str(e),
                }
            )

    accuracy = correct / total

    y_true = [r["expected"] for r in results]
    y_pred = [r["predicted"] for r in results]

    labels = list(set(y_true + y_pred))
    labels = sorted([l for l in labels if l != "ERROR" and l != "unknown"])

    metrics = calculate_metrics(y_true, y_pred, labels)
    cm = build_confusion_matrix(y_true, y_pred, labels)

    report_lines = ["分类报告:", f"准确率: {metrics['accuracy']:.2%}"]
    for label in labels:
        p = metrics.get(f"{label}_precision", 0)
        r = metrics.get(f"{label}_recall", 0)
        f = metrics.get(f"{label}_f1", 0)
        report_lines.append(f"  {label}: Precision={p:.2%}, Recall={r:.2%}, F1={f:.2%}")

    print(f"\n{'=' * 60}")
    print("评测结果")
    print(f"{'=' * 60}")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.2%}")
    print(f"精确率 (Precision): {metrics['precision']:.2%}")
    print(f"召回率 (Recall): {metrics['recall']:.2%}")
    print(f"F1分数: {metrics['f1']:.2%}")
    print(f"\n" + "\n".join(report_lines))
    print(f"\n混淆矩阵:")
    print(f"标签: {labels}")
    for i, row in enumerate(cm):
        print(f"  {labels[i]}: {row}")

    intent_stats = Counter([r["expected"] for r in results])
    print(f"\n各类别样本数: {dict(intent_stats)}")

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "total": total,
        "correct": correct,
        "report": "\n".join(report_lines),
        "confusion_matrix": cm,
        "labels": labels,
        "details": results,
        "intent_distribution": dict(intent_stats),
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
    output_path = output_dir / "intent_evaluation_result.json"

    test_data = load_test_data(str(data_path))
    intent_test_data = test_data.get("intent_test_data", [])

    results = evaluate_intent_recognition(intent_test_data[:10])
    save_results(results, str(output_path))

    print(f"\n{'=' * 60}")
    print("意图识别评测完成")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
