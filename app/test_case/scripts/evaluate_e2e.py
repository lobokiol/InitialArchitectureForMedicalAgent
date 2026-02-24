"""
端到端效果评测模块
评估Agent生成答案的正确性、相关性和完整性
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


def evaluate_answer_correctness(answer: str, criteria: List[str]) -> float:
    """
    评估答案正确性
    答案中包含关键信息得1分，否则0分
    """
    if not answer or not criteria:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for criterion in criteria if criterion.lower() in answer_lower)
    return hits / len(criteria)


def evaluate_answer_relevance(question: str, answer: str) -> float:
    """
    评估答案相关性
    简单评估：答案是否回应了问题
    """
    if not answer:
        return 0.0

    if not question:
        return 1.0

    question_keywords = set(question.lower().split())
    answer_words = set(answer.lower().split())

    common_words = question_keywords & answer_words

    if len(common_words) >= 2:
        return 1.0
    elif len(common_words) >= 1:
        return 0.5
    return 0.0


def evaluate_answer_completeness(answer: str, answer_type: str) -> float:
    """
    评估答案完整性
    根据不同答案类型评估完整性
    """
    if not answer:
        return 0.0

    answer_len = len(answer)

    if answer_type == "科室推荐":
        has_dept = any(kw in answer for kw in ["科", "门诊", "科室"])
        has_suggestion = any(kw in answer for kw in ["建议", "可以", "应该", "推荐"])
        if has_dept and has_suggestion:
            return 1.0
        elif has_dept:
            return 0.7
        return 0.3

    elif answer_type == "流程说明":
        has_steps = any(
            kw in answer for kw in ["步", "流程", "先", "后", "需要", "办理"]
        )
        has_location = any(kw in answer for kw in ["地", "位置", "在哪", "窗口"])
        if has_steps and has_location:
            return 1.0
        elif has_steps:
            return 0.7
        return 0.3

    elif answer_type == "建议":
        if answer_len > 20:
            return 1.0
        elif answer_len > 10:
            return 0.7
        return 0.3

    elif answer_type == "友好回应":
        greetings = ["你好", "您好", "欢迎", "有什么", "帮助"]
        if any(g in answer for g in greetings):
            return 1.0
        return 0.5

    elif answer_type == "礼貌回应":
        responses = ["不客气", "谢谢", "应该的", "随时"]
        if any(r in answer for r in responses):
            return 1.0
        return 0.5

    else:
        if answer_len > 30:
            return 1.0
        elif answer_len > 10:
            return 0.7
        return 0.3


def manual_evaluate_single(
    question: str,
    answer: str,
    expected_intent: str,
    criteria: List[str],
    answer_type: str = "",
) -> Dict[str, Any]:
    """
    人工评估单个问题

    返回评估结果和建议
    """
    correctness = evaluate_answer_correctness(answer, criteria)
    relevance = evaluate_answer_relevance(question, answer)
    completeness = evaluate_answer_completeness(answer, answer_type)

    overall = (correctness + relevance + completeness) / 3

    suggestions = []
    if correctness < 0.5:
        suggestions.append("答案缺少关键信息")
    if relevance < 0.5:
        suggestions.append("答案与问题不相关")
    if completeness < 0.5:
        suggestions.append("答案不够完整")

    return {
        "correctness": correctness,
        "relevance": relevance,
        "completeness": completeness,
        "overall": overall,
        "pass": overall >= 0.6,
        "suggestions": suggestions,
    }


def evaluate_e2e(
    test_data: List[Dict], user_id: str = "test_user_e2e"
) -> Dict[str, Any]:
    """
    端到端效果评测

    Args:
        test_data: 测试数据列表
        user_id: 测试用户ID

    Returns:
        评测结果字典
    """
    results = []

    correctness_scores = []
    relevance_scores = []
    completeness_scores = []
    overall_scores = []

    intent_correct = 0

    print(f"\n{'=' * 60}")
    print("端到端效果评测开始")
    print(f"{'=' * 60}")
    print(f"测试样本数: {len(test_data)}")
    print("\n注意: 此评测需要人工审核，请对每个答案进行评分")
    print("评分标准: 1-5分制，3分为及格\n")

    for i, item in enumerate(test_data):
        question = item["question"]
        expected_intent = item.get("expected_intent", "unknown")
        criteria = item.get("evaluation_criteria", [])

        try:
            response = chat_once(user_id=user_id, thread_id=None, message=question)

            answer = response.get("reply", "")
            intent_result = response.get("intent_result", {})
            predicted_intent = intent_result.get("intent_type", "unknown")

            answer_type = item.get("expected_answer_type", "")

            auto_eval = manual_evaluate_single(
                question, answer, expected_intent, criteria, answer_type
            )

            correctness_scores.append(auto_eval["correctness"])
            relevance_scores.append(auto_eval["relevance"])
            completeness_scores.append(auto_eval["completeness"])
            overall_scores.append(auto_eval["overall"])

            if predicted_intent == expected_intent:
                intent_correct += 1

            results.append(
                {
                    "id": item.get("id", i + 1),
                    "question": question,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "answer": answer,
                    "auto_correctness": auto_eval["correctness"],
                    "auto_relevance": auto_eval["relevance"],
                    "auto_completeness": auto_eval["completeness"],
                    "auto_overall": auto_eval["overall"],
                    "pass": auto_eval["pass"],
                    "suggestions": auto_eval["suggestions"],
                }
            )

            status = "✓" if auto_eval["pass"] else "✗"
            print(f"{status} [{i + 1}/{len(test_data)}] {question[:25]}...")
            print(f"   答案: {answer[:60]}...")
            print(
                f"   自评: 正确性={auto_eval['correctness']:.1f} 相关={auto_eval['relevance']:.1f} 完整={auto_eval['completeness']:.1f}"
            )
            if auto_eval["suggestions"]:
                print(f"   建议: {', '.join(auto_eval['suggestions'])}")
            print()

        except Exception as e:
            print(f"✗ [{i + 1}/{len(test_data)}] {question[:25]}... | 错误: {str(e)}")
            results.append(
                {"id": item.get("id", i + 1), "question": question, "error": str(e)}
            )
            correctness_scores.append(0)
            relevance_scores.append(0)
            completeness_scores.append(0)
            overall_scores.append(0)

    avg_correctness = (
        sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0
    )
    avg_relevance = (
        sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    )
    avg_completeness = (
        sum(completeness_scores) / len(completeness_scores)
        if completeness_scores
        else 0
    )
    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    pass_count = sum(1 for o in overall_scores if o >= 0.6)
    pass_rate = pass_count / len(overall_scores) if overall_scores else 0

    intent_accuracy = intent_correct / len(test_data) if test_data else 0

    print(f"\n{'=' * 60}")
    print("评测结果汇总")
    print(f"{'=' * 60}")
    print(f"意图准确率: {intent_accuracy:.2%}")
    print(f"答案正确率: {avg_correctness:.2%}")
    print(f"答案相关率: {avg_relevance:.2%}")
    print(f"答案完整率: {avg_completeness:.2%}")
    print(f"整体通过率: {pass_rate:.2%} ({pass_count}/{len(test_data)})")
    print(f"整体评分: {avg_overall:.2f}/1.0")

    print(f"\n--- 分类型统计 ---")
    intent_types = Counter(
        [r.get("expected_intent", "unknown") for r in results if "error" not in r]
    )
    for intent_type, count in intent_types.items():
        type_scores = [
            r["auto_overall"]
            for r in results
            if r.get("expected_intent") == intent_type and "error" not in r
        ]
        if type_scores:
            type_avg = sum(type_scores) / len(type_scores)
            print(f"{intent_type}: {type_avg:.2%} ({len(type_scores)}条)")

    return {
        "intent_accuracy": intent_accuracy,
        "correctness": avg_correctness,
        "relevance": avg_relevance,
        "completeness": avg_completeness,
        "overall": avg_overall,
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "total": len(test_data),
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
    output_path = output_dir / "e2e_evaluation_result.json"

    test_data = load_test_data(str(data_path))
    e2e_test_data = test_data.get("e2e_test_data", [])

    results = evaluate_e2e(e2e_test_data)
    save_results(results, str(output_path))

    print(f"\n{'=' * 60}")
    print("端到端效果评测完成")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
