"""
医院导诊Agent评测主入口
整合所有评测模块，生成完整评测报告
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          医院导诊Agent - 完整评测套件                            ║
║          Hospital Guidance Agent - Complete Test Suite        ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_intent_evaluation() -> Dict[str, Any]:
    """运行意图识别评测"""
    print_section("1. 意图识别评测")
    from app.test_case.scripts.evaluate_intent import main as intent_main

    return intent_main()


def run_retrieval_evaluation() -> Dict[str, Any]:
    """运行RAG检索质量评测"""
    print_section("2. RAG检索质量评测")
    from app.test_case.scripts.evaluate_retrieval import main as retrieval_main

    return retrieval_main()


def run_e2e_evaluation() -> Dict[str, Any]:
    """运行端到端效果评测"""
    print_section("3. 端到端效果评测")
    from app.test_case.scripts.evaluate_e2e import main as e2e_main

    return e2e_main()


def run_performance_evaluation() -> Dict[str, Any]:
    """运行性能评测"""
    print_section("4. 性能评测")
    from app.test_case.scripts.evaluate_performance import main as perf_main

    return perf_main()


def generate_summary_report(results: Dict[str, Any]) -> str:
    """生成评测汇总报告"""

    report_lines = [
        "# 医院导诊Agent评测报告",
        "",
        f"**评测时间**: {results.get('timestamp', 'N/A')}",
        "",
        "---",
        "",
        "## 评测结果汇总",
        "",
    ]

    if "intent" in results:
        intent = results["intent"]
        report_lines.extend(
            [
                "### 1. 意图识别",
                f"- 准确率: {intent.get('accuracy', 0):.2%}",
                f"- 精确率: {intent.get('precision', 0):.2%}",
                f"- 召回率: {intent.get('recall', 0):.2%}",
                f"- F1分数: {intent.get('f1', 0):.2%}",
                f"- 测试样本: {intent.get('total', 0)}条",
                "",
            ]
        )

    if "retrieval" in results:
        retrieval = results["retrieval"]
        milvus = retrieval.get("milvus", {})
        es = retrieval.get("es", {})
        report_lines.extend(
            [
                "### 2. RAG检索质量",
                f"- 整体MRR: {retrieval.get('mrr', 0):.2%}",
                "",
                "**Milvus (症状检索):**",
                f"- Recall@5: {milvus.get(5, {}).get('recall', 0):.2%}",
                f"- Precision@5: {milvus.get(5, {}).get('precision', 0):.2%}",
                "",
                "**ES (流程检索):**",
                f"- Recall@5: {es.get(5, {}).get('recall', 0):.2%}",
                f"- Precision@5: {es.get(5, {}).get('precision', 0):.2%}",
                "",
            ]
        )

    if "e2e" in results:
        e2e = results["e2e"]
        report_lines.extend(
            [
                "### 3. 端到端效果",
                f"- 意图准确率: {e2e.get('intent_accuracy', 0):.2%}",
                f"- 答案正确率: {e2e.get('correctness', 0):.2%}",
                f"- 答案相关率: {e2e.get('relevance', 0):.2%}",
                f"- 答案完整率: {e2e.get('completeness', 0):.2%}",
                f"- 整体通过率: {e2e.get('pass_rate', 0):.2%}",
                f"- 测试样本: {e2e.get('total', 0)}条",
                "",
            ]
        )

    if "performance" in results:
        perf = results["performance"]
        single = perf.get("single_request", {})
        concurrent = perf.get("concurrent", {})
        report_lines.extend(
            [
                "### 4. 性能评测",
                "",
                "**单请求性能:**",
                f"- P50延迟: {single.get('latency', {}).get('p50', 0):.2f}s",
                f"- P95延迟: {single.get('latency', {}).get('p95', 0):.2f}s",
                f"- P99延迟: {single.get('latency', {}).get('p99', 0):.2f}s",
                f"- 错误率: {single.get('error_rate', 0):.2%}",
                "",
                "**并发性能:**",
                f"- 实际QPS: {concurrent.get('actual_qps', 0):.2f}",
                f"- 错误率: {concurrent.get('error_rate', 0):.2%}",
                "",
            ]
        )

    report_lines.extend(["---", "", "## 达标评估", ""])

    standards = []
    issues = []

    if "intent" in results:
        if results["intent"].get("accuracy", 0) >= 0.85:
            standards.append("✓ 意图识别准确率 ≥ 85%")
        else:
            issues.append("✗ 意图识别准确率 < 85%")

    if "retrieval" in results:
        if results["retrieval"].get("mrr", 0) >= 0.70:
            standards.append("✓ 检索MRR ≥ 70%")
        else:
            issues.append("✗ 检索MRR < 70%")

    if "e2e" in results:
        if results["e2e"].get("pass_rate", 0) >= 0.80:
            standards.append("✓ 端到端通过率 ≥ 80%")
        else:
            issues.append("✗ 端到端通过率 < 80%")

    if "performance" in results:
        perf = results["performance"]
        single = perf.get("single_request", {})
        if single.get("error_rate", 1) <= 0.01:
            standards.append("✓ 错误率 ≤ 1%")
        else:
            issues.append("✗ 错误率 > 1%")

    if standards:
        report_lines.append("**达标项:**")
        for s in standards:
            report_lines.append(f"- {s}")
        report_lines.append("")

    if issues:
        report_lines.append("**未达标项:**")
        for i in issues:
            report_lines.append(f"- {i}")
        report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "## 结论",
            "",
            f"**上线评估**: {'✅ 达到上线标准' if len(standards) >= 3 and len(issues) == 0 else '⚠️ 需要改进'}",
            "",
            "---",
            "",
            "*报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*",
        ]
    )

    return "\n".join(report_lines)


def save_full_report(results: Dict[str, Any], report_content: str):
    """保存完整评测报告"""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"full_evaluation_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    md_path = output_dir / f"evaluation_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n评测结果已保存:")
    print(f"  - JSON: {json_path}")
    print(f"  - 报告: {md_path}")

    return str(md_path)


def main(eval_types: List[str] = None):
    """
    主评测入口

    Args:
        eval_types: 指定要运行的评测类型列表
                   可选: ["intent", "retrieval", "e2e", "performance"]
                   默认为全部
    """
    print_banner()

    if eval_types is None:
        eval_types = ["intent", "retrieval", "e2e", "performance"]

    results = {"timestamp": datetime.now().isoformat(), "eval_types": eval_types}

    try:
        if "intent" in eval_types:
            results["intent"] = run_intent_evaluation()

        if "retrieval" in eval_types:
            results["retrieval"] = run_retrieval_evaluation()

        if "e2e" in eval_types:
            results["e2e"] = run_e2e_evaluation()

        if "performance" in eval_types:
            results["performance"] = run_performance_evaluation()

    except KeyboardInterrupt:
        print("\n\n评测被用户中断")
        return results

    print_section("生成评测报告")
    report = generate_summary_report(results)
    print(report)

    report_path = save_full_report(results, report)

    print(f"\n{'=' * 70}")
    print("  评测完成！")
    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    eval_types = None
    if len(sys.argv) > 1:
        eval_types = sys.argv[1].split(",")

    main(eval_types)
