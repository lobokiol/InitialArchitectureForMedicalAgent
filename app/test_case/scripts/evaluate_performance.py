"""
性能评测模块
评估Agent的响应时间、吞吐量和稳定性
"""

import json
import sys
import time
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.chat_service import chat_once


SAMPLE_QUESTIONS = [
    "发烧了应该挂什么科？",
    "如何预约挂号？",
    "感冒咳嗽怎么办？",
    "住院需要带什么？",
    "体检流程是什么？",
    "头痛怎么办？",
    "医保怎么报销？",
    "肚子疼拉肚子",
    "你好呀",
    "谢谢",
]


def sync_chat_once_timed(
    user_id: str, thread_id: str | None, question: str
) -> Dict[str, Any]:
    """同步调用chat_once并记录时间"""
    start_time = time.time()
    try:
        response = chat_once(user_id, thread_id, question)
        elapsed = time.time() - start_time
        return {
            "success": True,
            "elapsed": elapsed,
            "question": question,
            "response_length": len(response.get("reply", "")),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed": elapsed,
            "question": question,
            "response_length": 0,
            "error": str(e),
        }


def evaluate_single_request_performance(num_requests: int = 20) -> Dict[str, Any]:
    """
    单请求性能评测

    Args:
        num_requests: 测试请求数量

    Returns:
        性能评测结果
    """
    print(f"\n{'=' * 60}")
    print("单请求性能评测")
    print(f"{'=' * 60}")
    print(f"测试请求数: {num_requests}")

    results = []

    for i in range(num_requests):
        question = SAMPLE_QUESTIONS[i % len(SAMPLE_QUESTIONS)]
        user_id = f"perf_test_user_{i}"

        result = sync_chat_once_timed(user_id, None, question)
        results.append(result)

        status = "✓" if result["success"] else "✗"
        print(
            f"{status} [{i + 1}/{num_requests}] {result['elapsed']:.2f}s | {question[:20]}..."
        )

    latencies = [r["elapsed"] for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    if not latencies:
        return {
            "error": "所有请求都失败了",
            "total_requests": num_requests,
            "success_count": 0,
            "error_count": num_requests,
        }

    latencies_sorted = sorted(latencies)
    n = len(latencies)

    p50 = latencies_sorted[int(n * 0.5)]
    p95 = latencies_sorted[int(n * 0.95)]
    p99 = latencies_sorted[int(n * 0.99)] if n >= 100 else latencies_sorted[-1]

    avg = statistics.mean(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0
    min_latency = min(latencies)
    max_latency = max(latencies)

    qps = (
        num_requests / (max(latencies) - min(latencies))
        if max(latencies) > min(latencies)
        else 0
    )

    error_rate = len(errors) / num_requests

    print(f"\n{'=' * 60}")
    print("单请求性能结果")
    print(f"{'=' * 60}")
    print(f"成功请求: {len(latencies)}/{num_requests}")
    print(f"失败请求: {len(errors)}/{num_requests}")
    print(f"错误率: {error_rate:.2%}")
    print(f"\n延迟统计:")
    print(f"  P50: {p50:.2f}s")
    print(f"  P95: {p95:.2f}s")
    print(f"  P99: {p99:.2f}s")
    print(f"  平均: {avg:.2f}s")
    print(f"  标准差: {std:.2f}s")
    print(f"  最小: {min_latency:.2f}s")
    print(f"  最大: {max_latency:.2f}s")
    print(f"  QPS: {qps:.2f}")

    if errors:
        print(f"\n错误详情:")
        for err in errors[:5]:
            print(f"  - {err['question'][:30]}: {err['error']}")

    return {
        "total_requests": num_requests,
        "success_count": len(latencies),
        "error_count": len(errors),
        "error_rate": error_rate,
        "latency": {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "avg": avg,
            "std": std,
            "min": min_latency,
            "max": max_latency,
        },
        "qps": qps,
        "details": results,
    }


def evaluate_concurrent_performance(
    num_users: int = 5, requests_per_user: int = 3
) -> Dict[str, Any]:
    """
    并发性能评测

    Args:
        num_users: 并发用户数
        requests_per_user: 每个用户的请求数

    Returns:
        并发评测结果
    """
    total_requests = num_users * requests_per_user

    print(f"\n{'=' * 60}")
    print("并发性能评测")
    print(f"{'=' * 60}")
    print(f"并发用户数: {num_users}")
    print(f"每用户请求数: {requests_per_user}")
    print(f"总请求数: {total_requests}")

    results = []
    lock = Lock()
    start_time = time.time()

    def user_requests(user_idx: int):
        user_results = []
        for req_idx in range(requests_per_user):
            question = SAMPLE_QUESTIONS[(user_idx + req_idx) % len(SAMPLE_QUESTIONS)]
            user_id = f"concurrent_user_{user_idx}"

            result = sync_chat_once_timed(user_id, None, question)
            user_results.append(result)

            status = "✓" if result["success"] else "✗"
            print(
                f"{status} 用户{user_idx} 请求{req_idx + 1}: {result['elapsed']:.2f}s"
            )

        return user_results

    with ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [executor.submit(user_requests, i) for i in range(num_users)]

        for future in as_completed(futures):
            try:
                user_results = future.result()
                with lock:
                    results.extend(user_results)
            except Exception as e:
                print(f"用户线程错误: {e}")

    total_time = time.time() - start_time

    latencies = [r["elapsed"] for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    if not latencies:
        return {
            "error": "所有请求都失败了",
            "total_requests": total_requests,
            "success_count": 0,
            "error_count": total_requests,
        }

    latencies_sorted = sorted(latencies)
    n = len(latencies)

    p50 = latencies_sorted[int(n * 0.5)]
    p95 = latencies_sorted[int(n * 0.95)]
    p99 = latencies_sorted[int(n * 0.99)] if n >= 100 else latencies_sorted[-1]

    avg = statistics.mean(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0
    min_latency = min(latencies)
    max_latency = max(latencies)

    actual_qps = total_requests / total_time

    error_rate = len(errors) / total_requests

    print(f"\n{'=' * 60}")
    print("并发性能结果")
    print(f"{'=' * 60}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"成功请求: {len(latencies)}/{total_requests}")
    print(f"失败请求: {len(errors)}/{total_requests}")
    print(f"错误率: {error_rate:.2%}")
    print(f"\n延迟统计:")
    print(f"  P50: {p50:.2f}s")
    print(f"  P95: {p95:.2f}s")
    print(f"  P99: {p99:.2f}s")
    print(f"  平均: {avg:.2f}s")
    print(f"  实际QPS: {actual_qps:.2f}")

    if errors:
        print(f"\n错误详情:")
        for err in errors[:5]:
            print(f"  - {err['question'][:30]}: {err['error']}")

    return {
        "config": {
            "num_users": num_users,
            "requests_per_user": requests_per_user,
            "total_requests": total_requests,
        },
        "total_time": total_time,
        "success_count": len(latencies),
        "error_count": len(errors),
        "error_rate": error_rate,
        "latency": {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "avg": avg,
            "std": std,
            "min": min_latency,
            "max": max_latency,
        },
        "actual_qps": actual_qps,
        "details": results,
    }


def evaluate_stability(duration_seconds: int = 60) -> Dict[str, Any]:
    """
    稳定性评测

    Args:
        duration_seconds: 测试持续时间(秒)

    Returns:
        稳定性评测结果
    """
    print(f"\n{'=' * 60}")
    print("稳定性评测")
    print(f"{'=' * 60}")
    print(f"测试时长: {duration_seconds}秒")

    results = []
    start_time = time.time()
    request_count = 0
    error_count = 0

    while time.time() - start_time < duration_seconds:
        question = SAMPLE_QUESTIONS[request_count % len(SAMPLE_QUESTIONS)]
        user_id = f"stability_user_{request_count}"

        result = sync_chat_once_timed(user_id, None, question)
        results.append(result)

        if not result["success"]:
            error_count += 1

        request_count += 1

        if request_count % 5 == 0:
            elapsed = time.time() - start_time
            current_qps = request_count / elapsed
            print(
                f"进度: {elapsed:.0f}s | 请求: {request_count} | 错误: {error_count} | QPS: {current_qps:.2f}"
            )

        time.sleep(0.5)

    latencies = [r["elapsed"] for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    total_time = time.time() - start_time
    actual_qps = request_count / total_time
    error_rate = error_count / request_count if request_count > 0 else 0

    print(f"\n{'=' * 60}")
    print("稳定性评测结果")
    print(f"{'=' * 60}")
    print(f"总请求数: {request_count}")
    print(f"错误数: {error_count}")
    print(f"错误率: {error_rate:.2%}")
    print(f"实际QPS: {actual_qps:.2f}")

    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        print(f"平均延迟: {statistics.mean(latencies):.2f}s")
        print(f"P99延迟: {latencies_sorted[int(n * 0.99)]:.2f}s")

    return {
        "duration": duration_seconds,
        "total_requests": request_count,
        "success_count": len(latencies),
        "error_count": error_count,
        "error_rate": error_rate,
        "actual_qps": actual_qps,
        "details": results[-20:],
    }


def save_results(results: Dict[str, Any], output_path: str):
    """保存评测结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


def main():
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("开始性能评测")
    print("=" * 60)

    single_results = evaluate_single_request_performance(num_requests=10)
    save_results(single_results, str(output_dir / "performance_single_result.json"))

    concurrent_results = evaluate_concurrent_performance(
        num_users=3, requests_per_user=2
    )
    save_results(
        concurrent_results, str(output_dir / "performance_concurrent_result.json")
    )

    all_results = {
        "single_request": single_results,
        "concurrent": concurrent_results,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'=' * 60}")
    print("性能评测完成")
    print(f"{'=' * 60}")

    return all_results


if __name__ == "__main__":
    main()
