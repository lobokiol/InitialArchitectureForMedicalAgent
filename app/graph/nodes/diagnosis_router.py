"""
诊断路由器节点

根据诊断状态决定下一步流向：
- emergency: 危急情况 → 紧急处理
- complete: 诊断完成 → 检索或生成答案
- in_progress: 诊断进行中 → 结束当前轮次，等待用户回复
"""

from app.core.logging import logger
from app.domain.models import AppState


def diagnosis_router_node(state: AppState) -> dict:
    """
    诊断路由器

    根据诊断状态返回下一步指令:
    - "emergency": 危急情况
    - "complete": 诊断完成
    - "in_progress": 诊断进行中
    """
    logger.info(">>> Enter node: diagnosis_router")

    # 检查是否是危急情况
    if state.diagnosis_risk_level == "critical" or state.diagnosis_type == "emergency":
        logger.info("diagnosis_router -> emergency (critical risk)")
        return {"diagnosis_next_step": "emergency"}

    # 检查诊断是否完成
    if state.diagnosis_completed or state.diagnosis_terminated:
        logger.info("diagnosis_router -> complete")
        return {"diagnosis_next_step": "complete"}

    # 诊断进行中
    logger.info("diagnosis_router -> in_progress")
    return {"diagnosis_next_step": "in_progress"}
