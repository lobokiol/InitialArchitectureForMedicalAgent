"""
医院导诊槽位提取模块
基于 LLM + LoRA 实现的医疗领域槽位提取器
"""

from .extractor import SlotExtractor

__all__ = ["SlotExtractor"]
