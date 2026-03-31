"""
基于 LLM 的症状提取器

使用 Qwen Turbo 模型进行结构化症状提取
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
import json


class ExtractedSymptoms(BaseModel):
    """LLM 提取的症状结构"""

    symptoms: List[str] = Field(
        default_factory=list,
        description="用户原始描述中的症状（保留原词，用于审计追溯）",
    )
    negative_symptoms: List[str] = Field(
        default_factory=list,
        description="排除的症状（用户明确说'没有'、'不'、'无'、'否认'等否定词修饰的症状，保留原词）",
    )
    location: str = Field(default="", description="疼痛或不适的部位")
    duration: str = Field(default="", description="症状持续时间")
    severity: str = Field(default="", description="严重程度")
    triggers: List[str] = Field(default_factory=list, description="可能的诱因")
    accompanying_symptoms: List[str] = Field(
        default_factory=list, description="伴随症状"
    )
    medical_history: List[str] = Field(default_factory=list, description="相关病史")
    urgency_level: str = Field(
        default="normal", description="紧急程度: normal/high/critical"
    )
    is_emergency: bool = Field(default=False, description="是否需要急诊")


class LLMSymptomExtractor:
    """基于 LLM 的症状提取器"""

    def __init__(self, model_name: str = "qwen-plus", temperature: float = 0.1):
        """
        初始化 LLM 提取器

        Args:
            model_name: 模型名称，默认为 qwen-turbo
            temperature: 温度参数
        """
        # 使用 DashScope API
        base_url = os.getenv(
            "DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        api_key = os.getenv("DASHSCOPE_API_KEY", "")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

        self.structured_llm = self.llm.with_structured_output(ExtractedSymptoms)

    def extract(self, user_input: str) -> ExtractedSymptoms:
        """
        从用户输入中提取症状信息

        Args:
            user_input: 用户的描述

        Returns:
            ExtractedSymptoms: 提取的结构化症状信息
        """
        prompt = self._build_prompt(user_input)

        try:
            result = self.structured_llm.invoke(prompt)
            return result
        except Exception as e:
            print(f"LLM 提取失败: {e}")
            # 返回空结果
            return ExtractedSymptoms()

    def _build_prompt(self, user_input: str) -> str:
        """构建提取提示词"""
        return f"""你是一个专业的医疗导诊预检助手。你的任务是从用户的描述中提取医学信息。

请从以下用户描述中提取症状信息：

用户描述：{user_input}

请严格按照以下JSON格式输出，注意所有字段都是必填的：
{{
    "symptoms": ["用户原始描述的症状（保留原词，如'头壳痛'、'肚子疼'）"],
    "negative_symptoms": ["排除的症状，用户明确说'没有'、'不'、'无'、'否认'等否定词修饰的症状（保留原词）"],
    "location": "疼痛部位",
    "duration": "持续时间",
    "severity": "严重程度",
    "triggers": ["诱因1"],
    "accompanying_symptoms": ["伴随症状"],
    "medical_history": ["相关病史，如'有糖尿病史'、'无外伤史'、'否认高血压史'等"],
    "urgency_level": "normal/high/critical",
    "is_emergency": true或false
}}

注意事项：
1. **最重要**：symptoms字段要提取用户描述中的所有症状原词，不要遗漏任何症状！
   - 例如用户说"头壳痛得要命"，symptoms应该是["头壳痛"]（保留原词，不要标准化）
   - 例如用户说"肚子疼，发烧，恶心"，symptoms应该是["肚子疼", "发烧", "恶心"]（保留原词）
   - 去除修饰词：如"要命"、"有点"、"稍微"等程度修饰词，只保留症状核心词
2. **否定词处理**（非常重要！）：
   - 症状否定：用户说"不咳嗽"、"没有发烧"、"不头疼"、"否认发烧"等等，表示排除该症状
     → 记录在 negative_symptoms（保留原词，如"发烧"、"咳嗽"）
     → 例如："我不咳嗽" → negative_symptoms=["咳嗽"]
     → 例如："没有发烧，也不头痛" → negative_symptoms=["发烧", "头痛"]
   - 病史否定：用户说"无外伤史"、"否认高血压史"、"没有糖尿病史"等等，表示否认该病史
     → 记录在 medical_history，格式为"否认XXX史"或"无XXX史"
     → 例如："无外伤史" → medical_history=["否认外伤史"]
     → 例如："否认高血压史" → medical_history=["否认高血压史"]
3. 如果发现以下"红旗征象"（任一），必须将 is_emergency 设为 true：胸痛，呼吸困难、意识不清、大出血、突发偏瘫、剧烈头痛、呕血、抽搐、昏迷
4. duration 填写具体时间，如"1天"、"3小时"、"1周"等
5. severity 可选：轻微、中等、严重、剧烈
6. 如果无法确定某字段，填写空字符串或空列表
7. 只输出JSON，不要输出其他内容"""


def get_llm_extractor() -> LLMSymptomExtractor:
    """获取 LLM 提取器单例"""
    global _llm_extractor
    if _llm_extractor is None:
        model_name = os.getenv("LLM_EXTRACTOR_MODEL", "qwen-turbo")
        _llm_extractor = LLMSymptomExtractor(model_name=model_name)
    return _llm_extractor


_llm_extractor: Optional[LLMSymptomExtractor] = None


def extract_symptoms_with_llm(user_input: str) -> Dict[str, Any]:
    """
    使用 LLM 提取症状（便捷函数）

    Args:
        user_input: 用户描述

    Returns:
        dict: 提取的症状信息
    """
    extractor = get_llm_extractor()
    result = extractor.extract(user_input)
    # 转换为字典
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"symptoms": [], "is_emergency": False}
