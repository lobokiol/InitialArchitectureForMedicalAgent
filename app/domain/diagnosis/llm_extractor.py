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

    symptoms: List[str] = Field(default_factory=list, description="识别的症状列表")
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
    standardized_symptoms: List[str] = Field(
        default_factory=list, description="标准化后的症状"
    )


class LLMSymptomExtractor:
    """基于 LLM 的症状提取器"""

    def __init__(self, model_name: str = "qwen-turbo", temperature: float = 0.1):
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
    "symptoms": ["所有症状，包括主要症状和伴随症状"],
    "location": "疼痛部位",
    "duration": "持续时间",
    "severity": "严重程度",
    "triggers": ["诱因1"],
    "accompanying_symptoms": ["伴随症状"],
    "medical_history": ["相关病史"],
    "urgency_level": "normal/high/critical",
    "is_emergency": true或false,
    "standardized_symptoms": ["标准化的症状1", "标准化的症状2"]
}}

注意事项：
1. **最重要**：symptoms字段要包含用户描述中的所有症状，不要遗漏任何症状！
   - 例如用户说"胸口疼还出汗"，symptoms应该是["胸痛", "出汗"]
   - 例如用户说"肚子疼，发烧，恶心"，symptoms应该是["腹痛", "发热", "恶心"]
2. 如果发现以下"红旗征象"（任一），必须将 is_emergency 设为 true：胸痛、呼吸困难、意识不清、大出血、突发偏瘫、剧烈头痛、呕血、抽搐、昏迷
3. 症状尽量使用标准医学术语（如"肚子疼"→"腹痛"，"心跳快"→"心悸"）
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
