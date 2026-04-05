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
        return f"""你是一个专业的医疗导诊预检助手。你的任务是从用户的描述中提取所有医学症状信息。

请从以下用户描述中提取症状信息：

用户描述：{user_input}

请严格按照以下JSON格式输出，注意所有字段都是必填的：
{{
    "symptoms": ["用户描述的所有症状（保留原词）"],
    "negative_symptoms": ["排除的症状"],
    "location": "疼痛或不适部位",
    "duration": "持续时间",
    "severity": "严重程度",
    "triggers": ["诱因"],
    "accompanying_symptoms": ["伴随症状"],
    "medical_history": ["相关病史"],
    "urgency_level": "normal/high/critical",
    "is_emergency": true或false
}}

**提取规则（非常重要）：**

1. **symptoms 字段必须提取用户描述的所有症状，包括：**
   - 主要症状：如"头痛"、"肚子疼"、"发烧"
   - 皮肤表现：**水泡、皮疹、红斑、瘙痒、刺痛、红肿、结痂、疱疹**等（这些是重要症状！）
   - 全身症状：**发热、乏力、头晕、恶心、呕吐、腹泻**等
   - 局部症状：**咳嗽、胸闷、气短、尿频、尿痛**等
   - 疼痛描述：**刺痛、胀痛、绞痛、隐痛、酸痛**等
   - 感觉异常：**痒、麻木、刺痛感、灼热感**等

2. **不要遗漏任何症状！**
   - 用户说"身上长了透明的小水泡，有点痒，挠破有刺痛感"
   - symptoms 应该是：["水泡", "痒", "刺痛感"] 或 ["透明小水泡", "瘙痒", "刺痛"]
   - **不能只提取"头痛"而忽略皮肤症状！**

3. **否定词处理（非常重要！）：**
   - 用户说"不咳嗽"、"没有发烧"、"否认头痛" → negative_symptoms=["咳嗽", "发烧", "头痛"]
   - 用户说"没有既往史"、"无外伤史" → medical_history=["无既往史"]
   - **只有明确用否定词修饰的症状才放入 negative_symptoms**

4. **duration 填写具体时间**，如"1天"、"3小时"、"1周"等
5. **severity 可选**：轻微、中等、严重、剧烈
6. **红旗征象检测**：胸痛、呼吸困难、意识不清、大出血、突发偏瘫、剧烈头痛、呕血、抽搐、昏迷 → is_emergency=true
7. **只输出JSON，不要输出其他内容**"""


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
