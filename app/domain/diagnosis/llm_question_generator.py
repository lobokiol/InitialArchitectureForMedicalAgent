"""
基于 LLM 的追问生成器

使用 Qwen Turbo 生成个性化、知识驱动的追问
"""

from typing import List, Optional, Dict, Any
import os


class LLMBasedQuestionGenerator:
    """基于 LLM 的追问生成器"""

    def __init__(self, model_name: str = "qwen-turbo", temperature: float = 0.3):
        """
        初始化追问生成器

        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        from langchain_openai import ChatOpenAI

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

    def generate_question(
        self,
        symptoms: List[str],
        missing_slots: List[str],
        associated_symptoms: Optional[List[str]] = None,
        recommended_departments: Optional[List[str]] = None,
        conversation_history: str = "",
    ) -> str:
        """
        生成个性化追问

        Args:
            symptoms: 已识别的症状列表
            missing_slots: 缺失的槽位
            associated_symptoms: 知识图谱返回的伴随症状
            recommended_departments: 推荐的科室
            conversation_history: 对话历史

        Returns:
            追问问题
        """
        # 获取判别性症状用于动态追问
        discriminative = self._get_discriminative_symptoms(
            symptoms, associated_symptoms
        )

        prompt = self._build_prompt(
            symptoms=symptoms,
            missing_slots=missing_slots,
            associated_symptoms=associated_symptoms,
            recommended_departments=recommended_departments,
            conversation_history=conversation_history,
            discriminative_symptoms=discriminative,
        )

        try:
            result = self.llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)

            # 清理输出
            content = content.strip()
            # 如果有JSON标记，尝试提取
            if "{" in content and "}" in content:
                import json
                import re

                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group())
                        return data.get("question", content)
                    except:
                        pass

            return content

        except Exception as e:
            print(f"LLM 追问生成失败: {e}")
            # 返回默认追问
            return self._fallback_question(missing_slots, associated_symptoms)

    def _get_discriminative_symptoms(
        self,
        known_symptoms: List[str],
        candidate_symptoms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取判别性症状 - 用于动态生成追问

        Args:
            known_symptoms: 已知的症状
            candidate_symptoms: 候选症状列表

        Returns:
            判别性症状列表
        """
        from app.tools.knowledge_graph_tool import (
            get_discriminative_symptoms as _get_disc,
        )

        candidates = candidate_symptoms if candidate_symptoms else []
        if not candidates:
            # 从知识图谱获取扩展症状
            from app.tools.knowledge_graph_tool import query_hybrid_retrieval

            result = query_hybrid_retrieval("", known_symptoms)
            candidates = result.get("expanded_symptoms", [])[:15]

        discriminative = _get_disc(known_symptoms, candidates, limit=5)
        return discriminative

    def _build_prompt(
        self,
        symptoms: List[str],
        missing_slots: List[str],
        associated_symptoms: Optional[List[str]] = None,
        recommended_departments: Optional[List[str]] = None,
        conversation_history: str = "",
        discriminative_symptoms: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建追问提示词"""

        symptoms_str = "、".join(symptoms) if symptoms else "无"
        associated_str = (
            "、".join(associated_symptoms[:5]) if associated_symptoms else "无"
        )
        departments_str = (
            "、".join(recommended_departments[:3]) if recommended_departments else "无"
        )
        missing_str = "、".join(missing_slots) if missing_slots else "无"

        # 构建判别性症状信息
        discriminative_str = "无"
        if discriminative_symptoms:
            discs = []
            for d in discriminative_symptoms:
                depts = "、".join(d.get("departments", []))
                discs.append(f"  - {d['name']}: 可区分科室[{depts}]")
            discriminative_str = "\n".join(discs)

        return f"""你是一个资深的导诊医生。根据已提取的症状和缺失信息，你需要通过追问来缩小疾病范围。

## 已收集信息
- 用户症状: {symptoms_str}
- 可能的伴随症状: {associated_str}
- 可能科室: {departments_str}
- 需要追问的信息: {missing_str}

## 判别性症状（可帮助区分科室）
{discriminative_str}

## 对话历史
{conversation_history or "（首次对话）"}

## 追问要求
1. 一次只问 1-2 个最关键的问题
2. 优先询问判别性症状（能帮助区分不同科室的症状）
3. 结合知识图谱的伴随症状进行针对性追问
4. 根据可能科室问该科室关注的重点问题
5. 问题要简洁、口语化
6. 不要重复用户已经说过的信息

## 科室关注点参考
- 消化内科: 关注腹痛性质、频率、进食关系、恶心呕吐
- 心血管内科: 关注胸痛性质、诱因、持续时间、出汗
- 呼吸内科: 关注咳嗽、咳痰、发热、呼吸困难
- 神经内科: 关注头痛性质、部位、伴随症状
- 骨科: 关注疼痛部位、活动受限、外伤史

## 输出格式
请直接输出问题，不要加任何格式前缀：
"""

    def _fallback_question(
        self,
        missing_slots: List[str],
        associated_symptoms: Optional[List[str]] = None,
    ) -> str:
        """默认追问（当LLM失败时）"""
        from app.domain.diagnosis.questions import get_next_question

        return get_next_question(missing_slots, associated_symptoms)


# 单例
_question_generator: Optional[LLMBasedQuestionGenerator] = None


def get_question_generator() -> LLMBasedQuestionGenerator:
    """获取追问生成器单例"""
    global _question_generator
    if _question_generator is None:
        _question_generator = LLMBasedQuestionGenerator()
    return _question_generator


def generate_question_with_llm(
    symptoms: List[str],
    missing_slots: List[str],
    associated_symptoms: Optional[List[str]] = None,
    recommended_departments: Optional[List[str]] = None,
    conversation_history: str = "",
) -> str:
    """
    使用 LLM 生成个性化追问（便捷函数）

    Args:
        symptoms: 已识别的症状
        missing_slots: 缺失的槽位
        associated_symptoms: 伴随症状
        recommended_departments: 推荐科室
        conversation_history: 对话历史

    Returns:
        追问问题
    """
    generator = get_question_generator()
    return generator.generate_question(
        symptoms=symptoms,
        missing_slots=missing_slots,
        associated_symptoms=associated_symptoms,
        recommended_departments=recommended_departments,
        conversation_history=conversation_history,
    )
