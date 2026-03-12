"""
知识图谱校验器 - Layer 4: 验证 + 扩展 + 消歧

功能:
1. 验证症状是否在知识图谱中存在
2. 用伴随症状关系扩展症状列表
3. 标记不确定症状
4. 处理同义词消歧

用法:
    from app.domain.diagnosis.kg_validator import KGValidator, get_kg_validator

    validator = get_kg_validator()
    result = validator.validate(symptoms=["头痛", "发热"], user_input="我头疼还发烧")
"""

from typing import List, Dict, Any, Optional, Set
from app.domain.diagnosis.slots import DiagnosisSlots


class ValidationResult:
    """校验结果"""

    def __init__(
        self,
        valid_symptoms: List[str] = None,
        uncertain_symptoms: List[str] = None,
        expanded_symptoms: List[str] = None,
        symptom_sources: Dict[str, str] = None,
        confidence_score: float = 1.0,
    ):
        self.valid_symptoms: List[str] = valid_symptoms or []
        self.uncertain_symptoms: List[str] = uncertain_symptoms or []
        self.expanded_symptoms: List[str] = expanded_symptoms or []
        self.symptom_sources: Dict[str, str] = symptom_sources or {}
        self.confidence_score: float = confidence_score

    def to_dict(self) -> dict:
        return {
            "valid_symptoms": self.valid_symptoms,
            "uncertain_symptoms": self.uncertain_symptoms,
            "expanded_symptoms": self.expanded_symptoms,
            "symptom_sources": self.symptom_sources,
            "confidence_score": self.confidence_score,
        }


class KGValidator:
    """知识图谱校验器"""

    def __init__(self):
        self._neo4j_client = None

    def _get_neo4j_client(self):
        """获取Neo4j客户端"""
        if self._neo4j_client is None:
            try:
                from app.infra.neo4j_client import get_neo4j_client as _get_client

                self._neo4j_client = _get_client()
            except Exception as e:
                print(f"Neo4j客户端初始化失败: {e}")
                self._neo4j_client = None
        return self._neo4j_client

    def is_available(self) -> bool:
        """检查KG是否可用"""
        client = self._get_neo4j_client()
        return client is not None and client._driver is not None

    def validate(
        self,
        symptoms: List[str],
        user_input: str = "",
        symptom_sources: Dict[str, str] = None,
    ) -> ValidationResult:
        """
        校验症状列表

        Args:
            symptoms: 待校验的症状列表
            user_input: 用户原始输入（用于消歧）
            symptom_sources: 症状来源映射 {"症状": "来源"}

        Returns:
            ValidationResult: 校验结果
        """
        if not self.is_available():
            return ValidationResult(
                valid_symptoms=symptoms,
                uncertain_symptoms=[],
                expanded_symptoms=[],
                symptom_sources=symptom_sources or {},
                confidence_score=0.5,  # KG不可用，降低置信度
            )

        client = self._get_neo4j_client()
        symptom_sources = symptom_sources or {}

        # Step 1: 验证症状是否在KG中存在
        valid, uncertain = self._verify_in_graph(client, symptoms)

        # Step 2: 扩展症状（基于伴随症状关系）
        expanded = self._expand_from_graph(client, valid)

        # Step 3: 计算置信度
        confidence = self._calculate_confidence(valid, uncertain, expanded)

        # Step 4: 合并来源信息
        final_sources = self._merge_sources(symptom_sources, valid, uncertain, expanded)

        return ValidationResult(
            valid_symptoms=valid,
            uncertain_symptoms=uncertain,
            expanded_symptoms=expanded,
            symptom_sources=final_sources,
            confidence_score=confidence,
        )

    def _verify_in_graph(
        self, client, symptoms: List[str]
    ) -> tuple[List[str], List[str]]:
        """验证症状是否在知识图谱中"""
        valid = []
        uncertain = []

        for symptom in symptoms:
            # 尝试精确匹配
            result = client.query_departments_by_symptom(symptom)
            if result:
                valid.append(symptom)
                continue

            # 尝试模糊匹配（关键词搜索）
            matched = client.query_symptoms_by_keyword(symptom)
            if matched:
                # 如果有精确匹配，使用匹配结果
                exact_match = [m for m in matched if m == symptom]
                if exact_match:
                    valid.append(symptom)
                else:
                    # 有近似匹配，标记为不确定
                    uncertain.append(symptom)
            else:
                # 完全没有匹配
                uncertain.append(symptom)

        return valid, uncertain

    def _expand_from_graph(self, client, symptoms: List[str]) -> List[str]:
        """用图关系扩展症状"""
        if not symptoms:
            return []

        expanded = []
        seen = set(symptoms)

        for symptom in symptoms:
            # 查询伴随症状
            associated = client.query_associated_symptoms(symptom)
            for item in associated[:5]:  # 最多取5个
                name = item.get("name")
                if name and name not in seen:
                    expanded.append(name)
                    seen.add(name)

        return expanded

    def _calculate_confidence(
        self,
        valid: List[str],
        uncertain: List[str],
        expanded: List[str],
    ) -> float:
        """计算置信度"""
        total = len(valid) + len(uncertain)
        if total == 0:
            return 0.5

        # 有效症状占比
        valid_ratio = len(valid) / total

        # 扩展症状作为加分项
        expansion_bonus = min(len(expanded) * 0.05, 0.2)

        confidence = valid_ratio + expansion_bonus
        return min(max(confidence, 0.0), 1.0)

    def _merge_sources(
        self,
        existing_sources: Dict[str, str],
        valid: List[str],
        uncertain: List[str],
        expanded: List[str],
    ) -> Dict[str, str]:
        """合并来源信息"""
        sources = dict(existing_sources)

        for s in valid:
            if s not in sources:
                sources[s] = "kg_validated"

        for s in uncertain:
            if s not in sources:
                sources[s] = "kg_uncertain"

        for s in expanded:
            if s not in sources:
                sources[s] = "kg_expanded"

        return sources

    def disambiguate(
        self,
        uncertain_symptom: str,
        context_symptoms: List[str],
    ) -> List[Dict[str, Any]]:
        """
        消歧 - 处理可能的同义词问题

        Args:
            uncertain_symptom: 不确定的症状
            context_symptoms: 上下文症状（已有的确认症状）

        Returns:
            可能的解释列表: [{"symptom": "症状名", "confidence": 0.8}]
        """
        if not self.is_available():
            return [{"symptom": uncertain_symptom, "confidence": 0.5}]

        client = self._get_neo4j_client()

        # 使用向量搜索找到相似症状
        matches = client.semantic_match_symptoms(
            uncertain_symptom, top_k=3, threshold=0.3
        )

        results = []
        for match in matches:
            results.append(
                {
                    "symptom": match["name"],
                    "confidence": match["score"],
                    "source": "vector",
                }
            )

        # 如果没有向量匹配，尝试关键词搜索
        if not results:
            matched = client.query_symptoms_by_keyword(uncertain_symptom)
            for m in matched[:3]:
                results.append({"symptom": m, "confidence": 0.5, "source": "keyword"})

        return results

    def apply_to_slots(
        self,
        slots: DiagnosisSlots,
        user_input: str = "",
    ) -> DiagnosisSlots:
        """
        将校验结果应用到 DiagnosisSlots

        Args:
            slots: 原始slots
            user_input: 用户输入

        Returns:
            更新后的slots
        """
        # 合并所有已知症状
        all_symptoms = (
            slots.symptoms + slots.uncertain_symptoms + slots.expanded_symptoms
        )

        # 获取来源
        sources = slots.symptom_sources.copy()

        # 校验
        result = self.validate(all_symptoms, user_input, sources)

        # 更新slots
        slots.symptoms = result.valid_symptoms
        slots.uncertain_symptoms = result.uncertain_symptoms
        slots.expanded_symptoms = result.expanded_symptoms
        slots.symptom_sources = result.symptom_sources
        slots.confidence_score = result.confidence_score
        slots.validated = True

        return slots


# 单例
_kg_validator: Optional[KGValidator] = None


def get_kg_validator() -> KGValidator:
    """获取KG校验器单例"""
    global _kg_validator
    if _kg_validator is None:
        _kg_validator = KGValidator()
    return _kg_validator


def validate_symptoms(
    symptoms: List[str],
    user_input: str = "",
) -> ValidationResult:
    """
    校验症状（便捷函数）

    Args:
        symptoms: 症状列表
        user_input: 用户输入

    Returns:
        ValidationResult
    """
    validator = get_kg_validator()
    return validator.validate(symptoms, user_input)
